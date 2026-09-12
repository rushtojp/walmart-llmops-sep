# %% [markdown]
# # L02 — Shipping formats: safetensors, AWQ/GPTQ, GGUF, ONNX
# **Objective 3.**
#
# **Northfield Grocers context:** Northfield wants the catalog attribute-extraction model on in-store handhelds and shelf-edge devices (CPU, GGUF), on the GPU pool for nightly batch (AWQ), and the taxonomy classifier as ONNX inside the PIM pipeline. Same model family, three shipping formats — and the quality of each must be measured, not assumed.
#
# **Retail use cases:** Shipping a compact model to store handhelds for offline product look-ups; shrinking the A100 footprint of the nightly supplier-sheet extraction job.
#
# **Platform:** Azure Databricks GPU cluster (NC24ads_A100_v4). Steps 1–3 are hand-rolled and run anywhere; Steps 4–6 need the GPU and model-hub access.
#
# **Done means:** hand-rolled int4 group quantiser reconstructs within tolerance; ONNX export round-trips; format table produced from measured sizes.
#
# ## Step 1 — Hand-roll group-wise weight quantisation
# *Why (pedagogy-first):* AWQ and GPTQ are both "group-wise int4 with a per-group scale, plus a cleverer choice of which weights to protect". If you can write the plain version you can read their papers.
# *What:* quantise a weight matrix in groups of 128 along the input dimension with a symmetric scale, dequantise, and measure relative error.
# %%
rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(1024, 4096)).astype(np.float32)   # a realistic-scale linear layer

def quant_groupwise(W, bits=4, group=128):
    """Symmetric per-group quantisation along axis=1. Returns (q_int, scales)."""
    # >>> SOLUTION reshape to (rows, groups, group); scale = absmax/(2^(bits-1)-1); q = round(W/scale) clipped
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(W.shape[0], -1, group)
    scales = np.abs(Wg).max(axis=2, keepdims=True) / qmax + 1e-12
    q = np.clip(np.round(Wg / scales), -qmax - 1, qmax).astype(np.int8)
    return q, scales.astype(np.float32)
    # <<< SOLUTION

def dequant(q, scales):
    return (q.astype(np.float32) * scales).reshape(q.shape[0], -1)

for bits in (8, 4, 3):
    q, s = quant_groupwise(W, bits=bits)
    rel = np.linalg.norm(W - dequant(q, s)) / np.linalg.norm(W)
    print(f"int{bits}: relative reconstruction error = {rel:.4f}; bytes ≈ {q.size*bits/8/1e6:.2f} MB vs fp16 {W.size*2/1e6:.2f} MB")
q4, s4 = quant_groupwise(W, 4)
q8, s8 = quant_groupwise(W, 8)
check(np.linalg.norm(W - dequant(q8, s8)) / np.linalg.norm(W) < 0.01 and np.linalg.norm(W - dequant(q4, s4)) / np.linalg.norm(W) < 0.15, "int8 error <1%, int4 group-128 error <15% (uniform int4 on Gaussian weights ≈ 11–12%)")

# %% [markdown]
# ## Step 2 — Why group size and outliers matter
# *Why:* a single outlier weight blows up the scale for its whole group; that is the problem AWQ (activation-aware scaling) and GPTQ (error-compensating rounding) solve differently.
# *What:* plant one outlier and watch group-256 error rise; shrink the group and watch it fall — the price of smaller groups is more scale values to store.
# %%
W2 = W.copy(); W2[0, 0] = 1.5   # planted outlier
for g in (256, 128, 64, 32):
    q, s = quant_groupwise(W2, 4, g); err = np.linalg.norm(W2[0] - dequant(q, s)[0]) / np.linalg.norm(W2[0])
    print(f"group {g:>3}: row-0 error {err:.4f}, scale values stored {s.size}")
q_big, s_big = quant_groupwise(W2, 4, 256); q_small, s_small = quant_groupwise(W2, 4, 32)
check(np.linalg.norm(W2[0]-dequant(q_small,s_small)[0]) < np.linalg.norm(W2[0]-dequant(q_big,s_big)[0]), "smaller groups reduce outlier damage")

# %% [markdown]
# ## Step 3 — ONNX export round-trip (encoder-style classifier)
# *Why:* ONNX is the format the catalog-NLP team ships encoder classifiers in (L08). Prove an export reproduces the source model's predictions before trusting any converter.
# *What:* in SMOKE mode export a scikit-learn model with `skl2onnx` and run it in `onnxruntime`; in GPU mode swap in the `transformers` → `optimum` export of the RoBERTa-class model (lab guide §Step 3).
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
cat = pd.read_csv(os.path.join(DATA_DIR, "catalog_items.csv")); cat = cat[cat.category != "UNKNOWN_CAT"]
X = cat.title.str.lower()   # lowercase in pandas: keeps the ONNX graph free of the locale-dependent StringNormalizer op
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=False), LogisticRegression(max_iter=500)).fit(X, cat.category)
ensure_packages(["onnx", "onnxruntime", "skl2onnx"])   # only these two labs need the ONNX stack; installs if absent, no restart
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnxruntime as ort
onnx_model = convert_sklearn(pipe, initial_types=[("title", StringTensorType([None, 1]))], options={id(pipe.steps[-1][1]): {"zipmap": False}})
onnx_path = "/tmp/northfield_taxonomy.onnx"; open(onnx_path, "wb").write(onnx_model.SerializeToString())
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
sample = X.head(200)
onnx_pred = sess.run(None, {"title": np.array(sample).reshape(-1, 1)})[0]
agree = (onnx_pred == pipe.predict(sample)).mean(); print(f"ONNX vs source agreement: {agree:.3f}; file {os.path.getsize(onnx_path)/1e6:.2f} MB")
check(agree >= 0.99, "ONNX export reproduces source predictions")

# %% [markdown]
# ## Step 4 — Real model: safetensors → AWQ → GGUF (GPU cluster)
# *Why:* the same measurement discipline on Gemma 4 E4B. Each converter is a separate library with its own version drift — pin versions in the cluster init script (lab guide §Setup).
# *What:* download the checkpoint (safetensors), quantise with AutoAWQ, convert to GGUF with llama.cpp's converter, record file sizes. Library APIs are version-sensitive: confirm signatures in the docs on the day.
# %%
if LAB_MODE == "GPU":
    gpu_only("Run the commands in lab guide §Step 4 (hf download → AWQ quantise → llama.cpp convert). Record sizes into `sizes` below.")
    sizes = {"safetensors_bf16": None, "awq_int4": None, "gguf_q4_k_m": None}   # fill from `ls -l` after each conversion
else:
    gpu_only("Model download and AWQ/GGUF conversion need a GPU cluster with model-hub access.")
    sizes = {"safetensors_bf16": None, "awq_int4": None, "gguf_q4_k_m": None}

# %% [markdown]
# ## Step 5 — Quality check per format
# *Why:* size without quality is meaningless. Run the same 50-prompt eval (from `eval_prompts.jsonl`) against each format's serving path and record exact-match on the attribute-extraction task.
# %%
evalset = [json.loads(l) for l in open(os.path.join(DATA_DIR, "eval_prompts.jsonl"))][:50]
def exact_match(pred, target): return json.loads(pred) == json.loads(target)
quality = {k: None for k in sizes}
if LAB_MODE == "GPU": gpu_only("Generate with each format's runtime and fill `quality` with exact-match over the 50 items.")
print("eval items loaded:", len(evalset))

# %% [markdown]
# ## Step 6 — The format table (the deliverable)
# *Why:* the team decides which format ships where — edge (GGUF), GPU serving (AWQ/safetensors), CPU batch (ONNX). The numbers are yours, not a vendor's.
# %%
table = pd.DataFrame({"format": list(sizes), "size_gb": list(sizes.values()), "exact_match_50": [quality.get(k) for k in sizes]})
print(table.to_string())
check(len(table) == 3, "format table has the three shipping formats")
print("L02 complete." + ("" if LAB_MODE == "GPU" else " (SMOKE: Steps 4–6 need the GPU cluster.)"))
