# %% [markdown]
# # L08 — Encoder classifiers at catalog scale (RoBERTa-class → ONNX → Spark)
# **Objective 9.**
#
# **Northfield Grocers context:** Every night Northfield classifies new and changed listings into its 8 top-level taxonomy categories. At this volume an encoder classifier on Spark costs cents per million items; an LLM zero-shot call costs dollars. Build the encoder path end-to-end and put the cost comparison on one line.
#
# **Retail use cases:** Nightly taxonomy classification of marketplace and private-label listings; policy screening of promo copy before publish.
#
# **Platform:** Databricks (T4 cluster; multi-node for Step 5). SMOKE mode trains a TF-IDF + logistic-regression classifier so the whole pipeline executes; GPU mode swaps in the RoBERTa-class fine-tune.
#
# **Done means:** macro-F1 above floor on held-out set; ONNX inference matches the trained model; Spark UDF scores the full set; cost-per-million-items derived from measured runtime.
#
# ## Step 1 — Clean and split the catalog
# *Why:* the catalog carries planted defects (D1 duplicates, D2 near-duplicates, D4 bad labels). A classifier trained on leaked duplicates reports a fake F1. Clean first, split by item, assert the cleaning.
# %%
from sklearn.model_selection import train_test_split
cat = pd.read_csv(os.path.join(DATA_DIR, "catalog_items.csv"))
n0 = len(cat)
# >>> SOLUTION drop exact dupes on item_id+title; normalise title (strip/lower) and drop dupes on normalised title; drop UNKNOWN_CAT
cat = cat.drop_duplicates(["item_id", "title"])
cat["title_norm"] = cat.title.str.strip().str.lower()
cat = cat.drop_duplicates("title_norm")
cat = cat[cat.category != "UNKNOWN_CAT"].reset_index(drop=True)
# <<< SOLUTION
print(f"{n0} → {len(cat)} rows after cleaning")
check((cat.category == "UNKNOWN_CAT").sum() == 0 and not cat.title_norm.duplicated().any(), "D1/D2/D4 removed")
tr, te = train_test_split(cat, test_size=0.2, stratify=cat.category, random_state=0)

# %% [markdown]
# ## Step 2 — Train the classifier
# *Why:* the encoder path. In GPU mode: `AutoModelForSequenceClassification` from a RoBERTa-class checkpoint, 3 epochs, `Trainer`. In SMOKE mode: the linear baseline — which is also the number a RoBERTa fine-tune must beat to justify its GPU cost.
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report
if LAB_MODE == "GPU":
    gpu_only("Fine-tune RoBERTa-class model per lab guide §Step 2 and export with optimum to /tmp/roberta_taxonomy.onnx; keep the sklearn baseline for comparison.")
# lowercase in pandas and export with lowercase=False: keeps the ONNX graph free of the locale-dependent StringNormalizer op
baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=False), LogisticRegression(max_iter=1000, C=5)).fit(tr.title_norm, tr.category)
pred = baseline.predict(te.title_norm); f1 = f1_score(te.category, pred, average="macro")
print(f"baseline macro-F1 = {f1:.3f}"); print(classification_report(te.category, pred, digits=3))
FLOOR = 0.90
check(f1 >= FLOOR, f"macro-F1 ≥ {FLOOR} floor")

# %% [markdown]
# ## Step 3 — Export to ONNX and verify equivalence
# *Why:* ONNX Runtime (CPU or TensorRT execution provider on GPU) is how the classifier ships. An export that silently changes predictions is a production incident; assert agreement on the full test set.
# %%
ensure_packages(["onnx", "onnxruntime", "skl2onnx"])   # only these two labs need the ONNX stack; installs if absent, no restart
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnxruntime as ort
onnx_path = "/tmp/northfield_taxonomy_l08.onnx"
onx = convert_sklearn(baseline, initial_types=[("title", StringTensorType([None, 1]))], options={id(baseline.steps[-1][1]): {"zipmap": False}})
open(onnx_path, "wb").write(onx.SerializeToString())
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_pred = sess.run(None, {"title": np.array(te.title_norm).reshape(-1, 1)})[0]
agree = (onnx_pred == pred).mean(); print(f"ONNX agreement on test set: {agree:.4f}")
check(agree >= 0.995, "ONNX matches trained model")

# %% [markdown]
# ## Step 4 — Measure single-node throughput
# *Why:* cost per million items comes from items/second, not from a vendor slide. Time the ONNX session on a 50k-row batch (titles tiled from the catalog).
# %%
big = pd.Series(np.tile(cat.title_norm.values, 20))   # ~50k normalised titles
t0 = time.perf_counter(); _ = sess.run(None, {"title": np.array(big).reshape(-1, 1)})[0]; dt = time.perf_counter() - t0
ips = len(big) / dt; print(f"{len(big):,} items in {dt:.2f}s → {ips:,.0f} items/s (single node, CPU)")
check(ips > 1000, "single-node throughput measured")

# %% [markdown]
# ## Step 5 — Scale out with a Spark pandas UDF
# *Why:* millions of items means Spark. A pandas UDF loads the ONNX session once per executor (module-level cache) and scores batches. Runs in local mode here; on Databricks the same code fans out over workers.
# *Watch:* Arrow batch size (`spark.sql.execution.arrow.maxRecordsPerBatch`) controls the per-call batch — too small wastes session overhead, too large risks executor memory.
# %%
ensure_packages(["pyarrow"])   # pandas UDFs need Arrow; present on Databricks, often absent on local venvs
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
spark = SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[1]")).appName("L08").config("spark.ui.enabled", "false").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "5000")
sdf = spark.createDataFrame(pd.DataFrame({"title": big}))

_SESS = {}
@pandas_udf(StringType())
def classify_udf(titles: pd.Series) -> pd.Series:
    # >>> SOLUTION cache session per executor process; run onnxruntime on the batch; return Series
    import onnxruntime as _ort, numpy as _np
    if "s" not in _SESS: _SESS["s"] = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = _SESS["s"].run(None, {"title": _np.array(titles).reshape(-1, 1)})[0]
    return pd.Series(out)
    # <<< SOLUTION

t0 = time.perf_counter(); scored = sdf.withColumn("category", classify_udf("title")); n_scored = scored.count(); dt_spark = time.perf_counter() - t0
dist = scored.groupBy("category").count().toPandas().sort_values("count", ascending=False)
print(f"Spark scored {n_scored:,} rows in {dt_spark:.1f}s"); print(dist.to_string(index=False))
check(n_scored == len(big) and dist["count"].sum() == len(big), "Spark UDF scored the full set")

# %% [markdown]
# ## Step 6 — Cost per million items, and the LLM zero-shot comparison
# *Why:* the business question. Encoder cost = (VM $/hr ÷ measured items/hr) × 1e6. LLM zero-shot cost = tokens per item × $/token (per-token or provisioned-throughput rate). Enter your measured encoder throughput on the GPU cluster and the LLM token cost from your platform's pricing page; the formula is what matters.
# *Watch:* the SKU price below comes from `content.py` and is approximate; the workbook holds the editable anchor.
# %%
sys.path.insert(0, os.path.join(DATA_DIR, "..")); from content import GPU_SKUS
vm_usd_hr = dict((s[0], s[3]) for s in GPU_SKUS)["Standard_NC4as_T4_v3"]
encoder_cost_per_M = vm_usd_hr / (ips * 3600) * 1e6
tokens_per_item, usd_per_M_tokens = 60, 0.50    # placeholder LLM economics — replace with the platform's published rates on the day
llm_cost_per_M = tokens_per_item * usd_per_M_tokens
print(f"Encoder (this node's measured rate, T4 SKU price): ${encoder_cost_per_M:,.2f} per million items")
print(f"LLM zero-shot at {tokens_per_item} tok/item and ${usd_per_M_tokens}/M tok: ${llm_cost_per_M:,.2f} per million items")
check(encoder_cost_per_M > 0, "cost per million items derived from measured runtime")
spark.stop(); print("L08 complete.")
