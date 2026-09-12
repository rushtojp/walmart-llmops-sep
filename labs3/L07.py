# %% [markdown]
# # L07 — Stand up Gemma end-to-end: LoRA fine-tune, MLflow, Unity Catalog, serve
# **Objective 8.**
#
# **Northfield Grocers context:** Supplier sheets arrive as free text; Northfield's catalog engine must extract brand, size and pack quantity as JSON for 38,000 SKUs, and later allergen and dietary flags. A LoRA-tuned Gemma 4 E4B, registered in Unity Catalog and served behind an OpenAI-compatible endpoint, is the target — gated by an eval harness the category team owns.
#
# **Retail use cases:** Attribute extraction from supplier product sheets; house-style product descriptions and promo copy that respect regulated-claim rules.
#
# **Platform:** Databricks GPU cluster (NC24ads_A100_v4), ML runtime with transformers, peft, trl, mlflow. Steps 1–3 run anywhere; Steps 4–7 need the cluster.
#
# **Done means:** hand-rolled LoRA update reproduces W+BA; fine-tuned endpoint passes the acceptance suite and the task-eval floor.
#
# ## Step 1 — Load and inspect the training data
# *Why:* the task is JSON attribute extraction from a title. Know the target schema before you tune anything.
# %%
train = [json.loads(l) for l in open(os.path.join(DATA_DIR, "train_prompts.jsonl"))]
evalset = [json.loads(l) for l in open(os.path.join(DATA_DIR, "eval_prompts.jsonl"))]
print(len(train), "train /", len(evalset), "eval"); print(train[0])
schema_ok = all(set(json.loads(t["target"])) == {"brand", "size", "pack_qty"} for t in train)
check(schema_ok, "every target has the 3-key schema")

# %% [markdown]
# ## Step 2 — LoRA by hand
# *Why (pedagogy-first):* LoRA freezes W and learns a low-rank update ΔW = B·A (r ≪ d). Show that the forward pass with the adapter equals the forward pass with the merged weight, which is exactly what "merge and unload" does before serving.
# %%
rng = np.random.default_rng(0); d_in, d_out, r = 512, 512, 8
W = rng.normal(0, 0.02, (d_out, d_in)).astype(np.float32)
A = rng.normal(0, 0.02, (r, d_in)).astype(np.float32); B = np.zeros((d_out, r), np.float32)   # B starts at zero, as in the paper
x = rng.normal(size=(16, d_in)).astype(np.float32)

def lora_forward(x, W, A, B, alpha=16, r=8):
    # >>> SOLUTION y = xWᵀ + (alpha/r) · x Aᵀ Bᵀ
    return x @ W.T + (alpha / r) * (x @ A.T) @ B.T
    # <<< SOLUTION

check(np.allclose(lora_forward(x, W, A, B), x @ W.T), "with B=0 the adapter is a no-op at init")
B = rng.normal(0, 0.02, (d_out, r)).astype(np.float32)   # pretend we trained
W_merged = W + (16 / 8) * B @ A
check(np.allclose(lora_forward(x, W, A, B), x @ W_merged.T, atol=1e-5), "adapter forward == merged-weight forward")
print(f"trainable params: LoRA {A.size + B.size:,} vs full {W.size:,} ({(A.size+B.size)/W.size:.2%})")

# %% [markdown]
# ## Step 3 — The evaluation harness
# *Why:* fine-tuning without a harness is guessing. Exact-match on parsed JSON, plus a per-key accuracy so you know *which* attribute is failing. This same harness is reused by L09's CI gate.
# %%
def evaluate(predict_fn, items):
    """predict_fn(prompt)->str JSON. Returns dict(exact_match, per_key_acc, parse_fail_rate)."""
    # >>> SOLUTION parse pred; count exact matches; per-key equality; parse failures
    em = 0; keys = {"brand": 0, "size": 0, "pack_qty": 0}; fail = 0
    for it in items:
        tgt = json.loads(it["target"])
        try: pred = json.loads(predict_fn(it["prompt"]))
        except Exception: fail += 1; continue
        em += pred == tgt
        for k in keys: keys[k] += pred.get(k) == tgt[k]
    n = len(items)
    return dict(exact_match=em / n, per_key_acc={k: v / n for k, v in keys.items()}, parse_fail_rate=fail / n)
    # <<< SOLUTION

def rule_based(prompt):   # a deterministic baseline so the harness is exercised in SMOKE mode
    t = prompt.split(": ", 1)[1].split()
    return json.dumps({"brand": t[0], "size": t[3] + " " + t[4], "pack_qty": int(t[-1])})

base = evaluate(rule_based, evalset); print(base)
check(base["parse_fail_rate"] == 0 and 0 < base["exact_match"] <= 1, "harness produces a valid baseline score")

# %% [markdown]
# ## Step 4 — Fine-tune with PEFT on the cluster
# *Why:* `peft.LoraConfig` on the attention and MLP projections of Gemma 4 E4B, `trl.SFTTrainer` over the 600 prompts, 2–3 epochs. Log everything to MLflow. API signatures are version-sensitive — the lab guide lists the exact calls to verify.
# %%
if LAB_MODE == "GPU":
    gpu_only("Follow lab guide §Step 4 (LoraConfig r=16, alpha=32, target_modules per model card; SFTTrainer; mlflow.autolog()).")
ft_metrics = {"train_loss_final": None, "epochs": None, "gpu_minutes": None}

# %% [markdown]
# ## Step 5 — Evaluate the fine-tuned model with the same harness
# *Why:* the acceptance floor is exact_match ≥ 0.80 on the 200-item eval (set by the trainer for this synthetic task). Compare with the rule-based baseline — a tuned LLM that loses to rules should not ship.
# %%
if LAB_MODE == "GPU":
    gpu_only("Wrap model.generate in predict_fn; run evaluate(); assign to `tuned`.")
tuned = None
FLOOR = 0.80
print("floor:", FLOOR, "| tuned:", tuned)

# %% [markdown]
# ## Step 6 — Register in Unity Catalog via MLflow
# *Why:* the registry is the hand-off point to serving and to the CI gate. `mlflow.set_registry_uri("databricks-uc")`, log the merged model with `mlflow.transformers.log_model`, then `mlflow.register_model(..., "northfield.llmops.gemma4_e4b_attr")`. Names are three-level (catalog.schema.model).
# %%
if LAB_MODE == "GPU":
    gpu_only("Register per lab guide §Step 6; record the version number here.")
registered_version = None

# %% [markdown]
# ## Step 7 — Serve and run the acceptance suite
# *Why:* Databricks Model Serving (GPU endpoint) or vLLM on the cluster exposes an OpenAI-compatible route. The acceptance suite is: 20 golden prompts pass, one malformed prompt returns a controlled error, p95 latency under the SLA you set in L03.
# %%
acceptance = {"golden_20_pass": None, "malformed_handled": None, "p95_ms_under_sla": None}
if LAB_MODE == "GPU":
    gpu_only("Run acceptance suite per lab guide §Step 7 and fill `acceptance`.")
else:
    print("SMOKE: harness, LoRA maths and data checks verified; endpoint steps run on the cluster.")
check(callable(evaluate) and W_merged.shape == W.shape, "core mechanics verified")
print("L07 complete.")
