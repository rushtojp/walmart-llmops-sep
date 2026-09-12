# %% [markdown]
# # L01 — Platform baseline & GPU sizing
# **Objectives 1–2.**
#
# **Northfield Grocers context:** Northfield Grocers is standing up three LLM systems — a store-ops SOP assistant for 24,000 associates on handhelds, a customer shopping assistant in the app, and a catalog engine for 38,000 SKUs. Before any model is chosen, platform engineering must size GPU memory for the dense, MoE and encoder workloads those systems need.
#
# **Retail use cases:** Sizing the GPU fleet for the customer assistant ahead of the holiday peak; deciding A100 vs H100 for the store-ops assistant under a 1-second TTFT target on handhelds.
#
# **Platform:** Azure ML compute instance *and* Azure Databricks GPU cluster (Standard_NC24ads_A100_v4). Steps 1–2 use the platform; Steps 3–5 are pure Python and run anywhere.
#
# **Done means:** memory-fit calculator returns the correct GPU count for all 4 planted workloads; nvidia-smi parser passes on the sample.
#
# ## Step 1 — Read GPU telemetry
# *Why:* every later lab starts from "how much memory do I actually have?". `nvidia-smi --query-gpu` gives a CSV we parse rather than eyeball.
# *What:* on GPU compute we run the real command; in SMOKE mode we parse the provided sample so the parser is tested either way.
# %%
def parse_nvidia_smi(text: str) -> pd.DataFrame:
    """Parse `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv` output."""
    # >>> SOLUTION Parse the CSV header, strip units ("MiB", "%") and return numeric columns total_mib, used_mib, util_pct
    rows = [r.strip() for r in text.strip().splitlines()]
    hdr = [h.strip() for h in rows[0].split(",")]
    df = pd.DataFrame([[c.strip() for c in r.split(",")] for r in rows[1:]], columns=hdr)
    out = pd.DataFrame({"index": df.iloc[:, 0].astype(int), "name": df.iloc[:, 1]})
    out["total_mib"] = df.iloc[:, 2].str.replace("MiB", "").astype(float)
    out["used_mib"]  = df.iloc[:, 3].str.replace("MiB", "").astype(float)
    out["util_pct"]  = df.iloc[:, 4].str.replace("%", "").astype(float)
    return out
    # <<< SOLUTION

if LAB_MODE == "GPU":
    import subprocess
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"],
                         capture_output=True, text=True).stdout
else:
    raw = open(os.path.join(DATA_DIR, "nvidia_smi_sample.csv")).read()
gpus = parse_nvidia_smi(raw); print(gpus)
check(gpus["total_mib"].iloc[0] > 0 and "used_mib" in gpus, "telemetry parsed with numeric memory columns")

# %% [markdown]
# ## Step 2 — Confirm the platform identity
# *Why:* the same notebook runs on Azure ML and Databricks; later labs branch on this. On Databricks `DATABRICKS_RUNTIME_VERSION` exists; on Azure ML compute instances `AZUREML_*`/`CI_NAME` variables exist.
# %%
platform = ("Databricks" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else
            "AzureML" if any(k.startswith("AZUREML") or k == "CI_NAME" for k in os.environ) else "Local/other")
print("Detected platform:", platform)
if platform == "Databricks":
    print("Runtime:", os.environ["DATABRICKS_RUNTIME_VERSION"])  # ML GPU runtime expected; see lab guide

# %% [markdown]
# ## Step 3 — Memory-fit calculator (weights + KV cache)
# *Why:* the single most common LLMOps failure is an OOM at deploy time. Weights are only part of the story: the KV cache grows with `batch × context × layers × 2 × (kv_heads × head_dim) × bytes`.
# *What:* implement the formula, then apply it to four planted workloads (dense 31B-class, MoE 26B/4B-active-class, an encoder classifier, and a long-context dense model).
# *Watch:* MoE models still need **all** expert weights resident — active parameters reduce compute, not memory.
# %%
def memory_fit_gb(params_b, dtype_bytes, n_layers, hidden, n_kv_heads, head_dim, ctx_len, batch, kv_dtype_bytes=2, overhead=0.10):
    """Return (weights_gb, kv_gb, total_gb). Uses grouped-query KV sizing (n_kv_heads*head_dim per layer per token, K and V)."""
    # >>> SOLUTION weights = params*bytes; kv per token per layer = 2 * n_kv_heads*head_dim*kv_bytes; total ×(1+overhead)
    weights = params_b * 1e9 * dtype_bytes / 1e9
    kv = 2 * n_layers * n_kv_heads * head_dim * kv_dtype_bytes * ctx_len * batch / 1e9
    return weights, kv, (weights + kv) * (1 + overhead)
    # <<< SOLUTION

def gpus_needed(total_gb, gpu_mem_gb, usable_fraction=0.90):
    return math.ceil(total_gb / (gpu_mem_gb * usable_fraction))

# Planted workloads — architecture numbers are ILLUSTRATIVE ("-class"); read the real model card before sizing production.
workloads = [
 dict(name="Dense 31B-class, bf16, ctx 8k, batch 8",  params_b=31, dtype_bytes=2, n_layers=60, hidden=5376, n_kv_heads=8,  head_dim=128, ctx_len=8192, batch=8),
 dict(name="MoE 26B total / 4B active, bf16, ctx 8k, batch 8", params_b=26, dtype_bytes=2, n_layers=40, hidden=3072, n_kv_heads=8, head_dim=128, ctx_len=8192, batch=8),
 dict(name="Encoder classifier 0.125B, fp16, seq 512, batch 256", params_b=0.125, dtype_bytes=2, n_layers=12, hidden=768, n_kv_heads=12, head_dim=64, ctx_len=512, batch=256),
 dict(name="Dense 31B-class, int4 weights, ctx 128k, batch 1", params_b=31, dtype_bytes=0.5, n_layers=60, hidden=5376, n_kv_heads=8, head_dim=128, ctx_len=131072, batch=1),
]
rows = []
for w in workloads:
    wg, kv, tot = memory_fit_gb(**{k: v for k, v in w.items() if k != "name"})
    rows.append(dict(workload=w["name"], weights_gb=round(wg, 1), kv_gb=round(kv, 1), total_gb=round(tot, 1),
                     a100_80=gpus_needed(tot, 80), h100_94=gpus_needed(tot, 94), t4_16=gpus_needed(tot, 16)))
fit = pd.DataFrame(rows); print(fit.to_string())
# Expected values derived by hand from the formula (see lab guide, Step 3 worked example)
check(list(fit.a100_80) == [2, 1, 1, 1], "A100-80GB counts match the hand-derived expectation")

# %% [markdown]
# ## Step 4 — Compare GPUs by workload attribute
# *Why:* memory decides feasibility; interconnect and compute decide cost per token. The table is built from the SKU sheet in `content.py` so it stays in sync with the cost workbook.
# *Watch:* prices are approximate, region-specific and change monthly — the workbook holds the editable anchor.
# %%
sys.path.insert(0, os.path.join(DATA_DIR, "..")); from content import GPU_SKUS
sku = pd.DataFrame(GPU_SKUS, columns=["sku", "gpus", "gpu_mem_gb", "vm_usd_hr", "source"])
sku["usd_per_gpu_hr"] = sku.vm_usd_hr / sku.gpus
sku["usd_per_gb_hr"] = sku.vm_usd_hr / (sku.gpus * sku.gpu_mem_gb)
print(sku[["sku", "gpus", "gpu_mem_gb", "vm_usd_hr", "usd_per_gpu_hr", "usd_per_gb_hr"]].round(4).to_string())

# %% [markdown]
# ## Step 5 — Write the decision record
# *Why:* the deliverable is a defensible choice per workload, not a table. One line each: the constraint that decided it.
# %%
decision = {w["name"]: ("fits one A100-80GB → single-GPU SKU" if fit.a100_80[i] == 1 else f"needs {fit.a100_80[i]}× A100 → multi-GPU SKU or int4/H100")
            for i, w in enumerate(workloads)}
for k, v in decision.items(): print(f"- {k}: {v}")
check(len(decision) == 4, "decision record has one line per workload")
print("L01 complete.")
