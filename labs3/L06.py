# %% [markdown]
# # L06 — Mixture-of-Experts serving
# **Objective 7.**
#
# **Northfield Grocers context:** One assistant model must serve store associates, customers and the contact centre at once. Northfield is weighing Gemma 4 26B A4B (MoE) against dense 31B: 31B-class quality at ~4B active compute — if the router stays balanced and the memory fits.
#
# **Retail use cases:** Multi-surface assistant (handhelds, app, contact centre) on a fixed GPU budget; choosing dense vs MoE for the nightly product-description job.
#
# **Platform:** Databricks GPU cluster (NC24ads_A100_v4). Steps 1–4 are NumPy; Steps 5–6 serve real models via vLLM.
#
# **Done means:** router load-imbalance metric computed; capacity-factor drop rate reported; MoE-vs-dense comparison table filled from measurements.
#
# ## Step 1 — A top-k router
# *Why:* an MoE layer is just N small MLPs plus a router that picks k of them per token. The router is a linear layer + softmax + top-k; write it.
# %%
rng = np.random.default_rng(0)
n_tokens, hidden, n_experts, k = 4096, 256, 8, 2
X = rng.normal(size=(n_tokens, hidden)).astype(np.float32)
W_router = rng.normal(0, 0.05, size=(hidden, n_experts)).astype(np.float32)

def route(X, W_router, k):
    """Return (expert_idx [n,k], gate_weights [n,k]) — softmax over experts, then top-k renormalised."""
    # >>> SOLUTION logits = X@W; p = softmax; idx = argpartition top-k; w = p[idx] / sum
    logits = X @ W_router
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    idx = np.argpartition(-p, k, axis=1)[:, :k]
    w = np.take_along_axis(p, idx, 1); w /= w.sum(1, keepdims=True)
    return idx, w
    # <<< SOLUTION

idx, gate = route(X, W_router, k)
check(idx.shape == (n_tokens, k) and np.allclose(gate.sum(1), 1), "router returns k experts per token with normalised gates")

# %% [markdown]
# ## Step 2 — Measure load imbalance
# *Why:* if one expert receives most tokens, the GPU holding it is the bottleneck and the others idle. Serving frameworks report this; you should be able to compute it: `max_load / mean_load` and the coefficient of variation.
# %%
def load_stats(idx, n_experts):
    # >>> SOLUTION counts per expert via bincount; return dict(counts, max_over_mean, cv)
    counts = np.bincount(idx.ravel(), minlength=n_experts)
    return dict(counts=counts, max_over_mean=counts.max() / counts.mean(), cv=counts.std() / counts.mean())
    # <<< SOLUTION

st = load_stats(idx, n_experts); print("expert token counts:", st["counts"], f"max/mean={st['max_over_mean']:.2f} cv={st['cv']:.2f}")
# plant a skewed router and show the metric catching it
W_skew = W_router.copy(); W_skew[:, 0] += 0.3
st_skew = load_stats(route(X, W_skew, k)[0], n_experts); print("skewed counts:", st_skew["counts"], f"max/mean={st_skew['max_over_mean']:.2f}")
check(st_skew["max_over_mean"] > st["max_over_mean"] * 1.5, "imbalance metric detects the skewed router")

# %% [markdown]
# ## Step 3 — Capacity factor and token dropping
# *Why:* training and some serving paths cap tokens per expert at `capacity = CF × tokens×k / experts`; overflow tokens are dropped (or routed to the next expert). Compute the drop rate for the balanced and skewed routers at CF = 1.0 and 1.25.
# %%
def drop_rate(idx, n_experts, cf):
    cap = int(cf * idx.size / n_experts)
    counts = np.bincount(idx.ravel(), minlength=n_experts)
    return np.clip(counts - cap, 0, None).sum() / idx.size

for cf in (1.0, 1.25):
    print(f"CF={cf}: balanced drop {drop_rate(idx, n_experts, cf):.3%}, skewed drop {drop_rate(route(X, W_skew, k)[0], n_experts, cf):.3%}")
check(drop_rate(route(X, W_skew, k)[0], n_experts, 1.0) > drop_rate(idx, n_experts, 1.0), "skew increases drops")

# %% [markdown]
# ## Step 4 — Active vs total parameters: compute and memory
# *Why:* the MoE promise is "31B quality at 4B FLOPs" — but memory holds all 26B. Compute per-token FLOPs (≈2×active params) and resident memory (all params) for the three lab models, and see which resource each is bound by.
# *Watch:* the parameter figures are from the public model cards as of Sep 2026; "~A4B" is Google's own notation for ~4B active. Re-verify.
# %%
models = pd.DataFrame([
  dict(model="Gemma 4 31B Dense", total_b=31, active_b=31),
  dict(model="Gemma 4 26B A4B (MoE)", total_b=26, active_b=4),
  dict(model="Nemotron 3 Nano (hybrid MoE, ~30B total, ~3B active)", total_b=30, active_b=3),
])
models["gflops_per_token"] = 2 * models.active_b        # 2 FLOPs per param per token, in GFLOP (params in B)
models["bf16_weights_gb"] = models.total_b * 2
models["compute_ratio_vs_dense"] = (models.active_b / 31).round(3)
print(models.to_string())
check(models.bf16_weights_gb.iloc[1] > 40 and models.compute_ratio_vs_dense.iloc[1] < 0.2, "MoE: memory close to dense, compute far below")

# %% [markdown]
# ## Step 5 — Serve the real models with vLLM (GPU cluster)
# *Why:* measure it. Start vLLM for each model (lab guide §Step 5 — expert parallelism flags are version-sensitive), then reuse the L03 harness at concurrency 8 and 32.
# %%
if LAB_MODE == "GPU":
    gpu_only("Serve google/gemma-4-26b-a4b-it, nvidia/nemotron-3-nano (verify ids), google/gemma-4-31b-it; run L03 harness; fill `serve`.")
serve = pd.DataFrame(columns=["model", "concurrency", "throughput_tok_s", "ttft_p95_ms", "gpu_mem_gb"])

# %% [markdown]
# ## Step 6 — MoE-vs-dense recommendation (the deliverable)
# *Why:* one page for one of the team's own workloads: which is cheaper per million tokens at the concurrency they actually run, and why.
# %%
print(serve.to_string()); print("Compute/memory table from Step 4 is the analytic half; Step 5 is the measured half.")
check("throughput_tok_s" in serve.columns, "comparison table schema in place")
print("L06 complete." + ("" if LAB_MODE == "GPU" else " (SMOKE: Steps 5–6 need the GPU cluster.)"))
