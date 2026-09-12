# %% [markdown]
# # L05 — GPU kernel optimisation: FlashAttention & a fused kernel
# **Objective 6.**
#
# **Northfield Grocers context:** The customer assistant's real-time ranker scores 200 substitution candidates per request; p99 latency is dominated by attention over long product contexts. Northfield's platform team needs to see *why* attention is memory-bound before touching FlashAttention or writing a fused kernel.
#
# **Retail use cases:** Cutting p99 latency of the substitution ranker on product pages; reducing GPU count for long-context supplier-sheet extraction.
#
# **Platform:** Databricks GPU cluster (NC24ads_A100_v4). Steps 1–3 are NumPy and run anywhere; Steps 4–6 need CUDA, torch, triton and Nsight.
#
# **Done means:** tiled attention equals naive within 1e-5 while reading O(N) rather than O(N²) from 'HBM'; before/after profiler traces captured.
#
# ## Step 1 — Naive attention, and count what it reads/writes
# *Why:* the N×N score matrix is materialised in memory. On a GPU that matrix lives in HBM, and moving it is what costs time — not the FLOPs.
# %%
class HBM:
    """Toy memory-traffic counter: every array we 'materialise' is counted in bytes."""
    def __init__(self): self.bytes = 0
    def touch(self, *arrays):
        for a in arrays: self.bytes += a.nbytes

def attention_naive(Q, K, V, mem):
    # >>> SOLUTION S = QKᵀ/√d (materialise N×N), P = softmax(S) (materialise), O = PV; count S and P
    d = Q.shape[1]
    S = Q @ K.T / np.sqrt(d); mem.touch(S)
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); mem.touch(P)
    return P @ V
    # <<< SOLUTION

rng = np.random.default_rng(0); N, d = 2048, 64
Q, K, V = (rng.normal(size=(N, d)).astype(np.float32) for _ in range(3))
m_naive = HBM(); O_naive = attention_naive(Q, K, V, m_naive)
print(f"naive: materialised {m_naive.bytes/1e6:.1f} MB for N={N}")

# %% [markdown]
# ## Step 2 — Tiled (flash-style) attention with online softmax
# *Why:* FlashAttention never materialises S or P. It streams K/V in blocks, keeps a running max and running sum per query row, and rescales the partial output as the max changes. Traffic becomes O(N·d) instead of O(N²).
# *What:* implement the online-softmax recurrence. This is the single most important algorithm in modern inference; write it once by hand.
# %%
def attention_tiled(Q, K, V, mem, block=256):
    """Flash-style attention: never materialise the N×N matrix. Only K/V blocks are 'read'."""
    # >>> SOLUTION for each K/V block: s = QKbᵀ/√d; m_new = max(m, rowmax s); rescale acc & l by exp(m-m_new); acc += exp(s-m_new)@Vb; l += rowsum
    N, d = Q.shape; O = np.zeros_like(Q); m = np.full(N, -np.inf, np.float32); l = np.zeros(N, np.float32)
    for j in range(0, N, block):
        Kb, Vb = K[j:j+block], V[j:j+block]; mem.touch(Kb, Vb)
        s = Q @ Kb.T / np.sqrt(d)
        m_new = np.maximum(m, s.max(1)); alpha = np.exp(m - m_new)
        p = np.exp(s - m_new[:, None])
        O = O * alpha[:, None] + p @ Vb; l = l * alpha + p.sum(1); m = m_new
    return O / l[:, None]
    # <<< SOLUTION

m_tiled = HBM(); O_tiled = attention_tiled(Q, K, V, m_tiled)
err = np.abs(O_naive - O_tiled).max()
print(f"tiled: traffic {m_tiled.bytes/1e6:.2f} MB (vs {m_naive.bytes/1e6:.1f} MB naive); max abs error {err:.2e}")
check(err < 1e-4, "tiled attention matches naive")
check(m_tiled.bytes * 10 < m_naive.bytes, "tiled reads an order of magnitude less memory")

# %% [markdown]
# ## Step 3 — Scaling: traffic vs N
# *Why:* the argument for FlashAttention is the slope, not one point. Naive traffic grows with N², tiled with N.
# %%
rows = []
for n in (512, 1024, 2048, 4096):
    q, k, v = (rng.normal(size=(n, d)).astype(np.float32) for _ in range(3))
    a, b = HBM(), HBM(); attention_naive(q, k, v, a); attention_tiled(q, k, v, b)
    rows.append(dict(N=n, naive_MB=round(a.bytes/1e6, 1), tiled_MB=round(b.bytes/1e6, 2)))
scal = pd.DataFrame(rows); print(scal.to_string())
check(scal.naive_MB.iloc[-1] / scal.naive_MB.iloc[0] > 50 and scal.tiled_MB.iloc[-1] / scal.tiled_MB.iloc[0] < 10, "naive ~N², tiled ~N")

# %% [markdown]
# ## Step 4 — Profile a real decode step (GPU cluster)
# *Why:* now measure it. Load Gemma 4 E4B in `torch` with `attn_implementation="eager"` and again with `"flash_attention_2"` (or SDPA), run 32 decode steps under `torch.profiler`, and compare kernel time share for attention.
# *Watch:* FlashAttention-2 requires Ampere or newer and specific `torch`/`flash-attn` version pairs — pin them in the init script; verify availability with `flash_attn.__version__` on the day.
# %%
if LAB_MODE == "GPU":
    gpu_only("Follow lab guide §Step 4: torch.profiler around generate(); export chrome trace; note attention kernel share for eager vs flash.")
else:
    gpu_only("Real profiling needs CUDA + torch + flash-attn.")
profile_notes = {"eager_attn_share_pct": None, "flash_attn_share_pct": None, "tokens_per_s_eager": None, "tokens_per_s_flash": None}

# %% [markdown]
# ## Step 5 — Write one fused kernel in Triton (GPU cluster)
# *Why:* fusion (e.g. RMSNorm + residual add) removes a full read/write of the activation tensor. The lab-guide listing gives the kernel skeleton; your job is to make it match the PyTorch reference within tolerance and beat it on a 4096×4096 tensor.
# %%
if LAB_MODE == "GPU":
    gpu_only("Implement fused_rmsnorm_residual per lab guide §Step 5 and compare with torch reference (allclose, atol=1e-3) and timing.")
kernel_result = {"max_abs_err": None, "speedup_vs_torch": None}

# %% [markdown]
# ## Step 6 — Before/after summary (the deliverable)
# *Why:* two profiler traces and one sentence: where did the time go, and why did it move.
# %%
summary = pd.DataFrame([profile_notes | kernel_result]); print(summary.T.to_string())
check(m_tiled.bytes < m_naive.bytes, "memory-traffic argument demonstrated by hand")
print("L05 complete." + ("" if LAB_MODE == "GPU" else " (SMOKE: Steps 4–6 need the GPU cluster.)"))
