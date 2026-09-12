# %% [markdown]
# # L11 — Distributed serving: tensor parallel & autoscaling on Ray
# **Objective 14.**
#
# **Northfield Grocers context:** Saturday 10:00 traffic on the customer assistant is 4× Tuesday 15:00. Northfield needs the dense 31B model served TP=4, an autoscaler that pre-warms before known peaks and keeps two replicas for availability, and proof the service survives losing a worker mid-peak.
#
# **Retail use cases:** Peak-hour scaling for the customer assistant; high availability for the store-ops assistant during trading hours.
#
# **Platform:** Databricks multi-GPU (NC96ads_A100_v4, 4× A100). Steps 1–3 are NumPy; Step 4 needs the cluster.
#
# **Done means:** sharded matmul equals full within 1e-6; autoscaler simulation keeps p95 queue time under the SLA; TP=4 endpoint survives a simulated worker loss.
#
# ## Step 1 — Tensor parallelism by hand
# *Why:* TP splits a weight matrix across GPUs. Column-parallel: each GPU computes part of the output columns, then concatenate. Row-parallel: each GPU holds part of the input dimension, then all-reduce (sum). Do both and prove they equal the unsharded matmul.
# %%
rng = np.random.default_rng(0); x = rng.normal(size=(8, 1024)).astype(np.float64); W = rng.normal(size=(1024, 4096)); TP = 4
full = x @ W

def column_parallel(x, W, tp):
    # >>> SOLUTION split W along axis=1; each shard computes x@W_i; concatenate
    return np.concatenate([x @ Wi for Wi in np.array_split(W, tp, axis=1)], axis=1)
    # <<< SOLUTION

def row_parallel(x, W, tp):
    # >>> SOLUTION split x along axis=1 and W along axis=0; partial = x_i@W_i; all-reduce = sum
    return sum(xi @ Wi for xi, Wi in zip(np.array_split(x, tp, axis=1), np.array_split(W, tp, axis=0)))
    # <<< SOLUTION

check(np.allclose(column_parallel(x, W, TP), full, atol=1e-6) and np.allclose(row_parallel(x, W, TP), full, atol=1e-6), "both shardings equal the full matmul")
print("bytes per GPU for W:", W.nbytes // TP, "vs full", W.nbytes)

# %% [markdown]
# ## Step 2 — What TP costs: the all-reduce
# *Why:* row-parallel needs a sum across GPUs every layer — that is why NVLink SKUs matter (L01). Count the communication volume per layer for the 8×4096 activation and compare it to the compute saved.
# %%
act_bytes = 8 * 4096 * 2   # bf16 activation
allreduce_bytes = 2 * (TP - 1) / TP * act_bytes   # ring all-reduce volume per GPU
print(f"all-reduce per layer per GPU ≈ {allreduce_bytes/1e3:.1f} KB; ×60 layers per token step")
check(allreduce_bytes > 0, "communication cost quantified")

# %% [markdown]
# ## Step 3 — Autoscaling simulation
# *Why:* before touching a real autoscaler, simulate one. Arrivals follow a daily curve; each replica serves `cap` requests/s; scale up when queue p95 exceeds the SLA, scale down when utilisation is low, with a minimum of 2 replicas for HA. At minute 300 a replica dies.
# %%
def simulate(minutes=720, cap=20, sla_s=2.0, min_rep=2, max_rep=8, fail_at=300, scale_up_cooldown=5):
    # >>> SOLUTION per-minute loop: arrivals = 60*rate(t); served = 60*cap*replicas; queue update; wait = queue/(cap*replicas); scaling with cooldown; drop replica at fail_at
    rng = np.random.default_rng(0); replicas, queue, last_up = min_rep, 0.0, -99; log = []
    for t in range(minutes):
        rate = 25 + 60 * max(0, np.sin(np.pi * t / minutes)) + rng.normal(0, 3)
        if t == fail_at: replicas = max(min_rep - 1, replicas - 1)
        arrivals = 60 * rate; served = 60 * cap * replicas
        queue = max(0.0, queue + arrivals - served); wait = queue / (cap * replicas)
        if wait > sla_s * 0.8 and replicas < max_rep and t - last_up >= scale_up_cooldown: replicas += 1; last_up = t
        elif wait == 0 and arrivals < 0.5 * served and replicas > min_rep: replicas -= 1
        log.append(dict(t=t, rate=rate, replicas=replicas, wait_s=wait))
    return pd.DataFrame(log)
    # <<< SOLUTION

sim = simulate(); p95_wait = sim.wait_s.quantile(.95)
print(f"replicas min/max {sim.replicas.min()}/{sim.replicas.max()}; p95 queue wait {p95_wait:.2f}s; wait at failure+1 min {sim.wait_s[301]:.2f}s")
check(p95_wait < 2.0 and sim.replicas.min() >= 1, "autoscaler keeps p95 wait under SLA and recovers from replica loss")

# %% [markdown]
# ## Step 4 — Real TP=4 serving on Ray-on-Databricks (cluster)
# *Why:* `ray.util.spark.setup_ray_cluster` starts Ray across Databricks workers; vLLM with `--tensor-parallel-size 4` shards Gemma 4 31B over the 4 A100s. Then kill one worker process and watch Ray Serve reschedule. Exact flags/APIs are version-sensitive — lab guide §Step 4.
# %%
if LAB_MODE == "GPU":
    gpu_only("Start Ray on Spark, launch vLLM TP=4, run the L03 harness at concurrency 32, simulate worker loss, record recovery time.")
recovery = {"tp4_throughput_tok_s": None, "recovery_s_after_worker_loss": None}
print(recovery); print("L11 complete." + ("" if LAB_MODE == "GPU" else " (SMOKE: Step 4 needs the multi-GPU cluster.)"))
