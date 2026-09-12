# %% [markdown]
# # L03 — Serving bake-off: vLLM vs TensorRT-LLM vs SGLang
# **Objective 4.**
#
# **Northfield Grocers context:** The customer shopping assistant must show the first words of an answer within 800 ms at Saturday-morning concurrency. Northfield will choose the serving engine on measured TTFT p95 and throughput for its own request mix — not a vendor benchmark.
#
# **Retail use cases:** Choosing the engine for the customer assistant under a p95 TTFT SLA at weekend peak; deciding whether a TensorRT-LLM engine build is worth it for the single long-lived store-ops model.
#
# **Platform:** Databricks GPU cluster (engines via cluster init script) and/or Azure ML managed online endpoint. Steps 1–3 validate the harness on a mock server (runs anywhere); Steps 4–6 point it at real engines.
#
# **Done means:** harness metrics match analytic values on the mock server; decision table has the team's own numbers at 3 concurrency levels.
#
# ## Step 1 — Load the request mix
# *Why:* benchmark numbers are only comparable if every engine sees the same prompts. `request_mix.jsonl` is 500 (prompt_tokens, max_new) pairs shaped like the ad-copy workload.
# %%
reqs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "request_mix.jsonl"))]
print(len(reqs), "requests; mean prompt tokens", np.mean([r["prompt_tokens"] for r in reqs]).round(1))

# %% [markdown]
# ## Step 2 — Build the async benchmark harness
# *Why:* TTFT and throughput must be measured under concurrency, streaming, with the first token timestamped separately. A harness that is wrong by one `await` gives wrong engine rankings.
# *What:* `run_benchmark(send, reqs, concurrency)` takes an async `send(req) -> async iterator of tokens`, drives it with a semaphore, and returns per-request TTFT and total time.
# %%
import asyncio

async def run_benchmark(send, reqs, concurrency):
    """Return DataFrame(ttft_s, total_s, tokens) and wall-clock seconds. `send` yields tokens asynchronously."""
    # >>> SOLUTION semaphore-limited workers; timestamp first token for TTFT; gather results; wall = t_end - t_start
    sem = asyncio.Semaphore(concurrency); results = []
    async def one(r):
        async with sem:
            t0 = time.perf_counter(); first = None; n = 0
            async for _tok in send(r):
                if first is None: first = time.perf_counter()
                n += 1
            results.append(dict(ttft_s=first - t0, total_s=time.perf_counter() - t0, tokens=n))
    t_start = time.perf_counter()
    await asyncio.gather(*(one(r) for r in reqs))
    return pd.DataFrame(results), time.perf_counter() - t_start
    # <<< SOLUTION

def summarise(df, wall, label):
    return dict(engine=label, throughput_tok_s=round(df.tokens.sum() / wall, 1),
                ttft_p50_ms=round(df.ttft_s.quantile(.5) * 1000, 1), ttft_p95_ms=round(df.ttft_s.quantile(.95) * 1000, 1),
                requests=len(df), wall_s=round(wall, 2))

# %% [markdown]
# ## Step 3 — Validate the harness on a mock server with known latencies
# *Why:* before pointing the harness at a real engine, prove it measures correctly. The mock yields its first token after exactly 50 ms and then one token per 5 ms, so TTFT p50 must be ≈50 ms and tokens/s ≈ concurrency × 200 (minus scheduling overhead).
# %%
async def mock_send(r, ttft=0.05, per_tok=0.005):
    await asyncio.sleep(ttft); yield "t"
    for _ in range(min(r["max_new"], 20) - 1):
        await asyncio.sleep(per_tok); yield "t"

def run(coro):
    """Run a coroutine whether or not an event loop is already running (Jupyter vs script)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()

df, wall = run(run_benchmark(mock_send, reqs[:64], concurrency=8))
s = summarise(df, wall, "mock"); print(s)
check(45 <= s["ttft_p50_ms"] <= 80, "harness TTFT p50 ≈ 50 ms on mock (measures first token, not last)")
check(s["requests"] == 64, "all requests completed")

# %% [markdown]
# ## Step 4 — Point the harness at real engines (GPU cluster)
# *Why:* vLLM, SGLang and TensorRT-LLM all expose an OpenAI-compatible `/v1/completions` streaming endpoint, so one `send` implementation covers all three. Start each server per lab guide §Step 4 (ports 8001/8002/8003), and note the cold-start and engine-build times by hand — they matter as much as throughput.
# *Watch:* SGLang's RadixAttention benefits from shared prefixes; use the same system prompt for every request or you will under-measure it.
# %%
async def openai_stream_send(r, base_url, model):
    """Stream tokens from an OpenAI-compatible completions endpoint. Requires `httpx` (pip install httpx)."""
    ensure_packages(["httpx"]); import httpx
    prompt = "You are the Northfield Grocers shopping assistant. Is oat drink 1 L in stock at the Riverside store and what can I substitute? " * max(1, r["prompt_tokens"] // 20)
    body = {"model": model, "prompt": prompt, "max_tokens": r["max_new"], "stream": True}
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{base_url}/v1/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield line

ENGINES = {"vllm": "http://localhost:8001", "sglang": "http://localhost:8002", "trtllm": "http://localhost:8003"}
MODEL = "google/gemma-4-e4b-it"   # verify exact model id on the hub on the day
results = []
if LAB_MODE == "GPU":
    for name, url in ENGINES.items():
        for conc in (1, 8, 32):
            try:
                d, w = run(run_benchmark(lambda r: openai_stream_send(r, url, MODEL), reqs[:128], conc))
                results.append({**summarise(d, w, name), "concurrency": conc})
            except Exception as e:
                print(f"{name}@{conc}: not reachable ({e}) — start the server per lab guide")
else:
    gpu_only("Engines need the GPU cluster; harness validated on mock in Step 3.")

# %% [markdown]
# ## Step 5 — Record cold start and engine-build time
# *Why:* TensorRT-LLM's engine build is a one-off cost that can dominate for short-lived deployments; vLLM/SGLang start in about a minute. Time them with a stopwatch and record here — this is the number that decides the "single long-lived model" case.
# %%
cold_start_s = {"vllm": None, "sglang": None, "trtllm_build": None, "trtllm_start": None}   # fill by hand on the day
print(cold_start_s)

# %% [markdown]
# ## Step 6 — Decision table (the deliverable)
# *Why:* one table, your numbers, three concurrency levels, plus the cold-start row. Write the recommendation under it in one sentence naming the constraint that decided it.
# %%
decision = pd.DataFrame(results) if results else pd.DataFrame(columns=["engine", "concurrency", "throughput_tok_s", "ttft_p50_ms", "ttft_p95_ms"])
print(decision.to_string())
check("ttft_p95_ms" in decision.columns, "decision table schema in place")
print("L03 complete." + ("" if results else " (SMOKE: real-engine rows are filled on the GPU cluster.)"))
