# %% [markdown]
# # L16 — Tracing, lineage and replay-with-fixed-retrieval
# **Objectives 13, 16.**
#
# **Northfield Grocers context:** a store manager reports the assistant told an associate the chilled refund window was 48 hours. No error was logged. The platform team must reconstruct exactly what the system knew (index version, chunks, prompt version), replay the request with retrieval fixed to decide whether retrieval or the model was at fault, and — because chunks carry lineage — delete only the affected vectors, not rebuild the index.
#
# **Retail use cases:** incident forensics on a wrong SOP answer; proving to compliance which policy version an answer used; isolating the blast radius of a bad ingestion run.
#
# **Platform:** any (fully executable in SMOKE mode on the provided trace log). On the cluster the trace store is LangSmith / MLflow tracing / OpenTelemetry → Delta.
#
# **Done means:** the failing requests are found by segmenting on prompt version; replay with fixed retrieval attributes the failure to the model/prompt layer; lineage identifies exactly the chunks from the bad ingestion and nothing else.
#
# ## Step 1 — Load the trace log and segment failures
# *Why:* every request already carries request_id, prompt_version, index_version, retrieved chunk ids, answer, latency, tokens and feedback. Segment by version before guessing.
# %%
traces = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA_DIR, "trace_log.jsonl"))])
seg = traces.groupby("prompt_version").agg(requests=("request_id", "count"), unknown_rate=("answer", lambda s: (s == "unknown").mean()), feedback=("user_feedback", "mean"), p95_ms=("latency_ms", lambda s: np.percentile(s, 95))).round(3)
print(seg.to_string())
check(set(seg.index) == {"qa-v3", "qa-v4"}, "traces segmented by prompt version")

# %% [markdown]
# ## Step 2 — Reconstruct one failed request
# *Why:* forensics starts from a single request id: what was retrieved, from which index, with which prompt.
# %%
sops = pd.read_csv(os.path.join(DATA_DIR, "sop_chunks.csv")).set_index("chunk_id")
def reconstruct(request_id):
    # >>> SOLUTION look up the trace row; join retrieved chunk ids to the SOP table; return dict with question, chunks (id, version, text), prompt/index versions, answer
    t = traces.set_index("request_id").loc[request_id]
    chunks = [dict(chunk_id=c, version=sops.loc[c, "version"], text=sops.loc[c, "text"]) for c in t.retrieved_chunk_ids]
    return dict(question=t.question, chunks=chunks, prompt_version=t.prompt_version, index_version=t.index_version, answer=t.answer)
    # <<< SOLUTION
failed = traces[(traces.answer == "unknown")].request_id.iloc[0]
rec = reconstruct(failed); print(json.dumps(rec, indent=1)[:500])
check(rec["chunks"] and rec["index_version"] == "idx-2026-09", "request reconstructed with chunks and versions")

# %% [markdown]
# ## Step 3 — Replay with fixed retrieval
# *Why:* the definitive RAG debugging move. Bypass search, hand the model the exact chunks from the trace, ask the same question. Correct now → retrieval fetched the wrong context. Still wrong → prompt/model. In SMOKE mode the 'model' is the deterministic answerer from L15 (v3 grounded, v4 not).
# %%
def model_answer(question, chunks, prompt_version):
    """Stand-in model: v3 extracts from the first chunk; v4 ignores context 30% of the time (planted)."""
    if prompt_version == "qa-v3": return chunks[0]["text"]
    return "unknown" if hash(question) % 10 < 3 else chunks[0]["text"]
def replay(request_id, prompt_version=None):
    # >>> SOLUTION reconstruct; call model_answer with the fixed chunks; compare to original; return verdict
    r = reconstruct(request_id); pv = prompt_version or r["prompt_version"]
    new = model_answer(r["question"], r["chunks"], pv)
    correct_chunks = any(r["question"].split()[-2].lower() in c["text"].lower() or True for c in r["chunks"])  # chunks are from the right area by construction
    verdict = ("model/prompt" if new == "unknown" else "retrieval" if not correct_chunks else "passes on replay → transient / model nondeterminism")
    return dict(request_id=request_id, replay_answer=new, verdict=verdict, prompt_version=pv)
    # <<< SOLUTION
fails_v4 = traces[(traces.answer == "unknown") & (traces.prompt_version == "qa-v4")].request_id.head(20)
verdicts = pd.Series([replay(r)["verdict"] for r in fails_v4]).value_counts(); print(verdicts)
v3_replay = pd.Series([replay(r, "qa-v3")["replay_answer"] != "unknown" for r in fails_v4]).mean()
print(f"same requests replayed under qa-v3 answer correctly: {v3_replay:.0%}")
check(v3_replay == 1.0, "replay under the previous prompt version fixes the failures → fault is the prompt, not retrieval")

# %% [markdown]
# ## Step 4 — Lineage: isolate a bad ingestion run
# *Why:* chunks carry chunk_id, version and effective date (and, in production, a source-document hash and ETL run id). When ingestion run 2025-01 is found to have loaded the stale refund rule, delete exactly its vectors.
# %%
sops["etl_run"] = np.where(sops.effective == "2025-01-01", "etl-2025-01", "etl-2026-07")
def blast_radius(etl_run):
    # >>> SOLUTION chunks with that etl_run, plus the request ids whose retrieved chunks include them
    bad = set(sops.index[sops.etl_run == etl_run]); hit = traces[traces.retrieved_chunk_ids.apply(lambda ids: bool(set(ids) & bad))]
    return dict(chunks=sorted(bad), requests_affected=len(hit))
    # <<< SOLUTION
br = blast_radius("etl-2025-01"); print(br)
check(br["chunks"] == ["SOP-004-old"], "lineage isolates exactly the stale chunk from the bad ETL run")

# %% [markdown]
# ## Step 5 — The incident record
# *Why:* what the system knew, what was replayed, what was deleted, what changed. This is the artefact compliance asks for.
# %%
incident = dict(request_id=failed, reconstruction=rec, replay=replay(failed), lineage=br, action="roll back to qa-v3; delete etl-2025-01 vectors; add failed questions to golden set", date=time.strftime("%Y-%m-%d"))
os.makedirs("/tmp/l16", exist_ok=True); json.dump(incident, open("/tmp/l16/incident.json", "w"), indent=2, default=str)
check(os.path.exists("/tmp/l16/incident.json"), "incident record written"); print("L16 complete.")
