# %% [markdown]
# # L15 — RAG for the store-ops assistant: retrieval, grounding and faithfulness evaluation
# **Objectives 5, 10.**
#
# **Northfield Grocers context:** 24,000 associates ask the store-ops assistant about food-safety temperatures, refund rules, breaks and planogram resets. Answers must come from the current SOPs with a citation, out-of-scope questions must be declined, and the team must measure faithfulness and answer relevancy on a golden set the store-ops function owns — before and after a prompt change.
#
# **Retail use cases:** SOP and HR-policy Q&A on handhelds; citing the SOP version in every answer for audit; catching a stale-index regression before associates act on old refund rules.
#
# **Platform:** any for Steps 1–5 (SMOKE uses lexical retrieval and a deterministic judge so the pipeline and metrics are fully exercised); GPU/cluster for Step 6 (embedding model + Ragas/LLM-as-judge).
#
# **Done means:** retrieval hits the right SOP area for ≥90% of golden questions; the stale chunk is never cited after the currency filter; out-of-scope question is declined; faithfulness and relevancy computed for two prompt versions and the regression detected.
#
# ## Step 1 — Ingest the SOP chunks and build the index
# *Why:* the index is an artefact: it carries the chunk ids, versions and effective dates that every answer will cite.
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
sops = pd.read_csv(os.path.join(DATA_DIR, "sop_chunks.csv")); golden = pd.read_csv(os.path.join(DATA_DIR, "sop_golden.csv"))
INDEX_VERSION = "idx-2026-09"
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(sops.text)   # stop words out: "what is the" must not score
E = normalize(vec.transform(sops.text)).toarray()
print(len(sops), "chunks indexed as", INDEX_VERSION)

# %% [markdown]
# ## Step 2 — Retrieve with a currency filter
# *Why:* L14's retrieval gate applied before ranking: superseded versions never reach the prompt.
# %%
def retrieve(question, k=2, current_version="v3"):
    """Return the top-k current chunks as a DataFrame with a score column."""
    # >>> SOLUTION embed question; cosine = dot; mask non-current versions to -1; take top-k
    q = normalize(vec.transform([question])).toarray()[0]
    sims = E @ q; sims = np.where(sops.version.values == current_version, sims, -1.0)
    idx = np.argsort(-sims)[:k]
    out = sops.iloc[idx].copy(); out["score"] = sims[idx]; return out
    # <<< SOLUTION
hits = [retrieve(q) for q in golden.question]
area_hit = np.mean([(h.area.iloc[0] == a) for h, a in zip(hits, golden.area) if a != "none"])
print(f"top-1 area accuracy on in-scope golden questions: {area_hit:.2f}")
check(area_hit >= 0.9 and not any("SOP-004-old" in h.chunk_id.values for h in hits), "retrieval hits the right SOP area; stale chunk never retrieved")

# %% [markdown]
# ## Step 3 — Build the prompt, generate, cite
# *Why:* the prompt carries the question, the retrieved chunks and the instruction to answer only from them. In SMOKE mode the "model" is a deterministic extractive answerer; in GPU mode call the served model.
# %%
MIN_SCORE = 0.15
def build_prompt(question, ctx, version="qa-v3"):
    rules = "Answer ONLY from the SOP text below. Cite the chunk id. If the answer is not in the text, say OUT_OF_SCOPE." if version == "qa-v3" else "Answer helpfully using the SOP text below."   # v4 planted regression: grounding rule dropped
    return f"[{version}] {rules}\n\n" + "\n".join(f"({c.chunk_id} {c.version}) {c.text}" for c in ctx.itertuples()) + f"\n\nQuestion: {question}"
def answer(question, version="qa-v3"):
    ctx = retrieve(question)
    if ctx.score.iloc[0] < MIN_SCORE: return dict(answer="OUT_OF_SCOPE", citations=[], grounded=True, prompt_version=version, index_version=INDEX_VERSION)
    prompt = build_prompt(question, ctx, version)
    if LAB_MODE == "GPU":
        gpu_only("Call the served model with `prompt`; parse answer + citation.")
    # deterministic stand-in: v3 quotes the best chunk; v4 'helpfully' adds an unsupported sentence
    text = ctx.text.iloc[0] + ("" if version == "qa-v3" else " Also, staff may extend this at their discretion.")
    return dict(answer=text, citations=[ctx.chunk_id.iloc[0]], grounded=(version == "qa-v3"), prompt_version=version, index_version=INDEX_VERSION)
a = answer(golden.question[0]); print(a)
oos = answer("What is the capital of France?"); print(oos["answer"])
check(oos["answer"] == "OUT_OF_SCOPE" and a["citations"], "out-of-scope declined; in-scope answer cites a chunk")

# %% [markdown]
# ## Step 4 — Faithfulness and answer relevancy, by hand
# *Why (pedagogy-first):* Ragas' faithfulness = fraction of answer claims supported by the context; answer relevancy = does the answer address the question. Implement deterministic versions (sentence-level support by lexical overlap; relevancy = expected phrase present) so the mechanics are clear before an LLM judge replaces them.
# %%
def faithfulness(answer_text, ctx_text):
    """Fraction of answer sentences with ≥60% token overlap with the context."""
    # >>> SOLUTION split answer into sentences; for each, tokens ∩ context tokens / tokens ≥ 0.6
    ctx_tok = set(re.findall(r"\w+", ctx_text.lower())); sents = [s for s in re.split(r"(?<=[.;])\s+", answer_text) if s.strip()]
    sup = [len(set(re.findall(r"\w+", s.lower())) & ctx_tok) / max(1, len(set(re.findall(r"\w+", s.lower())))) >= 0.6 for s in sents]
    return sum(sup) / len(sup)
    # <<< SOLUTION
def relevancy(answer_text, expected): return float(expected.lower() in answer_text.lower())
def evaluate(version):
    rows = []
    for q in golden.itertuples():
        r = answer(q.question, version); ctx = retrieve(q.question)
        rows.append(dict(question=q.question, faithfulness=faithfulness(r["answer"], " ".join(ctx.text)) if r["answer"] != "OUT_OF_SCOPE" else 1.0,
                         relevancy=relevancy(r["answer"], q.expected), cited=bool(r["citations"]) or q.expected == "OUT_OF_SCOPE"))
    return pd.DataFrame(rows)
e3, e4 = evaluate("qa-v3"), evaluate("qa-v4")
print("qa-v3:", e3[["faithfulness", "relevancy"]].mean().round(3).to_dict()); print("qa-v4:", e4[["faithfulness", "relevancy"]].mean().round(3).to_dict())
check(e3.faithfulness.mean() >= 0.95 and e4.faithfulness.mean() < e3.faithfulness.mean() - 0.1, "v4 prompt regression detected by faithfulness")

# %% [markdown]
# ## Step 5 — The golden-set report the store-ops owner signs
# *Why:* the deliverable is a per-area table that the function owner can read: which areas are weak, whether citations are present, and the version pair compared.
# %%
report = e3.merge(golden[["question", "area"]], on="question").groupby("area")[["faithfulness", "relevancy", "cited"]].mean().round(2)
print(report.to_string()); check(report.cited.min() == 1.0, "every in-scope answer carries a citation")

# %% [markdown]
# ## Step 6 — Real embeddings and an LLM judge (cluster)
# *Why:* replace TF-IDF with a sentence-embedding model and the deterministic judge with Ragas (faithfulness, answer_relevancy) using a judge model different from the serving model. Keep the golden set and the report format identical so the two runs are comparable.
# %%
if LAB_MODE == "GPU":
    gpu_only("Per lab guide: sentence-transformers encode(); ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy]) with a judge LLM; rebuild `report`.")
print("L15 complete." + ("" if LAB_MODE == "GPU" else " (SMOKE: Step 6 runs on the cluster.)"))
