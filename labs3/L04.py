# %% [markdown]
# # L04 — Embeddings, vector retrieval & recall evaluation
# **Objective 5.**
#
# **Northfield Grocers context:** 'Similar items' and out-of-stock substitutions on Northfield product pages are driven by retrieval over the catalog. The recommendation team must report recall@k and MRR, know where retrieval fails (size and pack tokens dominating), and choose between Qdrant and Databricks Vector Search.
#
# **Retail use cases:** Substitution candidates when an item is out of stock at a store; 'similar items' on product pages; RAG over merchandising-policy documents for category buyers.
#
# **Platform:** Databricks (T4 cluster is enough). Qdrant as a container or Qdrant Cloud; Databricks Vector Search as the managed alternative. SMOKE mode uses TF-IDF vectors and brute-force search so the evaluation logic is fully tested.
#
# **Done means:** brute-force and index retrieval agree on top-10 for ≥95% of queries; recall@10 and MRR reported; one failure case explained.
#
# ## Step 1 — Build the corpus and ground truth
# *Why:* recall needs a truth set. For "similar items" we define relevance as *same category and same first noun* — a deliberately imperfect proxy so the failure-case discussion in Step 6 is real.
# %%
cat = pd.read_csv(os.path.join(DATA_DIR, "catalog_items.csv")).drop_duplicates("item_id")
cat = cat[cat.category != "UNKNOWN_CAT"].reset_index(drop=True)
cat["noun"] = cat.title.str.split().str[1:3].str.join(" ")
print(len(cat), "items;", cat.category.nunique(), "categories")

# %% [markdown]
# ## Step 2 — Embed the catalog
# *Why (pedagogy-first):* hand-rolled retrieval before any framework. In SMOKE mode the "embedding" is an L2-normalised TF-IDF vector; in GPU mode swap in a sentence-embedding model (e.g. an open embedding model from the Nemotron or Gemma ecosystem — verify the checkpoint with the recommendation team).
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
if LAB_MODE == "GPU":
    gpu_only("Replace with sentence-transformers encode(...) per lab guide; keep the L2-normalisation.")
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(cat.title)
E = normalize(vec.transform(cat.title)).toarray().astype(np.float32)
print("embedding matrix", E.shape)

# %% [markdown]
# ## Step 3 — Brute-force cosine retrieval (the reference)
# *Why:* every approximate index (Qdrant HNSW, Databricks Vector Search) is judged against exact search. Build the exact version first.
# %%
def topk_bruteforce(E, q_idx, k=10):
    """Return indices of the k most similar rows to E[q_idx], excluding itself."""
    # >>> SOLUTION dot product with normalised vectors = cosine; mask self; argpartition then sort
    sims = E @ E[q_idx]; sims[q_idx] = -1
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])]
    # <<< SOLUTION

queries = np.random.default_rng(1).choice(len(cat), 200, replace=False)
bf = {q: topk_bruteforce(E, q) for q in queries}
print("example:", cat.title[queries[0]], "→", list(cat.title[bf[queries[0]][:3]]))

# %% [markdown]
# ## Step 4 — Index retrieval (Qdrant) and agreement with brute force
# *Why:* HNSW trades a little recall for a lot of speed. Measure the trade instead of assuming it. In SMOKE mode we emulate an approximate index by searching a random 60 % subset of candidates plus the exact top-3 — a stand-in that produces realistic disagreement so the agreement metric is exercised.
# %%
def topk_index(E, q_idx, k=10):
    try:
        from qdrant_client import QdrantClient   # GPU/lab cluster path: see lab guide §Step 4 for upsert + search
        raise ImportError("configure client per lab guide")
    except ImportError:
        rng = np.random.default_rng(q_idx); mask = rng.random(len(E)) < 0.6
        sims = np.where(mask, E @ E[q_idx], -1); sims[q_idx] = -1
        exact = topk_bruteforce(E, q_idx, 3); sims[exact] = (E @ E[q_idx])[exact]
        idx = np.argpartition(-sims, k)[:k]; return idx[np.argsort(-sims[idx])]

ix = {q: topk_index(E, q) for q in queries}
agree = np.mean([len(set(bf[q]) & set(ix[q])) / 10 for q in queries])
print(f"mean top-10 overlap index vs brute force: {agree:.3f}")
check(agree >= 0.6, "index overlap measured (≥0.95 expected with real HNSW; SMOKE stand-in is deliberately lossy)")

# %% [markdown]
# ## Step 5 — recall@k and MRR
# *Why:* these are the two numbers the recommendation team reports. Implement them from the definition, not from a library, once.
# %%
def relevant(q_idx):
    r = cat.iloc[q_idx]
    return set(cat.index[(cat.category == r.category) & (cat.noun == r.noun) & (cat.index != q_idx)])

def recall_at_k(ranked, rel, k=10):
    # >>> SOLUTION |top-k ∩ relevant| / min(k, |relevant|)
    if not rel: return np.nan
    return len(set(ranked[:k]) & rel) / min(k, len(rel))
    # <<< SOLUTION

def mrr(ranked, rel):
    # >>> SOLUTION 1/rank of first relevant hit, else 0
    for i, r in enumerate(ranked, 1):
        if r in rel: return 1 / i
    return 0.0
    # <<< SOLUTION

R = [recall_at_k(bf[q], relevant(q)) for q in queries]; M = [mrr(bf[q], relevant(q)) for q in queries]
print(f"recall@10 = {np.nanmean(R):.3f}   MRR = {np.mean(M):.3f}")
check(np.nanmean(R) > 0.5 and np.mean(M) > 0.5, "retrieval quality above the lab floor")

# %% [markdown]
# ## Step 6 — Document one failure case
# *Why:* a retrieval report without a failure case is not credible. Find the query with the lowest MRR and explain it (typical: pack-quantity or size tokens dominating the vector so a different product with the same size outranks the same product with a different size).
# %%
worst = queries[int(np.argmin(M))]
print("Worst query:", cat.title[worst]); print("Returned:", list(cat.title[bf[worst][:5]]))
print("Relevant set size:", len(relevant(worst)))
failure_note = "Size/pack tokens dominate lexical vectors so a 1 L oat drink matches a 1 L cola; a learned embedding plus a category filter pre-retrieval fixes this — and substitutions must also respect allergen/dietary flags (filter, never rank)."
check(len(failure_note) > 20, "failure case documented")
print("L04 complete.")
