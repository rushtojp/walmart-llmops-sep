# %% [markdown]
# # L12 — A/B testing & multi-armed bandits for rankers
# **Objective 15.**
#
# **Northfield Grocers context:** Northfield's recommendation team has a new substitution ranker (B) for out-of-stock items. A fixed A/B test sends half of shoppers to the worse ranker for two weeks; a Thompson-sampling bandit shifts traffic as evidence arrives. Run both on the click log and write the governance record that accompanies the champion-alias change.
#
# **Retail use cases:** Rolling out a substitution ranker with bounded regret; comparing two promo-copy models on click-through.
#
# **Platform:** Databricks (no GPU). Fully executable in SMOKE mode.
#
# **Done means:** bandit converges to the planted better arm; cumulative regret below A/B; decision record written.
#
# ## Step 1 — The click log
# *Why:* arm A and B are two ranker versions. The log has a planted truth (D6): B's CTR is ~30 % higher. Neither method may peek — they only see clicks as they arrive.
# %%
log = pd.read_csv(os.path.join(DATA_DIR, "click_log.csv"))
truth = log.groupby("arm").click.mean(); print("true CTRs (hidden from the algorithms):", truth.round(4).to_dict())

# %% [markdown]
# ## Step 2 — Fixed-split A/B test with a significance check
# *Why:* the classic approach: 50/50 for N impressions, then a two-proportion z-test. Regret = impressions sent to the worse arm × CTR gap.
# %%
from scipy.stats import norm
def ab_test(log, n):
    # >>> SOLUTION take first n rows per arm alternating order; compute CTRs, pooled z-test, p-value, regret
    a = log[log.arm == "A"].head(n); b = log[log.arm == "B"].head(n)
    pa, pb = a.click.mean(), b.click.mean(); p = (a.click.sum() + b.click.sum()) / (2 * n)
    z = (pb - pa) / np.sqrt(p * (1 - p) * 2 / n); pval = 2 * (1 - norm.cdf(abs(z)))
    regret = n * abs(truth["B"] - truth["A"])   # the worse arm got n impressions
    return dict(ctr_A=pa, ctr_B=pb, z=z, p_value=pval, regret=regret, winner="B" if pb > pa else "A")
    # <<< SOLUTION

ab = ab_test(log, 5000); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ab.items()})
check(ab["winner"] == "B", "A/B identifies B")

# %% [markdown]
# ## Step 3 — Thompson sampling bandit
# *Why:* keep a Beta(α, β) posterior per arm; each impression, sample from both, serve the larger. Traffic shifts to the better arm as evidence accumulates, so regret grows sub-linearly.
# %%
def thompson(log, n_impressions, seed=0):
    """Replay: at each step choose an arm, consume the next unseen click for that arm. Returns (allocation, regret)."""
    # >>> SOLUTION alpha/beta dicts; pools per arm; loop: sample, pick argmax, pop click, update; regret += gap if worse arm
    rng = np.random.default_rng(seed); alpha = {"A": 1, "B": 1}; beta = {"A": 1, "B": 1}
    pools = {a: list(log[log.arm == a].click) for a in ("A", "B")}; served = {"A": 0, "B": 0}; regret = 0.0
    for _ in range(n_impressions):
        s = {a: rng.beta(alpha[a], beta[a]) for a in ("A", "B")}; arm = max(s, key=s.get)
        c = pools[arm].pop(0); alpha[arm] += c; beta[arm] += 1 - c; served[arm] += 1
        if arm == "A": regret += truth["B"] - truth["A"]
    return served, regret
    # <<< SOLUTION

served, regret_ts = thompson(log, 10000)
print("bandit allocation:", served, f"| regret {regret_ts:.1f} vs A/B {ab['regret']:.1f}")
check(served["B"] > served["A"] * 2 and regret_ts < ab["regret"], "bandit converges to B with lower regret than A/B")

# %% [markdown]
# ## Step 4 — Governance record
# *Why:* whichever method you use, the decision needs a record: hypothesis, method, sample size, result, decision, owner, date. This is what the model registry alias change (`champion`) links to.
# %%
record = dict(hypothesis="Substitution ranker B raises click-through on out-of-stock substitutions by ≥10% over A", method="A/B z-test + Thompson replay", n_ab=5000, n_bandit=10000,
              ab=ab, bandit=dict(allocation=served, regret=regret_ts), decision="promote B to champion", owner="northfield-recommendation-team", date=time.strftime("%Y-%m-%d"))
os.makedirs("/tmp/l12", exist_ok=True); json.dump(record, open("/tmp/l12/decision.json", "w"), indent=2, default=float)
check(os.path.exists("/tmp/l12/decision.json"), "decision record written"); print("L12 complete.")
