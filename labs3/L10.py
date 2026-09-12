# %% [markdown]
# # L10 — Monitoring, drift & injected-fault alerting
# **Objective 13.**
#
# **Northfield Grocers context:** Promo weeks change what Northfield customers ask and how they phrase it; a holiday range launch changes the catalog. Week 4 in the lab is a promo week. Monitoring must flag the drifted input feature, stay silent on the stable one, and prove its latency alert fires on an injected fault before the real peak arrives.
#
# **Retail use cases:** Promo-week and holiday drift in customer questions; silent provider model updates raising cost per conversation; index staleness after a range change.
#
# **Platform:** Both platforms. Drift maths and alert logic run anywhere; managed monitors configured per lab guide.
#
# **Done means:** PSI flags the planted drifted feature (title_len, week 4) and not the stable one (attr_count); the injected latency fault raises the alert.
#
# ## Step 1 — Load the four-week window (week 4 is a promo week)
# %%
mon = pd.read_csv(os.path.join(DATA_DIR, "monitoring_window.csv"))
print(mon.groupby("week")[["title_len", "attr_count", "latency_ms"]].mean().round(2))

# %% [markdown]
# ## Step 2 — Population Stability Index by hand
# *Why:* PSI is what most monitoring products report; know its definition (Σ (p−q)·ln(p/q) over bins) and its usual thresholds (<0.1 stable, 0.1–0.25 watch, >0.25 drift).
# %%
def psi(ref, cur, bins=10):
    # >>> SOLUTION quantile bins from ref; histogram both; add epsilon; sum (p-q)*ln(p/q)
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + 1e-6; q = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((p - q) * np.log(p / q)))
    # <<< SOLUTION

ref = mon[mon.week == 1]
for f in ("title_len", "attr_count"):
    print(f, {w: round(psi(ref[f], mon[mon.week == w][f]), 3) for w in (2, 3, 4)})
check(psi(ref.title_len, mon[mon.week == 4].title_len) > 0.25 and psi(ref.attr_count, mon[mon.week == 4].attr_count) < 0.1, "PSI flags title_len week 4 only")

# %% [markdown]
# ## Step 3 — Cross-check with a KS test
# *Why:* PSI depends on binning; the two-sample KS statistic does not. Two methods agreeing is stronger evidence than one.
# %%
from scipy.stats import ks_2samp
ks = {f: ks_2samp(ref[f], mon[mon.week == 4][f]).pvalue for f in ("title_len", "attr_count")}
print("KS p-values week 4 vs week 1:", {k: f"{v:.2e}" for k, v in ks.items()})
check(ks["title_len"] < 0.01 < ks["attr_count"], "KS agrees with PSI")

# %% [markdown]
# ## Step 4 — Latency and cost-per-request, then inject a fault
# *Why:* an alert you have never seen fire is not an alert. Compute p95 latency and cost per request per week, then inject a +40 % latency fault into a copy of week 4 and confirm the rule triggers.
# %%
usd_per_gpu_hr, req_per_hr = 3.673, 20000   # A100 SKU price (approx, from content.py) and an assumed load
def metrics(df):
    return dict(p95_ms=float(np.percentile(df.latency_ms, 95)), cost_per_req_usd=usd_per_gpu_hr / req_per_hr)
weekly = {w: metrics(mon[mon.week == w]) for w in range(1, 5)}
baseline_p95 = np.mean([weekly[w]["p95_ms"] for w in (1, 2, 3)])
def alert(p95, baseline, threshold=1.25): return p95 > baseline * threshold
faulty = mon[mon.week == 4].copy(); faulty["latency_ms"] *= 1.4
print("week4 p95", round(weekly[4]["p95_ms"], 1), "| baseline", round(baseline_p95, 1), "| faulty p95", round(metrics(faulty)["p95_ms"], 1))
check(not alert(weekly[4]["p95_ms"], baseline_p95) and alert(metrics(faulty)["p95_ms"], baseline_p95), "alert silent on normal week, fires on injected fault")

# %% [markdown]
# ## Step 5 — Configure the managed monitor (platform)
# *Why:* the hand-rolled version explains the numbers; the managed one runs unattended. Databricks: `Lakehouse Monitoring` on an inference table with a baseline table = week 1. Azure ML: model monitor with data-drift signal on the deployment's collected data. Both are configured through the UI/SDK per lab guide §Step 5 — API names are version-sensitive.
# %%
if LAB_MODE == "GPU":
    gpu_only("Create the monitor per lab guide §Step 5 and screenshot the drift chart for the deliverable.")
else:
    print("Local: drift maths and alert logic verified.")
print("L10 complete.")
