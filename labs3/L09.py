# %% [markdown]
# # L09 — Evaluation harness & CI/CD gating — with a zero-tolerance allergen check
# **Objectives 10, 12.**
#
# **Northfield Grocers context:** A new extraction model is proposed for Northfield's catalog engine. Two things can block it: a regression on the attribute golden set, and — non-negotiable — any error on allergen questions, which are answered by structured lookup and evaluated by exact match, never by a judge. The gate must record why it decided.
#
# **Retail use cases:** Gating a model swap in the catalog-extraction service; allergen accuracy as a blocking release check; prompt-version drift between releases.
#
# **Platform:** Azure ML (pipeline job with an eval component that fails the run) — MLflow registry promotion on Databricks is the equivalent. Harness and gate execute fully in SMOKE mode.
#
# **Done means:** harness blocks the planted regressed candidate and passes the good one; allergen zero-tolerance check blocks any error; the gate decision is written to the run.
#
# ## Step 1 — Golden set and candidates
# *Why:* a gate compares a candidate to a baseline on a fixed golden set. We simulate two candidates with deterministic predictors: `good` (correct except pack_qty parsing edge cases) and `regressed` (drops the "size" key 15 % of the time — a realistic prompt-template regression).
# %%
golden = [json.loads(l) for l in open(os.path.join(DATA_DIR, "eval_prompts.jsonl"))]
def _base(prompt):
    t = prompt.split(": ", 1)[1].split(); return {"brand": t[0], "size": t[3] + " " + t[4], "pack_qty": int(t[-1])}
def good(prompt): return json.dumps(_base(prompt))
def regressed(prompt):
    o = _base(prompt)
    if hash(prompt) % 100 < 15: o.pop("size")
    return json.dumps(o)
print(len(golden), "golden items")

# %% [markdown]
# ## Step 2 — The harness: exact match, per-key, schema, latency
# *Why:* one metric hides regressions. Schema validity catches missing keys even when other fields are right; per-key accuracy tells you *what* broke; latency guards cost.
# %%
def harness(predict_fn, items):
    # >>> SOLUTION compute exact_match, schema_valid_rate, per_key accuracy, p95 latency
    REQ = {"brand", "size", "pack_qty"}; em = schema = 0; keys = dict.fromkeys(REQ, 0); lat = []
    for it in items:
        t0 = time.perf_counter(); pred = json.loads(predict_fn(it["prompt"])); lat.append(time.perf_counter() - t0)
        tgt = json.loads(it["target"]); schema += set(pred) == REQ; em += pred == tgt
        for k in REQ: keys[k] += pred.get(k) == tgt[k]
    n = len(items)
    return dict(exact_match=em / n, schema_valid=schema / n, **{f"acc_{k}": v / n for k, v in keys.items()}, p95_ms=float(np.percentile(lat, 95) * 1000))
    # <<< SOLUTION

m_good, m_reg = harness(good, golden), harness(regressed, golden)
print("good:", {k: round(v, 3) for k, v in m_good.items()}); print("regressed:", {k: round(v, 3) for k, v in m_reg.items()})
check(m_reg["schema_valid"] < m_good["schema_valid"], "harness sees the regression")

# %% [markdown]
# ## Step 3 — The gate
# *Why:* a gate is a policy: absolute floors *and* no regression beyond a tolerance versus the current champion. Write it so the reasons are recorded, not just a boolean.
# %%
def gate(candidate, champion, floors={"exact_match": 0.80, "schema_valid": 0.98}, tolerance=0.02):
    """Return (passed: bool, reasons: list[str])."""
    # >>> SOLUTION check each floor; check candidate >= champion - tolerance for exact_match and schema_valid
    reasons = []
    for k, f in floors.items():
        if candidate[k] < f: reasons.append(f"{k}={candidate[k]:.3f} below floor {f}")
    for k in ("exact_match", "schema_valid"):
        if candidate[k] < champion[k] - tolerance: reasons.append(f"{k} regressed vs champion by {champion[k]-candidate[k]:.3f}")
    return (len(reasons) == 0, reasons or ["all checks passed"])
    # <<< SOLUTION

ok_good, r_good = gate(m_good, m_good); ok_reg, r_reg = gate(m_reg, m_good)
print("good →", ok_good, r_good); print("regressed →", ok_reg, r_reg)
check(ok_good and not ok_reg, "gate passes good candidate and blocks regressed one")

# %% [markdown]
# ## Step 4 — Zero-tolerance class: allergen answers by structured lookup
# *Why:* an allergen error is a safety incident, not a quality dip. Northfield never lets a model *generate* an allergen statement — the answer is a structured lookup in the product master — and the gate evaluates it by exact match against the allergen golden set. One error blocks the release, regardless of every other metric.
# *What:* implement the lookup, run the 100-question allergen golden set, and add a blocking rule to the gate. Then plant one wrong row in a copy of the product master to prove the rule fires.
# %%
catalog = pd.read_csv(os.path.join(DATA_DIR, "catalog_items.csv")).drop_duplicates("item_id").set_index("item_id")
allergen_golden = pd.read_csv(os.path.join(DATA_DIR, "allergen_golden.csv"))
def allergen_lookup(item_id, allergen, master=catalog):
    """Structured answer: 'yes' if the allergen is listed for the item in the product master, else 'no'."""
    # >>> SOLUTION split the allergens field on ';' and test membership
    return "yes" if allergen in str(master.loc[item_id, "allergens"]).split(";") else "no"
    # <<< SOLUTION
def allergen_check(master):
    pred = [allergen_lookup(r.item_id, r.allergen, master) for r in allergen_golden.itertuples()]
    errors = int((pd.Series(pred) != allergen_golden.answer).sum())
    return dict(allergen_errors=errors, allergen_accuracy=1 - errors / len(allergen_golden))
a_ok = allergen_check(catalog); print("product master:", a_ok)
tampered = catalog.copy(); _yes = allergen_golden[allergen_golden.answer == "yes"].item_id.iloc[0]; tampered.loc[_yes, "allergens"] = "none"   # planted wrong row (an item that DOES contain the allergen)
a_bad = allergen_check(tampered); print("tampered master:", a_bad)
def gate_with_allergen(candidate, champion, allergen_result):
    passed, reasons = gate(candidate, champion)
    if allergen_result["allergen_errors"] > 0:
        passed = False; reasons = [r for r in reasons if r != "all checks passed"] + [f"BLOCKING: {allergen_result['allergen_errors']} allergen error(s) — zero tolerance"]
    return passed, reasons
print("good + clean master →", gate_with_allergen(m_good, m_good, a_ok))
print("good + tampered master →", gate_with_allergen(m_good, m_good, a_bad))
check(a_ok["allergen_errors"] == 0 and gate_with_allergen(m_good, m_good, a_ok)[0], "allergen lookup is exact on the clean master")
check(a_bad["allergen_errors"] >= 1 and not gate_with_allergen(m_good, m_good, a_bad)[0], "one planted allergen error blocks an otherwise-passing release")

# %% [markdown]
# ## Step 5 — Write the decision record
# *Why:* CI must leave evidence. A JSON record with metrics, thresholds, reasons and a timestamp is what an auditor (and your future self) reads.
# %%
record = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), candidate="regressed-v2", champion="good-v1",
              metrics=m_reg, champion_metrics=m_good, allergen=a_ok, passed=ok_reg, reasons=r_reg)
os.makedirs("/tmp/l09", exist_ok=True); json.dump(record, open("/tmp/l09/gate_decision.json", "w"), indent=2)
check(os.path.exists("/tmp/l09/gate_decision.json"), "decision record written")

# %% [markdown]
# ## Step 6 — Wire it into Azure ML (workspace required)
# *Why:* as a pipeline component, a non-zero exit fails the job and blocks the downstream registration step. The component YAML and `azure-ai-ml` SDK calls are in the lab guide §Step 6 (SDK v2; signatures are version-sensitive — verify). On Databricks: run this notebook as a job task; on failure the next task (`mlflow.register_model` + alias `champion`) is skipped.
# %%
if os.environ.get("AZUREML_RUN_ID") or LAB_MODE == "GPU":
    gpu_only("Submit the pipeline per lab guide §Step 5; confirm the eval component fails for 'regressed'.")
else:
    print("Local: gate logic verified; pipeline submission needs the Azure ML workspace.")
if not ok_reg: print("Gate would exit non-zero here → registration blocked.")
print("L09 complete.")
