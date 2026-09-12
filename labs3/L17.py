# %% [markdown]
# # L17 — Token economics, model routing and capacity planning
# **Objectives 2, 4, 15.**
#
# **Northfield Grocers context:** finance has asked what the customer assistant will cost per conversation, per month and at the Christmas peak, and whether routing simple questions to a small self-hosted model beats sending everything to a hosted frontier model. The answer must be a formula-driven model the product owner can defend, built from measured numbers.
#
# **Retail use cases:** cost per resolved customer conversation; GPU replica plan for Saturday and holiday peaks; the routing policy between a small self-hosted model and a hosted model.
#
# **Platform:** any (fully executable in SMOKE mode). The rates below are placeholders — replace with measured throughput (L03) and the provider's current price list.
#
# **Done means:** cost per conversation computed for hosted, self-hosted and routed; break-even utilisation found; peak replica count derived from the arrival curve; routing policy justified with numbers.
#
# ## Step 1 — Conversation token profile from the request mix
# %%
reqs = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA_DIR, "request_mix.jsonl"))])
turns_per_conv = 4
profile = dict(prompt_tok=reqs.prompt_tokens.mean() * turns_per_conv, out_tok=reqs.max_new.mean() * 0.6 * turns_per_conv)   # ~60% of max_new used on average (assumption)
print({k: round(v) for k, v in profile.items()})

# %% [markdown]
# ## Step 2 — Hosted cost per conversation
# *Watch:* prices are placeholders; use the provider's list on the day.
# %%
HOSTED = dict(frontier=dict(in_per_M=2.50, out_per_M=10.0), small=dict(in_per_M=0.15, out_per_M=0.60))   # USD per million tokens — PLACEHOLDERS
def hosted_cost(profile, price):
    # >>> SOLUTION prompt tokens × in price + output tokens × out price, per million
    return profile["prompt_tok"] * price["in_per_M"] / 1e6 + profile["out_tok"] * price["out_per_M"] / 1e6
    # <<< SOLUTION
hc = {k: hosted_cost(profile, v) for k, v in HOSTED.items()}; print({k: round(v, 5) for k, v in hc.items()})
check(hc["frontier"] > hc["small"] * 5, "hosted cost per conversation computed for both tiers")

# %% [markdown]
# ## Step 3 — Self-hosted cost per conversation and break-even utilisation
# *Why:* a GPU costs the same idle or busy. Cost per conversation = GPU $/hr ÷ conversations per hour at the utilisation you actually achieve.
# %%
sys.path.insert(0, os.path.join(DATA_DIR, "..")); from content import GPU_SKUS
gpu_usd_hr = dict((s[0], s[3]) for s in GPU_SKUS)["Standard_NC24ads_A100_v4"]
measured_tok_s = 1800   # PLACEHOLDER: output tokens/s at concurrency 32 from the L03 harness on this SKU
def self_hosted_cost(profile, tok_s, utilisation, usd_hr=gpu_usd_hr):
    # >>> SOLUTION conv/hr = tok_s×3600×utilisation / out_tok; cost = usd_hr / conv/hr
    conv_per_hr = tok_s * 3600 * utilisation / profile["out_tok"]
    return usd_hr / conv_per_hr
    # <<< SOLUTION
for u in (0.1, 0.3, 0.6, 0.9): print(f"utilisation {u:.0%}: ${self_hosted_cost(profile, measured_tok_s, u):.5f} per conversation")
breakeven = next(u for u in np.arange(0.01, 1.0, 0.01) if self_hosted_cost(profile, measured_tok_s, u) <= hc["small"])
print(f"break-even vs hosted small tier at ~{breakeven:.0%} utilisation")
check(0 < breakeven < 1, "break-even utilisation found")

# %% [markdown]
# ## Step 4 — Routing policy: small model first, frontier for the hard 20%
# *Why:* most shopping questions are simple (stock, aisle, price). Route by a cheap classifier; escalate the rest. Compute blended cost.
# %%
def routed_cost(share_simple, simple_cost, hard_cost): return share_simple * simple_cost + (1 - share_simple) * hard_cost
blend = {s: routed_cost(s, self_hosted_cost(profile, measured_tok_s, 0.6), hc["frontier"]) for s in (0.5, 0.7, 0.8, 0.9)}
print({f"{k:.0%} simple": round(v, 5) for k, v in blend.items()})
check(blend[0.8] < hc["frontier"] * 0.4, "routing 80% of traffic to the small model cuts cost by more than 60%")

# %% [markdown]
# ## Step 5 — Capacity plan for the peak
# *Why:* replicas are sized on the arrival curve, not the average. Saturday 10:00 ≈ 4× Tuesday 15:00; Christmas week ≈ 2× a normal Saturday.
# %%
base_conv_per_hr = 3000
curve = {"Tue 15:00": 1.0, "Sat 10:00": 4.0, "Xmas Sat 10:00": 8.0}
def replicas_needed(conv_per_hr, tok_s, target_util=0.6, min_rep=2):
    # >>> SOLUTION tokens/hr needed / (tok_s×3600×target_util), ceil, at least min_rep
    need = conv_per_hr * profile["out_tok"] / (tok_s * 3600 * target_util)
    return max(min_rep, math.ceil(need))
    # <<< SOLUTION
plan = {k: replicas_needed(base_conv_per_hr * m, measured_tok_s) for k, m in curve.items()}; print(plan)
monthly = sum(blend[0.8] * base_conv_per_hr * 24 * 30 * 1.6 for _ in [0])   # 1.6 = average load factor over the week (assumption)
print(f"indicative monthly cost at 80% routing: ${monthly:,.0f} (placeholder rates)")
check(plan["Xmas Sat 10:00"] >= plan["Sat 10:00"] >= plan["Tue 15:00"], "replica plan grows with the arrival curve")
print("L17 complete. Replace every PLACEHOLDER with measured numbers before the finance review.")
