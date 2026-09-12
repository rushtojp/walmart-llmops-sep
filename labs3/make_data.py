"""Synthetic NORTHFIELD GROCERS datasets (fictional supermarket chain) with planted defects D1-D8 (asserted in test_data.py)."""
import numpy as np, pandas as pd, json, os, random
rng = np.random.default_rng(42); random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(OUT, exist_ok=True)

CATS = ["Dairy","Bakery","Produce","Frozen","Beverages","Snacks","Household","Personal Care"]
NOUNS = {"Dairy":["whole milk","greek yogurt","cheddar cheese","salted butter","double cream"],
 "Bakery":["sourdough loaf","wholemeal bread","butter croissant","blueberry muffin","seeded bagel"],
 "Produce":["gala apple","banana bunch","baby spinach","cherry tomato","avocado ripe"],
 "Frozen":["garden peas","fish fingers","margherita pizza","vanilla icecream","mixed berries"],
 "Beverages":["orange juice","sparkling water","cold brew","oat drink","cola zero"],
 "Snacks":["salted crisps","dark chocolate","trail mix","rice cakes","tortilla chips"],
 "Household":["dish soap","paper towel","laundry pods","bin liners","glass cleaner"],
 "Personal Care":["shampoo daily","toothpaste mint","hand soap","body lotion","razor blades"]}
BRANDS=["Northfield","Harvest","Valley","Orchard","Bluebell","Meadow"]
SIZES=[("500","g"),("1","kg"),("1","L"),("250","ml"),("330","ml"),("2","L"),("750","g"),("6","pk")]
ALLERGENS={"Dairy":"milk","Bakery":"gluten;egg","Produce":"none","Frozen":"gluten;fish","Beverages":"none","Snacks":"nuts;milk","Household":"none","Personal Care":"none"}

def title(cat):
    b=rng.choice(BRANDS); n=rng.choice(NOUNS[cat]); s=SIZES[rng.integers(len(SIZES))]; q=int(rng.choice([1,2,4,6,12]))
    return f"{b} {n} {s[0]} {s[1]} pack of {q}", b, f"{s[0]} {s[1]}", q   # tokens: brand p1 p2 sizeNum sizeUnit pack of n

N=3000; rows=[]
for i in range(N):
    c=CATS[i%len(CATS)]; t,b,s,q=title(c)
    rows.append(dict(item_id=f"N{200000+i}", title=t, category=c, attributes=json.dumps({"brand":b,"size":s,"pack_qty":q}),
                     allergens=ALLERGENS[c], unit_price=round(float(rng.uniform(0.4,24)),2), store_format=str(rng.choice(["supermarket","express","online"]))))
df=pd.DataFrame(rows)
d1=df.sample(40,random_state=1); df=pd.concat([df,d1])                                  # D1 exact duplicates
d2=df.sample(25,random_state=2).copy(); d2["title"]=d2["title"].str.upper()+"  "; d2["item_id"]=d2["item_id"]+"N"; df=pd.concat([df,d2])  # D2 near-dupes
bad=df.sample(15,random_state=3).index; df.loc[bad,"category"]="UNKNOWN_CAT"          # D4 bad labels
df=df.sample(frac=1,random_state=4).reset_index(drop=True)
df.to_csv(f"{OUT}/catalog_items.csv",index=False)

# supplier product sheets (free text) with D3 PII (30)
uq=df.drop_duplicates("item_id").reset_index(drop=True)
desc=[]
for i,r in uq.head(1200).iterrows():
    txt=f"Supplier sheet: {r['title']}. Store at {'chilled' if r['category'] in ('Dairy','Produce') else 'ambient' if r['category']!='Frozen' else '-18C'}. Shelf life {int(rng.integers(3,180))} days. Allergens: {r['allergens']}."
    desc.append(dict(item_id=r["item_id"],description=txt))
pii_idx=rng.choice(len(desc),30,replace=False)
for k,j in enumerate(pii_idx):
    desc[j]["description"]+= (f" Rep contact rep{k}@example-supplier.test" if k%2==0 else f" Call +1-555-01{k:02d}-{1000+k}")
pd.DataFrame(desc).to_csv(f"{OUT}/supplier_descriptions.csv",index=False)

# extraction prompts: train 600 / eval 200 with D5 leakage (10)
mk=lambda i,pre: dict(id=f"{pre}{i}",prompt=f"Extract attributes from item {uq.iloc[i]['item_id']}: {uq.iloc[i]['title']}",target=uq.iloc[i]['attributes'])
train=[mk(i,"T") for i in range(0,600)]; evalp=[mk(i,"E") for i in range(2000,2200)]
for i in range(10): evalp[i]["prompt"]=train[i]["prompt"]; evalp[i]["target"]=train[i]["target"]
with open(f"{OUT}/train_prompts.jsonl","w") as f: [f.write(json.dumps(x)+"\n") for x in train]
with open(f"{OUT}/eval_prompts.jsonl","w") as f: [f.write(json.dumps(x)+"\n") for x in evalp]

# allergen golden set (zero-tolerance class) — 100 items, structured lookup truth
ag=[dict(item_id=r.item_id,question=f"Does {r.title} contain {a}?",allergen=a,answer=("yes" if a in r.allergens.split(";") else "no"))
    for r in uq.sample(100,random_state=7).itertuples() for a in [str(rng.choice(["milk","gluten","nuts","egg","fish"]))]]
pd.DataFrame(ag).to_csv(f"{OUT}/allergen_golden.csv",index=False)

# click log D6: substitution ranker B has CTR ~40% higher
ctr={"A":0.040,"B":0.056}
cl=[dict(ts=i,arm=a,click=int(rng.random()<ctr[a])) for i,a in enumerate(rng.choice(["A","B"],20000))]
pd.DataFrame(cl).to_csv(f"{OUT}/click_log.csv",index=False)

# monitoring window D7: title_len drifts in promo week 4
mon=[]
for wk in range(1,5):
    for i in range(500):
        tl=rng.normal(42 if wk<4 else 58,6); ac=rng.normal(3,0.8)
        mon.append(dict(week=wk,title_len=round(float(tl),1),attr_count=round(float(ac),2),latency_ms=round(float(rng.gamma(4,30)),1)))
pd.DataFrame(mon).to_csv(f"{OUT}/monitoring_window.csv",index=False)

# SOP corpus for the store-ops assistant (RAG lab) + golden Q&A; D8: one SOP chunk is stale (superseded version present)
sops=[]
areas=[("food-safety","Chilled products must be held at or below 5 C; record temperatures twice per shift."),
 ("food-safety","Frozen products must be held at -18 C or colder; discard if thawed above -12 C for more than 2 hours."),
 ("allergens","Allergen labels must be checked on every delivery; any unlabelled item is quarantined until the supplier confirms."),
 ("promotions","Promotional price tags are applied Wednesday night and removed Tuesday night; mismatches are escalated to the duty manager."),
 ("returns","Refunds for chilled goods are accepted within 24 hours with a receipt; frozen goods are not resaleable and are disposed of."),
 ("returns","Refunds over 50 in value require duty-manager approval."),
 ("hr-policy","Breaks: a 30 minute unpaid break for shifts over 6 hours; a further 15 minute paid break for shifts over 9 hours."),
 ("hr-policy","Sickness must be reported to the store manager at least 2 hours before shift start."),
 ("planogram","Planogram resets follow the weekly reset sheet; end-caps are reset every Wednesday before opening."),
 ("safety","Spillages are marked with a wet-floor sign immediately and cleaned within 10 minutes.")]
for i,(area,text) in enumerate(areas):
    sops.append(dict(chunk_id=f"SOP-{i:03d}",area=area,version="v3",effective="2026-07-01",text=text))
sops.append(dict(chunk_id="SOP-004-old",area="returns",version="v2",effective="2025-01-01",text="Refunds for chilled goods are accepted within 48 hours with a receipt."))  # D8 stale
pd.DataFrame(sops).to_csv(f"{OUT}/sop_chunks.csv",index=False)
gq=[dict(question="What temperature must chilled products be held at?",expected="5 C",area="food-safety"),
    dict(question="How long is the refund window for chilled goods?",expected="24 hours",area="returns"),
    dict(question="When are promotional price tags removed?",expected="Tuesday night",area="promotions"),
    dict(question="How long is the unpaid break for a 7 hour shift?",expected="30 minute",area="hr-policy"),
    dict(question="What temperature must frozen products be held at?",expected="-18 C",area="food-safety"),
    dict(question="When are end-caps reset?",expected="Wednesday",area="planogram"),
    dict(question="Who approves refunds over 50?",expected="duty-manager",area="returns"),
    dict(question="How quickly must a spillage be marked and cleaned?",expected="10 minutes",area="safety"),
    dict(question="What is the capital of France?",expected="OUT_OF_SCOPE",area="none"),
    dict(question="How much notice for sickness reporting?",expected="2 hours",area="hr-policy")]
pd.DataFrame(gq).to_csv(f"{OUT}/sop_golden.csv",index=False)

# request mix for serving labs (customer assistant traffic shape) + nvidia-smi sample + trace log for tracing lab
req=[dict(prompt_tokens=int(rng.integers(80,600)),max_new=int(rng.integers(32,256))) for _ in range(500)]
with open(f"{OUT}/request_mix.jsonl","w") as f: [f.write(json.dumps(x)+"\n") for x in req]
open(f"{OUT}/nvidia_smi_sample.csv","w").write("index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%]\n0, NVIDIA A100 80GB PCIe, 81920 MiB, 22144 MiB, 37 %\n1, NVIDIA A100 80GB PCIe, 81920 MiB, 0 MiB, 0 %\n")
traces=[]
for i in range(300):
    q=gq[i%9]; retrieved=[s for s in sops if s["area"]==q["area"]][:2] or [sops[0]]
    ok=rng.random()<0.85
    traces.append(dict(request_id=f"req-{i:05d}",prompt_version="qa-v3" if i<200 else "qa-v4",question=q["question"],retrieved_chunk_ids=[r["chunk_id"] for r in retrieved],
                       index_version="idx-2026-09",answer=(q["expected"] if ok else "unknown"),latency_ms=float(rng.gamma(4,40)),tokens_out=int(rng.integers(20,120)),user_feedback=int(ok or rng.random()<0.2)))
with open(f"{OUT}/trace_log.jsonl","w") as f: [f.write(json.dumps(x)+"\n") for x in traces]
open(f"{OUT}/README.md","w").write("Synthetic NORTHFIELD GROCERS teaching data (fictional supermarket chain). NOT real. Planted defects D1-D8 documented in content.py.\n")
print("data written")
