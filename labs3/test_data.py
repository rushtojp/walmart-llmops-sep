import pandas as pd, json, re, os
D=os.path.join(os.path.dirname(__file__),"data")
cat=pd.read_csv(f"{D}/catalog_items.csv")
assert cat.duplicated(["item_id","title"]).sum()==40, "D1"
norm=cat.title.str.strip().str.lower(); assert (norm.duplicated().sum()-40)>=25, "D2"
sup=pd.read_csv(f"{D}/supplier_descriptions.csv"); assert sup.description.str.contains(r"@|\+1-555",regex=True).sum()==30, "D3"
assert (cat.category=="UNKNOWN_CAT").sum()==15, "D4"
tr={json.loads(l)["prompt"] for l in open(f"{D}/train_prompts.jsonl")}; ev=[json.loads(l)["prompt"] for l in open(f"{D}/eval_prompts.jsonl")]
assert sum(p in tr for p in ev)==10, "D5"
cl=pd.read_csv(f"{D}/click_log.csv"); r=cl.groupby("arm").click.mean(); assert r["B"]>r["A"]*1.15, "D6"
m=pd.read_csv(f"{D}/monitoring_window.csv"); g=m.groupby("week").title_len.mean(); assert g[4]-g[1]>10 and abs(m.groupby("week").attr_count.mean().diff().abs().max())<0.3, "D7"
s=pd.read_csv(f"{D}/sop_chunks.csv"); assert (s.version=="v2").sum()==1 and (s.area=="returns").sum()==3, "D8"
t=cat.title.iloc[0].split(); assert len(t)==8 and t[5:7]==["pack","of"], "title schema"
print("All planted defects D1-D8 asserted OK")
