# %% [markdown]
# # L13 — Data preparation pipelines: dedup, PII, data cards (Spark + Delta)
# **Objective 16.**
#
# **Northfield Grocers context:** Supplier sheets and catalog extracts feed both the extraction fine-tune and its eval set. Northfield's pipeline must remove duplicate listings, quarantine bad taxonomy labels, redact supplier rep contact details, and guarantee no eval item leaked into training — then write Delta tables with a data card the governance team can read.
#
# **Retail use cases:** Preparing supplier-sheet corpora for extraction fine-tuning; eval-set hygiene before a model swap; supplier confidentiality.
#
# **Platform:** Databricks (no GPU). Runs in local Spark here; Delta write is exercised on Databricks.
#
# **Done means:** all planted defects (D1–D5) are detected and removed; leakage check passes; data card generated.
#
# ## Step 1 — Load with Spark
# %%
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[1]")).appName("L13").config("spark.ui.enabled", "false").getOrCreate()
cat = spark.read.csv(os.path.join(DATA_DIR, "catalog_items.csv"), header=True, inferSchema=True)
sup = spark.read.csv(os.path.join(DATA_DIR, "supplier_descriptions.csv"), header=True)
n_cat0, n_sup0 = cat.count(), sup.count(); print(n_cat0, "catalog rows;", n_sup0, "supplier rows")

# %% [markdown]
# ## Step 2 — Exact and near-duplicate removal (D1, D2)
# *Why:* duplicates in a fine-tune corpus over-weight examples; duplicates across train/eval inflate scores. Exact first (item_id + title), then a normalised key (lower, collapse whitespace).
# %%
# >>> SOLUTION dropDuplicates on [item_id,title]; add title_norm = lower(trim(regexp_replace(title,'\s+',' '))); dropDuplicates on title_norm
cat1 = cat.dropDuplicates(["item_id", "title"])
cat1 = cat1.withColumn("title_norm", F.lower(F.trim(F.regexp_replace("title", r"\s+", " "))))
cat2 = cat1.dropDuplicates(["title_norm"])
# <<< SOLUTION
n1, n2 = cat1.count(), cat2.count()
print(f"exact dupes removed: {n_cat0-n1}; near-dupes removed: {n1-n2}")
check(n_cat0 - n1 == 40 and n1 - n2 >= 25, "D1 (40 exact) and D2 (≥25 near) removed")

# %% [markdown]
# ## Step 3 — Label validation (D4)
# *Why:* an out-of-vocabulary label silently becomes a new class. Enforce the allowed set and quarantine the rest.
# %%
ALLOWED = ["Dairy", "Bakery", "Produce", "Frozen", "Beverages", "Snacks", "Household", "Personal Care"]
quarantine = cat2.filter(~F.col("category").isin(ALLOWED)); cat3 = cat2.filter(F.col("category").isin(ALLOWED))
nq = quarantine.count(); print("quarantined:", nq)
check(nq >= 1 and cat3.filter(F.col("category") == "UNKNOWN_CAT").count() == 0, "D4 quarantined")

# %% [markdown]
# ## Step 4 — PII redaction (D3)
# *Why:* supplier notes leak contact details; a fine-tuned model will happily reproduce them. Regex redaction for emails and phone numbers is the minimum; log the count so the data card can state it.
# %%
EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"; PHONE = r"\+?\d[\d\-\s]{7,}\d"
# >>> SOLUTION flag rows matching either regex; redact with regexp_replace to [EMAIL]/[PHONE]
sup1 = sup.withColumn("had_pii", F.col("description").rlike(EMAIL) | F.col("description").rlike(PHONE))
sup2 = sup1.withColumn("description", F.regexp_replace(F.regexp_replace("description", EMAIL, "[EMAIL]"), PHONE, "[PHONE]"))
# <<< SOLUTION
n_pii = sup2.filter("had_pii").count(); leftover = sup2.filter(F.col("description").rlike(EMAIL) | F.col("description").rlike(PHONE)).count()
print("PII rows redacted:", n_pii, "| leftover matches:", leftover)
check(n_pii == 30 and leftover == 0, "D3: all 30 PII rows redacted")

# %% [markdown]
# ## Step 5 — Train/eval leakage check (D5)
# *Why:* ten eval prompts were planted verbatim in train. Detect them with a join on the prompt text and remove them from eval (never from train — eval must stay clean and independent).
# %%
tr = spark.read.json(os.path.join(DATA_DIR, "train_prompts.jsonl")); ev = spark.read.json(os.path.join(DATA_DIR, "eval_prompts.jsonl"))
# >>> SOLUTION leaked = ev join tr on prompt (inner); ev_clean = ev left_anti tr on prompt
leaked = ev.join(tr.select("prompt"), "prompt", "inner"); ev_clean = ev.join(tr.select("prompt"), "prompt", "left_anti")
# <<< SOLUTION
nl, ne = leaked.count(), ev_clean.count(); print("leaked eval items:", nl, "| clean eval size:", ne)
check(nl == 10 and ne == ev.count() - 10, "D5: 10 leaked items removed from eval")

# %% [markdown]
# ## Step 6 — Write Delta and the data card
# *Why:* Delta gives versioned, ACID tables in Unity Catalog; the data card records provenance and every transformation count. On Databricks write `format("delta")` to `northfield.llmops.catalog_clean`; locally we write Parquet and keep the same card.
# %%
out = "/tmp/l13"; os.makedirs(out, exist_ok=True)
fmt = "delta" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else "parquet"
cat3.drop("title_norm").write.mode("overwrite").format(fmt).save(f"{out}/catalog_clean")
sup2.drop("had_pii").write.mode("overwrite").format(fmt).save(f"{out}/supplier_clean")
card = dict(dataset="Northfield Grocers catalog + supplier sheets (synthetic)", built=time.strftime("%Y-%m-%d"), source_rows=dict(catalog=n_cat0, supplier=n_sup0),
            transformations=dict(exact_dupes_removed=n_cat0 - n1, near_dupes_removed=n1 - n2, labels_quarantined=nq, pii_rows_redacted=n_pii, eval_leakage_removed=nl),
            final_rows=dict(catalog=cat3.count(), supplier=sup2.count(), eval=ne), intended_use="fine-tune & eval for attribute extraction", not_for="any production decision — synthetic data")
json.dump(card, open(f"{out}/data_card.json", "w"), indent=2); print(json.dumps(card, indent=2))
check(all(v > 0 for v in card["transformations"].values()), "data card records every transformation")
spark.stop(); print("L13 complete.")
