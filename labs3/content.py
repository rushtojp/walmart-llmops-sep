"""Single source of truth for the LLMOps Advanced Labs package (Azure ML + Azure Databricks)."""
PROGRAMME = "LLMOps for Advanced Practitioners — Azure ML & Azure Databricks Lab Package"
VERSION = "v2.0 (built 2026-09-11) — Northfield Grocers edition"
SCENARIO = ("Northfield Grocers — fictional supermarket chain (420 stores, online grocery, loyalty programme, 38,000 SKUs). "
            "Three LLM systems anchor the labs: a store-ops SOP assistant on handhelds, a customer shopping assistant in the app, "
            "and a catalog/merchandising engine. All data is synthetic and labelled as such.")

# GPU SKUs used in cost model (Azure, East US, Linux PAYG, checked 2026-09-06 — VERIFY before delivery)
GPU_SKUS = [
    # name, gpus, gpu_mem_gb, vm_usd_hr, source_note
    ("Standard_NC4as_T4_v3",     1, 16, 0.526, "1x T4 — thundercompute.com Azure GPU guide, reviewed 1 Sep 2026 (approx)"),
    ("Standard_NC24ads_A100_v4", 1, 80, 3.673, "1x A100 80GB — azurespeed / vantage / holori agree, Jul–Sep 2026"),
    ("Standard_NC48ads_A100_v4", 2, 160, 7.35, "2x A100 — vantage.sh, approx"),
    ("Standard_NC96ads_A100_v4", 4, 320, 14.69, "4x A100 — thundercompute.com, approx"),
    ("Standard_NC40ads_H100_v5", 1, 94, 6.98, "1x H100 NVL — thundercompute.com, approx"),
]
DBU_RATE_ALL_PURPOSE = 0.55   # USD per DBU, Azure Premium All-Purpose (multiple sources, approx)
DBU_RATE_JOBS = 0.15          # USD per DBU, Azure Premium Jobs compute (approx)
DBU_PER_HOUR_PLACEHOLDER = 6.0  # UNVERIFIED — DBU/hr for GPU VM types not found in any source; edit in workbook

LABS = [
 dict(id="L01", title="Platform baseline & GPU sizing", objectives=[1,2], platform="Both",
   sku="Standard_NC24ads_A100_v4", hours=1.5, gpu_hours=0.75,
   summary="Stand up the two lab platforms, read GPU telemetry, and size memory for dense, MoE and encoder workloads before touching a model.",
   use_cases=['Sizing the GPU fleet for the customer assistant ahead of the holiday peak', 'A100 vs H100 for the store-ops assistant under a handheld TTFT target'],
   done="Memory-fit calculator returns the correct GPU count for all 4 planted workloads; nvidia-smi parser passes on the provided sample."),
 dict(id="L02", title="Shipping formats: safetensors, AWQ/GPTQ, GGUF, ONNX", objectives=[3], platform="Databricks",
   sku="Standard_NC24ads_A100_v4", hours=2.0, gpu_hours=1.25,
   summary="Convert a Gemma 4 checkpoint through the shipping formats and measure size vs quality; hand-roll group-wise quantisation first so the trade-off is understood, not assumed.",
   use_cases=['Compact model on store handhelds for offline product look-ups (GGUF)', 'Shrinking the A100 footprint of the nightly supplier-sheet extraction job (AWQ)', 'Taxonomy classifier shipped as ONNX inside the PIM pipeline'],
   done="Hand-rolled int4 group quantiser reconstructs within tolerance; ONNX export round-trips; format table produced from measured sizes."),
 dict(id="L03", title="Serving bake-off: vLLM vs TensorRT-LLM vs SGLang", objectives=[4], platform="Both",
   sku="Standard_NC24ads_A100_v4", hours=2.5, gpu_hours=2.0,
   summary="Serve the same Gemma model on three engines and benchmark throughput, TTFT p50/p95 and cold start with a harness you built and validated against a mock server first.",
   use_cases=['Engine choice for the customer assistant under a p95 TTFT SLA at weekend peak', 'Whether a TensorRT-LLM engine build is worth it for the long-lived store-ops model'],
   done="Harness metrics match analytic values on the mock server; decision memo table has the team's own numbers at 3 concurrency levels."),
 dict(id="L04", title="Embeddings, vector retrieval & recall evaluation", objectives=[5], platform="Databricks",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=1.0,
   summary="Build retrieval for a recommendation feed: embed a catalog, load Qdrant (Databricks Vector Search as the managed alternative), rerank, and measure recall@k / MRR with a documented failure case.",
   use_cases=['Substitution candidates for out-of-stock items', "'Similar items' on product pages", 'RAG over merchandising-policy documents'],
   done="Brute-force and index retrieval agree on top-10 for 95% of queries; recall@10 and MRR reported; one failure case explained."),
 dict(id="L05", title="GPU kernel optimisation: FlashAttention & a fused kernel", objectives=[6], platform="Databricks",
   sku="Standard_NC24ads_A100_v4", hours=2.5, gpu_hours=1.5,
   summary="Hand-roll naive vs tiled (flash-style) attention to see why memory traffic, not FLOPs, is the bottleneck; then profile a real decode step, enable FlashAttention and write one fused Triton kernel.",
   use_cases=['p99 latency of the substitution ranker scoring 200 candidates per request', 'Fewer GPUs for long-context supplier-sheet extraction'],
   done="Tiled attention equals naive within 1e-5 while reading O(N) rather than O(N^2) from 'HBM'; before/after profiler traces captured."),
 dict(id="L06", title="Mixture-of-Experts serving", objectives=[7], platform="Databricks",
   sku="Standard_NC24ads_A100_v4", hours=2.0, gpu_hours=1.5,
   summary="Implement a top-k router and measure expert load imbalance; then serve Gemma 4 26B A4B and Nemotron 3 Nano and compare active-parameter cost to dense Gemma 4 31B.",
   use_cases=['One multi-surface assistant (handhelds, app, contact centre) on a fixed GPU budget', 'Dense vs MoE for the nightly product-description job'],
   done="Router load-imbalance metric computed; capacity-factor drop rate reported; MoE-vs-dense serving comparison table filled from measurements."),
 dict(id="L07", title="Stand up Gemma end-to-end: LoRA fine-tune, MLflow, Unity Catalog, serve", objectives=[8], platform="Databricks",
   sku="Standard_NC24ads_A100_v4", hours=3.0, gpu_hours=2.5,
   summary="Pull Gemma 4 E4B, LoRA fine-tune on Northfield supplier-sheet attribute data, evaluate with a harness, register in Unity Catalog via MLflow, and serve behind an OpenAI-compatible endpoint.",
   use_cases=['Attribute extraction (brand, size, pack, later allergen/dietary) from supplier sheets', 'House-style product descriptions respecting regulated-claim rules'],
   done="Hand-rolled LoRA update reproduces W+BA; fine-tuned endpoint passes the acceptance suite and the task eval floor."),
 dict(id="L08", title="Encoder classifiers at catalog scale (RoBERTa-class → ONNX → Spark)", objectives=[9], platform="Databricks",
   sku="Standard_NC4as_T4_v3", hours=3.0, gpu_hours=1.5,
   summary="Fine-tune a RoBERTa-class classifier for taxonomy, export to ONNX, accelerate with TensorRT/onnxruntime, and score a large item set with a Spark pandas UDF; compare cost and latency with LLM zero-shot.",
   use_cases=['Nightly taxonomy classification of 38k listings', 'Promo-copy policy screening before publish'],
   done="Macro-F1 above floor on held-out set; ONNX inference matches the trained model; Spark UDF scores the full set; cost-per-million-items derived from measured runtime."),
 dict(id="L09", title="Evaluation harness & CI/CD gating (with allergen zero-tolerance check)", objectives=[10,12], platform="Azure ML",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=0.5,
   summary="Build an eval harness with exact-match, rubric and regression checks; wire it as a gate in an Azure ML pipeline / MLflow registry promotion so a regressed model cannot ship.",
   use_cases=['Gating a model swap in the catalog-extraction service', 'Allergen accuracy as a blocking, zero-tolerance release check'],
   done="Harness blocks the planted regressed candidate and passes the good one; gate decision is written to the run."),
 dict(id="L10", title="Monitoring, drift & injected-fault alerting", objectives=[13], platform="Both",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=0.5,
   summary="Compute PSI/KS drift, latency and cost-per-request metrics; build a Lakehouse Monitoring / Azure ML model-monitor view and prove one alert fires on an injected fault.",
   use_cases=['Promo-week and holiday drift in customer questions', 'Silent provider model update raising cost per conversation'],
   done="PSI flags the planted drifted feature and not the stable ones; injected latency fault raises the alert."),
 dict(id="L11", title="Distributed serving: tensor parallel & autoscaling on Ray", objectives=[14], platform="Databricks",
   sku="Standard_NC96ads_A100_v4", hours=2.5, gpu_hours=1.5,
   summary="Shard a matmul by hand to see tensor parallelism; serve Gemma 4 31B with TP=4 via vLLM on Ray-on-Databricks; simulate autoscaling and a node loss.",
   use_cases=['Saturday-morning and Christmas peak scaling for the customer assistant', 'Two-replica availability for the store-ops assistant in trading hours'],
   done="Sharded matmul equals full within 1e-6; autoscaler simulation keeps p95 queue time under the SLA; TP=4 endpoint survives a simulated worker loss."),
 dict(id="L12", title="A/B testing & multi-armed bandits for rankers", objectives=[15], platform="Databricks",
   sku="Standard_NC4as_T4_v3", hours=1.5, gpu_hours=0.0,
   summary="Run an A/B test and a Thompson-sampling bandit over a simulated click log for two ranker versions; document the decision and the governance record.",
   use_cases=['Rolling out a substitution ranker with bounded regret', 'Comparing two promo-copy models on click-through'],
   done="Bandit converges to the planted better arm; cumulative regret below A/B; decision record written."),
 dict(id="L13", title="Data preparation pipelines: dedup, PII, data cards (Spark + Delta)", objectives=[16], platform="Databricks",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=0.0,
   summary="Build the fine-tune/eval corpus pipeline in PySpark: exact and near-duplicate removal, PII redaction, split leakage checks, Delta write, and a data card — all planted defects asserted.",
   use_cases=['Supplier-sheet corpus for extraction fine-tuning', 'Eval-set hygiene before a model swap; supplier rep PII redaction'],
   done="All 5 planted defects (D1–D5) are detected and removed; leakage check passes; data card generated."),
 dict(id="L14", title="Guardrails and structured outputs: input, retrieval and output gates", objectives=[10,13,16], platform="Both",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=0.0,
   summary="Build the three guardrail gates for the customer assistant — PII masking and injection detection at input, currency/trust filtering at retrieval, schema/claims/allergen-provenance validation at output — with every decision recorded on the response.",
   use_cases=["Blocking prompt-injection probes for staff discount codes","Masking customer phone numbers before a hosted model call","Refusing generated allergen claims not backed by the product master"],
   done="input gate masks all planted PII and flags 5 injection probes; retrieval gate drops the stale SOP chunk; output gate rejects 3 planted bad outputs and passes 3 good ones; decisions recorded."),
 dict(id="L15", title="RAG for the store-ops assistant: retrieval, grounding and faithfulness evaluation", objectives=[5,10], platform="Both",
   sku="Standard_NC4as_T4_v3", hours=2.5, gpu_hours=0.5,
   summary="Index the SOP corpus, retrieve with a currency filter, answer with citations and an out-of-scope decline, then hand-roll faithfulness and answer relevancy on the store-ops golden set and catch a planted prompt regression.",
   use_cases=["SOP and HR-policy Q&A on associate handhelds with SOP-version citations","Catching a stale-index or prompt regression before associates act on it"],
   done="retrieval hits the right SOP area for ≥90% of golden questions; stale chunk never cited; out-of-scope declined; v4 prompt regression detected by faithfulness."),
 dict(id="L16", title="Tracing, lineage and replay-with-fixed-retrieval", objectives=[13,16], platform="Both",
   sku="Standard_NC4as_T4_v3", hours=2.0, gpu_hours=0.0,
   summary="Forensics on a wrong SOP answer: segment the trace log by prompt version, reconstruct one request, replay it with retrieval fixed to attribute the fault, and use chunk lineage to isolate a bad ingestion run's blast radius.",
   use_cases=["Incident forensics on a wrong refund-window answer","Proving to compliance which policy version an answer used","Deleting only the vectors from a bad ETL run"],
   done="failures segmented by prompt version; replay under the previous prompt fixes them (fault = prompt); lineage isolates exactly the stale chunk."),
 dict(id="L17", title="Token economics, model routing and capacity planning", objectives=[2,4,15], platform="Both",
   sku="Standard_NC4as_T4_v3", hours=1.5, gpu_hours=0.0,
   summary="Cost per conversation for hosted, self-hosted and routed serving; break-even GPU utilisation; a small-model-first routing policy; and a replica plan from the Saturday and Christmas arrival curve — all formula-driven from measured inputs.",
   use_cases=["Cost per resolved customer conversation for the finance review","GPU replica plan for Saturday and holiday peaks","Routing policy between a small self-hosted model and a hosted frontier model"],
   done="cost per conversation for three serving options; break-even utilisation found; routing 80% simple traffic cuts cost >60%; replica plan grows with the arrival curve."),
]

PLANTED_DEFECTS = {
 "D1": "catalog_items.csv — 40 exact-duplicate rows (same item_id and title)",
 "D2": "catalog_items.csv — 25 near-duplicate titles differing only by case/whitespace",
 "D3": "supplier_descriptions.csv — 30 supplier sheets containing synthetic rep PII (emails, phone numbers)",
 "D4": "catalog_items.csv — 15 rows with taxonomy label outside the allowed set ('UNKNOWN_CAT')",
 "D5": "eval_prompts.jsonl — 10 eval items that also appear verbatim in train (leakage)",
 "D6": "click_log.csv — substitution ranker B has a true CTR ~40% higher than A (bandit must find it)",
 "D7": "monitoring_window.csv — feature 'title_len' drifts in week 4 (promo week); 'attr_count' does not",
 "D8": "sop_chunks.csv — one superseded SOP chunk (v2, 48-hour refund rule) sits alongside the current v3 (24 hours); retrieval gate / currency filter must drop it",
}

CORRECTIONS = [
 ("C1","make_data.py","Imperial size attribute emitted as 'in' instead of e.g. '1/4 in'; eval targets wrong (L07/L09 baseline scored 0.40).","Fixed size = tokens[3] (+ ' in'); regenerated data; L07 rule_based and L09 _base aligned."),
 ("C2","L02 Step 1","int4 reconstruction threshold set to 8% — too tight; uniform int4 on Gaussian weights is ~11–12%.","Threshold 15% for int4, added int8 <1% check; explanation in the check message."),
 ("C3","L02 Step 3 / L08 Step 3","onnxruntime StringNormalizer op requires an en_US.UTF-8 locale; absent on minimal images and some Databricks runtimes.","Lowercase in pandas, export with lowercase=False so the graph has no locale-dependent op."),
]

CORRECTIONS += [
 ("C6","header / L02 / L08","ONNX stack was a hard requirement in every lab header, so all 17 labs failed on a cluster missing packages only L02/L08 use; pandas UDFs also need pyarrow (absent on clean venvs).","Header no longer requires extras. ensure_packages() installs lazily, only in the lab that needs it, via sys.executable -m pip (no restart). Verified in a clean venv lacking onnx/skl2onnx/pyarrow: all 17 labs pass."),
 ("C4","header / Databricks","Auto pip-install + dbutils.library.restartPython() in the header aborted Run All and looped when the install landed in a different Python; header variables then undefined (NameError: pd).","Header no longer installs or restarts: it checks imports and points to L00_setup (one-time %pip + restart). DATA_DIR resolution now tries env → package-relative → UC volume and fails with a clear message."),
 ("C5","make_data D6","Regenerated RNG stream reduced the planted CTR gap below the assertion margin.","B CTR raised to 0.056 (~40% above A)."),
]
