from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
DEEP,NAVY="065A82","21295C"; H=Font(name="Cambria",size=11,bold=True,color="FFFFFF"); B=Font(name="Calibri",size=11)
WRAP=Alignment(wrap_text=True,vertical="top"); thin=Side(style="thin",color="BFBFBF"); BOX=Border(top=thin,bottom=thin,left=thin,right=thin)
# (lab, difficulty, scenario/question, A, B, C, D, answer, rationale)
PRE=[
("L01","Easy","Your team plans to serve a 31B-parameter dense model in bf16 on a single A100 80GB. Before any KV cache, roughly how much memory do the weights alone need?","~31 GB","~62 GB","~124 GB","~15 GB","B","bf16 = 2 bytes per parameter → 31 × 2 = 62 GB, already over one 80GB card once KV cache and overhead are added."),
("L01","Medium","A colleague says a 26B-total / 4B-active MoE model needs only about 8 GB of GPU memory because 'only 4B parameters are active'. What is wrong?","Nothing — active parameters determine memory","All expert weights must be resident; active parameters reduce compute, not memory","MoE models cannot run on a single GPU","KV cache is not needed for MoE","B","Routing picks experts per token, but every expert's weights must be loaded. ~26B × 2 bytes ≈ 52 GB in bf16."),
("L02","Easy","The edge team wants to run a small model on in-store devices with CPU only, using llama.cpp. Which shipping format do they need?","ONNX","safetensors bf16","GGUF","AWQ int4 for vLLM","C","GGUF is llama.cpp's format and is the usual choice for CPU/edge inference."),
("L02","Medium","After int4 quantisation with group size 256, one linear layer shows a much larger error than the others. Inspection shows a single very large weight. What is the most likely cause and the simplest first fix?","Rounding bug; re-run the converter","An outlier inflates its group's scale; use a smaller group size (e.g. 64 or 128)","int4 is unusable for that layer; keep it in fp32","The layer needs a bigger learning rate","B","One outlier sets the absmax scale for the whole group; smaller groups limit the damage (AWQ/GPTQ address this more cleverly)."),
("L03","Easy","For an ad-copy generation API, product managers care most about how quickly the first words appear on screen. Which metric should the serving benchmark report for that?","Throughput (tokens/s)","Time to first token (TTFT) p50/p95","Cold-start time","GPU utilisation","B","TTFT is the user-perceived responsiveness metric; throughput matters for cost, not first-word latency."),
("L03","Medium","You need to serve many different Hugging Face models that change weekly, and the deployment must be online within two minutes of a model swap. Which engine choice best fits?","TensorRT-LLM with compiled engines","vLLM (or SGLang) serving directly from the checkpoint","Export to ONNX and use onnxruntime","Run the model in a Spark UDF","B","TensorRT-LLM's engine build step can take tens of minutes; vLLM/SGLang load checkpoints directly with a much faster cold start."),
("L04","Easy","The recommendation team reports retrieval quality with recall@10 = 0.82. What does this number mean?","82% of queries return at least one result","On average 82% of the relevant items appear in the top 10 results","82% of retrieved items are relevant","The index answers in 82 ms","B","recall@k = |top-k ∩ relevant| / |relevant| (capped at k), averaged over queries."),
("L05","Easy","Why is standard attention on a long sequence usually limited by memory bandwidth rather than arithmetic?","It materialises an N×N score matrix that must be written to and read from HBM","Softmax is expensive","GPUs have too few cores","Matrix multiplication is slow on GPUs","A","The N² score/probability matrices are the traffic; FlashAttention avoids materialising them."),
("L06","Medium","An MoE serving dashboard shows expert 0 receiving 40% of tokens while the other seven get ~8% each. What operational effect should you expect?","Lower memory use","Higher accuracy","The GPU holding expert 0 becomes the bottleneck while others idle; latency rises","No effect — routing is free","C","Load imbalance leaves capacity unused on other experts/GPUs; capacity factors and load-balancing losses exist to fight this."),
("L07","Easy","You fine-tune a 4B model with LoRA (rank 16) on attention projections. Roughly what fraction of parameters is trainable?","About half","Well under 1%","All of them","Exactly 16%","B","LoRA trains only small A and B matrices; trainable parameters are typically a fraction of a percent of the base model."),
("L08","Easy","Nightly you must classify 5 million catalog titles into 8 taxonomy categories on a fixed budget. Which is the most cost-effective default?","Zero-shot prompting of a 31B LLM per item","A fine-tuned RoBERTa-class encoder classifier exported to ONNX and run in batch","Manual labelling","Embedding search against 8 category names with a 31B model","B","Encoder classifiers run thousands of items/second per node; LLM zero-shot costs tokens per item and is orders of magnitude more expensive for high-volume simple classification."),
("L09","Medium","A new model version scores exact-match 0.86 (floor 0.80) but its output is missing the 'size' key in 15% of responses. Your CI gate only checks exact-match. What is the risk and the fix?","No risk — it passed the floor","Downstream JSON parsers break; add a schema-validity check and a no-regression-vs-champion rule","Lower the floor to 0.70","Retrain with more epochs","B","A single metric hides structural regressions; gates should check schema validity and compare to the champion within a tolerance."),
("L10","Easy","Monitoring shows PSI = 0.32 for input title length this week versus the baseline, and 0.03 for attribute count. How should this be read?","Both features drifted","Title length drifted materially (>0.25); attribute count is stable (<0.1)","Neither drifted","PSI cannot be compared across features","B","Common PSI thresholds: <0.1 stable, 0.1–0.25 watch, >0.25 drift."),
("L12","Medium","You want to roll out ranker B against ranker A while limiting the traffic sent to the worse version. Which approach adapts traffic as evidence accumulates?","Fixed 50/50 A/B test for two weeks","Thompson-sampling (multi-armed bandit)","Deploy B to 100% and watch","Shadow mode with no traffic","B","Bandits shift allocation toward the better arm, giving lower cumulative regret than a fixed split."),
("L13","Easy","Ten evaluation prompts also appear verbatim in the fine-tuning training set. What should the data pipeline do?","Remove them from the training set","Remove them from the evaluation set so eval stays independent","Keep both — more data is better","Duplicate them to balance","B","Eval must be clean and independent; train can keep its rows, the leaked items leave eval."),
]
POST=[
("L01","Medium","Using the lab's memory-fit formula, a dense 31B bf16 model with a KV cache of ~16 GB (batch 8, 8k context) and 10% overhead totals ~86 GB. With 90% usable memory per GPU, how many A100-80GB cards are needed?","1","2","3","4","B","86 / (80 × 0.9 = 72) = 1.19 → ceil = 2. The KV cache and overhead push it over one card."),
("L02","Medium","In L02 you exported a classifier to ONNX and checked prediction agreement with the source model on the full test set. Why is this check mandatory before shipping?","ONNX files are always smaller","A converter can silently change behaviour; equivalence must be proven, not assumed","onnxruntime requires it","It speeds up inference","B","Export equivalence is a release gate; a silent mismatch is a production incident."),
("L03","Medium","Your harness measured TTFT p50 = 52 ms on a mock server configured for 50 ms first-token latency. The real vLLM run shows p95 TTFT rising sharply at concurrency 32. What is the most likely interpretation?","The harness is broken","Requests are queueing under continuous batching at that load; you are near the engine's capacity","vLLM does not support concurrency","TTFT is not affected by load","B","The mock validated the harness; the p95 growth is the real engine saturating — the point of testing three concurrency levels."),
("L03","Easy","In the bake-off you must record TensorRT-LLM's engine-build time separately from its cold start. Why?","Engine build is billed differently","Engine build is a one-off cost that changes the decision for short-lived vs long-lived deployments","It is always zero","Only vLLM has a cold start","B","A long build is acceptable for a single long-lived model but not for weekly model swaps."),
("L04","Medium","Brute-force search and your HNSW index agree on only 88% of top-10 results, below the 95% target. What is the right next action?","Switch to keyword search","Tune index parameters (e.g. ef_search / M) and re-measure agreement against brute force","Accept it — approximate search is always lossy","Reduce k to 5","B","Exact search is the reference; index recall is tuned by its parameters and re-measured, not assumed."),
("L05","Medium","In the tiled attention you wrote, why must the accumulated output be rescaled by exp(m_old − m_new) when a new block raises the running max?","To normalise the output to unit length","Because softmax is computed relative to the max; earlier partial sums were scaled with the old max and must be corrected","To reduce memory","It is optional and only improves speed","B","This is the online-softmax recurrence that makes streaming K/V blocks exact."),
("L05","Easy","Your profiler trace shows attention kernels at 61% of decode time with eager attention and 34% with FlashAttention, with the same output. What changed?","FLOPs were reduced","HBM traffic for the N×N matrices was eliminated, so the kernel became less memory-bound","The model got smaller","Batch size increased","B","Same maths, less memory movement — the FlashAttention argument demonstrated by measurement."),
("L06","Medium","At capacity factor 1.0 your skewed router drops 9% of tokens; at 1.25 it drops 1%. What is the trade-off of raising the capacity factor?","No trade-off — always use the highest","More padding/compute and memory per expert for headroom versus fewer dropped tokens","Lower accuracy","It disables routing","B","Capacity factor buys reliability with wasted capacity; balancing the router reduces the need for it."),
("L06","Easy","Your MoE-vs-dense table shows Gemma 4 26B A4B at ~52 GB resident memory and ~13% of the dense 31B model's per-token compute. Which resource decides whether it fits on one A100-80GB?","Compute","Memory (weights + KV cache)","Network bandwidth","Number of experts","B","Feasibility is decided by resident memory; compute decides throughput and cost once it fits."),
("L07","Medium","After LoRA training you 'merge and unload' before registering the model in Unity Catalog. What does the merge do and why is it safe?","Deletes the adapter; behaviour changes","Adds B·A (scaled) into W so the merged forward pass equals the adapter forward pass — proven in Step 2 of the lab","Compresses the model to int4","Converts to ONNX","B","W_merged = W + (α/r)·B·A; the lab asserted adapter and merged outputs are equal."),
("L07","Easy","Your fine-tuned Gemma model scores exact-match 0.78 on the eval set, while the rule-based baseline scores 0.95. Per the lab's acceptance policy, what happens?","Ship the LLM — it is more modern","Do not ship; a tuned model that loses to the rule baseline fails the gate","Ship both","Lower the floor to 0.75","B","The floor and the baseline comparison exist precisely to stop this."),
("L08","Medium","Your Spark pandas UDF for ONNX scoring runs slower than the single-node loop. The code creates a new InferenceSession inside the UDF function. What is the fix?","Use more executors","Cache the session at module level so it is created once per executor process, and raise the Arrow batch size","Switch to a Python UDF","Disable Arrow","B","Session creation per batch dominates; the lab caches the session and tunes maxRecordsPerBatch."),
("L09","Easy","Your Azure ML pipeline has train → eval_gate → register. The eval_gate component exits with code 1 for the 'regressed' candidate. What is the intended outcome?","Registration proceeds with a warning","The pipeline fails at eval_gate and registration never runs","The model is registered as champion","The component retries","B","A non-zero exit fails the job and blocks the downstream registration step — that is the gate."),
("L10","Medium","You inject a +40% latency fault into a copy of week 4 and the p95 alert (threshold 1.25× baseline) fires, while the unmodified week 4 stays silent. Why is this test part of the lab deliverable?","Because alerts are optional","An alert that has never been seen firing is unproven; the injected fault demonstrates both sensitivity and no false positive","To increase cost","To test the GPU","B","Monitoring is only trustworthy once you have watched it detect a known fault."),
("L11","Medium","In the autoscaling simulation a replica dies at minute 300 with a minimum of 2 replicas configured. Queue wait spikes briefly then recovers. Which two mechanisms produced the recovery?","Spot pricing and caching","Minimum-replica floor for HA plus scale-up when queue wait approaches the SLA","Tensor parallelism and quantisation","Larger batch size and FlashAttention","B","HA minimum keeps service alive; the scale-up rule restores capacity — the L11 simulation's two levers."),
]
import random
def shuffle(rows, seed):
    rng=random.Random(seed); out=[]
    for r in rows:
        opts=list(r[3:7]); correct=opts["ABCD".index(r[7])]; rng.shuffle(opts)
        out.append((r[0],r[1],r[2],*opts,"ABCD"[opts.index(correct)],r[8]))
    return out
PRE=shuffle(PRE,11); POST=shuffle(POST,23)
wb=Workbook()
def sheet(ws,title,rows):
    ws["A1"]=title; ws["A1"].font=Font(name="Cambria",size=14,bold=True,color=NAVY)
    ws["A2"]="Scenario-based multiple choice. Difficulty easy–medium. Mapped to the LLMOps Advanced Labs (Azure ML / Azure Databricks). Answer key and rationale in columns I–J (hide before distribution)."; ws["A2"].font=Font(name="Calibri",size=11,italic=True)
    hdr=["#","Lab","Difficulty","Scenario / question","A","B","C","D","Correct","Rationale"]
    for i,h in enumerate(hdr,1):
        c=ws.cell(row=4,column=i,value=h); c.font=H; c.fill=PatternFill("solid",fgColor=DEEP); c.alignment=WRAP; c.border=BOX
    for r,row in enumerate(rows,5):
        vals=[r-4,*row]
        for i,v in enumerate(vals,1):
            c=ws.cell(row=r,column=i,value=v); c.font=B; c.alignment=WRAP; c.border=BOX
            if r%2==0: c.fill=PatternFill("solid",fgColor="F2F2F2")
    for col,w in zip("ABCDEFGHIJ",(5,7,11,60,28,34,28,28,9,60)): ws.column_dimensions[col].width=w
    ws.freeze_panes="A5"
sheet(wb.active,"Pre-Test Question Set (15) — LLMOps for Advanced Practitioners",PRE); wb.active.title="Pre-Test"
sheet(wb.create_sheet("Post-Test"),"Post-Test Question Set (15) — LLMOps for Advanced Practitioners",POST)
# answer sheet template
for nm in ("Pre","Post"):
    ws=wb.create_sheet(f"{nm}-Test Answer Sheet")
    ws["A1"]=f"{nm}-Test answer sheet — participant enters A/B/C/D in column B; score computed by formula"; ws["A1"].font=Font(name="Cambria",size=14,bold=True,color=NAVY)
    ws["A2"]="Participant name:"; ws["B2"]=""; ws["B2"].fill=PatternFill("solid",fgColor="FFFF00")
    for i,h in enumerate(["#","Your answer","Correct","Mark"],1):
        c=ws.cell(row=4,column=i,value=h); c.font=H; c.fill=PatternFill("solid",fgColor=DEEP); c.border=BOX
    dv=DataValidation(type="list",formula1='"A,B,C,D"',allow_blank=True); ws.add_data_validation(dv)
    for q in range(1,16):
        r=q+4; ws.cell(row=r,column=1,value=q); ws.cell(row=r,column=2).fill=PatternFill("solid",fgColor="FFFF00"); dv.add(ws.cell(row=r,column=2))
        ws.cell(row=r,column=3,value=f"='{nm}-Test'!I{r}").font=Font(name="Calibri",size=11,color="008000")
        ws.cell(row=r,column=4,value=f'=IF(B{r}="","",IF(B{r}=C{r},1,0))')
        for c in range(1,5): ws.cell(row=r,column=c).border=BOX; ws.cell(row=r,column=c).alignment=WRAP
        if ws.cell(row=r,column=2).font.size!=11: ws.cell(row=r,column=2).font=B
        ws.cell(row=r,column=1).font=B; ws.cell(row=r,column=4).font=B
    ws["A21"]="Score"; ws["A21"].font=Font(name="Calibri",size=11,bold=True); ws["B21"]="=SUM(D5:D19)"; ws["B21"].font=Font(name="Calibri",size=11,bold=True)
    ws["C21"]="of 15"; ws["C21"].font=B; ws["D21"]='=IF(COUNT(D5:D19)=0,"",B21/15)'; ws["D21"].number_format="0%"; ws["D21"].font=B
    ws.column_dimensions["A"].width=8; ws.column_dimensions["B"].width=14; ws.column_dimensions["C"].width=10; ws.column_dimensions["D"].width=10
    ws.sheet_properties.tabColor="E0A800"
out="docs/LLMOps_Pre_Post_Test_Questions.xlsx"; wb.save(out)
import subprocess; print(subprocess.run(["python3","/mnt/skills/public/xlsx/scripts/recalc.py",out,"60"],capture_output=True,text=True).stdout)
# QA: answers are valid letters, correct answer text nonempty, no two sets share a question
assert all(r[7] in "ABCD" for r in PRE+POST); assert len({r[2] for r in PRE}&{r[2] for r in POST})==0
from collections import Counter; print("Pre answers:",Counter(r[7] for r in PRE),"Post:",Counter(r[7] for r in POST)); print("Pre difficulty:",Counter(r[1] for r in PRE),"Post:",Counter(r[1] for r in POST))
