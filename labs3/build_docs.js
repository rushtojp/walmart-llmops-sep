// Builds: Lab_Catalogue.docx, Facilitator_Guide.docx, and one Lab Guide per lab — all from docs/_docs.json
const fs = require('fs'), path = require('path');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel,
        ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber, TableLayoutType } = require('docx');
const D = JSON.parse(fs.readFileSync('docs/_docs.json'));
const NAVY='21295C', DEEP='065A82', TEAL='1C7293', GOLD='E0A800', GREY='F2F2F2', RED='C00000', LIGHT='EAF3F6';
const H='Cambria', B='Calibri', M='Consolas';
const run=(t,o={})=>new TextRun({text:t,font:o.mono?M:B,size:o.size??22,bold:o.bold,italics:o.italics,color:o.color});
const p=(t,o={})=>new Paragraph({spacing:{after:o.after??110},children:Array.isArray(t)?t:[run(t,o)]});
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:300,after:130},children:[new TextRun({text:t,font:H,size:30,bold:true,color:NAVY})]});
const h2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:200,after:90},children:[new TextRun({text:t,font:H,size:25,bold:true,color:DEEP})]});
const h3=t=>new Paragraph({heading:HeadingLevel.HEADING_3,spacing:{before:140,after:60},children:[new TextRun({text:t,font:H,size:23,bold:true,color:TEAL})]});
const bullet=(t)=>new Paragraph({numbering:{reference:'bul',level:0},spacing:{after:60},children:Array.isArray(t)?t:[run(t)]});
const num=(t)=>new Paragraph({numbering:{reference:'num',level:0},spacing:{after:70},children:Array.isArray(t)?t:[run(t)]});
const bd={style:BorderStyle.SINGLE,size:4,color:'BFBFBF'}, borders={top:bd,bottom:bd,left:bd,right:bd};
function cell(t,w,o={}){const paras=(Array.isArray(t)?t:[t]).map(x=>new Paragraph({spacing:{after:30},children:[typeof x==='string'?new TextRun({text:x,font:o.mono?M:B,size:o.size??19,bold:o.bold,color:o.color}):x]}));
  return new TableCell({width:{size:w,type:WidthType.DXA},borders,margins:{top:50,bottom:50,left:80,right:80},shading:o.fill?{type:ShadingType.CLEAR,fill:o.fill,color:'auto'}:undefined,children:paras});}
function table(widths,head,rows){const tw=widths.reduce((a,b)=>a+b,0);
  const hr=new TableRow({tableHeader:true,children:head.map((t,i)=>cell(t,widths[i],{bold:true,color:'FFFFFF',fill:DEEP}))});
  const body=rows.map((r,ri)=>new TableRow({children:r.map((t,i)=>{const o={fill:ri%2?GREY:undefined};if(t&&typeof t==='object'&&t.text!==undefined){Object.assign(o,t);t=t.text;}return cell(String(t??''),widths[i],o);})}));
  return new Table({width:{size:tw,type:WidthType.DXA},columnWidths:widths,layout:TableLayoutType.FIXED,rows:[hr,...body]});}
const gap=()=>p('',{after:60});
const pb=()=>new Paragraph({children:[new PageBreak()]});
function title(kicker,main,sub){return [
 new Paragraph({spacing:{after:50},children:[new TextRun({text:kicker,font:B,size:18,bold:true,color:TEAL})]}),
 new Paragraph({spacing:{after:70},children:[new TextRun({text:main,font:H,size:40,bold:true,color:NAVY})]}),
 new Paragraph({spacing:{after:180},border:{bottom:{style:BorderStyle.SINGLE,size:12,color:GOLD,space:4}},children:[new TextRun({text:sub,font:B,size:22,italics:true,color:DEEP})]})];}
function doc(children,footerText){return new Document({styles:{default:{document:{run:{font:B,size:22}}}},
 numbering:{config:[{reference:'bul',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:270}}}}]},
   {reference:'num',levels:[{level:0,format:LevelFormat.DECIMAL,text:'%1.',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:360}}}}]}]},
 sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1200,bottom:1200,left:1440,right:1440}}},
  footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:footerText+' · Page ',font:B,size:16,color:'808080'}),new TextRun({children:[PageNumber.CURRENT],font:B,size:16,color:'808080'})]})]})},children}]});}
async function save(d,fn){fs.writeFileSync(fn,await Packer.toBuffer(d));console.log('wrote',fn);}

const statusTxt=t=>t.status==='EXECUTED'?`Executed in SMOKE mode: ${t.checks_passed} checks PASS${t.gpu_steps_skipped?`, ${t.gpu_steps_skipped} GPU-only step(s) deferred`:''}; starter ${t.starter.toLowerCase()}`:`NOT EXECUTED: ${t.error}`;

(async()=>{
 // ===================== LAB CATALOGUE
 let c=[...title('ADVANCED LABS · AZURE ML & AZURE DATABRICKS',D.programme.split(' — ')[0],'Lab catalogue, platform mapping, industry use cases and cost — '+D.version)];
 c.push(h1('1. How the package is organised'));
 c.push(p('Seventeen labs (plus a one-time L00 setup notebook) cover the v2 learning objectives — L01–L13 map one-to-one, L14–L17 deepen guardrails, RAG evaluation, tracing/lineage and cost economics. Every lab is a Jupyter notebook in two files — a starter (`Lxx_lab.ipynb`, with TODO stubs) and a solution (`Lxx_solution.ipynb`) — plus a Word lab guide. Starters are generated from solutions so they cannot drift. Each notebook auto-detects whether it is on GPU compute; on any machine without a GPU it runs the SMOKE path (hand-rolled algorithms on synthetic data), which is how the whole package was tested here.'));
 c.push(p([run('Scenario: ',{bold:true}),run(D.scenario)]));
 c.push(p([run('Pedagogy-first sequencing: ',{bold:true}),run('every lab hand-rolls the core algorithm (group-wise quantisation, online-softmax attention, top-k routing, LoRA, PSI, Thompson sampling, tensor-parallel matmul) before the framework call, so participants can read the framework\'s docs with understanding rather than faith.')]));
 c.push(h1('2. Lab list'));
 c.push(table([700,3200,1300,2100,900,1160],['Lab','Title','Platform','SKU (single node)','Hours','GPU-h'],D.labs.map(l=>[l.id,l.title,l.platform,l.sku,String(l.hours),String(l.gpu_hours)])));
 c.push(gap());
 const th=D.labs.reduce((a,l)=>a+l.hours,0), tg=D.labs.reduce((a,l)=>a+l.gpu_hours,0);
 c.push(p(`Total: ${th} lab hours, ${tg} GPU-hours per person (lab time only). Fits a 5-day programme at ~7 lab hours/day, or a 6-day track with L12, L13 and L17 (no GPU) set as pre-work / evening work.`));
 c.push(h1('3. Objective mapping and "done means"'));
 c.push(table([700,1100,4300,3260],['Lab','v2 objective(s)','Done means (binary acceptance check in the notebook)','Industry use cases'],D.labs.map(l=>[l.id,l.objectives.join(', '),l.done,l.use_cases.join(' · ')])));
 c.push(gap());
 c.push(h1('4. Northfield Grocers — where each lab lands'));
 c.push(table([2600,6760],['System','Labs'],[
  ['Store-ops SOP assistant (24k associates, handhelds)','L01 sizing · L02 GGUF for handhelds · L14 retrieval gate · L15 RAG + faithfulness on the SOP golden set · L16 incident forensics and lineage · L06 shared MoE serving'],
  ['Customer shopping assistant (app / web)','L03 engine bake-off under TTFT SLA · L04 substitutions and similar items · L05 ranker latency · L11 Saturday/Christmas scaling · L12 substitution-ranker bandit · L14 input/output guardrails incl. allergen provenance · L17 cost per conversation and routing'],
  ['Catalog & merchandising engine (38k SKUs)','L02 AWQ for nightly batch, ONNX in PIM · L07 LoRA extraction from supplier sheets · L08 taxonomy at scale on Spark · L09 release gate with allergen zero-tolerance · L13 supplier-sheet prep, PII, leakage · L10 promo-week drift'],
  ['Walmart mapping','Ad publishing ≈ promo/product copy paths (L07, L08, L12); recommendation ≈ substitutions/similar items (L04, L11, L12); catalog NLP ≈ taxonomy/attribute extraction (L07, L08, L09, L13)']]));
 c.push(gap());
 c.push(h1('5. Cost per person (Azure Databricks) — summary'));
 c.push(p('The full formula-driven model is in Cost_Estimate_Azure_Databricks_Labs.xlsx (Inputs → Lab Costs → Summary → Sensitivity). With the inputs as built (East US PAYG, 1.6× cluster overhead, one cluster per person):'));
 c.push(bullet('VM charges ≈ $94 per person (≈$56 A100 labs, ≈$35 the one 4×A100 lab, ≈$3 T4 labs).'));
 c.push(bullet([run('DBU charges: '),run('the DBU/hr multiplier for GPU VM types was not found in any consulted source and is a flagged placeholder (6 DBU/hr → ≈$77). ',{bold:true,color:RED}),run('The Sensitivity sheet shows the total from $99 (0 DBU/hr) to ≈$300 (15 DBU/hr). Confirm from the Databricks instance-types page or system.billing.usage after a 10-minute test run.')]));
 c.push(bullet('Azure ML compute-instance alternative (VM only): ≈$99 per person.'));
 c.push(bullet('Pairing two participants per cluster halves both components; spot pricing is not recommended for live labs (eviction).'));
 c.push(h1('6. Planted defects'));
 c.push(table([800,8560],['ID','Defect (all asserted programmatically by test_data.py and by the lab that must find it)'],Object.entries(D.defects).map(([k,v])=>[k,v])));
 c.push(gap());
 c.push(h1('7. Build corrections log'));
 c.push(table([700,2000,3600,3060],['ID','Where','Defect found during build','Fix'],D.corrections.map(r=>[r[0],r[1],r[2],r[3]])));
 c.push(gap());
 c.push(h1('8. Test status (this build environment)'));
 c.push(p('Environment: Linux sandbox, 1 vCPU, 3 GB RAM, no GPU, no model-hub network access. Every solution notebook (17) was executed by nbclient in SMOKE mode; every starter was compile-checked. Dependency remediation (C4/C6): the header requires nothing beyond numpy/pandas/scikit-learn; L02, L08 and L03-GPU install their own extras lazily via ensure_packages() with no restart. Verified twice: in a clean venv lacking onnx/onnxruntime/skl2onnx/pyarrow (packages self-installed, 17/17 pass) and in the system Python (17/17). DATA_DIR resolves env → package-relative → UC volume → workspace search, with an explicit message if none succeed.'));
 c.push(table([700,8560],['Lab','Result'],D.labs.map(l=>[l.id,{text:statusTxt(l.test),color:l.test.status==='EXECUTED'?undefined:RED}])));
 c.push(gap());
 c.push(p([run('What this does and does not prove: ',{bold:true}),run('the SMOKE run proves data loading, every hand-rolled algorithm, the eval harness, the gate, drift maths, the bandit, the Spark pipelines, ONNX export/inference and the cost formulas. It does not prove any GPU step (model download, vLLM/TensorRT-LLM/SGLang serving, PEFT fine-tune, Triton kernels, Ray TP=4, managed monitors). Those cells are syntax-checked only and are listed per lab in the Verification Register of each lab guide; run them on the target cluster before delivery.')]));
 await save(doc(c,'LLMOps Advanced Labs · Lab Catalogue'),'docs/Lab_Catalogue.docx');

 // ===================== FACILITATOR GUIDE
 let f=[...title('FACILITATOR GUIDE','Running the LLMOps labs on Azure ML & Databricks','Setup, timing, verification register and troubleshooting — '+D.version)];
 f.push(h1('1. Environment setup'));
 for(const [plat,items] of Object.entries(D.common_setup)){f.push(h2(plat));items.forEach(i=>f.push(bullet(i)));}
 f.push(h1('2. Suggested 5-day shape'));
 f.push(table([1200,4200,3960],['Day','Labs','Taught block before the labs'],[
  ['Day 1','L00 setup, L01, L02, L13','Lifecycle, GPU anatomy, shipping formats, data hygiene (Northfield supplier sheets)'],
  ['Day 2','L03, L04, L15','Serving engines, embeddings and substitutions, RAG for the store-ops assistant with faithfulness eval'],
  ['Day 3','L05, L06, L14','Memory-bound attention, FlashAttention, MoE; the three guardrail gates incl. allergen provenance'],
  ['Day 4','L07, L08, L09','LoRA/PEFT extraction, MLflow + Unity Catalog, encoder at catalog scale, release gate with allergen zero-tolerance'],
  ['Day 5','L10, L11, L16, L12, L17 (+ capstone scoping)','Monitoring and promo-week drift, Ray TP/autoscaling, tracing/lineage forensics, bandits, cost and routing']]));
 f.push(gap());
 f.push(h1('3. Verification register (version-sensitive claims)'));
 f.push(p('Grades: Verified = confirmed by execution here or by a primary source on 2026-09-06. Version-sensitive = correct in principle, exact API/flag names must be re-checked on the runtime. Unverified = could not be checked here.'));
 f.push(table([3400,1500,4460],['Claim / artefact','Grade','Re-verify action before delivery'],[
  ['All 17 solution notebooks execute end-to-end in SMOKE mode; 70 check() assertions pass — in a clean venv (self-install path) AND the system Python','Verified (executed twice)','Re-run test_labs.py (LAB_KERNEL=<clean kernel>) after any edit'],
  ['Planted defects D1–D8 present and detected','Verified (executed)','python test_data.py'],
  ['Cost workbook: 120 formulas, 0 errors, hand cross-check of VM and DBU totals','Verified (executed)','recalc.py after any input change'],
  ['Gemma 4 model ids (e4b-it, 26b-a4b-it, 31b-it) and Nemotron 3 Nano id','Version-sensitive','Confirm exact hub ids and licence acceptance'],
  ['vLLM / SGLang / TensorRT-LLM CLI flags; llama.cpp converter script; AutoAWQ API','Version-sensitive','Run each command on the pinned runtime; record versions in the init script'],
  ['peft LoraConfig target_modules for Gemma 4; trl SFTTrainer/SFTConfig signature','Version-sensitive','model.named_modules(); trl docs'],
  ['mlflow.transformers.log_model + UC registry; Databricks Model Serving GPU sizes','Version-sensitive','Databricks docs; pricing page for GPU_MEDIUM etc.'],
  ['Lakehouse Monitoring API module path; Azure ML model-monitor SDK','Version-sensitive','Both moved in 2025; check current docs'],
  ['ray.util.spark.setup_ray_cluster args; vLLM --distributed-executor-backend ray','Version-sensitive','Ray-on-Databricks docs'],
  ['flash-attn wheel availability for the runtime torch/CUDA','Unverified','Install on cluster; fall back to SDPA and record'],
  ['Azure VM prices and DBU rates','Approx / UNVERIFIED (DBU/hr)','Azure pricing calculator; Databricks instance-types page'],
  ['Databricks supports NC A100 v4 and NC H100 v5 instance types','Verified (Microsoft Learn, 2026-09-06)','Same page']]));
 f.push(gap());
 f.push(h1('4. Troubleshooting'));
 f.push(table([2600,6760],['Symptom','Action'],D.troubleshoot.map(r=>[r[0],r[1]])));
 f.push(gap());
 f.push(h1('5. Regenerating the package'));
 ['python make_data.py && python test_data.py — synthetic data + defect assertions','python build_notebooks.py — starters and solutions from src/','python test_labs.py — executes every solution (SMOKE) and writes TEST_REPORT.md','python build_cost_workbook.py — cost model with recalc + QA','python export_docs_json.py && node build_docs.js — all Word documents'].forEach(s=>f.push(num([run(s,{mono:true,size:20})])));
 await save(doc(f,'LLMOps Advanced Labs · Facilitator Guide'),'docs/Facilitator_Guide.docx');

 // ===================== PER-LAB GUIDES
 for(const l of D.labs){
  let g=[...title(`LAB GUIDE · ${l.id} · OBJECTIVE ${l.objectives.join(', ')}`,`${l.id} — ${l.title}`,l.summary)];
  g.push(table([2400,6960],['Field','Value'],[
   ['Platform',l.platform],['Compute SKU',l.sku],['Duration',`${l.hours} h (of which ≈${l.gpu_hours} GPU-hours)`],
   ['Files',`labs/${l.id}/${l.id}_lab.ipynb (starter) · ${l.id}_solution.ipynb · data/ folder`],
   ['Done means',l.done],['Industry use cases',l.use_cases.join(' · ')],
   ['Build test status',{text:statusTxt(l.test),color:l.test.status==='EXECUTED'?undefined:RED}]]));
  g.push(gap());
  g.push(h1('1. Before you start'));
  g.push(bullet('Open the starter notebook on the compute named above; confirm the header cell prints LAB_MODE=GPU (on a GPU node) and finds DATA_DIR.'));
  g.push(bullet('Every code cell that ends with check(...) is an acceptance test: it prints PASS or stops the notebook. Do not edit the check lines.'));
  g.push(bullet('TODO blocks in the starter carry a one-line hint. The solution notebook is released after the session — attempt first.'));
  g.push(h1('2. Steps'));
  g.push(p('Each step below mirrors a section of the notebook. "Why" is the concept, "What" is the task, "Watch" is the common mistake.'));
  l.steps.forEach((s,i)=>{g.push(h3(s.title));s.notes.forEach(n=>{const m=n.match(/^\*(Why|What|Watch)[^*]*\*:?\s*(.*)$/);g.push(m?p([run(m[1]+': ',{bold:true,color:DEEP}),run(m[2])]):p(n.replace(/\*/g,'')));});
   const key=Object.keys(l.platform_steps).find(k=>k.startsWith(s.title.split(' — ')[0]));
   if(key){g.push(p([run('On the platform: ',{bold:true,color:TEAL}),run(l.platform_steps[key])]));}});
  const extra=Object.entries(l.platform_steps).filter(([k])=>!l.steps.some(s=>k.startsWith(s.title.split(' — ')[0])));
  if(extra.length){g.push(h2('Additional platform notes'));extra.forEach(([k,v])=>g.push(p([run(k+': ',{bold:true}),run(v)])));}
  g.push(h1('3. Version-sensitive items in this lab'));
  g.push(p('Commands and API names quoted in the platform notes were correct as of the build date but change frequently. Before delivery run each GPU step on the pinned runtime and record library versions in the Facilitator Guide register.',{italics:true}));
  g.push(h1('4. Timing'));
  g.push(table([3000,1500,4860],['Block','Minutes','Notes'],[
   ['Intro + concept',String(Math.round(l.hours*60*0.2)),'Trainer-led, hand-rolled algorithm on screen'],
   ['Individual build',String(Math.round(l.hours*60*0.5)),'Starter notebook; hints at the halfway point'],
   ['GPU steps / team extension',String(Math.round(l.hours*60*0.2)),'Real model or platform step; teams of 3–4'],
   ['Review',String(Math.round(l.hours*60*0.1)),'One team demos; done-means checked live']]));
  await save(doc(g,`LLMOps Advanced Labs · ${l.id} Lab Guide`),`labs/${l.id}/${l.id}_Lab_Guide.docx`);
 }
})();
