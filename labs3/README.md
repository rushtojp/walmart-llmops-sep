# LLMOps Advanced Labs — Azure ML & Azure Databricks (v2.0, built 2026-09-11, Northfield Grocers edition)

17 labs (+ L00 setup) mapped to the v2 learning objectives, framed around a fictional supermarket chain. Synthetic data with planted defects D1–D8.

**Databricks first run:** upload `data/` to a Unity Catalog volume (or import the whole package folder) and open any lab → Run All. Labs install their own extra packages lazily (no restart): L02/L08 need onnx, onnxruntime, skl2onnx; L08 pyarrow; L03-GPU httpx. `labs/L00_Setup/L00_setup.ipynb` is optional (cluster-wide install so nothing installs during a session).

```
content.py            single source of truth (labs, SKUs, rates, defects, corrections)
platform_steps.py     cluster setup + every GPU-only step referenced from the notebooks
make_data.py / test_data.py     synthetic data + defect assertions
src/Lxx.py            solution sources (cell-marker format; SOLUTION blocks)
build_notebooks.py    -> labs/Lxx/Lxx_solution.ipynb + Lxx_lab.ipynb (starter generated from solution)
test_labs.py          executes every solution in SMOKE mode -> TEST_REPORT.md, *_solution_executed.ipynb
build_cost_workbook.py-> docs/Cost_Estimate_Azure_Databricks_Labs.xlsx (formula-driven, recalc'd, QA'd)
export_docs_json.py + build_docs.js -> docs/Lab_Catalogue.docx, docs/Facilitator_Guide.docx, labs/Lxx/Lxx_Lab_Guide.docx
```

Rebuild everything: `python make_data.py && python test_data.py && python build_notebooks.py && python test_labs.py && python build_cost_workbook.py && python export_docs_json.py && node build_docs.js`

## What was tested here
- All 17 solution notebooks executed end-to-end in SMOKE mode twice: in a clean venv that lacked onnx/onnxruntime/skl2onnx/pyarrow (labs self-installed them) and in the system Python — 70 `check()` assertions pass each time. Starters compile. See TEST_REPORT.md and TEST_REPORT_cleanenv.md.
- Planted defects D1–D8 asserted. Cost workbook: 120 formulas, 0 errors, hand cross-checked.
- **Not tested here (syntax-checked only):** every GPU/platform step — model download, vLLM/SGLang/TensorRT-LLM serving, AWQ/GGUF conversion, PEFT fine-tune, Triton kernel, torch profiler, Ray TP=4, Databricks Model Serving, Lakehouse Monitoring, Azure ML pipeline submission. See Facilitator_Guide.docx §3 and TEST_REPORT.md.

## Run on the target platform
Databricks: upload `data/` to `/Volumes/northfield/llmops/labdata` (or import the whole package so `data/` is package-relative), import `labs/`, GPU ML runtime with the init script in Facilitator_Guide §1. Notebooks auto-detect `LAB_MODE=GPU`.
