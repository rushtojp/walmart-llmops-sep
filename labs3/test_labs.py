"""Execute every solution notebook in SMOKE mode; write TEST_REPORT.md. Also compile-check every starter."""
import os, glob, sys, time, json, ast, nbformat
from nbclient import NotebookClient
ROOT = os.path.dirname(os.path.abspath(__file__)); os.environ["LAB_MODE"] = "SMOKE"; os.environ["DATA_DIR"] = f"{ROOT}/data"
rows = []
for nbp in sorted(glob.glob(f"{ROOT}/labs/L*/L*_solution.ipynb")):
    lab = os.path.basename(nbp)[:3]; t0 = time.time()
    nb = nbformat.read(nbp, as_version=4)
    try:
        NotebookClient(nb, timeout=600, kernel_name=os.environ.get("LAB_KERNEL", "python3"), resources={"metadata": {"path": os.path.dirname(nbp)}}).execute()
        outs = "\n".join("".join(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")
        n_pass = outs.count("PASS "); n_skip = outs.count("GPU-only step not executed")
        status = "EXECUTED"; err = ""
        nbformat.write(nb, nbp.replace("_solution.ipynb", "_solution_executed.ipynb"))
    except Exception as e:
        status = "FAILED"; err = str(e)[:400]; n_pass = n_skip = 0
    # starter compile check
    st = nbformat.read(nbp.replace("_solution", "_lab"), as_version=4)
    try:
        for c in st.cells:
            if c.cell_type == "code": ast.parse(c.source)
        starter = "COMPILES"
    except SyntaxError as e: starter = f"SYNTAX ERROR {e}"
    rows.append(dict(lab=lab, status=status, checks_passed=n_pass, gpu_steps_skipped=n_skip, seconds=round(time.time() - t0, 1), starter=starter, error=err))
    print(rows[-1])
with open(f"{ROOT}/TEST_REPORT.md", "w") as f:
    f.write("# Lab test report (SMOKE mode, sandbox: 1 CPU, no GPU, no model-hub access)\n\n")
    f.write("| Lab | Solution notebook | `check()` PASSes | GPU-only steps skipped | Seconds | Starter |\n|---|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r['lab']} | {r['status']} | {r['checks_passed']} | {r['gpu_steps_skipped']} | {r['seconds']} | {r['starter']} |\n")
    f.write("\nFAILED details:\n" + "\n".join(f"- {r['lab']}: {r['error']}" for r in rows if r["status"] != "EXECUTED") + "\n")
json.dump(rows, open(f"{ROOT}/test_results.json", "w"), indent=2)
sys.exit(0 if all(r["status"] == "EXECUTED" and r["starter"] == "COMPILES" for r in rows) else 1)
