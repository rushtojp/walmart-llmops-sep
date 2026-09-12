"""Convert src/Lxx.py (cell-marker format) into labs/Lxx/Lxx_solution.ipynb and Lxx_lab.ipynb.
Starters are generated from solutions: every '# >>> SOLUTION hint' ... '# <<< SOLUTION' block becomes a TODO stub."""
import re, os, glob, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
ROOT = os.path.dirname(os.path.abspath(__file__))
HEADER = open(f"{ROOT}/src/_header.py").read()

def parse_cells(text):
    cells, cur, kind = [], [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind: cells.append((kind, "\n".join(cur).strip("\n")))
            kind = "md" if "[markdown]" in line else "code"; cur = []
        else:
            cur.append(line)
    if kind: cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for k, body in cells:
        if k == "md": body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        out.append((k, body))
    return out

SOL_RE = re.compile(r"^(\s*)# >>> SOLUTION ?(.*?)\n(.*?)^\s*# <<< SOLUTION[^\n]*\n?", re.S | re.M)
def to_starter(code):
    def repl(m):
        ind, hint = m.group(1), m.group(2).strip()
        stub = f"{ind}# TODO: {hint or 'implement this block (see lab guide)'}\n{ind}raise NotImplementedError('complete this step')\n"
        return stub
    return SOL_RE.sub(repl, code)

def to_solution(code):
    return re.sub(r"^\s*# (>>>|<<<) SOLUTION[^\n]*\n?", "", code, flags=re.M)

def build(src_path):
    lab = os.path.basename(src_path)[:-3]
    cells = parse_cells(HEADER + "\n" + open(src_path).read())
    # move header code cell after the title markdown so the notebook opens with its title
    hdr, title = cells[0], cells[1]; rest = cells[2:]
    ordered = [title, hdr] + rest
    outdir = f"{ROOT}/labs/{lab}"; os.makedirs(outdir, exist_ok=True)
    for variant, fn in (("solution", to_solution), ("lab", to_starter)):
        nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}})
        for k, body in ordered:
            nb.cells.append(new_markdown_cell(body) if k == "md" else new_code_cell(fn(body)))
        nbformat.write(nb, f"{outdir}/{lab}_{variant}.ipynb")
    n_todo = sum(fn_body.count("# TODO:") for k, fn_body in [(k, to_starter(b)) for k, b in ordered if k == "code"])
    return lab, len(ordered), n_todo

if __name__ == "__main__":
    for p in sorted(glob.glob(f"{ROOT}/src/L*.py")):
        lab, ncells, ntodo = build(p); print(f"{lab}: {ncells} cells, {ntodo} TODO blocks in starter")
