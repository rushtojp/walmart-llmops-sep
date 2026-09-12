"""Export everything the docx builder needs (content.py + platform_steps + notebook step headings + test results) to docs/_docs.json."""
import json, re, os, glob
from content import *; from platform_steps import COMMON_SETUP, STEPS, TROUBLESHOOT
ROOT = os.path.dirname(os.path.abspath(__file__))
def steps_from_src(lab):
    txt = open(f"{ROOT}/src/{lab}.py").read(); out = []
    for m in re.finditer(r"^# ## (Step \d+ — .*?)\n((?:# \*.*?\n)*)", txt, re.M):
        title = m.group(1); notes = [re.sub(r"^# ", "", l).strip() for l in m.group(2).splitlines() if l.strip()]
        out.append(dict(title=title, notes=notes))
    return out
tests = {r["lab"]: r for r in json.load(open(f"{ROOT}/test_results.json"))}
labs = []
for L in LABS:
    labs.append({**L, "steps": steps_from_src(L["id"]), "platform_steps": STEPS.get(L["id"], {}), "test": tests.get(L["id"], {})})
json.dump(dict(programme=PROGRAMME, version=VERSION, scenario=SCENARIO, labs=labs, common_setup=COMMON_SETUP, troubleshoot=TROUBLESHOOT,
               defects=PLANTED_DEFECTS, corrections=CORRECTIONS, gpu_skus=GPU_SKUS), open(f"{ROOT}/docs/_docs.json", "w"), indent=1)
print("exported", len(labs), "labs; steps per lab:", [len(l["steps"]) for l in labs])
