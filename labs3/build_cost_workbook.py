"""Formula-driven cost estimate for running the labs per person on Azure Databricks (and Azure ML VM-only)."""
import os, subprocess, json
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.comments import Comment
from content import LABS, GPU_SKUS, DBU_RATE_ALL_PURPOSE, DBU_RATE_JOBS, DBU_PER_HOUR_PLACEHOLDER
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = f"{ROOT}/docs/Cost_Estimate_Azure_Databricks_Labs.xlsx"
NAVY, DEEP, TEAL, MINT, GOLD = "21295C", "065A82", "1C7293", "16A0A0", "E0A800"
H = Font(name="Cambria", size=11, bold=True, color="FFFFFF"); B = Font(name="Calibri", size=11); BLUE = Font(name="Calibri", size=11, color="0000FF")
RED = Font(name="Calibri", size=11, bold=True, color="C00000"); WRAP = Alignment(wrap_text=True, vertical="top")
thin = Side(style="thin", color="BFBFBF"); BOX = Border(top=thin, bottom=thin, left=thin, right=thin)
YEL = PatternFill("solid", fgColor="FFFF00"); REDF = PatternFill("solid", fgColor="F8CBAD")

def hdr(ws, row, labels, fill=DEEP):
    for i, l in enumerate(labels, 1):
        c = ws.cell(row=row, column=i, value=l); c.font = H; c.fill = PatternFill("solid", fgColor=fill); c.alignment = WRAP; c.border = BOX
def put(ws, ref, v, font=None, fmt=None, fill=None, comment=None):
    c = ws[ref]; c.value = v; c.font = font or B; c.alignment = WRAP; c.border = BOX
    if fmt: c.number_format = fmt
    if fill: c.fill = fill
    if comment: c.comment = Comment(comment, "build")
    return c

wb = Workbook()
# ---------------- Inputs
ws = wb.active; ws.title = "Inputs"
ws["A1"] = "Cost model inputs — edit BLUE cells only. Yellow = key assumption. Red = UNVERIFIED, must be confirmed before quoting."; ws["A1"].font = Font(name="Cambria", size=14, bold=True, color=NAVY)
ws["A2"] = "All prices USD, Azure East US, Linux pay-as-you-go, as published by third-party trackers on/around 2026-09-06 (approx). Re-check on Azure pricing calculator before delivery."; ws["A2"].font = Font(name="Calibri", size=11, italic=True)
hdr(ws, 4, ["SKU", "GPUs", "GPU memory (GB)", "VM $/hr (PAYG)", "Source / note"])
for i, (n, g, m, p, src) in enumerate(GPU_SKUS, 5):
    put(ws, f"A{i}", n); put(ws, f"B{i}", g); put(ws, f"C{i}", m); put(ws, f"D{i}", p, BLUE, "$#,##0.000"); put(ws, f"E{i}", src)
r = 5 + len(GPU_SKUS) + 1
labels = [
 ("DBU rate — All-Purpose compute, Premium ($/DBU)", DBU_RATE_ALL_PURPOSE, "Multiple 2026 pricing guides agree ≈$0.55; confirm on Azure Databricks pricing page", YEL),
 ("DBU rate — Jobs compute, Premium ($/DBU)", DBU_RATE_JOBS, "≈$0.15 per several 2026 guides; confirm", YEL),
 ("DBUs consumed per hour by the GPU VM (single node)", DBU_PER_HOUR_PLACEHOLDER, "UNVERIFIED PLACEHOLDER — not published in any source consulted. Read from databricks.com instance-types page or system.billing.usage after a 10-minute test run. See Sensitivity sheet.", REDF),
 ("Cluster-running overhead factor (cluster up ÷ lab GPU-hours)", 1.6, "Cluster runs during instruction, debugging and idle gaps; 1.6× is a trainer estimate — measure on cohort 1", YEL),
 ("Participants sharing one single-node GPU cluster", 1, "1 = one cluster per person. Set 2–3 if participants pair on a cluster (halves/thirds VM+DBU cost)", YEL),
 ("Cohort size (people)", 12, "For the cohort total on Summary", YEL),
 ("Storage + networking allowance per person ($)", 5, "Trainer allowance: Unity Catalog volume, model downloads, egress; small relative to GPU time", YEL),
 ("Azure ML alternative — compute-instance only (no DBU): apply same VM $/hr", 1, "1 = yes. Azure ML compute instances bill VM only (no DBU); managed endpoints bill VM-hours too", YEL),
]
hdr(ws, r, ["Assumption", "Value", "Note"]); NAMES = {}
for j, (lab, val, note, fill) in enumerate(labels, r + 1):
    put(ws, f"A{j}", lab); c = put(ws, f"B{j}", val, RED if fill is REDF else BLUE, fill=fill); put(ws, f"C{j}", note)
    NAMES[lab] = f"Inputs!$B${j}"
DBU_AP, DBU_JOBS, DBU_HR, OVH, SHARE, COHORT, STOR, AML = [NAMES[l[0]] for l in labels]
for col, w in zip("ABCDE", (52, 14, 16, 16, 90)): ws.column_dimensions[col].width = w
ws.freeze_panes = "A5"

# ---------------- Lab Costs
lc = wb.create_sheet("Lab Costs")
lc["A1"] = "Per-person cost by lab — every number is a formula from Inputs"; lc["A1"].font = Font(name="Cambria", size=14, bold=True, color=NAVY)
lc["A2"] = "VM cost = GPU-hours × overhead × VM $/hr ÷ sharing.  DBU cost = GPU-hours × overhead × DBU/hr × DBU rate ÷ sharing.  Azure ML column = VM cost only."; lc["A2"].font = Font(name="Calibri", size=11, italic=True)
cols = ["Lab", "Title", "Platform", "SKU", "Lab hours", "GPU-hours (lab)", "VM $/hr", "Cluster hours (×overhead)", "VM cost ($)", "DBU cost ($)", "Databricks total ($)", "Azure ML VM-only ($)"]
hdr(lc, 4, cols)
first = 5
for i, L in enumerate(LABS, first):
    put(lc, f"A{i}", L["id"]); put(lc, f"B{i}", L["title"]); put(lc, f"C{i}", L["platform"]); put(lc, f"D{i}", L["sku"])
    put(lc, f"E{i}", L["hours"], BLUE, "0.0"); put(lc, f"F{i}", L["gpu_hours"], BLUE, "0.00")
    put(lc, f"G{i}", f"=INDEX(Inputs!$D$5:$D${4+len(GPU_SKUS)},MATCH(D{i},Inputs!$A$5:$A${4+len(GPU_SKUS)},0))", Font(name="Calibri", size=11, color="008000"), "$#,##0.000")
    put(lc, f"H{i}", f"=F{i}*{OVH}", fmt="0.00")
    put(lc, f"I{i}", f"=H{i}*G{i}/{SHARE}", fmt="$#,##0.00")
    put(lc, f"J{i}", f"=H{i}*{DBU_HR}*{DBU_AP}/{SHARE}", fmt="$#,##0.00")
    put(lc, f"K{i}", f"=I{i}+J{i}", fmt="$#,##0.00")
    put(lc, f"L{i}", f"=I{i}*{AML}", fmt="$#,##0.00")
last = first + len(LABS) - 1; tot = last + 1
put(lc, f"A{tot}", "TOTAL", Font(name="Calibri", size=11, bold=True))
for col in "EFHIJKL":
    put(lc, f"{col}{tot}", f"=SUM({col}{first}:{col}{last})", Font(name="Calibri", size=11, bold=True), "$#,##0.00" if col in "IJKL" else "0.00")
for col, w in zip("ABCDEFGHIJKL", (7, 52, 12, 26, 10, 12, 11, 14, 12, 12, 14, 14)): lc.column_dimensions[col].width = w
lc.freeze_panes = "C5"; lc.auto_filter.ref = f"A4:L{last}"

# ---------------- Summary
sm = wb.create_sheet("Summary")
sm["A1"] = "Cost summary — per person and per cohort"; sm["A1"].font = Font(name="Cambria", size=14, bold=True, color=NAVY)
hdr(sm, 3, ["Metric", "Value", "Formula basis"])
rows = [
 ("Total lab hours per person", f"='Lab Costs'!E{tot}", "0.0", "Sum of lab durations"),
 ("Total GPU-hours per person (lab time only)", f"='Lab Costs'!F{tot}", "0.00", "Sum of per-lab GPU-hours"),
 ("Cluster hours per person (with overhead)", f"='Lab Costs'!H{tot}", "0.00", "GPU-hours × overhead factor"),
 ("Databricks — VM cost per person", f"='Lab Costs'!I{tot}", "$#,##0.00", "Azure VM charges"),
 ("Databricks — DBU cost per person", f"='Lab Costs'!J{tot}", "$#,##0.00", "UNVERIFIED DBU/hr input — see Sensitivity"),
 ("Databricks — storage/network allowance per person", f"={STOR}", "$#,##0.00", "Inputs"),
 ("Databricks — TOTAL per person", f"=B7+B8+B9", "$#,##0.00", "VM + DBU + allowance"),
 ("Azure ML compute-instance alternative — per person (VM only)", f"='Lab Costs'!L{tot}+{STOR}", "$#,##0.00", "No DBU component"),
 ("Cohort total — Databricks", f"=B10*{COHORT}", "$#,##0.00", "× cohort size"),
 ("Cohort total — Azure ML", f"=B11*{COHORT}", "$#,##0.00", "× cohort size"),
 ("Spot-pricing sensitivity — Databricks VM cost per person at 30% of PAYG", f"=B7*0.3", "$#,##0.00", "Spot for A100 SKUs has run ~$0.68–1.15/hr vs $3.67 PAYG; eviction risk unsuitable for live labs — reference only"),
]
for j, (lab, f, fmt, basis) in enumerate(rows, 4):
    put(sm, f"A{j}", lab, Font(name="Calibri", size=11, bold=(j in (10, 11)))); put(sm, f"B{j}", f, fmt=fmt, font=Font(name="Calibri", size=11, color="008000", bold=(j in (10, 11)))); put(sm, f"C{j}", basis)
for col, w in zip("ABC", (60, 18, 80)): sm.column_dimensions[col].width = w
sm.freeze_panes = "A4"

# ---------------- Sensitivity
se = wb.create_sheet("Sensitivity")
se["A1"] = "Sensitivity of the per-person Databricks total to the unverified DBU/hr input"; se["A1"].font = Font(name="Cambria", size=14, bold=True, color=NAVY)
hdr(se, 3, ["DBU/hr assumed", "DBU cost per person ($)", "Databricks total per person ($)", "Cohort total ($)"])
for j, d in enumerate([0, 2, 4, 6, 8, 10, 12, 15], 4):
    put(se, f"A{j}", d, BLUE, "0.0")
    put(se, f"B{j}", f"='Lab Costs'!H{tot}*A{j}*{DBU_AP}/{SHARE}", fmt="$#,##0.00")
    put(se, f"C{j}", f"='Lab Costs'!I{tot}+B{j}+{STOR}", fmt="$#,##0.00")
    put(se, f"D{j}", f"=C{j}*{COHORT}", fmt="$#,##0.00")
for col, w in zip("ABCD", (18, 24, 28, 20)): se.column_dimensions[col].width = w

# ---------------- Sources
so = wb.create_sheet("Sources & Verification")
hdr(so, 1, ["Item", "Status", "Source (checked 2026-09-06)", "Re-verify action"])
src = [
 ("NC24ads_A100_v4 $3.673/hr East US", "Verified (3 trackers agree)", "azurespeed.com, instances.vantage.sh, calculator.holori.com", "Azure pricing calculator, chosen region"),
 ("NC40ads_H100_v5 ≈$6.98/hr", "Approx (1 tracker)", "thundercompute.com Azure GPU guide, reviewed 1 Sep 2026", "Azure pricing calculator"),
 ("NC96ads_A100_v4 ≈$14.69/hr; NC48ads ≈$7.35", "Approx", "thundercompute.com / vantage.sh", "Azure pricing calculator"),
 ("NC4as_T4_v3 ≈$0.526/hr", "Approx", "thundercompute.com", "Azure pricing calculator"),
 ("All-Purpose Premium DBU ≈$0.55; Jobs ≈$0.15", "Approx (several guides)", "lucentinnovation, doit.com, brilworks 2026 guides", "azure.microsoft.com/pricing/details/databricks"),
 ("DBU/hr for GPU VM types", "UNVERIFIED — placeholder", "Not published in any consulted source", "databricks.com/product/pricing/product-pricing/instance-types, or query system.billing.usage after a test run"),
 ("Overhead factor 1.6×, sharing 1, allowance $5", "Trainer assumption", "—", "Measure on cohort 1 from system.billing.usage"),
 ("Databricks GPU instance support (NC A100 v4, NC H100 v5)", "Verified", "learn.microsoft.com/azure/databricks/compute/gpu", "Same page before delivery"),
]
for j, row in enumerate(src, 2):
    for k, v in enumerate(row, 1): put(so, f"{get_column_letter(k)}{j}", v, RED if "UNVERIFIED" in v else B)
for col, w in zip("ABCD", (46, 26, 60, 60)): so.column_dimensions[col].width = w

for sh in wb.worksheets:
    sh.page_setup.orientation = "landscape"; sh.sheet_view.showGridLines = True
wb.save(OUT)
res = subprocess.run(["python3", "/mnt/skills/public/xlsx/scripts/recalc.py", OUT, "60"], capture_output=True, text=True); print(res.stdout)
v = load_workbook(OUT, data_only=True)
print("Summary values:"); [print(" ", v["Summary"][f"A{j}"].value[:55].ljust(56), v["Summary"][f"B{j}"].value) for j in range(4, 15)]
# QA: font + wrap on every populated cell
bad = [(s.title, c.coordinate) for s in wb.worksheets for row in s.iter_rows() for c in row if c.value is not None and (c.font.size != 11 and c.row > 2 or not c.alignment.wrap_text and c.row > 2)]
print("font/wrap violations:", bad[:5], len(bad))
