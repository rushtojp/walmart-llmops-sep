# %%
# === Lab environment header (identical in every lab) ===
import os, sys, json, time, math, shutil, re, subprocess, importlib, importlib.util
import numpy as np, pandas as pd

def ensure_packages(pkgs):
    """Install any missing pip packages into THIS Python (same mechanism as %pip on Databricks) and import them.
    Fresh packages are importable immediately — no restart. Only labs that need extras call this (L02, L08)."""
    missing = [p for p in pkgs if importlib.util.find_spec(p.replace("-", "_")) is None]
    if not missing:
        print("Packages present:", pkgs); return
    print("Installing missing packages into", sys.executable, ":", missing)
    cmd = [sys.executable, "-m", "pip", "install", "-q", *missing]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(cmd + ["--break-system-packages"], capture_output=True, text=True)   # local system Pythons
    if r.returncode != 0:
        raise ImportError("pip could not install " + str(missing) + ". Ask the admin to add them as cluster libraries "
                          "(Compute → Libraries → PyPI) or use an internal index. pip said: " + r.stderr[-600:])
    importlib.invalidate_caches()
    for p in missing: importlib.import_module(p.replace("-", "_"))
    print("Installed and imported:", missing)

# Mode: "GPU" runs the full lab on Azure GPU compute; "SMOKE" runs the CPU/synthetic path anywhere.
LAB_MODE = os.environ.get("LAB_MODE") or ("GPU" if shutil.which("nvidia-smi") else "SMOKE")

# Data folder: env override → package-relative (../../data) → Unity Catalog volume → search the workspace once
_candidates = [os.environ.get("DATA_DIR"), os.path.abspath(os.path.join(os.getcwd(), "..", "..", "data")), "/Volumes/northfield/llmops/labdata"]
DATA_DIR = next((c for c in _candidates if c and os.path.exists(os.path.join(c, "catalog_items.csv"))), None)
if DATA_DIR is None:
    import glob
    _hits = [h for root in ("/Workspace", "/Volumes", os.path.expanduser("~")) if os.path.isdir(root)
             for h in glob.glob(os.path.join(root, "**", "catalog_items.csv"), recursive=True)][:1]
    DATA_DIR = os.path.dirname(_hits[0]) if _hits else None
if DATA_DIR is None:
    raise FileNotFoundError("Lab data not found. Upload the package's data/ folder to a Unity Catalog volume and set "
                            "os.environ['DATA_DIR'] = '/Volumes/<catalog>/<schema>/<volume>' in a cell above this one.")
def gpu_only(msg):
    """Called wherever a step needs a GPU / model download that the smoke path cannot run."""
    print(f"[{LAB_MODE}] GPU-only step not executed here: {msg}")
def check(cond, msg):
    """Binary 'done means' assertion — prints PASS/FAIL and raises on FAIL so the notebook stops."""
    print(("PASS " if cond else "FAIL ") + msg); assert cond, msg
print(f"LAB_MODE={LAB_MODE}  DATA_DIR={DATA_DIR}  python={sys.version.split()[0]}")
