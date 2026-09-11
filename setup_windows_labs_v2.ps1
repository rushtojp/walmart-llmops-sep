<#
.SYNOPSIS
  One-shot setup of a basic Windows VM (no GPU) as the participant workstation for the LLMOps Advanced Labs.
.DESCRIPTION
  Installs: Python 3.12, Git, Temurin JDK 17, VS Code (+ Python/Jupyter extensions), Azure CLI (+ ml extension),
  Databricks CLI, Hadoop winutils for PySpark; creates a venv and installs the SMOKE/platform Python stack;
  sets JAVA_HOME / HADOOP_HOME; runs verification checks and (optionally) the L12 smoke lab.
  Idempotent: safe to re-run. Requires: Windows 10/11 or Server 2022, winget, Administrator PowerShell, internet.
.PARAMETER LabRoot
  Folder where the lab package (LLMOps_Advanced_Labs) is or will be unzipped. Default: C:\llmops
.PARAMETER IncludeOptional
  Also install Node.js LTS, Docker Desktop, LibreOffice.
.PARAMETER RunSmokeTest
  After setup, execute labs/L12/L12_solution.ipynb in SMOKE mode as an end-to-end check.
.EXAMPLE
  Set-ExecutionPolicy -Scope Process Bypass; .\setup_windows_labs.ps1 -LabRoot C:\llmops -RunSmokeTest
#>
[CmdletBinding()]
param(
  [string]$LabRoot = "C:\llmops",
  [switch]$IncludeOptional,
  [switch]$RunSmokeTest
)
$ErrorActionPreference = "Stop"
$log = Join-Path $env:TEMP "llmops_setup_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Start-Transcript -Path $log | Out-Null
function Step($m){ Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Ok($m){ Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow }

# ---------- 0. Preconditions
Step "Preconditions"
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw "Run this script from an elevated (Administrator) PowerShell." }
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "winget not found. Install 'App Installer' from the Microsoft Store, then re-run." }
Ok "Administrator + winget present. Log: $log"

function Install-Winget($id, $name){
  $listed = winget list --id $id --exact --accept-source-agreements 2>$null | Out-String
  if ($listed -match [regex]::Escape($id)) { Ok "$name already installed"; return }
  Write-Host "  installing $name ..."
  winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements | Out-Null
  Ok "$name installed"
}

# ---------- 1. Core software via winget
Step "Core software"
Install-Winget "Python.Python.3.12"            "Python 3.12"
Install-Winget "Git.Git"                        "Git for Windows"
Install-Winget "EclipseAdoptium.Temurin.17.JDK" "Temurin JDK 17"
Install-Winget "Microsoft.VisualStudioCode"     "VS Code"
Install-Winget "Microsoft.AzureCLI"             "Azure CLI"
Install-Winget "Databricks.DatabricksCLI"       "Databricks CLI"
if ($IncludeOptional) {
  Install-Winget "OpenJS.NodeJS.LTS"      "Node.js LTS"
  Install-Winget "Docker.DockerDesktop"   "Docker Desktop"
  Install-Winget "TheDocumentFoundation.LibreOffice" "LibreOffice"
}
# refresh PATH for this session
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

# ---------- 2. JAVA_HOME
Step "JAVA_HOME"
$jdk = Get-ChildItem "C:\Program Files\Eclipse Adoptium" -Directory -ErrorAction SilentlyContinue | Where-Object Name -like "jdk-17*" | Select-Object -First 1
if (-not $jdk) { $jdk = Get-ChildItem "C:\Program Files\Java" -Directory -ErrorAction SilentlyContinue | Where-Object Name -like "*17*" | Select-Object -First 1 }
if (-not $jdk) { throw "JDK 17 folder not found; set JAVA_HOME manually." }
[Environment]::SetEnvironmentVariable("JAVA_HOME", $jdk.FullName, "Machine"); $env:JAVA_HOME = $jdk.FullName
Ok "JAVA_HOME = $($jdk.FullName)"

# ---------- 3. Hadoop winutils for PySpark on Windows
Step "Hadoop winutils (PySpark on Windows)"
$hadoopHome = "C:\hadoop"; $bin = Join-Path $hadoopHome "bin"; New-Item -ItemType Directory -Force -Path $bin | Out-Null
# Source: community-maintained winutils builds (cdarlint/winutils on GitHub). Verify the Hadoop version matches your PySpark build.
$hv = "hadoop-3.3.6"
foreach ($f in @("winutils.exe","hadoop.dll")) {
  $dst = Join-Path $bin $f
  if (-not (Test-Path $dst)) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/cdarlint/winutils/master/$hv/bin/$f" -OutFile $dst -UseBasicParsing
  }
}
[Environment]::SetEnvironmentVariable("HADOOP_HOME", $hadoopHome, "Machine"); $env:HADOOP_HOME = $hadoopHome
$mp = [Environment]::GetEnvironmentVariable("Path","Machine"); if ($mp -notlike "*$bin*") { [Environment]::SetEnvironmentVariable("Path", "$mp;$bin", "Machine") }
$env:Path += ";$bin"
Copy-Item (Join-Path $bin "hadoop.dll") "C:\Windows\System32\hadoop.dll" -Force -ErrorAction SilentlyContinue
Ok "HADOOP_HOME = $hadoopHome ($hv)"

# ---------- 4. Lab folder + venv + Python packages
Step "Python virtual environment"
New-Item -ItemType Directory -Force -Path $LabRoot | Out-Null
$venv = Join-Path $LabRoot ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) { py -3.12 -m venv $venv }
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --upgrade pip wheel | Out-Null
$req = Join-Path $PSScriptRoot "requirements-smoke.txt"
if (-not (Test-Path $req)) { throw "requirements-smoke.txt not found next to this script." }
& $py -m pip install -r $req
& $py -m ipykernel install --user --name llmops --display-name "Python (llmops)" | Out-Null
Ok "venv at $venv; kernel 'Python (llmops)' registered"

# ---------- 5. Azure CLI ml extension + VS Code extensions
Step "CLI extensions"
try { az extension add -n ml --upgrade --yes 2>$null | Out-Null; Ok "az ml extension" } catch { Warn "az ml extension failed: $_" }
foreach ($ext in @("ms-python.python","ms-toolsai.jupyter")) { try { code --install-extension $ext --force 2>$null | Out-Null; Ok "VS Code extension $ext" } catch { Warn "VS Code extension $ext failed" } }

# ---------- 6. Verification
Step "Verification"
$verify = Join-Path $env:TEMP "llmops_verify.py"
@'
import importlib, sys, os, tempfile
mods = ["numpy","pandas","scipy","sklearn","matplotlib","openpyxl","onnx","onnxruntime","skl2onnx","pyspark","nbformat","nbclient","httpx","azure.ai.ml","databricks.sdk","mlflow"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print("  [OK] " + m)
    except Exception as e:
        bad.append(m); print("  [FAIL] " + m + ": " + str(e))
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"])
    assert df.count() == 2
    p = os.path.join(tempfile.mkdtemp(), "t")
    df.write.mode("overwrite").parquet(p)
    print("  [OK] PySpark local + parquet write (winutils working)")
    spark.stop()
except Exception as e:
    bad.append("pyspark-runtime"); print("  [FAIL] PySpark runtime: " + str(e))
sys.exit(1 if bad else 0)
'@ | Set-Content -Path $verify -Encoding UTF8
& $py $verify
if ($LASTEXITCODE -ne 0) { Warn "Some checks failed - see above and the Troubleshooting section." } else { Ok "Python stack verified" }
foreach ($c in @("git --version", "java -version", "az version", "databricks --version")) {
  try { $out = (cmd /c "$c 2>&1" | Select-Object -First 1); Ok "$c -> $out" } catch { Warn "$c failed" }
}

# ---------- 7. Optional smoke test of one lab
if ($RunSmokeTest) {
  Step "Smoke test: L12 solution notebook"
  $nbItem = Get-ChildItem -Path $LabRoot, "C:\setup" -Recurse -Filter "L12_solution.ipynb" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($nbItem) {
    $nb = $nbItem.FullName
    $pkg = Split-Path (Split-Path (Split-Path $nb -Parent) -Parent) -Parent   # ...\<package>\labs\L12\file -> <package>
    $env:LAB_MODE = "SMOKE"; $env:DATA_DIR = Join-Path $pkg "data"
    Ok "Found lab package at $pkg"
    $smoke = Join-Path $env:TEMP "llmops_smoke.py"
    @'
import nbformat, os, sys
from nbclient import NotebookClient
p = sys.argv[1]
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=600, kernel_name="llmops", resources={"metadata": {"path": os.path.dirname(p)}}).execute()
print("  [OK] L12 executed end-to-end in SMOKE mode")
'@ | Set-Content -Path $smoke -Encoding UTF8
    & $py $smoke $nb
    if ($LASTEXITCODE -ne 0) { Warn "L12 smoke test failed - see output above." }
  } else { Warn "L12_solution.ipynb not found under $LabRoot or C:\setup - unzip the lab package there and re-run with -RunSmokeTest." }
}

Step "Done"
Write-Host @"
Next steps:
  1. Unzip the lab package (LLMOps_Advanced_Labs.zip / walmart-llmops-sep-master.zip) into $LabRoot (if not already).
  2. Open VS Code -> select kernel 'Python (llmops)' -> run labs\L01\L01_lab.ipynb.
  3. az login   and   databricks configure   (workspace URL + token) for the cloud steps.
  4. On the trainer rehearsal box only: pip freeze > requirements-lock.txt and record versions in the verification register.
Open a NEW terminal so JAVA_HOME / HADOOP_HOME / PATH changes take effect.
Log: $log
"@
Stop-Transcript | Out-Null
