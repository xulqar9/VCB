# VCB launcher. Run through VCB.bat (double-click) -- no Python install needed.
#
# First run: downloads uv (a small, fast Python package manager), a private
# Python 3.12, the right PyTorch build for this PC (NVIDIA CUDA or CPU), and all
# VCB dependencies into .runtime\ next to this repo, then opens the settings
# window. Later runs start VCB straight away; dependencies are only reinstalled
# when the requirements files change. Delete .runtime\ to uninstall all of it.
#
# Launcher switches (anything else is passed through to main.py):
#   --settings   open the settings window and exit
#   --extras     also install the optional TTS engines (xtts / orpheus / csm)
#   --reinstall  force a clean dependency reinstall

# Native tools report failure via exit codes (checked below); "Stop" would turn
# their harmless stderr output into terminating errors on Windows PowerShell.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"     # Invoke-WebRequest is very slow with it on

$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".runtime"
$Uv = Join-Path $Runtime "uv\uv.exe"
$Venv = Join-Path $Runtime "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Stamp = Join-Path $Runtime "installed.txt"
$TorchVersion = "2.7.1"

$SettingsOnly = $false; $Extras = $false; $Reinstall = $false
$Pass = @()
foreach ($a in $args) {
    switch ($a) {
        "--settings"  { $SettingsOnly = $true }
        "--extras"    { $Extras = $true }
        "--reinstall" { $Reinstall = $true }
        default       { $Pass += $a }
    }
}

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# Keep uv's Python and cache inside .runtime so nothing is spread over the PC
# (set UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR yourself to share them instead).
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python" }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $Runtime "cache" }

New-Item -ItemType Directory -Force $Runtime | Out-Null

# -- uv -----------------------------------------------------------------------
if (-not (Test-Path $Uv)) {
    Step "Downloading uv (one-time)"
    $zip = Join-Path $Runtime "uv.zip"
    $url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    try { Invoke-WebRequest $url -OutFile $zip -UseBasicParsing -ErrorAction Stop }
    catch { Fail "could not download uv from $url ($_). Check your internet connection." }
    Expand-Archive $zip (Join-Path $Runtime "uv") -Force
    Remove-Item $zip
    if (-not (Test-Path $Uv)) { Fail "uv download looks incomplete; delete .runtime and retry" }
}

# -- python + venv ------------------------------------------------------------
if ($Reinstall -and (Test-Path $Venv)) {
    Step "Removing old environment"
    Remove-Item -Recurse -Force $Venv
    Remove-Item -Force $Stamp -ErrorAction SilentlyContinue
}
if (-not (Test-Path $Py)) {
    Step "Setting up Python 3.12 (one-time)"
    & $Uv venv $Venv --python 3.12 --seed --quiet
    if ($LASTEXITCODE -ne 0) { Fail "could not create the Python environment" }
}

# -- settings (only needs Python's built-in tkinter, so it runs before the big
#    dependency install: the user can configure VCB while nothing heavy exists) --
Set-Location $Root
function Ensure-SettingsDeps {
    # sounddevice (+ numpy, which vcb.audio imports) lists mics/speakers in the
    # window; small, so install them upfront.
    & $Py -c "import vcb.audio.devices" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $Uv pip install --python $Py --quiet -c (Join-Path $Root "requirements-lock.txt") sounddevice "numpy<2"
    }
}
if ($SettingsOnly) {
    Ensure-SettingsDeps
    & $Py -m vcb.settings
    exit 0
}
if (-not (Test-Path (Join-Path $Root ".env"))) {
    Ensure-SettingsDeps
    Step "First run: choose your model and voice"
    & $Py -m vcb.settings
    if ($LASTEXITCODE -ne 0) { Write-Host "Settings not saved - run VCB.bat again when you're ready."; exit 0 }
}

# -- dependencies -------------------------------------------------------------
$HasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if (-not $HasNvidia) { $HasNvidia = Test-Path "$env:SystemRoot\System32\nvidia-smi.exe" }
$Flavor = if ($HasNvidia) { "cu128" } else { "cpu" }
$Previous = if (Test-Path $Stamp) { (Get-Content $Stamp -Raw).Trim() } else { "" }
$ExtrasWanted = $Extras -or ($Previous -match "extras=True")

$reqFiles = @("requirements.txt", "requirements-lock.txt", "requirements-overrides.txt")
if ($ExtrasWanted) { $reqFiles += "requirements-extras.txt" }
$hash = ($reqFiles | ForEach-Object { (Get-FileHash (Join-Path $Root $_) -Algorithm SHA256).Hash }) -join ""
$want = "torch=$TorchVersion+$Flavor`nextras=$ExtrasWanted`nreqs=$hash"

if ($Previous -ne $want) {
    if ($HasNvidia) { Step "Installing VCB dependencies for your NVIDIA GPU (first run downloads ~5 GB, please wait)" }
    else { Step "Installing VCB dependencies (no NVIDIA GPU found: CPU mode; first run downloads ~1.5 GB)" }
    $common = @(
        "pip", "install", "--python", $Py,
        "-c", (Join-Path $Root "requirements-lock.txt"),
        "--override", (Join-Path $Root "requirements-overrides.txt"),
        "--extra-index-url", "https://download.pytorch.org/whl/$Flavor",
        "--index-strategy", "unsafe-best-match"
    )
    # Antivirus scanners briefly lock freshly downloaded files, which can make
    # uv's cache rename fail ("Access is denied"). uv keeps what it already
    # fetched, so simply retrying gets past it.
    function Invoke-UvInstall {
        for ($try = 1; $try -le 3; $try++) {
            & $Uv @common @args | Out-Host
            if ($LASTEXITCODE -eq 0) { return $true }
            if ($try -lt 3) { Write-Host "Install hiccup, retrying ($($try + 1)/3)..." -ForegroundColor Yellow; Start-Sleep 3 }
        }
        return $false
    }
    if (-not (Invoke-UvInstall "torch==$TorchVersion" "torchaudio==$TorchVersion" -r (Join-Path $Root "requirements.txt"))) {
        Fail "dependency install failed (see above). Re-run VCB.bat to retry; if it keeps failing, temporarily pause your antivirus or exclude this folder."
    }
    if ($ExtrasWanted) {
        Step "Installing optional TTS engines"
        if (-not (Invoke-UvInstall -r (Join-Path $Root "requirements-extras.txt"))) {
            Fail "optional engine install failed (see above)"
        }
        # fastembed pulls plain onnxruntime, which shadows onnxruntime-gpu (Orpheus).
        & $Uv pip uninstall --python $Py onnxruntime
        if (-not (Invoke-UvInstall --reinstall-package onnxruntime-gpu "onnxruntime-gpu==1.20.2")) {
            Fail "onnxruntime-gpu install failed (see above)"
        }
    }
    Set-Content $Stamp $want -NoNewline
}

# -- run ----------------------------------------------------------------------
Step "Starting VCB (the first start also downloads the speech models, a few GB)"
& $Py (Join-Path $Root "main.py") @Pass
exit $LASTEXITCODE
