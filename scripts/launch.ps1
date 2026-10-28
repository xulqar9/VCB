# VCB launcher. Run through VCB.bat (double-click) -- no Python install needed.
#
# First run: downloads uv (a small, fast Python package manager), a private
# Python 3.12, the right PyTorch build for this PC (NVIDIA CUDA, AMD ROCm or
# CPU), VCB's base dependencies (Kokoro preset voice, Whisper, memory) and the
# voice engine chosen in the settings into .runtime\ next to this repo. Later
# runs start VCB straight away; dependencies are only (re)installed when the
# requirements files change or a different voice engine was chosen. Choosing
# the IndexTTS voice adds its own isolated environment (.runtime\indextts).
# Delete .runtime\ to uninstall all of it.
#
# Launcher switches (anything else is passed through to main.py):
#   --settings   open the settings window and exit
#   --update     get the latest VCB from GitHub (no git needed), then start
#   --extras     install all optional voice engines at once (normally the one
#                you pick in the settings is installed on the next start)
#   --reinstall  force a clean dependency reinstall
# Environment:
#   VCB_TORCH_FLAVOR=cu128|rocm|cpu   force the PyTorch build instead of detecting
#   VCB_NO_UPDATE_CHECK=1             skip the "update available" check at start

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
# AMD Radeon: AMD publishes its Windows PyTorch builds only on repo.radeon.com
# (Python 3.12). The torch/torchaudio wheel URLs live in
# requirements-overrides-rocm.txt; these are the ROCm runtime they need.
$RocmTorchVersion = "2.9.1"
$RocmRelease = "7.2.1"
$RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-$RocmRelease"
$RocmRuntime = @(
    "rocm-sdk-core @ $RocmBase/rocm_sdk_core-$RocmRelease-py3-none-win_amd64.whl",
    "rocm-sdk-devel @ $RocmBase/rocm_sdk_devel-$RocmRelease-py3-none-win_amd64.whl",
    "rocm-sdk-libraries-custom @ $RocmBase/rocm_sdk_libraries_custom-$RocmRelease-py3-none-win_amd64.whl",
    "rocm[libraries,devel] @ $RocmBase/rocm-$RocmRelease.tar.gz"
)

$SettingsOnly = $false; $Extras = $false; $Reinstall = $false; $Update = $false
$Pass = @()
foreach ($a in $args) {
    switch ($a) {
        "--settings"  { $SettingsOnly = $true }
        "--update"    { $Update = $true }
        "--extras"    { $Extras = $true }
        "--reinstall" { $Reinstall = $true }
        default       { $Pass += $a }
    }
}

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# -- updates (for zip downloads; git checkouts use git pull) -------------------
$IsGitCheckout = Test-Path (Join-Path $Root ".git")
if ($Update) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "update.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    # The update may have replaced this very script: start VCB with the new one.
    $rest = @(); if ($Extras) { $rest += "--extras" }; if ($Reinstall) { $rest += "--reinstall" }
    $env:VCB_NO_UPDATE_CHECK = "1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath @rest @Pass
    exit $LASTEXITCODE
}
if (-not $IsGitCheckout -and -not $env:VCB_NO_UPDATE_CHECK) {
    # Quick, quiet check: a different latest commit on GitHub than the one
    # this copy was updated to, with a peek at what changed so the user can
    # decide whether it's worth it. Offline or rate-limited -> say nothing.
    $versionFile = Join-Path $Runtime "version.txt"
    try {
        $hdr = @{ "User-Agent" = "VCB" }
        $latest = (Invoke-RestMethod "https://api.github.com/repos/xulqar9/VCB/commits/main" `
            -Headers ($hdr + @{ Accept = "application/vnd.github.sha" }) -TimeoutSec 3).Trim()
        $current = if (Test-Path $versionFile) { (Get-Content $versionFile -First 1).Trim() } else { "" }
        if ($latest -and $latest -ne $current) {
            Write-Host "A newer VCB is available - double-click 'VCB Update.bat' to get it (optional; your settings are kept)." -ForegroundColor Yellow
            if ($current) {
                $cmp = Invoke-RestMethod "https://api.github.com/repos/xulqar9/VCB/compare/$current...$latest" -Headers $hdr -TimeoutSec 3
                $lines = @($cmp.commits | ForEach-Object { ($_.commit.message -split "`n")[0].Trim() } | Where-Object { $_ })
                [array]::Reverse($lines)
                $lines | Select-Object -First 5 | ForEach-Object { Write-Host "  - $_" -ForegroundColor DarkGray }
                if ($lines.Count -gt 5) { Write-Host "  ... and $($lines.Count - 5) more" -ForegroundColor DarkGray }
            }
        }
    } catch { }
}

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
    # sounddevice (+ numpy, which vcb.audio imports) lists mics/speakers and
    # tzdata lists time zones in the window; small, so install them upfront.
    & $Py -c "import vcb.audio.devices, tzdata" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $Uv pip install --python $Py --quiet -c (Join-Path $Root "requirements-lock.txt") sounddevice "numpy<2" tzdata
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
# PyTorch build: NVIDIA -> CUDA; a Radeon that AMD supports with PyTorch on
# Windows (Windows 11 only) -> AMD's ROCm build, so speech recognition and the
# voice run on the GPU; otherwise CPU.
$HasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if (-not $HasNvidia) { $HasNvidia = Test-Path "$env:SystemRoot\System32\nvidia-smi.exe" }
function Get-RocmGpu {
    # Keep the model list in step with vcb\gpu.py (rocm_supported).
    $names = try { @((Get-CimInstance Win32_VideoController -ErrorAction Stop).Name) } catch { @() }
    foreach ($n in $names) {
        if ($n -match 'Radeon' -and
            $n -match '\bRX 9070( XT)?\b|\bRX 9060 XT\b|\bAI PRO R9700\b|\bRX 7900 XTX\b|\bPRO W7900\b|\bRX 7700\b(?! XT)') {
            return $n
        }
    }
    return $null
}
$AmdGpu = if ($HasNvidia) { $null } else { Get-RocmGpu }
$IsWin11 = [Environment]::OSVersion.Version.Build -ge 22000
if ($env:VCB_TORCH_FLAVOR) {
    $Flavor = $env:VCB_TORCH_FLAVOR
    if ($Flavor -notin @("cu128", "rocm", "cpu")) { Fail "VCB_TORCH_FLAVOR must be cu128, rocm or cpu (got '$Flavor')" }
} elseif ($HasNvidia) { $Flavor = "cu128" }
elseif ($AmdGpu -and $IsWin11) { $Flavor = "rocm" }
else {
    if ($AmdGpu) { Write-Host "$AmdGpu found, but AMD's PyTorch build needs Windows 11: CPU mode." -ForegroundColor Yellow }
    $Flavor = "cpu"
}
$Overrides = if ($Flavor -eq "rocm") { "requirements-overrides-rocm.txt" } else { "requirements-overrides.txt" }
$WantTorch = if ($Flavor -eq "rocm") { $RocmTorchVersion } else { $TorchVersion }
$Previous = if (Test-Path $Stamp) { (Get-Content $Stamp -Raw).Trim() } else { "" }

# -- voice engine packs ---------------------------------------------------------
# The base install only has the Kokoro preset voice. The engine chosen in the
# settings is installed on the next start (and stays installed when you switch
# again). --extras installs all of them. IndexTTS is special: see below.
function Get-TtsEngine {
    if ($env:VCB_TTS_ENGINE) { return $env:VCB_TTS_ENGINE }
    $envFile = Join-Path $Root ".env"
    if (Test-Path $envFile) {
        $m = Select-String -Path $envFile -Pattern '^\s*VCB_TTS_ENGINE\s*=\s*"?([A-Za-z0-9_-]+)' | Select-Object -Last 1
        if ($m) { return $m.Matches[0].Groups[1].Value }
    }
    return "chatterbox-turbo"
}
$Engine = Get-TtsEngine
# engine -> (requirements file, download size shown to the user)
$PackFiles = [ordered]@{
    "chatterbox-turbo" = "requirements-chatterbox.txt"
    "xtts"             = "requirements-xtts.txt"
    "orpheus"          = "requirements-orpheus.txt"
    "csm"              = "requirements-csm.txt"
}
$PackNames = @{ "chatterbox-turbo" = "Chatterbox"; "xtts" = "XTTS"; "orpheus" = "Orpheus"; "csm" = "Sesame CSM" }
$Packs = @()
if ($Previous -match "packs=([^\r\n]*)") { $Packs += ($Matches[1].Split(",") | Where-Object { $_ }) }
if ($Extras -or $Previous -match "extras=True") { $Packs += @($PackFiles.Keys) }
if ($PackFiles.Contains($Engine)) { $Packs += $Engine }
# Without an NVIDIA GPU, main.py swaps IndexTTS/Orpheus for Chatterbox (AMD) or
# the preset voice (CPU): make sure what it falls back to is installed.
if ($Flavor -eq "rocm" -and $Engine -in @("indextts", "orpheus")) { $Packs += "chatterbox-turbo" }
if ($Flavor -ne "cu128") { $Packs = @($Packs | Where-Object { $_ -ne "orpheus" }) }   # CUDA-only wheels
$Packs = @($Packs | Sort-Object -Unique)

$reqFiles = @("requirements.txt", "requirements-lock.txt", $Overrides) + @($Packs | ForEach-Object { $PackFiles[$_] })
$hash = ($reqFiles | ForEach-Object { (Get-FileHash (Join-Path $Root $_) -Algorithm SHA256).Hash }) -join ""
$want = "torch=$WantTorch+$Flavor`npacks=$($Packs -join ',')`nreqs=$hash"

if ($Previous -ne $want) {
    $fresh = -not ($Previous -match "^torch=")
    if ($fresh) {
        switch ($Flavor) {
            "cu128" { Step "Installing VCB dependencies for your NVIDIA GPU (first run downloads ~4 GB, please wait)" }
            "rocm"  { Step "Installing VCB dependencies for your AMD GPU ($(if ($AmdGpu) { $AmdGpu } else { 'VCB_TORCH_FLAVOR=rocm' }); first run downloads ~4 GB, please wait)" }
            default { Step "Installing VCB dependencies (no supported GPU found: CPU mode; first run downloads ~1.5 GB)" }
        }
    } else { Step "Updating VCB dependencies" }
    $common = @(
        "pip", "install", "--python", $Py,
        "-c", (Join-Path $Root "requirements-lock.txt"),
        "--override", (Join-Path $Root $Overrides),
        "--index-strategy", "unsafe-best-match"
    )
    # AMD's wheels are fetched by URL (from the overrides file); the PyTorch
    # index only serves the CUDA and CPU builds for Windows.
    if ($Flavor -ne "rocm") { $common += @("--extra-index-url", "https://download.pytorch.org/whl/$Flavor") }
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
    $torchReqs = @("torch==$TorchVersion", "torchaudio==$TorchVersion")
    if ($Flavor -eq "rocm") { $torchReqs = $RocmRuntime + @("torch", "torchaudio") }
    if (-not (Invoke-UvInstall @torchReqs -r (Join-Path $Root "requirements.txt"))) {
        Fail "dependency install failed (see above). Re-run VCB.bat to retry; if it keeps failing, temporarily pause your antivirus or exclude this folder."
    }
    foreach ($pack in $Packs) {
        Step "Installing the $($PackNames[$pack]) voice engine"
        if (-not (Invoke-UvInstall -r (Join-Path $Root $PackFiles[$pack]))) {
            Fail "$($PackNames[$pack]) install failed (see above). Re-run VCB.bat to retry, or pick another voice in the settings."
        }
    }
    if ($Flavor -eq "cu128") {
        # NVIDIA: speech recognition (parakeet) and Orpheus' vocoder run through
        # onnxruntime's CUDA provider. fastembed pulls the plain CPU onnxruntime,
        # which shadows onnxruntime-gpu (same import name), so swap it out.
        & $Uv pip uninstall --python $Py onnxruntime
        if (-not (Invoke-UvInstall --reinstall-package onnxruntime-gpu "onnxruntime-gpu==1.20.2")) {
            Fail "onnxruntime-gpu install failed (see above)"
        }
    }
    Set-Content $Stamp $want -NoNewline
}

# -- IndexTTS voice engine (only when selected) --------------------------------
# IndexTTS pins Python 3.11, torch 2.8 and an older transformers, so it gets its
# own isolated environment from its upstream lockfile (VCB talks to it through
# vcb\tts\indextts_worker.py). Installed on the first start after choosing it.
$IndexDir = Join-Path $Runtime "indextts"
$IndexCommit = "d9e41aac89fd00b3d71497fddb287b7f24613712"   # index-tts/index-tts, IndexTTS-2.5


function Invoke-Retry([scriptblock]$Action) {
    # Antivirus can briefly lock fresh downloads; retrying gets past it.
    for ($try = 1; $try -le 3; $try++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return $true }
        if ($try -lt 3) { Write-Host "Install hiccup, retrying ($($try + 1)/3)..." -ForegroundColor Yellow; Start-Sleep 3 }
    }
    return $false
}

function Ensure-IndexTTS {
    $stamp = Join-Path $IndexDir "installed.txt"
    $want = "commit=$IndexCommit"
    if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $want)) { return }
    if (-not $HasNvidia) {
        if ($Flavor -eq "rocm") { Write-Host "IndexTTS needs an NVIDIA GPU; VCB will clone your voice with Chatterbox instead." -ForegroundColor Yellow }
        else { Write-Host "IndexTTS needs an NVIDIA GPU; VCB will use the Kokoro preset voice instead." -ForegroundColor Yellow }
        return
    }
    Step "Installing the IndexTTS voice engine (one-time, ~8 GB download, please wait)"
    New-Item -ItemType Directory -Force $IndexDir | Out-Null
    $src = Join-Path $IndexDir "src"
    if (-not (Test-Path (Join-Path $src ".vcb-$IndexCommit"))) {
        if (Test-Path $src) { Remove-Item -Recurse -Force $src }
        $zip = Join-Path $IndexDir "src.zip"
        $url = "https://github.com/index-tts/index-tts/archive/$IndexCommit.zip"
        try { Invoke-WebRequest $url -OutFile $zip -UseBasicParsing -ErrorAction Stop }
        catch { Fail "could not download IndexTTS from $url ($_)" }
        Expand-Archive $zip $IndexDir -Force
        Remove-Item $zip
        Rename-Item (Join-Path $IndexDir "index-tts-$IndexCommit") $src
        New-Item -ItemType File (Join-Path $src ".vcb-$IndexCommit") | Out-Null
    }
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $IndexDir "venv"
    $ok = Invoke-Retry { & $Uv sync --project $src --frozen --no-dev | Out-Host }
    Remove-Item Env:\UV_PROJECT_ENVIRONMENT
    if (-not $ok) { Fail "IndexTTS install failed (see above). Re-run VCB.bat to retry." }

    $ipy = Join-Path $IndexDir "venv\Scripts\python.exe"
    $ckpt = Join-Path $IndexDir "checkpoints"
    Step "Downloading the IndexTTS-2.5 model"
    $dl = "from huggingface_hub import snapshot_download as d; " +
          "d('IndexTeam/IndexTTS-2.5', local_dir=r'$ckpt', ignore_patterns=['qwen0.6bemo4-merge/*'])"
    if (-not (Invoke-Retry { & $ipy -c $dl | Out-Host })) { Fail "IndexTTS model download failed (see above)" }
    $worker = Join-Path $Root "vcb\tts\indextts_worker.py"
    if (-not (Invoke-Retry { & $ipy $worker --model-dir $ckpt --prepare | Out-Host })) {
        Fail "IndexTTS helper model download failed (see above)"
    }
    Set-Content $stamp $want -NoNewline
}

if ($Engine -eq "indextts") { Ensure-IndexTTS }

# -- run ----------------------------------------------------------------------
Step "Starting VCB (the first start also downloads the speech models, a few GB)"
& $Py (Join-Path $Root "main.py") @Pass
exit $LASTEXITCODE
