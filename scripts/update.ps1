# VCB self-updater for people without git: downloads the latest VCB from
# GitHub and lays it over this folder. Run through "VCB Update.bat" (or
# VCB.bat --update). Your settings and files are never touched: .env, user\
# (voices, prompts, model presets), logs\, models\, .runtime\ aren't part of
# the download, and are excluded here as well in case that ever changes.
#
# .runtime\version.txt   commit this copy was updated to (for the "update
#                        available" check at start-up)
# .runtime\manifest.txt  files the last update installed; files that a newer
#                        version removed are deleted on the next update, so
#                        no stale code lingers (only files VCB itself installed)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".runtime"
$Repo = "xulqar9/VCB"
$Branch = "main"
$VersionFile = Join-Path $Runtime "version.txt"
$ManifestFile = Join-Path $Runtime "manifest.txt"
# Never overwritten or deleted, whatever the download contains.
$Protected = @(".env", "user", "logs", "models", ".runtime", ".git", ".venv", ".vscode")

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (Test-Path (Join-Path $Root ".git")) {
    Write-Host "This VCB folder is a git checkout - update it with 'git pull' instead." -ForegroundColor Yellow
    exit 0
}

Step "Checking GitHub for the latest VCB"
$latest = $null
try {
    $latest = (Invoke-RestMethod "https://api.github.com/repos/$Repo/commits/$Branch" `
        -Headers @{ Accept = "application/vnd.github.sha"; "User-Agent" = "VCB-updater" } -TimeoutSec 15).Trim()
} catch {
    Fail "could not reach GitHub ($_). Check your internet connection and try again."
}
$current = if (Test-Path $VersionFile) { (Get-Content $VersionFile -First 1).Trim() } else { "" }
if ($latest -eq $current) {
    Write-Host "VCB is already up to date ($($latest.Substring(0, 7)))." -ForegroundColor Green
    exit 0
}

# What's new since this copy: one line per commit (newest first). Best effort,
# the update itself doesn't depend on it.
$changes = @()
if ($current) {
    try {
        $cmp = Invoke-RestMethod "https://api.github.com/repos/$Repo/compare/$current...$latest" `
            -Headers @{ "User-Agent" = "VCB-updater" } -TimeoutSec 15
        $changes = @($cmp.commits | ForEach-Object { ($_.commit.message -split "`n")[0].Trim() } | Where-Object { $_ })
        [array]::Reverse($changes)
    } catch { }
}
if ($changes.Count) {
    Write-Host "What's new ($($changes.Count) change$(if ($changes.Count -ne 1) { 's' })):" -ForegroundColor Cyan
    $changes | Select-Object -First 15 | ForEach-Object { Write-Host "  - $_" }
    if ($changes.Count -gt 15) { Write-Host "  ... and $($changes.Count - 15) more: https://github.com/$Repo/commits/$Branch" }
}

Step "Downloading VCB $($latest.Substring(0, 7))"
New-Item -ItemType Directory -Force $Runtime | Out-Null
$zip = Join-Path $Runtime "update.zip"
$stage = Join-Path $Runtime "update"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
try {
    Invoke-WebRequest "https://github.com/$Repo/archive/$latest.zip" -OutFile $zip -UseBasicParsing
} catch {
    Fail "download failed ($_). Try again in a moment."
}
Expand-Archive $zip $stage -Force
Remove-Item $zip
# The archive holds one top-level folder (VCB-<sha>): that's the new tree.
$src = Get-ChildItem $stage -Directory | Select-Object -First 1
if (-not $src -or -not (Test-Path (Join-Path $src.FullName "VCB.bat"))) {
    Fail "the download doesn't look like VCB; delete .runtime\update and retry"
}

Step "Installing the update"
$newFiles = Get-ChildItem $src.FullName -Recurse -File |
    ForEach-Object { $_.FullName.Substring($src.FullName.Length + 1) }
# Files the previous update installed that no longer exist upstream.
if (Test-Path $ManifestFile) {
    $keep = @{}; foreach ($f in $newFiles) { $keep[$f] = $true }
    foreach ($old in Get-Content $ManifestFile) {
        $old = $old.Trim()
        if (-not $old -or $keep.ContainsKey($old)) { continue }
        $top = $old.Split("\")[0]
        if ($Protected -contains $top) { continue }
        $path = Join-Path $Root $old
        if (Test-Path $path) { Remove-Item -Force $path }
    }
}
# robocopy merges directories properly (Copy-Item nests them); exit codes
# below 8 mean success.
$xd = $Protected | ForEach-Object { Join-Path $src.FullName $_ }
& robocopy $src.FullName $Root /E /NFL /NDL /NJH /NJS /NP /R:2 /W:2 /XF .env /XD @xd | Out-Null
if ($LASTEXITCODE -ge 8) { Fail "copying the update failed (robocopy code $LASTEXITCODE)" }
# Drop directories emptied by the deletions above (deepest first).
Get-ChildItem $Root -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $Protected -notcontains $_.FullName.Substring($Root.Length + 1).Split("\")[0] } |
    Sort-Object { $_.FullName.Length } -Descending |
    Where-Object { -not (Get-ChildItem $_.FullName -Force | Select-Object -First 1) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Set-Content $ManifestFile ($newFiles -join "`n") -NoNewline
Set-Content $VersionFile "$latest`nupdated $(Get-Date -Format s)" -NoNewline
Remove-Item -Recurse -Force $stage
Write-Host "VCB updated to $($latest.Substring(0, 7)). Your settings, voices and memory were kept." -ForegroundColor Green
Write-Host "Full history: https://github.com/$Repo/commits/$Branch" -ForegroundColor DarkGray
