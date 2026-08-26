[CmdletBinding()]
param(
    [string]$Repository = "SamikSwarupBiswal/veriswarm-rescue-person-models",
    [string]$Tag = "candidate-2026-08-23",
    [string]$ModelDirectory = ""
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ModelDirectory)) {
    $ModelDirectory = Join-Path $RepositoryRoot "models"
}
elseif (-not [System.IO.Path]::IsPathRooted($ModelDirectory)) {
    $ModelDirectory = Join-Path $RepositoryRoot $ModelDirectory
}
New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null

$null = Get-Command gh -ErrorAction Stop
$assets = @("yolov8n-640-best.pt", "yolov8n-960-best.pt")
foreach ($asset in $assets) {
    $destination = Join-Path $ModelDirectory $asset
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Write-Host "Keeping existing checkpoint for hash verification: $destination"
        continue
    }
    gh release download $Tag --repo $Repository --pattern $asset --dir $ModelDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub release download failed for $asset"
    }
}

$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Run scripts\setup-local.ps1 first so the verifier is available."
}
& $Python -m veriswarm_rescue_models.cli `
    --manifest (Join-Path $RepositoryRoot "models\manifest.json") `
    verify `
    --models-dir $ModelDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint verification failed"
}
