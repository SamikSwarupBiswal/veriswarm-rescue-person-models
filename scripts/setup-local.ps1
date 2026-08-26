[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $Python -m venv (Join-Path $RepositoryRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python environment"
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Push-Location $RepositoryRoot
try {
    & $VenvPython -m pip install -e ".[local,test]"
    if ($LASTEXITCODE -ne 0) { throw "local package installation failed" }
    & (Join-Path $PSScriptRoot "download-models.ps1")
    & $VenvPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw "automated tests failed" }
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Run: .\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli verify"
