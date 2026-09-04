[CmdletBinding()]
param(
    [string]$Branch = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$stopScript = Join-Path $repoRoot "scripts\stop-local.ps1"

Set-Location $repoRoot
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot ".git") -PathType Container)) {
    throw "Deployment root is not a Git working tree: $repoRoot"
}
if ((& git status --porcelain).Count -gt 0) {
    throw "Deployment working tree has uncommitted changes. Resolve them before deployment."
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot ".env") -PathType Leaf)) {
    throw "Deployment .env is missing."
}

# Stop the current UI/API with the version that launched it, then move the code forward.
& $stopScript
& git fetch origin $Branch
if ($LASTEXITCODE -ne 0) { throw "git fetch failed." }
& git merge-base --is-ancestor HEAD "origin/$Branch"
if ($LASTEXITCODE -ne 0) {
    throw "Deployment branch is not a fast-forward of the installed revision."
}
& git merge --ff-only "origin/$Branch"
if ($LASTEXITCODE -ne 0) { throw "git fast-forward failed." }

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}
& $pythonPath -m pip install -e ".[web]"
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
& docker compose -f deploy/docker-compose.yml pull qdrant open-webui
if ($LASTEXITCODE -ne 0) { throw "Container image pull failed." }
& (Join-Path $repoRoot "scripts\start-local.ps1")
