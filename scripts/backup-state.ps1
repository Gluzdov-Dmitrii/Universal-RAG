[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$DestinationRoot,

    [switch]$IncludeOpenWebUI
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$destination = [IO.Path]::GetFullPath($DestinationRoot)
$repoPrefix = $repoRoot.TrimEnd('\') + '\'
if ($destination.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Backups must be stored outside the Git working tree."
}

$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$composePath = Join-Path $repoRoot "deploy\docker-compose.yml"
$localEnvironmentPath = Join-Path $repoRoot ".env"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment is missing: $pythonPath"
}

if (Test-Path -LiteralPath $localEnvironmentPath -PathType Leaf) {
    foreach ($rawLine in Get-Content -LiteralPath $localEnvironmentPath) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $parts = $line.Split("=", 2)
        if ($parts.Count -eq 2 -and $parts[0] -match '^[A-Za-z_][A-Za-z0-9_]*$') {
            Set-Item -LiteralPath "Env:$($parts[0])" -Value $parts[1]
        }
    }
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$snapshotRoot = Join-Path $destination "universal-rag-$timestamp"
New-Item -ItemType Directory -Force -Path $snapshotRoot | Out-Null

$configJson = & $pythonPath -c @'
import json
from secure_rag.config import load_config
c = load_config()
print(json.dumps({
    "schema_version": c.schema_version,
    "manifest_path": str(c.manifest_path),
    "qdrant_url": c.qdrant.url,
    "collection": c.qdrant.collection_name,
}))
'@
if ($LASTEXITCODE -ne 0) { throw "Configuration lookup failed." }
$config = $configJson | ConvertFrom-Json
if (-not $config.qdrant_url) { throw "State backup currently requires Qdrant server mode." }

$manifestDestination = Join-Path $snapshotRoot "documents.sqlite"
$backupCode = @'
import sqlite3
import sys
source = sqlite3.connect(sys.argv[1])
target = sqlite3.connect(sys.argv[2])
try:
    source.backup(target)
finally:
    target.close()
    source.close()
'@
& $pythonPath -c $backupCode ([string]$config.manifest_path) $manifestDestination
if ($LASTEXITCODE -ne 0) { throw "SQLite manifest backup failed." }

$headers = @{}
if ($env:SECURE_RAG_QDRANT_API_KEY) {
    $headers["api-key"] = $env:SECURE_RAG_QDRANT_API_KEY
}
$collection = [Uri]::EscapeDataString([string]$config.collection)
$snapshotResponse = Invoke-RestMethod -Method Post `
    -Uri "$($config.qdrant_url)/collections/$collection/snapshots" -Headers $headers
$snapshotName = [string]$snapshotResponse.result.name
if (-not $snapshotName) { throw "Qdrant did not return a snapshot name." }
$qdrantDestination = Join-Path $snapshotRoot $snapshotName
Invoke-WebRequest -UseBasicParsing `
    -Uri "$($config.qdrant_url)/collections/$collection/snapshots/$snapshotName" `
    -Headers $headers -OutFile $qdrantDestination

if ($IncludeOpenWebUI) {
    & docker compose -f $composePath stop open-webui
    if ($LASTEXITCODE -ne 0) { throw "Could not stop Open WebUI for a consistent backup." }
    try {
        & docker run --rm `
            --volume "universal_rag_open_webui_data:/data:ro" `
            --volume "${snapshotRoot}:/backup" `
            alpine:3.22 tar -czf /backup/open-webui-data.tgz -C /data .
        if ($LASTEXITCODE -ne 0) { throw "Open WebUI volume backup failed." }
    }
    finally {
        & docker compose -f $composePath up -d open-webui
    }
}

$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestDestination).Hash
$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
$metadata = [ordered]@{
    created_at = [DateTimeOffset]::Now.ToString("o")
    git_commit = $gitCommit
    config_schema = [int]$config.schema_version
    qdrant_collection = [string]$config.collection
    qdrant_snapshot = $snapshotName
    manifest_file = "documents.sqlite"
    manifest_sha256 = $manifestHash
    open_webui_included = [bool]$IncludeOpenWebUI
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $snapshotRoot "snapshot.json") -Encoding UTF8

Write-Host "State backup created: $snapshotRoot"
