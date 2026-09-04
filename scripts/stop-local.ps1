[CmdletBinding()]
param(
    [switch]$StopQdrant,

    [ValidateRange(1, 60)]
    [int]$ApiTimeoutSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$composePath = Join-Path $repoRoot "deploy\docker-compose.yml"
$pythonPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv\Scripts\python.exe"))
$pidFile = Join-Path $repoRoot "runtime\run\pids\api.json"
$localEnvironmentPath = Join-Path $repoRoot ".env"

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

if (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker compose -f $composePath stop open-webui
    if ($LASTEXITCODE -ne 0) { throw "Open WebUI compose stop failed." }
}

if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    $metadata = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $trackedProcessId = [int]$metadata.process_id
    $processRecord = Get-CimInstance Win32_Process -Filter "ProcessId = $trackedProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $processRecord) {
        Remove-Item -LiteralPath $pidFile -Force
    }
    else {
        $actualExecutable = [IO.Path]::GetFullPath([string]$processRecord.ExecutablePath)
        $identityMatches = (
            [string]::Equals($actualExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
            $processRecord.CommandLine -and
            $processRecord.CommandLine.IndexOf("-m uvicorn", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $processRecord.CommandLine.IndexOf("secure_rag.api.web:app", [StringComparison]::OrdinalIgnoreCase) -ge 0
        )
        if (-not $identityMatches) {
            throw "PID $trackedProcessId does not match this deployment's API. Nothing was stopped."
        }
        Stop-Process -Id $trackedProcessId
        $deadline = [DateTimeOffset]::UtcNow.AddSeconds($ApiTimeoutSeconds)
        do {
            Start-Sleep -Milliseconds 250
            $processRecord = Get-CimInstance Win32_Process -Filter "ProcessId = $trackedProcessId" -ErrorAction SilentlyContinue
        } while ($null -ne $processRecord -and [DateTimeOffset]::UtcNow -lt $deadline)
        if ($null -ne $processRecord) {
            throw "Universal RAG API PID $trackedProcessId did not stop in time."
        }
        Remove-Item -LiteralPath $pidFile -Force
    }
}

if ($StopQdrant) {
    & docker compose -f $composePath stop qdrant
    if ($LASTEXITCODE -ne 0) { throw "Qdrant compose stop failed." }
}

Write-Host "Open WebUI and Universal RAG API stopped. Persistent data was preserved."
if ($StopQdrant) { Write-Host "Qdrant stopped; its named volume was preserved." }
