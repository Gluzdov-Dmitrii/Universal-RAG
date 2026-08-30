[CmdletBinding()]
param(
    [switch]$StopQdrant,

    [ValidateRange(1, 60)]
    [int]$AppTimeoutSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$composePath = Join-Path $repoRoot "deploy\docker-compose.yml"
$pythonPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv\Scripts\python.exe"))
$appPath = [IO.Path]::GetFullPath((Join-Path $repoRoot "src\secure_rag\web_app.py"))
$pidFile = Join-Path $repoRoot "runtime\pids\streamlit.json"
$appHealthUrl = "http://127.0.0.1:8501/_stcore/health"

function Test-HttpHealth {
    param([Parameter(Mandatory)][string]$Url)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Get-ProcessRecord {
    param([Parameter(Mandatory)][int]$ProcessId)

    return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    try {
        $metadata = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    }
    catch {
        throw "PID file is invalid. No process was changed: $pidFile"
    }

    $trackedProcessId = [int]$metadata.process_id
    $processRecord = Get-ProcessRecord -ProcessId $trackedProcessId
    if ($null -eq $processRecord) {
        Remove-Item -LiteralPath $pidFile -Force
        Write-Host "Secure RAG was already stopped; removed stale PID metadata."
    }
    else {
        $metadataExecutable = [IO.Path]::GetFullPath([string]$metadata.executable_path)
        $actualExecutable = if ($processRecord.ExecutablePath) {
            [IO.Path]::GetFullPath([string]$processRecord.ExecutablePath)
        }
        else {
            ""
        }
        $identityMatches = (
            [string]::Equals($metadataExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
            [string]::Equals($actualExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
            $processRecord.CommandLine -and
            $processRecord.CommandLine.IndexOf("-m streamlit", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $processRecord.CommandLine.IndexOf($appPath, [StringComparison]::OrdinalIgnoreCase) -ge 0
        )
        if (-not $identityMatches) {
            throw "PID $trackedProcessId does not match this repository's Streamlit process. Nothing was stopped."
        }

        Stop-Process -Id $trackedProcessId
        $deadline = [DateTimeOffset]::UtcNow.AddSeconds($AppTimeoutSeconds)
        do {
            Start-Sleep -Milliseconds 250
            $processRecord = Get-ProcessRecord -ProcessId $trackedProcessId
        } while ($null -ne $processRecord -and [DateTimeOffset]::UtcNow -lt $deadline)

        if ($null -ne $processRecord) {
            throw "Streamlit PID $trackedProcessId did not stop within $AppTimeoutSeconds seconds."
        }
        Remove-Item -LiteralPath $pidFile -Force
        Write-Host "Secure RAG Streamlit app stopped."
    }
}
elseif (Test-HttpHealth -Url $appHealthUrl) {
    throw "A healthy untracked service is listening on port 8501. It was not stopped."
}
else {
    Write-Host "Secure RAG Streamlit app is already stopped."
}

if ($StopQdrant) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI was not found; Qdrant was not stopped."
    }
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Engine is unavailable; Qdrant was not stopped."
    }
    & docker compose -f $composePath stop qdrant
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose stop failed with exit code $LASTEXITCODE."
    }
    Write-Host "Qdrant stopped; the named volume and vector index were preserved."
}
