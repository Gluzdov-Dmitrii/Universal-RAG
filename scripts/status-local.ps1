[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv\Scripts\python.exe"))
$appPath = [IO.Path]::GetFullPath((Join-Path $repoRoot "src\secure_rag\api\web.py"))
$pidFile = Join-Path $repoRoot "runtime\run\pids\streamlit.json"
$appHealthUrl = "http://127.0.0.1:8501/_stcore/health"
$qdrantHealthUrl = "http://127.0.0.1:6333/readyz"

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

$appHealthy = Test-HttpHealth -Url $appHealthUrl
$appState = "stopped"
$appProcessId = $null
$stdoutLog = $null
$stderrLog = $null

if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    try {
        $metadata = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        $appProcessId = [int]$metadata.process_id
        $stdoutLog = [string]$metadata.stdout_log
        $stderrLog = [string]$metadata.stderr_log
        $processRecord = Get-ProcessRecord -ProcessId $appProcessId
        if ($null -eq $processRecord) {
            $appState = "stale PID metadata"
        }
        else {
            $actualExecutable = if ($processRecord.ExecutablePath) {
                [IO.Path]::GetFullPath([string]$processRecord.ExecutablePath)
            }
            else {
                ""
            }
            $identityMatches = (
                [string]::Equals($actualExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
                $processRecord.CommandLine -and
                $processRecord.CommandLine.IndexOf("-m streamlit", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $processRecord.CommandLine.IndexOf($appPath, [StringComparison]::OrdinalIgnoreCase) -ge 0
            )
            if (-not $identityMatches) {
                $appState = "PID identity mismatch (no process will be changed)"
            }
            elseif ($appHealthy) {
                $appState = "running and healthy"
            }
            else {
                $appState = "running but unhealthy"
            }
        }
    }
    catch {
        $appState = "invalid PID metadata (no process will be changed)"
    }
}
elseif ($appHealthy) {
    $appState = "untracked service on port 8501"
}

$dockerAvailable = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    $dockerAvailable = ($LASTEXITCODE -eq 0)
}
$qdrantHealthy = Test-HttpHealth -Url $qdrantHealthUrl
$qdrantState = if ($qdrantHealthy) {
    "running and healthy"
}
elseif ($dockerAvailable) {
    "not healthy"
}
else {
    "not reachable; Docker Engine unavailable"
}

[PSCustomObject]@{
    Component = "Secure RAG Streamlit"
    State = $appState
    PID = $appProcessId
    URL = "http://127.0.0.1:8501/"
} | Format-List

[PSCustomObject]@{
    Component = "Qdrant"
    State = $qdrantState
    PID = $null
    URL = "http://127.0.0.1:6333/dashboard"
} | Format-List

if ($stdoutLog -or $stderrLog) {
    Write-Host "Streamlit logs:"
    if ($stdoutLog) { Write-Host "  stdout: $stdoutLog" }
    if ($stderrLog) { Write-Host "  stderr: $stderrLog" }
}
