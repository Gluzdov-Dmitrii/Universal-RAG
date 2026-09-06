[CmdletBinding()]
param()

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

$apiPort = if ($env:SECURE_RAG_API_PORT) { [int]$env:SECURE_RAG_API_PORT } else { 8000 }
$webUiPort = if ($env:OPEN_WEBUI_PORT) { [int]$env:OPEN_WEBUI_PORT } else { 3000 }
$agentWorkspace = if ($env:SECURE_RAG_AGENT_WORKSPACE_ROOT) {
    [IO.Path]::GetFullPath($env:SECURE_RAG_AGENT_WORKSPACE_ROOT)
}
else { $null }
$agentWorkspaceState = if ($null -eq $agentWorkspace) {
    "not configured"
}
elseif (-not (Test-Path -LiteralPath $agentWorkspace -PathType Container)) {
    "missing"
}
elseif (-not (Test-Path -LiteralPath (Join-Path $agentWorkspace ".rag-workspace-manifest.json") -PathType Leaf)) {
    "manifest missing"
}
else {
    "ready; saved Codex project root must match this path"
}

function Test-HttpHealth {
    param([Parameter(Mandatory)][string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch { return $false }
}

$apiProcessId = $null
$apiState = "stopped"
$stdoutLog = $null
$stderrLog = $null
if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    try {
        $metadata = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
        $apiProcessId = [int]$metadata.process_id
        $stdoutLog = [string]$metadata.stdout_log
        $stderrLog = [string]$metadata.stderr_log
        $processRecord = Get-CimInstance Win32_Process -Filter "ProcessId = $apiProcessId" -ErrorAction SilentlyContinue
        if ($null -eq $processRecord) {
            $apiState = "stale PID metadata"
        }
        else {
            $actualExecutable = [IO.Path]::GetFullPath([string]$processRecord.ExecutablePath)
            $identityMatches = (
                [string]::Equals($actualExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
                $processRecord.CommandLine -and
                $processRecord.CommandLine.IndexOf("-m uvicorn", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $processRecord.CommandLine.IndexOf("secure_rag.api.web:app", [StringComparison]::OrdinalIgnoreCase) -ge 0
            )
            if (-not $identityMatches) { $apiState = "PID identity mismatch" }
            elseif (Test-HttpHealth -Url "http://127.0.0.1:$apiPort/healthz") { $apiState = "healthy" }
            else { $apiState = "running but unhealthy" }
        }
    }
    catch { $apiState = "invalid PID metadata" }
}

$dockerState = "unavailable"
if (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $dockerState = "available"
        & docker compose --env-file $localEnvironmentPath -f $composePath ps
    }
}

[PSCustomObject]@{
    Component = "Universal RAG API"
    State = $apiState
    PID = $apiProcessId
    URL = "http://127.0.0.1:$apiPort/healthz"
} | Format-List

[PSCustomObject]@{
    Component = "Open WebUI"
    State = if (Test-HttpHealth -Url "http://127.0.0.1:$webUiPort/health") { "healthy" } else { "not healthy" }
    PID = $null
    URL = "http://localhost:$webUiPort/"
} | Format-List

[PSCustomObject]@{
    Component = "Qdrant"
    State = if (Test-HttpHealth -Url "http://127.0.0.1:6333/readyz") { "healthy" } else { "not healthy" }
    PID = $null
    URL = "http://127.0.0.1:6333/dashboard"
} | Format-List

[PSCustomObject]@{
    Component = "Codex RAG project workspace"
    State = $agentWorkspaceState
    PID = $null
    URL = $agentWorkspace
} | Format-List

Write-Host "Docker Engine: $dockerState"
if ($stdoutLog -or $stderrLog) {
    Write-Host "API logs:"
    if ($stdoutLog) { Write-Host "  stdout: $stdoutLog" }
    if ($stderrLog) { Write-Host "  stderr: $stderrLog" }
}
