[CmdletBinding()]
param(
    [ValidateRange(10, 600)]
    [int]$DockerTimeoutSeconds = 120,

    [ValidateRange(10, 300)]
    [int]$QdrantTimeoutSeconds = 90,

    [ValidateRange(5, 300)]
    [int]$ApiTimeoutSeconds = 90,

    [ValidateRange(10, 600)]
    [int]$WebUiTimeoutSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$composePath = Join-Path $repoRoot "deploy\docker-compose.yml"
$pythonPath = [IO.Path]::GetFullPath((Join-Path $repoRoot ".venv\Scripts\python.exe"))
$runtimeRoot = Join-Path $repoRoot "runtime"
$pidDirectory = Join-Path $runtimeRoot "run\pids"
$logDirectory = Join-Path $runtimeRoot "diagnostics\logs"
$pidFile = Join-Path $pidDirectory "api.json"
$localEnvironmentPath = Join-Path $repoRoot ".env"

function Import-DotEnv {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing .env. Copy .env.example to .env and set machine-specific values."
    }
    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            throw "Invalid .env line. Expected NAME=VALUE without shell expressions."
        }
        Set-Item -LiteralPath "Env:$($parts[0])" -Value $parts[1]
    }
}

function Test-HttpHealth {
    param([Parameter(Mandatory)][string]$Url)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch { return $false }
}

function Wait-Until {
    param(
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][int]$TimeoutSeconds
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if (& $Condition) { return $true }
        Start-Sleep -Milliseconds 1000
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    return $false
}

function Test-DockerEngine {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-ProcessRecord {
    param([Parameter(Mandatory)][int]$ProcessId)
    return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-TrackedApiIdentity {
    param([Parameter(Mandatory)]$ProcessRecord)

    if (-not $ProcessRecord -or -not $ProcessRecord.ExecutablePath -or -not $ProcessRecord.CommandLine) {
        return $false
    }
    $actualExecutable = [IO.Path]::GetFullPath([string]$ProcessRecord.ExecutablePath)
    return (
        [string]::Equals($actualExecutable, $pythonPath, [StringComparison]::OrdinalIgnoreCase) -and
        $ProcessRecord.CommandLine.IndexOf("-m uvicorn", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $ProcessRecord.CommandLine.IndexOf("secure_rag.api.web:app", [StringComparison]::OrdinalIgnoreCase) -ge 0
    )
}

foreach ($requiredPath in @($composePath, $pythonPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}

Import-DotEnv -Path $localEnvironmentPath
foreach ($secretName in @("SECURE_RAG_API_KEY", "OPEN_WEBUI_SECRET_KEY")) {
    $secretValue = [Environment]::GetEnvironmentVariable($secretName)
    if (-not $secretValue -or $secretValue.StartsWith("replace-with-")) {
        throw "$secretName must contain a generated secret in .env."
    }
}

$apiHost = if ($env:SECURE_RAG_API_HOST) { $env:SECURE_RAG_API_HOST } else { "0.0.0.0" }
$apiPort = if ($env:SECURE_RAG_API_PORT) { [int]$env:SECURE_RAG_API_PORT } else { 8000 }
$webUiPort = if ($env:OPEN_WEBUI_PORT) { [int]$env:OPEN_WEBUI_PORT } else { 3000 }
$apiHealthUrl = "http://127.0.0.1:$apiPort/healthz"
$qdrantHealthUrl = "http://127.0.0.1:6333/readyz"
$webUiHealthUrl = "http://127.0.0.1:$webUiPort/health"

New-Item -ItemType Directory -Force -Path $pidDirectory, $logDirectory | Out-Null

if (-not (Test-DockerEngine)) {
    $dockerDesktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktopPath -PathType Leaf)) {
        throw "Docker Engine is unavailable and Docker Desktop was not found."
    }
    if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $dockerDesktopPath -WindowStyle Hidden | Out-Null
    }
    if (-not (Wait-Until -TimeoutSeconds $DockerTimeoutSeconds -Condition { Test-DockerEngine })) {
        throw "Docker Engine did not become ready within $DockerTimeoutSeconds seconds."
    }
}

& docker compose -f $composePath up -d qdrant
if ($LASTEXITCODE -ne 0) { throw "Qdrant compose start failed with exit code $LASTEXITCODE." }
if (-not (Wait-Until -TimeoutSeconds $QdrantTimeoutSeconds -Condition {
            Test-HttpHealth -Url $qdrantHealthUrl
        })) {
    throw "Qdrant did not become healthy within $QdrantTimeoutSeconds seconds."
}

$apiRunning = $false
if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    $metadata = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $trackedProcess = Get-ProcessRecord -ProcessId ([int]$metadata.process_id)
    if ($null -eq $trackedProcess) {
        Remove-Item -LiteralPath $pidFile -Force
    }
    elseif (-not (Test-TrackedApiIdentity -ProcessRecord $trackedProcess)) {
        throw "Tracked API PID does not belong to this deployment. Nothing was changed."
    }
    elseif (-not (Test-HttpHealth -Url $apiHealthUrl)) {
        throw "Tracked Universal RAG API process is running but unhealthy."
    }
    else {
        $apiRunning = $true
    }
}
elseif (Test-HttpHealth -Url $apiHealthUrl) {
    throw "Port $apiPort has an untracked healthy service. It was not changed."
}

if (-not $apiRunning) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutPath = Join-Path $logDirectory "api-$timestamp.stdout.log"
    $stderrPath = Join-Path $logDirectory "api-$timestamp.stderr.log"
    $arguments = @(
        "-m", "uvicorn", "secure_rag.api.web:app",
        "--host", $apiHost,
        "--port", [string]$apiPort,
        "--workers", "1"
    )
    $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
        -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $metadata = [ordered]@{
        process_id = $apiProcess.Id
        executable_path = $pythonPath
        repo_root = $repoRoot
        started_at = [DateTimeOffset]::Now.ToString("o")
        url = $apiHealthUrl
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
    }
    $temporaryPidFile = "$pidFile.tmp"
    $metadata | ConvertTo-Json | Set-Content -LiteralPath $temporaryPidFile -Encoding UTF8
    Move-Item -LiteralPath $temporaryPidFile -Destination $pidFile -Force
    if (-not (Wait-Until -TimeoutSeconds $ApiTimeoutSeconds -Condition {
                Test-HttpHealth -Url $apiHealthUrl
            })) {
        throw "Universal RAG API did not become healthy. Logs: $stdoutPath and $stderrPath"
    }
}

& docker compose -f $composePath up -d open-webui
if ($LASTEXITCODE -ne 0) { throw "Open WebUI compose start failed with exit code $LASTEXITCODE." }
if (-not (Wait-Until -TimeoutSeconds $WebUiTimeoutSeconds -Condition {
            Test-HttpHealth -Url $webUiHealthUrl
        })) {
    throw "Open WebUI did not become healthy within $WebUiTimeoutSeconds seconds."
}

$displayUrl = if ($env:OPEN_WEBUI_URL) { $env:OPEN_WEBUI_URL } else { "http://localhost:$webUiPort" }
Write-Host "Universal RAG is ready: $displayUrl"
Write-Host "Backend health: $apiHealthUrl"
Write-Host "Qdrant dashboard (server only): http://127.0.0.1:6333/dashboard"
