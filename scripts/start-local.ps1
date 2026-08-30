[CmdletBinding()]
param(
    [ValidateRange(10, 600)]
    [int]$DockerTimeoutSeconds = 120,

    [ValidateRange(10, 300)]
    [int]$QdrantTimeoutSeconds = 90,

    [ValidateRange(5, 180)]
    [int]$AppTimeoutSeconds = 45
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$composePath = Join-Path $repoRoot "deploy\docker-compose.yml"
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $repoRoot "src\secure_rag\web_app.py"
$runtimeRoot = Join-Path $repoRoot "runtime"
$pidDirectory = Join-Path $runtimeRoot "pids"
$logDirectory = Join-Path $runtimeRoot "logs"
$pidFile = Join-Path $pidDirectory "streamlit.json"
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

function Test-DockerEngine {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Wait-Until {
    param(
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][int]$TimeoutSeconds,
        [int]$PollMilliseconds = 1000
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if (& $Condition) {
            return $true
        }
        Start-Sleep -Milliseconds $PollMilliseconds
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    return $false
}

function Get-TrackedApp {
    if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
        return $null
    }

    try {
        return Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    }
    catch {
        throw "PID file is invalid. No process was changed: $pidFile"
    }
}

function Get-ProcessRecord {
    param([Parameter(Mandatory)][int]$ProcessId)

    return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-TrackedProcessIdentity {
    param(
        [Parameter(Mandatory)]$ProcessRecord,
        [Parameter(Mandatory)]$Metadata
    )

    if (-not $ProcessRecord -or -not $ProcessRecord.ExecutablePath -or -not $ProcessRecord.CommandLine) {
        return $false
    }
    $expectedExecutable = [IO.Path]::GetFullPath([string]$Metadata.executable_path)
    $actualExecutable = [IO.Path]::GetFullPath([string]$ProcessRecord.ExecutablePath)
    if (-not [string]::Equals($actualExecutable, $expectedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    if ($ProcessRecord.CommandLine.IndexOf("-m streamlit", [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        return $false
    }
    return ($ProcessRecord.CommandLine.IndexOf($appPath, [StringComparison]::OrdinalIgnoreCase) -ge 0)
}

foreach ($requiredPath in @($composePath, $pythonPath, $appPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $pidDirectory, $logDirectory | Out-Null

if (-not (Test-DockerEngine)) {
    $dockerDesktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktopPath -PathType Leaf)) {
        throw "Docker Engine is unavailable and Docker Desktop was not found: $dockerDesktopPath"
    }

    if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
        Write-Host "Starting Docker Desktop..."
        Start-Process -FilePath $dockerDesktopPath -WindowStyle Hidden | Out-Null
    }
    else {
        Write-Host "Docker Desktop is running; waiting for Docker Engine..."
    }

    if (-not (Wait-Until -TimeoutSeconds $DockerTimeoutSeconds -Condition { Test-DockerEngine })) {
        throw "Docker Engine did not become ready within $DockerTimeoutSeconds seconds."
    }
}

Write-Host "Starting Qdrant..."
& docker compose -f $composePath up -d qdrant
if ($LASTEXITCODE -ne 0) {
    throw "docker compose failed with exit code $LASTEXITCODE."
}
if (-not (Wait-Until -TimeoutSeconds $QdrantTimeoutSeconds -Condition {
            Test-HttpHealth -Url $qdrantHealthUrl
        })) {
    throw "Qdrant did not become healthy within $QdrantTimeoutSeconds seconds."
}

$tracked = Get-TrackedApp
if ($null -ne $tracked) {
    $trackedProcessId = [int]$tracked.process_id
    $processRecord = Get-ProcessRecord -ProcessId $trackedProcessId
    if ($null -ne $processRecord) {
        if (-not (Test-TrackedProcessIdentity -ProcessRecord $processRecord -Metadata $tracked)) {
            throw "PID $trackedProcessId belongs to another process. No process was changed. Inspect $pidFile."
        }
        if (Test-HttpHealth -Url $appHealthUrl) {
            Write-Host "Secure RAG is already running: http://127.0.0.1:8501/ (PID $trackedProcessId)"
            exit 0
        }
        throw "Tracked Streamlit process $trackedProcessId is running but is not healthy. Run scripts\status-local.ps1."
    }

    Remove-Item -LiteralPath $pidFile -Force
}

if (Test-HttpHealth -Url $appHealthUrl) {
    throw "Port 8501 has an untracked healthy service. It was not changed."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$stdoutPath = Join-Path $logDirectory "streamlit-$timestamp.stdout.log"
$stderrPath = Join-Path $logDirectory "streamlit-$timestamp.stderr.log"
$arguments = @(
    "-m",
    "streamlit",
    "run",
    $appPath,
    "--server.address",
    "127.0.0.1",
    "--server.port",
    "8501",
    "--server.headless",
    "true"
)

# The prototype uses pinned, pre-downloaded model revisions. Prevent an ordinary app
# start from contacting Hugging Face or silently fetching different artifacts.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$appProcess = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$metadata = [ordered]@{
    schema_version   = 1
    process_id       = $appProcess.Id
    executable_path  = [IO.Path]::GetFullPath($pythonPath)
    app_path         = [IO.Path]::GetFullPath($appPath)
    repo_root        = $repoRoot
    started_at       = [DateTimeOffset]::Now.ToString("o")
    url              = "http://127.0.0.1:8501/"
    stdout_log       = $stdoutPath
    stderr_log       = $stderrPath
}
$temporaryPidFile = "$pidFile.tmp"
$metadata | ConvertTo-Json | Set-Content -LiteralPath $temporaryPidFile -Encoding UTF8
Move-Item -LiteralPath $temporaryPidFile -Destination $pidFile -Force

if (-not (Wait-Until -TimeoutSeconds $AppTimeoutSeconds -Condition {
            Test-HttpHealth -Url $appHealthUrl
        })) {
    $stillRunning = Get-ProcessRecord -ProcessId $appProcess.Id
    if ($null -eq $stillRunning) {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
    throw "Streamlit did not become healthy within $AppTimeoutSeconds seconds. Logs: $stdoutPath and $stderrPath"
}

Write-Host "Secure RAG is ready: http://127.0.0.1:8501/"
Write-Host "PID: $($appProcess.Id)"
Write-Host "Logs: $stdoutPath and $stderrPath"
