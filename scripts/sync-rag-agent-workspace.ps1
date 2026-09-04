[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("Push", "Check")]
    [string]$Mode = "Push",

    [string]$TargetRoot = "C:\Dev\LLM\Codex_RAG_Test"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$templateRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "llm-workspaces\rag-test"))
$target = [IO.Path]::GetFullPath($TargetRoot)
$templatePrefix = $templateRoot.TrimEnd("\") + "\"
$targetPrefix = $target.TrimEnd("\") + "\"
$repoPrefix = $repoRoot.TrimEnd("\") + "\"

if (-not (Test-Path -LiteralPath $templateRoot -PathType Container)) {
    throw "RAG agent workspace template is missing: $templateRoot"
}
if ([string]::Equals($target, $repoRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $target.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase) -or
    $repoRoot.StartsWith($targetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Target workspace must not overlap the Secure RAG repository: $target"
}
if ([string]::Equals($target, [IO.Path]::GetPathRoot($target), [StringComparison]::OrdinalIgnoreCase)) {
    throw "Target workspace must not be a filesystem root: $target"
}

$manifestPath = Join-Path $templateRoot ".rag-workspace-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.schema_version -ne 1 -or -not $manifest.managed_files) {
    throw "Unsupported or empty RAG workspace manifest."
}

$drift = [Collections.Generic.List[string]]::new()
foreach ($relativeValue in $manifest.managed_files) {
    $relative = [string]$relativeValue
    if ([IO.Path]::IsPathRooted($relative) -or
        ($relative -split '[\\/]') -contains "..") {
        throw "Invalid managed path in workspace manifest: $relative"
    }
    $source = [IO.Path]::GetFullPath((Join-Path $templateRoot $relative))
    $destination = [IO.Path]::GetFullPath((Join-Path $target $relative))
    if (-not $source.StartsWith($templatePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith($targetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Managed workspace path escaped its root: $relative"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Managed template file is missing: $source"
    }

    $matches = (
        (Test-Path -LiteralPath $destination -PathType Leaf) -and
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -eq
        (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    )
    if ($matches) {
        continue
    }
    if ($Mode -eq "Check") {
        $drift.Add($relative)
        continue
    }

    if ($PSCmdlet.ShouldProcess($destination, "Deploy managed RAG agent file")) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

if ($Mode -eq "Check" -and $drift.Count -gt 0) {
    Write-Error "RAG agent workspace drift: $($drift -join ', ')"
    exit 2
}
if ($Mode -eq "Check") {
    Write-Host "RAG agent workspace matches the canonical template."
}
else {
    Write-Host "RAG agent workspace deployed. Unmanaged target files were preserved."
}
