<#
.SYNOPSIS
    Runs a PyPSA-Eur Snakemake target, archives the results afterwards, and
    appends a one-line entry to experiments/log.md -- so successive
    iterations never overwrite each other and you always have a record of
    what each one was.

.PARAMETER Label
    Short label for this experiment, e.g. "dk-baseline" or "dk-detailed-v3".
    Used to build the archive folder name and the log entry.

.PARAMETER Description
    One-line note on what this run actually tests (e.g. "detailed grid,
    5 clusters, DK only"). Goes straight into experiments/log.md.

.PARAMETER ConfigFile
    Optional path to a config file to pass to Snakemake via --configfile.
    Leave unset to just run "snakemake <Target>" and let the Snakefile's
    own default config-loading behaviour apply (this is what you want if
    config.default.yaml is your only config file).

.PARAMETER Target
    Snakemake target/rule to run. Defaults to "all".

.EXAMPLE
    .\run_and_archive.ps1 -Label "dk-baseline" -Description "original single-link grid, 5 clusters, DK only"

.EXAMPLE
    .\run_and_archive.ps1 -Label "dk-detailed-v3" -Description "feeder splitting fix" -ConfigFile config/config.default.yaml -Target all
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Label,

    [Parameter(Mandatory = $true)]
    [string]$Description,

    [string]$ConfigFile = "",

    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$archiveDir = "experiments/${timestamp}_${Label}"

# --- run Snakemake -----------------------------------------------------
if ($ConfigFile -ne "") {
    Write-Host "=== Running: snakemake $Target --configfile $ConfigFile ==="
    snakemake $Target --configfile $ConfigFile
    $effectiveConfigFile = $ConfigFile
} else {
    Write-Host "=== Running: snakemake $Target ==="
    snakemake $Target
    $effectiveConfigFile = "config/config.default.yaml"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Snakemake run failed (exit code $LASTEXITCODE) -- nothing archived." -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- figure out where results actually landed (run.name aware) --------
$resultsSource = "results"
if (Test-Path $effectiveConfigFile) {
    $runNameLine = Select-String -Path $effectiveConfigFile -Pattern '^\s*name:\s*"?([^"\r\n]*)"?\s*$' |
        Select-Object -First 1
    if ($runNameLine) {
        $runName = $runNameLine.Matches[0].Groups[1].Value.Trim()
        if ($runName -ne "") {
            $resultsSource = "results/$runName"
        }
    }
}

Write-Host "=== Snakemake finished successfully. Archiving from '$resultsSource' to $archiveDir ==="
New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null

if (Test-Path $effectiveConfigFile) {
    Copy-Item $effectiveConfigFile "$archiveDir/config_used.yaml"
}

if (Test-Path $resultsSource) {
    Copy-Item -Recurse -Force $resultsSource "$archiveDir/results"
} else {
    Write-Host "Expected results at '$resultsSource' but it doesn't exist -- check run.name in $effectiveConfigFile and adjust manually." -ForegroundColor Yellow
}

# Optional: also snapshot intermediate resources (can be large for
# sector-coupled runs) -- uncomment if you want full reproducibility.
# $resourcesSource = $resultsSource -replace '^results', 'resources'
# if (Test-Path $resourcesSource) {
#     Copy-Item -Recurse -Force $resourcesSource "$archiveDir/resources"
# }

# --- append to the experiment log --------------------------------------
$logFile = "experiments/log.md"
if (-not (Test-Path $logFile)) {
    New-Item -ItemType Directory -Force -Path "experiments" | Out-Null
    "| Timestamp | Label | Description | Archive |" | Out-File -FilePath $logFile -Encoding utf8
    "|---|---|---|---|" | Out-File -FilePath $logFile -Append -Encoding utf8
}
"| $timestamp | $Label | $Description | $archiveDir |" | Out-File -FilePath $logFile -Append -Encoding utf8

Write-Host "=== Archived to $archiveDir ===" -ForegroundColor Green
Write-Host "=== Logged to $logFile ===" -ForegroundColor Green