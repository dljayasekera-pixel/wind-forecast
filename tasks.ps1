<#
Windows equivalent of the Makefile.

Usage (from the repo folder, in PowerShell):
    .\tasks.ps1 install
    .\tasks.ps1 test
    .\tasks.ps1 run -Config configs/kelmarsh.yaml
    .\tasks.ps1 app -Artifacts artifacts/kelmarsh

If Windows blocks the script ("running scripts is disabled on this system"), run once per window:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "lint", "test", "test-fast", "run", "backtest", "diagnose", "app", "clean")]
    [string]$Task = "help",
    [string]$Config = "configs/synthetic.yaml",
    [string]$Artifacts = "artifacts/synthetic"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Every tool is called through "python -m" so it works even when Python's Scripts folder is not on PATH.
function Invoke-Python {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed: python $($Arguments -join ' ')" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Test-PythonVersion {
    $version = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if (-not $version) {
        Write-Host "Python was not found. Install Python 3.12 or 3.13 from python.org and tick 'Add python.exe to PATH'." -ForegroundColor Red
        exit 1
    }
    $parts = $version.Split(".")
    if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 11)) {
        Write-Host "Python $version found; this project needs 3.11 or newer (3.12 or 3.13 recommended)." -ForegroundColor Red
        exit 1
    }
    Write-Host "Using Python $version"
}

switch ($Task) {
    "install" {
        Test-PythonVersion
        Invoke-Python @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Python @("-m", "pip", "install", "-r", "requirements.lock")
        Invoke-Python @("-m", "pip", "install", "-e", ".", "--no-deps")
    }
    "lint"      { Invoke-Python @("-m", "ruff", "check", "src", "tests", "app", "infra") }
    "test"      { Invoke-Python @("-m", "pytest") }
    "test-fast" { Invoke-Python @("-m", "pytest", "-m", "not slow") }
    "run"       { Invoke-Python @("-m", "windcast.cli", "run-all", "--config", $Config) }
    "backtest"  { Invoke-Python @("-m", "windcast.cli", "backtest", "--config", $Config) }
    "diagnose"  { Invoke-Python @("scripts/diagnose_data.py", "--config", $Config) }
    "app"       { Invoke-Python @("-m", "streamlit", "run", "app/streamlit_app.py", "--", "--artifacts", $Artifacts) }
    "clean" {
        foreach ($path in @("data", "artifacts", ".pytest_cache", ".ruff_cache")) {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
        }
    }
    default {
        Write-Host "Tasks: install, lint, test, test-fast, run, backtest, diagnose, app, clean"
        Write-Host "Options: -Config <yaml> (run, backtest, diagnose)   -Artifacts <folder> (app)"
    }
}
