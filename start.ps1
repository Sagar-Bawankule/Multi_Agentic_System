param(
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host 'Virtual environment not found. Create with: python -m venv .venv' -ForegroundColor Yellow
}

$env:PYTHONUNBUFFERED = 1
if ($Reload.IsPresent) {
    $reloadFlag = '--reload'
} else {
    $reloadFlag = ''
}
$cmd = "python -m uvicorn app:app --host 0.0.0.0 --port $Port $reloadFlag".Trim()

if ($Background.IsPresent) {
    Write-Host "Starting server in background on port $Port..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList '-NoLogo','-NoProfile','-Command',"cd `"$scriptDir`"; . .\.venv\Scripts\Activate.ps1; $cmd" -WindowStyle Normal
    Write-Host 'Use Get-Process -Name python to see it or Stop-Process to kill.' -ForegroundColor Green
} else {
    Write-Host "Running server (Ctrl+C to stop) -> http://127.0.0.1:$Port" -ForegroundColor Green
    Invoke-Expression $cmd
}

Pop-Location
