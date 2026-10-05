# PowerShell Overnight E9 Ensemble Runner with Tee-Object logging
param (
    [string]$Profile = "laptop"
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogsDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDir "ensemble_overnight_$Timestamp.log"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "PENUMBRA OVERNIGHT ENSEMBLE TRAINING" -ForegroundColor Cyan
Write-Host "Profile: $Profile" -ForegroundColor Cyan
Write-Host "Log file: $LogFile" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$Datasets = @("unsw", "nslkdd", "cicids")

foreach ($ds in $Datasets) {
    Write-Host "`n>>> Starting dataset: $ds (Profile: $Profile) at $(Get-Date -Format 'HH:mm:ss') <<<" -ForegroundColor Yellow
    
    # Run penumbra with tee logging
    penumbra ensemble -d $ds --profile $Profile 2>&1 | Tee-Object -FilePath $LogFile -Append
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] Training failed on dataset: $ds with code $LASTEXITCODE" -ForegroundColor Red
        break
    }
    Write-Host "`n[SUCCESS] Completed dataset: $ds at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
}

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT RUN FINISHED at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "Check reports in artifacts/reports/:" -ForegroundColor Cyan
Get-ChildItem -Path (Join-Path $RepoRoot "artifacts\reports\ensemble_*.json") | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Write-Host "Log file saved to: $LogFile" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
