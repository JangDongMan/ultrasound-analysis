# ============================================================
# VB5K Capture - Fetch files from server
# Usage:
#   .\fetch_from_server.ps1
#   .\fetch_from_server.ps1 -ServerHost 192.168.1.100
# ============================================================

param(
    [string]$User       = "dmjang",
    [string]$ServerHost = "spark-a70d",
    [string]$RemoteDir  = "~/work/ultrasound_analysis/utils3",
    [string]$LocalDir   = $PSScriptRoot
)

$Server = "${User}@${ServerHost}"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " VB5K Capture - Fetch from server"          -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Server : $Server"
Write-Host "Remote : $RemoteDir/*"
Write-Host "Local  : $LocalDir"
Write-Host ""

scp -r "${Server}:${RemoteDir}/*" "$LocalDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done: all files fetched successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  Build exe  : .\build_exe.bat"
    Write-Host "  Run direct : py -3 adc_capture_gui.py"
} else {
    Write-Host ""
    Write-Host "FAILED (exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  Test connection : ssh $Server"
    Write-Host "  Add SSH key     : ssh-copy-id $Server"
}

Write-Host ""
