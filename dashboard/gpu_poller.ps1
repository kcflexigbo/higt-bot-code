# Host-side GPU poller (PowerShell). Writes ..\data\inspection_logs\gpu_status.txt every 15 s.
# Line 1: ISO-8601 timestamp; following lines: util %, mem used, mem total, temp C (one per GPU).
# Usage (detached, from project dir): Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\dashboard\gpu_poller.ps1`""
param([int]$Interval = 15)
$out = Join-Path (Split-Path -Parent $PSScriptRoot) 'data\inspection_logs\gpu_status.txt'
$tmp = "$out.tmp"
while ($true) {
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz')
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
    $text = (@($ts) + @($gpu | ForEach-Object { "$_" })) -join "`n"
    [System.IO.File]::WriteAllText($tmp, $text + "`n", (New-Object System.Text.UTF8Encoding $false))
    Move-Item -Force $tmp $out
    Start-Sleep -Seconds $Interval
}
