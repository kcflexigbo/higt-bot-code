# Training dashboard

1. Start: `cd dashboard && docker compose up -d` (Docker Desktop must be running).
2. GPU line (host side, from the project dir): Git Bash `bash dashboard/gpu_poller.sh &`, or PowerShell `Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\dashboard\gpu_poller.ps1`""`.
3. Open http://localhost:8088 (auto-refreshes every 10 s). Stop: `cd dashboard && docker compose down` (stop the poller: `Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | ? CommandLine -like '*gpu_poller*' | % { Stop-Process -Id $_.ProcessId }`).
