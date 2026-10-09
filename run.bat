@echo off
rem IoTong AI Waste Detection Runner
rem Tutup serial monitor jika masih aktif agar port serial dapat dibuka oleh script deteksi
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'device monitor' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0detect_trash.py" %*
) else (
    python "%~dp0detect_trash.py" %*
)
