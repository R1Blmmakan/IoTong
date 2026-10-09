@echo off
rem PlatformIO Dynamic Uploader
rem Tutup proses serial monitor atau deteksi yang mengunci port sebelum upload
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'device monitor|detect_trash' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

where pio >nul 2>nul
if %errorlevel% equ 0 (
    pio run -t upload %*
) else if exist "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" (
    "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" run -t upload %*
) else (
    echo [ERROR] PlatformIO CLI tidak ditemukan di PATH atau di %%USERPROFILE%%\.platformio\penv\Scripts!
    exit /b 1
)
