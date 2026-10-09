@echo off
rem PlatformIO Dynamic Serial Monitor
rem Tutup script deteksi yang sedang mengunci port serial agar monitor dapat terbuka
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%%'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'detect_trash' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

where pio >nul 2>nul
if %errorlevel% equ 0 (
    pio device monitor %*
) else if exist "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" (
    "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" device monitor %*
) else (
    echo [ERROR] PlatformIO CLI tidak ditemukan di PATH atau di %%USERPROFILE%%\.platformio\penv\Scripts!
    exit /b 1
)
