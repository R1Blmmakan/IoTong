@echo off
rem PlatformIO Dynamic Serial Monitor
where pio >nul 2>nul
if %errorlevel% equ 0 (
    pio device monitor %*
) else if exist "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" (
    "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" device monitor %*
) else (
    echo [ERROR] PlatformIO CLI tidak ditemukan di PATH atau di %%USERPROFILE%%\.platformio\penv\Scripts!
    exit /b 1
)
