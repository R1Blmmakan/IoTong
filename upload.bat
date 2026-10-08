@echo off
rem PlatformIO Dynamic Uploader
where pio >nul 2>nul
if %errorlevel% equ 0 (
    pio run -t upload %*
) else if exist "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" (
    "%USERPROFILE%\.platformio\penv\Scripts\platformio.exe" run -t upload %*
) else (
    echo [ERROR] PlatformIO CLI tidak ditemukan di PATH atau di %%USERPROFILE%%\.platformio\penv\Scripts!
    exit /b 1
)
