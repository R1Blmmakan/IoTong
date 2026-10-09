# PlatformIO Dynamic Serial Monitor
# Tutup script deteksi yang sedang mengunci port serial agar monitor dapat terbuka
$busyProcesses = Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match 'detect_trash'
}
if ($busyProcesses) {
    Write-Host "[INFO] Menutup detect_trash.py yang sedang menggunakan port serial..."
    $busyProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 400
}

$pio = Get-Command pio, platformio -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $pio -or -not (Test-Path $pio)) {
    $pio = Join-Path $HOME ".platformio\penv\Scripts\platformio.exe"
}

if (Test-Path $pio) {
    & $pio device monitor $args
} else {
    Write-Error "[ERROR] PlatformIO CLI tidak ditemukan di PATH atau di $HOME\.platformio\penv\Scripts\"
}
