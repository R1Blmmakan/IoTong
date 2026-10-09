# PlatformIO Dynamic Uploader
# Hentikan proses yang mengunci port serial (serial monitor atau script deteksi)
Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match 'device monitor|detect_trash'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$pio = Get-Command pio, platformio -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $pio -or -not (Test-Path $pio)) {
    $pio = Join-Path $HOME ".platformio\penv\Scripts\platformio.exe"
}

if (Test-Path $pio) {
    & $pio run -t upload $args
} else {
    Write-Error "[ERROR] PlatformIO CLI tidak ditemukan di PATH atau di $HOME\.platformio\penv\Scripts\"
}
