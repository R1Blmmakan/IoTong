# PlatformIO Dynamic Uploader
$pio = Get-Command pio, platformio -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $pio -or -not (Test-Path $pio)) {
    $pio = Join-Path $HOME ".platformio\penv\Scripts\platformio.exe"
}

if (Test-Path $pio) {
    & $pio run -t upload $args
} else {
    Write-Error "[ERROR] PlatformIO CLI tidak ditemukan di PATH atau di $HOME\.platformio\penv\Scripts\"
}
