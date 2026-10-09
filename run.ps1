# IoTong AI Waste Detection Runner
# Tutup serial monitor jika masih aktif agar port serial dapat dibuka oleh script deteksi
Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match 'device monitor'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy (Join-Path $PSScriptRoot "detect_trash.py") @args
} else {
    python (Join-Path $PSScriptRoot "detect_trash.py") @args
}
