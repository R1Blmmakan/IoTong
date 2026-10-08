# IoTong (Smart Trash Bin AI) 🗑️🤖

Sistem pemilah sampah otomatis berbasis Computer Vision (YOLO11) dan Mikrokontroler ESP32 dengan kontrol Dual Servo.

---

## 🚀 Panduan Setup Cepat (Untuk Tim / Kolaborator)

Setelah melakukan `git clone` atau `git pull`, lakukan langkah-langkah berikut:

### 1. Setup Python Environment & Dependensi
Pastikan kamu menggunakan Python 3.10+ (disarankan Python 3.11 atau 3.13):
```bash
# Buat virtual environment (opsional tapi disarankan)
python -m venv .venv

# Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install semua dependensi
pip install -r requirements.txt
```

### 2. Konfigurasi Environment (`.env`)
Salin file template `.env.example` menjadi `.env`:
```bash
# Windows:
copy .env.example .env

# Linux / Mac:
cp .env.example .env
```

Buka file `.env` dan sesuaikan dengan perangkat kamu:
| Variabel | Default | Keterangan |
| :--- | :--- | :--- |
| `SERIAL_PORT` | `AUTO` | Port ESP32 (`AUTO` mendeteksi otomatis, atau isi `COM3`, `COM4`, `/dev/ttyUSB0`) |
| `BAUD_RATE` | `115200` | Kecepatan baud komunikasi serial |
| `CAMERA_SOURCE` | `0` | `0` untuk webcam laptop, `1` untuk webcam USB kedua, atau URL stream IP Cam |
| `CONF_THRESHOLD`| `0.55` | Ambang batas akurasi deteksi sampah (0.1 - 1.0) |
| `ACTION_COOLDOWN`| `3.0` | Jeda waktu (detik) antar gerakan servo |
| `MODEL_PATH` | `yolo11n.pt` | Path model YOLO (akan diunduh otomatis pada run pertama) |
| `SHOW_GUI` | `true` | Menampilkan jendela preview kamera |

---

### 3. Upload Firmware ke ESP32
1. Colok ESP32 ke laptop menggunakan kabel USB data.
2. Jalankan upload menggunakan salah satu cara:
   * **VS Code:** Buka tab PlatformIO lalu klik ikon Upload (`→`), **atau**
   * **Terminal (PowerShell):** `.\upload.ps1`, **atau**
   * **Terminal (CMD):** `upload.bat`, **atau**
   * **PlatformIO CLI:** `pio run -t upload`

---

### 4. Jalankan Deteksi AI
```bash
python detect_trash.py
```
* Sistem akan otomatis menghubungkan port serial ESP32 (jika dicolok) atau berjalan dalam **Mode Simulasi** jika ESP32 belum terhubung.
* Tekan tombol **`q`** pada jendela kamera untuk keluar.

---

### 5. Training Model YOLO di ZimaOS / Server Docker

Dataset `dataset_raw/` sudah tersedia langsung di dalam repositori. Cukup clone repositori ini ke server ZimaOS lalu jalankan:

#### Menggunakan Docker Compose (Direkomendasikan)


1. **GPU AMD RX Vega 56 / 64 (ROCm 6.1 Compatibility):**
   ```bash
   docker compose run --rm train-gpu-vega
   ```
2. **GPU AMD Modern (RDNA 2/3, CDNA via ROCm 10):**
   ```bash
   docker compose run --rm train-gpu
   ```
3. **CPU Fallback (Paling Stabil & Langsung Jalan):**
   ```bash
   docker compose run --rm train-cpu
   ```

Hasil training akan tersimpan di folder `runs/detect/train/weights/best.pt`.
Salin bobot tersebut ke root direktori dan atur `MODEL_PATH=best.pt` pada `.env`.

