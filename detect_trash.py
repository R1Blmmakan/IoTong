import os
import sys
import time
import cv2
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# Muat variabel environment dari file .env jika tersedia
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# KONFIGURASI IOTONG (DIBACA DARI ENVIRONMENT / .env)
# ==============================================================================
SERIAL_PORT = os.getenv("SERIAL_PORT", "AUTO").strip()
BAUD_RATE = int(os.getenv("BAUD_RATE", "115200"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
ACTION_COOLDOWN = float(os.getenv("ACTION_COOLDOWN", "3.0"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt").strip()
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
SHOW_GUI = os.getenv("SHOW_GUI", "true").lower() in ("true", "1", "yes")

# Parsing Sumber Kamera: int jika angka (webcam lokal), str jika URL RTSP/HTTP
_raw_cam = os.getenv("CAMERA_SOURCE", "0").strip()
CAMERA_SOURCE = int(_raw_cam) if _raw_cam.isdigit() else _raw_cam

# Mapping Kategori Sampah (COCO Dataset Default) ke Sudut Servo:
# Sudut: 0 = Organik, 90 = Plastik, 180 = Kertas / Anorganik Lainnya
TRASH_MAP = {
    # Organik -> 0 Derajat
    "banana":     ("Organik", 0),
    "apple":      ("Organik", 0),
    "orange":     ("Organik", 0),
    "broccoli":   ("Organik", 0),
    "carrot":     ("Organik", 0),
    "sandwich":   ("Organik", 0),
    "pizza":      ("Organik", 0),
    "donut":      ("Organik", 0),

    # Plastik -> 90 Derajat
    "bottle":     ("Plastik", 90),
    "cup":        ("Plastik", 90),

    # Kertas / Logam / Anorganik -> 180 Derajat
    "book":       ("Kertas", 180),
    "fork":       ("Logam/Anorganik", 180),
    "knife":      ("Logam/Anorganik", 180),
    "spoon":      ("Logam/Anorganik", 180),
    "cell phone": ("Elektronik", 180),
}

def find_esp32_port():
    """Mencari port serial ESP32 / USB UART secara otomatis."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    # Prioritas 1: Port dengan identifier chip UART ESP32 umum
    keywords = ["cp210", "ch340", "ch341", "ftdi", "silicon labs", "usb-serial", "esp32", "espressif"]
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        if any(k in desc or k in mfg for k in keywords):
            return p.device

    # Prioritas 2: Jika hanya ada 1 port serial yang terhubung, gunakan port tersebut
    if len(ports) == 1:
        return ports[0].device

    return None

def init_serial(port_name, baud):
    actual_port = port_name
    if not port_name or port_name.upper() == "AUTO":
        detected = find_esp32_port()
        if detected:
            print(f"[SERIAL AUTO] Port terdeteksi otomatis: {detected}")
            actual_port = detected
        else:
            print("[SERIAL WARNING] Tidak ada port serial/ESP32 yang terdeteksi.")
            print("[SERIAL] Mode Simulasi Berjalan (Hanya visualisasi webcam).")
            return None

    try:
        ser = serial.Serial(actual_port, baud, timeout=1)
        time.sleep(2)  # Tunggu ESP32 siap setelah reboot serial
        print(f"[SERIAL] Berhasil terhubung ke ESP32 di {actual_port}!")
        return ser
    except Exception as e:
        print(f"[SERIAL WARNING] Tidak bisa membuka {actual_port}: {e}")
        print("[SERIAL] Mode Simulasi Berjalan (Hanya visualisasi webcam).")
        return None

def send_servo_command(ser, angle):
    if ser and ser.is_open:
        cmd = f"pos {angle}\n"
        ser.write(cmd.encode())
        print(f">> [ESP32 SENT] Mengirim perintah: {cmd.strip()}")

def main():
    print("==================================================")
    print("      IoTong: YOLO11n Real-Time Trash Detector    ")
    print("==================================================")
    print(f"[ENV] Config Serial   : Port={SERIAL_PORT}, Baud={BAUD_RATE}")
    print(f"[ENV] Config Kamera   : Source={CAMERA_SOURCE} ({FRAME_WIDTH}x{FRAME_HEIGHT})")
    print(f"[ENV] Config AI Model : Model={MODEL_PATH}, Conf={CONF_THRESHOLD*100:.0f}%, Cooldown={ACTION_COOLDOWN}s")
    print("--------------------------------------------------")

    # 1. Hubungkan Serial ke ESP32
    ser = init_serial(SERIAL_PORT, BAUD_RATE)

    # 2. Muat Model YOLO
    print(f"[YOLO] Memuat model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Model siap digunakan!")

    # 3. Buka Kamera
    print(f"[CAMERA] Membuka sumber kamera: {CAMERA_SOURCE}...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Kamera '{CAMERA_SOURCE}' tidak dapat dibuka!")
        print("Tip: Cek koneksi webcam atau sesuaikan CAMERA_SOURCE di file .env")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    last_action_time = 0
    current_status = "Standby (Menunggu Sampah)"

    if SHOW_GUI:
        print("\nTekan tombol 'q' pada jendela kamera untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Gagal membaca frame kamera.")
                time.sleep(0.1)
                continue

            # Inferensi YOLO
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = frame.copy() if SHOW_GUI else None

            detected_trash = None
            target_category = None
            target_angle = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])

                    if cls_name in TRASH_MAP:
                        category, angle = TRASH_MAP[cls_name]
                        detected_trash = cls_name
                        target_category = category
                        target_angle = angle

                        if SHOW_GUI:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{category} ({cls_name}): {conf*100:.1f}%"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif SHOW_GUI:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                        label = f"{cls_name}: {conf*100:.1f}%"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Logika Pengiriman Sinyal ke ESP32 (dengan jeda cooldown)
            now = time.time()
            if target_angle is not None and (now - last_action_time > ACTION_COOLDOWN):
                last_action_time = now
                current_status = f"Terdeteksi: {target_category} -> Servo {target_angle} deg"
                send_servo_command(ser, target_angle)

            if SHOW_GUI:
                # Tampilkan Status OSD di layar
                cv2.putText(annotated_frame, f"Status: {current_status}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("IoTong AI Detection (YOLO11n)", annotated_frame)

                # Tekan 'q' untuk keluar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Dihentikan oleh pengguna.")
    finally:
        cap.release()
        if SHOW_GUI:
            cv2.destroyAllWindows()
        if ser and ser.is_open:
            # Kembalikan ke posisi netral sebelum keluar
            send_servo_command(ser, 90)
            ser.close()
        print("[EXIT] Program deteksi selesai.")

if __name__ == "__main__":
    main()
