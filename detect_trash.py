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

SERIAL_PORT = os.getenv("SERIAL_PORT", "AUTO").strip()
BAUD_RATE = int(os.getenv("BAUD_RATE", "115200"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
SERVO_TRIGGER_CONF = float(os.getenv("SERVO_TRIGGER_CONF", "0.80"))
ACTION_COOLDOWN = float(os.getenv("ACTION_COOLDOWN", "3.0"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt").strip()
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
SHOW_GUI = os.getenv("SHOW_GUI", "true").lower() in ("true", "1", "yes")

_raw_cam = os.getenv("CAMERA_SOURCE", "0").strip()
CAMERA_SOURCE = int(_raw_cam) if _raw_cam.isdigit() else _raw_cam
FLIP_HORIZONTAL = os.getenv("FLIP_HORIZONTAL", "true").lower() in ("true", "1", "yes")
FILTER_PERSON = os.getenv("FILTER_PERSON", "true").lower() in ("true", "1", "yes")

MIN_BOX_AREA_RATIO = float(os.getenv("MIN_BOX_AREA_RATIO", "0.015"))
MAX_BOX_AREA_RATIO = float(os.getenv("MAX_BOX_AREA_RATIO", "0.35"))
USE_ROI = os.getenv("USE_ROI", "true").lower() in ("true", "1", "yes")
STABLE_FRAMES_REQUIRED = int(os.getenv("STABLE_FRAMES_REQUIRED", "3"))

# Mapping class model ke: (nama_sampah, jenis_kategori, sudut_servo, bgr_color)
TRASH_MAP = {
    # Model sampah khusus
    "paper":        ("Kertas", "Anorganik", 180, (0, 215, 255)),
    "cardboard":    ("Kardus", "Anorganik", 180, (0, 165, 255)),
    "plastic":      ("Plastik", "Anorganik", 90, (255, 191, 0)),
    "organic":      ("Sisa Makanan", "Organik", 0, (0, 255, 0)),
    "metal":        ("Logam / Kaleng", "Anorganik", 90, (200, 200, 200)),
    "glass":        ("Kaca", "Anorganik", 90, (255, 144, 30)),
    "bulky":        ("Sampah Campuran", "Anorganik", 90, (180, 105, 255)),
    "trash":        ("Sampah Umum", "Anorganik", 90, (180, 105, 255)),

    # Fallback model COCO
    "banana":       ("Pisang", "Organik", 0, (0, 255, 0)),
    "apple":        ("Apel", "Organik", 0, (0, 255, 0)),
    "orange":       ("Jeruk", "Organik", 0, (0, 255, 0)),
    "broccoli":     ("Sayuran", "Organik", 0, (0, 255, 0)),
    "carrot":       ("Wortel", "Organik", 0, (0, 255, 0)),
    "sandwich":     ("Makanan", "Organik", 0, (0, 255, 0)),
    "pizza":        ("Makanan", "Organik", 0, (0, 255, 0)),
    "donut":        ("Makanan", "Organik", 0, (0, 255, 0)),
    "bottle":       ("Botol Plastik", "Anorganik", 90, (255, 191, 0)),
    "cup":          ("Gelas Plastik", "Anorganik", 90, (255, 191, 0)),
    "book":         ("Buku / Kertas", "Anorganik", 180, (0, 215, 255)),
    "fork":         ("Garpu Logam", "Anorganik", 90, (200, 200, 200)),
    "knife":        ("Pisau Logam", "Anorganik", 90, (200, 200, 200)),
    "spoon":        ("Sendok Logam", "Anorganik", 90, (200, 200, 200)),
    "cell phone":   ("Elektronik / HP", "B3", 180, (0, 0, 255)),
}

def find_esp32_port():
    """Mencari port serial ESP32 / USB UART secara otomatis."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    # Cek chip USB UART yang biasa digunakan pada modul ESP32
    keywords = ["cp210", "ch340", "ch341", "ftdi", "silicon labs", "usb-serial", "esp32", "espressif"]
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        if any(k in desc or k in mfg for k in keywords):
            return p.device

    # Fallback saat hanya ada satu serial port terhubung
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
        time.sleep(2)  # Delay inisialisasi boot serial ESP32
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
    print(f"[ENV] Serial: {SERIAL_PORT} ({BAUD_RATE} baud) | Cam: {CAMERA_SOURCE} ({FRAME_WIDTH}x{FRAME_HEIGHT})")
    print(f"[ENV] Model: {MODEL_PATH} | Deteksi: {CONF_THRESHOLD*100:.0f}% | Trigger Servo: {SERVO_TRIGGER_CONF*100:.0f}%")

    ser = init_serial(SERIAL_PORT, BAUD_RATE)

    print(f"[YOLO] Memuat model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Model siap digunakan!")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Kamera '{CAMERA_SOURCE}' tidak dapat dibuka!")
        print("Tip: Cek koneksi webcam atau sesuaikan CAMERA_SOURCE di file .env")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    total_area = FRAME_WIDTH * FRAME_HEIGHT
    roi_x1 = int(FRAME_WIDTH * 0.15)
    roi_y1 = int(FRAME_HEIGHT * 0.15)
    roi_x2 = int(FRAME_WIDTH * 0.85)
    roi_y2 = int(FRAME_HEIGHT * 0.90)

    last_action_time = 0
    current_status = "Standby (Menunggu Sampah)"
    is_flipped = FLIP_HORIZONTAL

    candidate_name = None
    candidate_type = None
    candidate_angle = None
    candidate_count = 0

    if SHOW_GUI:
        print("\nKontrol Keyboard:")
        print(" - 'm' : Toggle flip horizontal kamera")
        print(" - 'q' : Keluar dari program\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Gagal membaca frame kamera.")
                time.sleep(0.1)
                continue

            if is_flipped:
                frame = cv2.flip(frame, 1)

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = frame.copy() if SHOW_GUI else None

            if SHOW_GUI and USE_ROI:
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (70, 70, 70), 1)
                cv2.putText(annotated_frame, "[ DROP ZONE / AREA SAMPAH ]", (roi_x1 + 10, roi_y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

            best_conf = 0.0
            best_name = None
            best_type = None
            best_angle = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    box_w = max(0, x2 - x1)
                    box_h = max(0, y2 - y1)
                    box_area = box_w * box_h
                    area_ratio = box_area / total_area

                    # Abaikan bounding box terlalu besar untuk menyaring badan dan pakaian manusia
                    if area_ratio > MAX_BOX_AREA_RATIO:
                        if SHOW_GUI and not (FILTER_PERSON and cls_name == "person"):
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
                            cv2.putText(annotated_frame, f"Abaikan ({cls_name} > max)", (x1, max(y1 - 5, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
                        continue

                    if area_ratio < MIN_BOX_AREA_RATIO:
                        continue

                    # Centroid harus berada di dalam area pembuangan (drop zone)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if USE_ROI and not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                        continue

                    if cls_name in TRASH_MAP:
                        nama_sampah, jenis_sampah, sudut_servo, color = TRASH_MAP[cls_name]

                        if conf > best_conf:
                            best_conf = conf
                            best_name = nama_sampah
                            best_type = jenis_sampah
                            best_angle = sudut_servo

                        if SHOW_GUI:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{nama_sampah} [{jenis_sampah}]: {conf*100:.1f}%"
                            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    elif SHOW_GUI:
                        if FILTER_PERSON and cls_name == "person":
                            continue

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                        label = f"{cls_name}: {conf*100:.1f}%"
                        cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # Eksekusi servo hanya saat confidence mencapai target minimal (misal >= 80%)
            now = time.time()
            if best_conf >= SERVO_TRIGGER_CONF:
                if best_name == candidate_name:
                    candidate_count += 1
                else:
                    candidate_name = best_name
                    candidate_type = best_type
                    candidate_angle = best_angle
                    candidate_count = 1

                if (candidate_count >= STABLE_FRAMES_REQUIRED) and (now - last_action_time > ACTION_COOLDOWN):
                    last_action_time = now
                    current_status = f"{candidate_name} [{candidate_type}] {best_conf*100:.0f}% -> Servo {candidate_angle} deg"
                    send_servo_command(ser, candidate_angle)
                    candidate_count = 0
            elif best_name is not None:
                candidate_count = 0
                current_status = f"{best_name} [{best_type}] {best_conf*100:.0f}% (menunggu >= {int(SERVO_TRIGGER_CONF*100)}%)"
            else:
                candidate_count = max(0, candidate_count - 1)

            if SHOW_GUI:
                cv2.rectangle(annotated_frame, (10, 10), (630, 80), (20, 20, 20), -1)
                cv2.putText(annotated_frame, f"Status: {current_status}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                flip_txt = "AKTIF" if is_flipped else "NONAKTIF"
                trigger_pct = int(SERVO_TRIGGER_CONF * 100)
                cv2.putText(annotated_frame, f"Model: {os.path.basename(MODEL_PATH)} | Trigger: >={trigger_pct}% | Flip [M]: {flip_txt}", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

                cv2.imshow("IoTong AI Detection", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('m'), ord('M')):
                    is_flipped = not is_flipped
                    print(f"[CAMERA] Flip horizontal diubah: {'AKTIF' if is_flipped else 'NONAKTIF'}")
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
