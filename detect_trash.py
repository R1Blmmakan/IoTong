import os
import sys
import time
import argparse
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
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
SERVO_TRIGGER_CONF = float(os.getenv("SERVO_TRIGGER_CONF", "0.60"))
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

# Pemetaan sudut servo pemutar wadah (Servo 2) per kategori sampah
# Posisi netral / standby diatur ke 0 derajat
ANGLE_ORGANIK = int(os.getenv("ANGLE_ORGANIK", "45"))
ANGLE_ANORGANIK = int(os.getenv("ANGLE_ANORGANIK", "90"))
ANGLE_KERTAS = int(os.getenv("ANGLE_KERTAS", "135"))
ANGLE_B3 = int(os.getenv("ANGLE_B3", "180"))

# 1. Pemetaan Eksplisit (Exact Class Match dari berbagai dataset YOLO)
EXPLICIT_TRASH_MAP = {
    # --- E-Waste & Kabel (B3 / Elektronik) ---
    "cable":            ("Kabel Listrik / USB", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "cables":           ("Kabel Listrik / USB", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "wire":             ("Kabel / Kawat", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "wires":            ("Kabel / Kawat", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "cable charger":    ("Kabel Charger", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "patch cord":       ("Kabel Jaringan LAN", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "charger":          ("Charger / Adaptor", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "battery":          ("Baterai Bekas", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "batteries":        ("Baterai Bekas", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "e-waste":          ("Sampah Elektronik", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "electronics":      ("Komponen Elektronik", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "electronic":       ("Komponen Elektronik", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "pcb":              ("Papan Sirkuit PCB", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "circuit board":    ("Papan Sirkuit PCB", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "cell phone":       ("Handphone / Gadget", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "phone":            ("Handphone / Gadget", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "mouse":            ("Mouse Komputer", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "keyboard":         ("Keyboard Komputer", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "hazardous waste":  ("Limbah Berbahaya (B3)", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "medical-waste":    ("Limbah Medis / Masker", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),
    "aerosols":         ("Kaleng Semprot / Aerosol", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)),

    # --- Logam, Stainless, & Kaleng (Anorganik Logam) ---
    "stainless":        ("Stainless Steel", "Anorganik (Logam)", ANGLE_ANORGANIK, (220, 220, 220)),
    "stainless steel":  ("Stainless Steel", "Anorganik (Logam)", ANGLE_ANORGANIK, (220, 220, 220)),
    "metal":            ("Logam / Besi", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "scrap metal":      ("Besi / Logam Bekas", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "metal shavings":   ("Serpihan Logam", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "aluminum can":     ("Kaleng Aluminium", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "aluminum caps":    ("Tutup Aluminium", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "tin":              ("Kaleng Timah", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "iron utensils":    ("Peralatan Besi", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "foil":             ("Aluminium Foil", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "fork":             ("Garpu Logam", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "knife":            ("Pisau Logam", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),
    "spoon":            ("Sendok Logam", "Anorganik (Logam)", ANGLE_ANORGANIK, (200, 200, 200)),

    # --- Plastik, Kaca, & Anorganik Umum ---
    "plastic":          ("Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "plastic bottle":   ("Botol Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "bottle":           ("Botol Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "cup":              ("Gelas Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "plastic cup":      ("Gelas Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "plastic bag":      ("Kantong Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "plastic caps":     ("Tutup Botol Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "glass":            ("Kaca", "Anorganik", ANGLE_ANORGANIK, (255, 144, 30)),
    "glass bottle":     ("Botol Kaca", "Anorganik", ANGLE_ANORGANIK, (255, 144, 30)),
    "ceramic":          ("Keramik", "Anorganik", ANGLE_ANORGANIK, (255, 144, 30)),
    "recyclable":       ("Sampah Daur Ulang", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)),
    "trash":            ("Sampah Anorganik", "Anorganik", ANGLE_ANORGANIK, (180, 105, 255)),
    "clothes":          ("Kain / Tekstil", "Anorganik", ANGLE_ANORGANIK, (180, 105, 255)),
    "textile":          ("Kain / Tekstil", "Anorganik", ANGLE_ANORGANIK, (180, 105, 255)),

    # --- Kertas & Kardus ---
    "paper":            ("Kertas", "Kertas / Karton", ANGLE_KERTAS, (0, 215, 255)),
    "cardboard":        ("Kardus", "Kertas / Karton", ANGLE_KERTAS, (0, 165, 255)),
    "book":             ("Buku / Kertas", "Kertas / Karton", ANGLE_KERTAS, (0, 215, 255)),
    "paper bag":        ("Kantong Kertas", "Kertas / Karton", ANGLE_KERTAS, (0, 215, 255)),
    "paper cups":       ("Gelas Kertas", "Kertas / Karton", ANGLE_KERTAS, (0, 215, 255)),
    "tetra pack":       ("Karton Kemasan", "Kertas / Karton", ANGLE_KERTAS, (0, 165, 255)),

    # --- Organik ---
    "organic":          ("Sisa Makanan", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "biological":       ("Sampah Organik", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "food":             ("Sisa Makanan", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "banana":           ("Kulit Buah", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "apple":            ("Sisa Buah", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "orange":           ("Sisa Buah", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "sandwich":         ("Sisa Makanan", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "pizza":            ("Sisa Makanan", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "donut":            ("Sisa Makanan", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "vegetable":        ("Sayuran", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
    "wood":             ("Kayu / Ranting", "Organik", ANGLE_ORGANIK, (0, 255, 0)),
}

# 2. Aturan Kata Kunci Semantik (Dynamic Keyword Fallback)
KEYWORD_RULES = [
    # (daftar_kata_kunci, nama_display, kategori_display, sudut_servo, bgr_color)
    (
        ["cable", "wire", "cord", "charger", "usb", "patch", "lan", "lead"],
        "Kabel / Kabel Charger", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)
    ),
    (
        ["battery", "batteries", "accumulator", "cell", "powerbank"],
        "Baterai / Sel Daya", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)
    ),
    (
        ["electronic", "pcb", "circuit", "chip", "phone", "hardware", "laptop", "mouse", "keyboard", "e-waste", "display", "screen"],
        "Elektronik / E-Waste", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)
    ),
    (
        ["hazardous", "toxic", "chemical", "medical", "mask", "syringe", "aerosol"],
        "Limbah B3 Berbahaya", "B3 / Elektronik", ANGLE_B3, (0, 0, 255)
    ),
    (
        ["stainless", "steel", "metal", "iron", "aluminum", "aluminium", "tin", "can", "cans", "foil", "copper", "brass", "scrap"],
        "Stainless / Logam", "Anorganik (Logam)", ANGLE_ANORGANIK, (220, 220, 220)
    ),
    (
        ["paper", "cardboard", "carton", "book", "box", "magazine", "newspaper", "cellulose"],
        "Kertas & Karton", "Kertas / Karton", ANGLE_KERTAS, (0, 215, 255)
    ),
    (
        ["plastic", "bottle", "cup", "straw", "poly", "film", "bag", "pet", "hdpe", "pvc"],
        "Plastik", "Anorganik", ANGLE_ANORGANIK, (255, 191, 0)
    ),
    (
        ["glass", "jar", "ceramic", "porcelain"],
        "Kaca & Keramik", "Anorganik", ANGLE_ANORGANIK, (255, 144, 30)
    ),
    (
        ["organic", "food", "fruit", "vegetable", "leaf", "leaves", "wood", "biological", "bread", "meat", "rice"],
        "Sampah Organik", "Organik", ANGLE_ORGANIK, (0, 255, 0)
    ),
]

def resolve_trash_category(raw_class_name):
    """
    Menentukan (nama_sampah, jenis_kategori, sudut_servo, bgr_color)
    secara dinamis dari nama class YOLO.
    """
    clean_name = str(raw_class_name).strip().lower()

    # 1. Cek Exact Match
    if clean_name in EXPLICIT_TRASH_MAP:
        return EXPLICIT_TRASH_MAP[clean_name]

    # 2. Cek Berdasarkan Kata Kunci Semantik (Keyword Substring Match)
    for keywords, display_name, kategori, angle, color in KEYWORD_RULES:
        if any(kw in clean_name for kw in keywords):
            formatted_name = f"{display_name} ({clean_name})"
            return (formatted_name, kategori, angle, color)

    # 3. Fallback Umum: Default ke Anorganik jika tidak diketahui
    return (clean_name.capitalize(), "Anorganik (Umum)", ANGLE_ANORGANIK, (160, 160, 160))

def probe_available_cameras(max_search=4):
    available = []
    for idx in range(max_search):
        # DirectShow on Windows avoids MSMF delay on non-existent devices
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available

def open_camera(source, width=640, height=480):
    if isinstance(source, int):
        # Prefer DirectShow on Windows for reliable index binding
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def resolve_camera_source(configured_source, force_select=False):
    is_interactive = force_select or str(configured_source).strip().upper() in ("SELECT", "CHOOSE", "MENU", "AUTO")
    if is_interactive:
        print("[CAMERA] Memindai kamera yang terhubung...")
        available = probe_available_cameras()
        if not available:
            print("[CAMERA WARNING] Tidak ada kamera yang terdeteksi otomatis.")
            val = input("Masukkan index kamera (default 0): ").strip()
            return int(val) if val.isdigit() else 0

        print("\nPerangkat Kamera Terdeteksi:")
        for idx in available:
            label = "Webcam Eksternal" if idx > 0 else "Kamera Utama / Built-in"
            print(f"  [{idx}] Index {idx} ({label})")

        choice = input(f"\nPilih nomor kamera {available} [default {available[0]}]: ").strip()
        if choice.isdigit():
            return int(choice)
        return available[0]

    raw = str(configured_source).strip()
    return int(raw) if raw.isdigit() else raw

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
        ser.reset_input_buffer()
        print(f"[SERIAL] Berhasil terhubung ke ESP32 di {actual_port}!")
        return ser
    except Exception as e:
        print(f"[SERIAL WARNING] Tidak bisa membuka {actual_port}: {e}")
        print("[SERIAL] Mode Simulasi Berjalan (Hanya visualisasi webcam).")
        return None

def send_sort_command(ser, bin_angle, trash_name="", trash_type=""):
    if ser and ser.is_open:
        cmd = f"sort {bin_angle}\n"
        ser.write(cmd.encode())
        print(f">> [ESP32 SENT] {cmd.strip()} (Wadah/Servo2: {bin_angle} deg -> Trapdoor: buka 0 deg, tutup 180 deg -> wadah kembali ke 0 deg)")
    else:
        print(f">> [SIMULASI DUAL SERVO] '{trash_name}' [{trash_type}]:")
        print(f"   1. Wadah (Servo 2) memutar ke {bin_angle} deg.")
        print(f"   2. Wadah berhenti, jeda 1 detik.")
        print(f"   3. Trapdoor (Servo 1) membuka ke 0 deg (putar 180 deg).")
        print(f"   4. Trapdoor (Servo 1) menutup kembali ke 180 deg.")
        print(f"   5. Wadah (Servo 2) kembali ke posisi netral (0 deg).")

def send_servo_command(ser, angle):
    send_sort_command(ser, angle)

def main():
    parser = argparse.ArgumentParser(description="IoTong AI Waste Detection")
    parser.add_argument("--cam", "-c", default=None, help="Index kamera atau URL stream")
    parser.add_argument("--select-cam", action="store_true", help="Pilih kamera secara interaktif saat mulai")
    args, _ = parser.parse_known_args()

    target_cam = args.cam if args.cam is not None else CAMERA_SOURCE
    active_cam = resolve_camera_source(target_cam, force_select=args.select_cam)

    print(f"[ENV] Serial: {SERIAL_PORT} ({BAUD_RATE} baud) | Cam: {active_cam} ({FRAME_WIDTH}x{FRAME_HEIGHT})")
    print(f"[ENV] Model: {MODEL_PATH} | Deteksi: {CONF_THRESHOLD*100:.0f}% | Trigger Servo: {SERVO_TRIGGER_CONF*100:.0f}%")

    ser = init_serial(SERIAL_PORT, BAUD_RATE)

    model_to_use = MODEL_PATH
    if (model_to_use == "yolo11n.pt" or not os.path.exists(model_to_use)) and os.path.exists("best.pt"):
        print("[MODEL AUTO] File bobot custom 'best.pt' terdeteksi di direktori! Menggunakan 'best.pt'.")
        model_to_use = "best.pt"

    print(f"[YOLO] Memuat model {model_to_use}...")
    model = YOLO(model_to_use)
    print("[YOLO] Model siap digunakan!")

    cap = open_camera(active_cam, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap or not cap.isOpened():
        print(f"[ERROR] Kamera '{active_cam}' tidak dapat dibuka!")
        print("[CAMERA] Mencari kamera alternatif yang terhubung...")
        available = probe_available_cameras()
        fallback_found = False
        for alt in available:
            if alt != active_cam:
                cap = open_camera(alt, FRAME_WIDTH, FRAME_HEIGHT)
                if cap and cap.isOpened():
                    print(f"[CAMERA] Beralih otomatis ke kamera alternatif: Index {alt}")
                    active_cam = alt
                    fallback_found = True
                    break
        if not fallback_found:
            print("Tip: Cek koneksi USB webcam atau pastikan webcam tidak sedang dipakai aplikasi lain.")
            return

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
        print(" - 'c' : Ganti ke kamera berikutnya (Webcam <-> Built-in)")
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

                    if FILTER_PERSON and cls_name == "person":
                        continue

                    nama_sampah, jenis_sampah, sudut_servo, color = resolve_trash_category(cls_name)

                    if conf > best_conf:
                        best_conf = conf
                        best_name = nama_sampah
                        best_type = jenis_sampah
                        best_angle = sudut_servo

                    if SHOW_GUI:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{nama_sampah} [{jenis_sampah}]: {conf*100:.1f}%"
                        cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            if ser and ser.in_waiting > 0:
                try:
                    esp_response = ser.readline().decode("utf-8", errors="ignore").strip()
                    if esp_response:
                        print(f"[ESP32] {esp_response}")
                except Exception:
                    pass

            # Eksekusi servo hanya saat confidence mencapai target minimal
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
                    current_status = f"{candidate_name} [{candidate_type}] {best_conf*100:.0f}% -> Wadah {candidate_angle} deg + Trapdoor (180->0 deg)"
                    send_sort_command(ser, candidate_angle, candidate_name, candidate_type)
                    candidate_count = 0
            elif best_name is not None:
                candidate_count = 0
                current_status = f"{best_name} [{best_type}] {best_conf*100:.0f}% (menunggu >= {int(SERVO_TRIGGER_CONF*100)}%)"
            else:
                candidate_count = max(0, candidate_count - 1)

            if SHOW_GUI:
                cv2.rectangle(annotated_frame, (10, 10), (630, 80), (20, 20, 20), -1)
                cv2.putText(annotated_frame, f"Status: {current_status}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
                cam_label = f"Cam [{active_cam}]"
                trigger_pct = int(SERVO_TRIGGER_CONF * 100)
                cv2.putText(annotated_frame, f"{cam_label} | Trigger >={trigger_pct}% | Wadah(S2) + Trapdoor 180->0 deg(S1) | [C] Cam | [M] Flip", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

                cv2.imshow("IoTong AI Detection", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('c'), ord('C')):
                    if isinstance(active_cam, int):
                        next_cam = 1 if active_cam == 0 else 0
                        print(f"[CAMERA] Beralih ke kamera Index {next_cam}...")
                        new_cap = open_camera(next_cam, FRAME_WIDTH, FRAME_HEIGHT)
                        if new_cap and new_cap.isOpened():
                            ret_test, _ = new_cap.read()
                            if ret_test:
                                cap.release()
                                cap = new_cap
                                active_cam = next_cam
                                print(f"[CAMERA] Aktif di kamera Index {active_cam}.")
                            else:
                                new_cap.release()
                                print(f"[CAMERA WARNING] Kamera Index {next_cam} tidak mengirim frame.")
                        else:
                            print(f"[CAMERA WARNING] Gagal membuka kamera Index {next_cam}.")
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
            ser.write(b"trapdoor close\n")
            ser.write(b"pos2 0\n")
            ser.close()
        print("[EXIT] Program deteksi selesai.")

if __name__ == "__main__":
    main()
