import os
import sys
import time
import cv2
from dotenv import load_dotenv

load_dotenv()

def probe_available_cameras(max_search=4):
    available = []
    for idx in range(max_search):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available

def open_camera(source):
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)

raw_cam = os.getenv("CAMERA_SOURCE", "0").strip()
if raw_cam.upper() in ("SELECT", "CHOOSE", "MENU", "AUTO"):
    print("[CAMERA] Memindai kamera yang terhubung...")
    cams = probe_available_cameras()
    if cams:
        print(f"Perangkat kamera ditemukan: {cams}")
        choice = input(f"Pilih nomor kamera {cams} [default {cams[0]}]: ").strip()
        CAMERA_SOURCE = int(choice) if choice.isdigit() else cams[0]
    else:
        CAMERA_SOURCE = 0
else:
    CAMERA_SOURCE = int(raw_cam) if raw_cam.isdigit() else raw_cam

FLIP_HORIZONTAL = os.getenv("FLIP_HORIZONTAL", "true").lower() in ("true", "1", "yes")

OUTPUT_DIR = "dataset_raw"
CATEGORIES = {
    "1": "paper",
    "2": "plastic",
    "3": "organic",
    "4": "b3",
}

for cat in CATEGORIES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

cap = open_camera(CAMERA_SOURCE)
if not cap or not cap.isOpened():
    print(f"[ERROR] Gagal membuka kamera {CAMERA_SOURCE}")
    exit(1)

print("Kamera pengumpul dataset aktif.")
print("Tekan angka untuk simpan foto ke kategori:")
for key, cat in CATEGORIES.items():
    print(f" - Tombol '{key}' : Kategori '{cat}'")
print(" - Tombol 'q' : Selesai dan keluar\n")

count_saved = {cat: 0 for cat in CATEGORIES.values()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)

    display_frame = frame.copy()
    h, w = frame.shape[:2]

    # Area fokus drop zone
    rx1, ry1 = int(w * 0.15), int(h * 0.15)
    rx2, ry2 = int(w * 0.85), int(h * 0.90)
    cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)

    info_text = " | ".join([f"{k}:{cat}({count_saved[cat]})" for k, cat in CATEGORIES.items()])
    cv2.rectangle(display_frame, (10, 10), (w - 10, 45), (30, 30, 30), -1)
    cv2.putText(display_frame, info_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imshow("IoTong Dataset Collector", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in CATEGORIES:
        selected_cat = CATEGORIES[chr(key)]
        timestamp = int(time.time() * 1000)
        filename = f"{selected_cat}_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_DIR, selected_cat, filename)

        # Simpan frame asli tanpa overlay teks
        cv2.imwrite(filepath, frame)
        count_saved[selected_cat] += 1
        print(f"[SAVED] {filepath} (Total: {count_saved[selected_cat]})")

cap.release()
cv2.destroyAllWindows()
print("\nPengumpulan data selesai. Foto tersimpan di folder 'dataset_raw/'.")
