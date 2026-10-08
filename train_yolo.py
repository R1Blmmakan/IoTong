import os
import sys

# PyTorch ROCm 6.1 (PyTorch 2.1.2) dikompilasi dengan C-API NumPy 1.x.
# NumPy 2.x menyebabkan 'RuntimeError: Numpy is not available' saat tensor_numpy dipanggil.
try:
    import numpy as np
    if np.__version__.startswith("2."):
        print(f"[HOTFIX] NumPy {np.__version__} terdeteksi! ROCm PyTorch 2.1.2 membutuhkan NumPy < 2.0.0.")
        print("[HOTFIX] Mendowngrade NumPy ke 1.26.4 secara otomatis di dalam container...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy<2.0.0"])
        print("[HOTFIX] NumPy berhasil didowngrade. Merestart proses training...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception as e:
    print(f"[WARN] Pengecekan NumPy hotfix: {e}")

# Konfigurasi stabilitas AMD ROCm untuk arsitektur Vega (gfx900)
# 1. HSA_ENABLE_SDMA=0: Matikan SDMA hardware engine yang sering race condition / desync PCIe
# 2. MIOPEN_FIND_MODE=2: Gunakan mode Fast FindDb untuk mencegah benchmarking crash pada kernel konvolusi
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")

# Cegah Ultralytics mencoba download/update package mendadak saat runtime
os.environ["YOLO_AUTOINSTALL"] = "False"

import torch
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from ultralytics import YOLO

# Prioritaskan dataset gabungan (43 kelas) jika ada, fallback ke TrashType
default_dataset = "dataset_raw/yolo_waste_merged/data.yaml"
if not os.path.exists(default_dataset):
    default_dataset = "dataset_raw/TrashType_Image_Dataset/data.yaml"

data_cfg = os.getenv("DATASET_CONFIG", default_dataset)
dataset_config = os.path.abspath(data_cfg)

if not os.path.exists(dataset_config):
    print(f"[ERROR] Dataset configuration file not found: {dataset_config}")
    exit(1)

model_base = os.getenv("MODEL_BASE", "yolo11n.pt")
epochs = int(os.getenv("EPOCHS", "50"))
batch_size = int(os.getenv("BATCH", "16"))
img_size = int(os.getenv("IMGSZ", "640"))

default_device = "0" if torch.cuda.is_available() else "cpu"
device = os.getenv("DEVICE", default_device)
amp_enabled = os.getenv("AMP", "True").lower() not in ("false", "0", "no")

workers = int(os.getenv("WORKERS", "2"))

print(f"[TRAIN] Model base: {model_base}")
print(f"[TRAIN] Device: {device} (cuda/rocm available: {torch.cuda.is_available()})")
print(f"[TRAIN] Dataset: {dataset_config}")
print(f"[TRAIN] Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}, Workers: {workers}")
print(f"[TRAIN] AMP (Mixed Precision): {amp_enabled}")

model = YOLO(model_base)
results = model.train(
    data=dataset_config,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    workers=workers,
    device=device,
    amp=amp_enabled,
    plots=True
)

print("\nTraining completed.")
print("Weights saved in: runs/detect/train*/weights/best.pt")