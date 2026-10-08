import os
import torch
from ultralytics import YOLO

# Cegah Ultralytics mencoba download/update package mendadak saat runtime
os.environ["YOLO_AUTOINSTALL"] = "False"

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

print(f"[TRAIN] Model base: {model_base}")
print(f"[TRAIN] Device: {device} (cuda/rocm available: {torch.cuda.is_available()})")
print(f"[TRAIN] Dataset: {dataset_config}")
print(f"[TRAIN] Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
print(f"[TRAIN] AMP (Mixed Precision): {amp_enabled}")

model = YOLO(model_base)
results = model.train(
    data=dataset_config,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    amp=amp_enabled,
    plots=True
)

print("\nTraining completed.")
print("Weights saved in: runs/detect/train*/weights/best.pt")