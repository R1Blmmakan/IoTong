import os
import shutil
from ultralytics.data.utils import check_det_dataset

def merge_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ds_old = os.path.join(base_dir, "dataset_raw", "TrashType_Image_Dataset")
    ds_new = os.path.join(base_dir, "dataset_raw", "yolo_waste_v1")
    ds_merged = os.path.join(base_dir, "dataset_raw", "yolo_waste_merged")

    if not os.path.exists(ds_old):
        raise FileNotFoundError(f"Dataset lama tidak ditemukan: {ds_old}")
    if not os.path.exists(ds_new):
        raise FileNotFoundError(f"Dataset baru tidak ditemukan: {ds_new}")

    print("[1/5] Menyiapkan direktori dataset gabungan...")
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(ds_merged, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(ds_merged, split, "labels"), exist_ok=True)

    print("[2/5] Menyalin data dari yolo_waste_v1 (13.104 gambar)...")
    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(ds_new, split, "images")
        src_lbl_dir = os.path.join(ds_new, split, "labels")
        dst_img_dir = os.path.join(ds_merged, split, "images")
        dst_lbl_dir = os.path.join(ds_merged, split, "labels")

        for f in os.listdir(src_img_dir):
            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))
        for f in os.listdir(src_lbl_dir):
            shutil.copy2(os.path.join(src_lbl_dir, f), os.path.join(dst_lbl_dir, f))

    # Pemetaan Kelas Lama (0-5) ke Kelas Baru (43 kelas)
    # 0: cardboard -> 3 (Cardboard)
    # 1: glass     -> 12 (Glass bottle)
    # 2: metal     -> 1 (Aluminum can)
    # 3: paper     -> 18 (Paper)
    # 4: plastic   -> 24 (Plastic bottle)
    # 5: trash     -> 42 (Trash)
    CLASS_REMAP = {
        0: 3,   # Cardboard
        1: 12,  # Glass bottle
        2: 1,   # Aluminum can
        3: 18,  # Paper
        4: 24,  # Plastic bottle
        5: 42   # Trash
    }

    print("[3/5] Menggabungkan dan mere-mapping TrashType_Image_Dataset (2.528 gambar)...")
    for split, txt_file in [("train", "train.txt"), ("valid", "val.txt")]:
        list_path = os.path.join(ds_old, txt_file)
        dst_img_dir = os.path.join(ds_merged, split, "images")
        dst_lbl_dir = os.path.join(ds_merged, split, "labels")

        with open(list_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f" - Memproses {len(lines)} gambar untuk {split}...")
        for rel_img in lines:
            img_src = os.path.join(ds_old, rel_img)
            lbl_src = os.path.splitext(img_src)[0] + ".txt"

            if not os.path.exists(img_src):
                continue

            base_name = os.path.basename(img_src)
            # Prefix untuk memastikan tidak ada konflik nama
            unique_name = f"trashtype_{base_name}"
            unique_lbl_name = os.path.splitext(unique_name)[0] + ".txt"

            # Salin gambar
            shutil.copy2(img_src, os.path.join(dst_img_dir, unique_name))

            # Remap label
            if os.path.exists(lbl_src):
                with open(lbl_src, "r") as lf:
                    lbl_lines = lf.readlines()

                new_lbl_lines = []
                for ll in lbl_lines:
                    parts = ll.strip().split()
                    if parts:
                        old_id = int(parts[0])
                        new_id = CLASS_REMAP.get(old_id, 42)
                        new_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                        new_lbl_lines.append(new_line)

                with open(os.path.join(dst_lbl_dir, unique_lbl_name), "w") as out_lf:
                    out_lf.writelines(new_lbl_lines)

    print("[4/5] Membuat data.yaml untuk dataset gabungan...")
    names = [
        'Aerosols', 'Aluminum can', 'Aluminum caps', 'Cardboard', 'Cellulose', 'Ceramic',
        'Combined plastic', 'Container for household chemicals', 'Disposable tableware',
        'Electronics', 'Foil', 'Furniture', 'Glass bottle', 'Iron utensils', 'Liquid',
        'Metal shavings', 'Milk bottle', 'Organic', 'Paper', 'Paper bag', 'Paper cups',
        'Paper shavings', 'Papier mache', 'Plastic bag', 'Plastic bottle', 'Plastic can',
        'Plastic canister', 'Plastic caps', 'Plastic cup', 'Plastic shaker', 'Plastic shavings',
        'Plastic toys', 'Postal packaging', 'Printing industry', 'Scrap metal', 'Stretch film',
        'Tetra pack', 'Textile', 'Tin', 'Unknown plastic', 'Wood', 'Zip plastic bag', 'Trash'
    ]

    yaml_content = f"""# IoTong - Unified Mega Waste Dataset (TrashNet + Roboflow 43-Classes)
path: dataset_raw/yolo_waste_merged
train: train/images
val: valid/images
test: test/images

nc: {len(names)}
names: {names}
"""
    yaml_path = os.path.join(ds_merged, "data.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print("[5/5] Melakukan verifikasi dataset menggunakan Ultralytics...")
    data = check_det_dataset(yaml_path)
    print("\n" + "="*50)
    print("SUCCESS! DATASET BERHASIL DIGABUNGKAN!")
    print(f"Total Kelas     : {len(data['names'])} kelas")
    print(f"Lokasi data.yaml: {yaml_path}")
    print(f"Train Images    : {len(os.listdir(os.path.join(ds_merged, 'train', 'images')))}")
    print(f"Val Images      : {len(os.listdir(os.path.join(ds_merged, 'valid', 'images')))}")
    print(f"Test Images     : {len(os.listdir(os.path.join(ds_merged, 'test', 'images')))}")
    print(f"TOTAL GAMBAR    : {len(os.listdir(os.path.join(ds_merged, 'train', 'images'))) + len(os.listdir(os.path.join(ds_merged, 'valid', 'images'))) + len(os.listdir(os.path.join(ds_merged, 'test', 'images')))}")
    print("="*50)

if __name__ == "__main__":
    merge_datasets()
