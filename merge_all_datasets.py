import os
import shutil
import argparse

try:
    from ultralytics.data.utils import check_det_dataset
except ImportError:
    check_det_dataset = None

def merge_all_datasets(output_dir="dataset_raw/yolo_waste_master"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ds_merged_43 = os.path.join(base_dir, "dataset_raw", "yolo_waste_merged")
    ds_garbage_11 = os.path.join(base_dir, "dataset_raw", "garbage_v1i")
    target_dir = os.path.abspath(output_dir)

    print("=" * 65)
    print("  IoTong Master Dataset Merger (43-Class + 11-Class)")
    print("=" * 65)
    print(f"Dataset A (43 classes) : {ds_merged_43}")
    print(f"Dataset B (11 classes) : {ds_garbage_11}")
    print(f"Output Master Dataset  : {target_dir}")
    print("-" * 65)

    if not os.path.exists(ds_merged_43):
        print(f"[ERROR] Dataset A tidak ditemukan di: {ds_merged_43}")
        print("Pastikan folder dataset_raw/yolo_waste_merged sudah diekstrak.")
        return

    if not os.path.exists(ds_garbage_11):
        print(f"[ERROR] Dataset B tidak ditemukan di: {ds_garbage_11}")
        print("Jalankan 'python download_ndjson_dataset.py' terlebih dahulu untuk mendownload dataset B.")
        return

    # Siapkan direktori target
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

    # Definisi 46 Master Classes
    master_classes = [
        # 0 - 42: Kelas eksisting dari yolo_waste_merged (43 kelas)
        'Aerosols', 'Aluminum can', 'Aluminum caps', 'Cardboard', 'Cellulose', 'Ceramic',
        'Combined plastic', 'Container for household chemicals', 'Disposable tableware',
        'Electronics', 'Foil', 'Furniture', 'Glass bottle', 'Iron utensils', 'Liquid',
        'Metal shavings', 'Milk bottle', 'Organic', 'Paper', 'Paper bag', 'Paper cups',
        'Paper shavings', 'Papier mache', 'Plastic bag', 'Plastic bottle', 'Plastic can',
        'Plastic canister', 'Plastic caps', 'Plastic cup', 'Plastic shaker', 'Plastic shavings',
        'Plastic toys', 'Postal packaging', 'Printing industry', 'Scrap metal', 'Stretch film',
        'Tetra pack', 'Textile', 'Tin', 'Unknown plastic', 'Wood', 'Zip plastic bag', 'Trash',
        # 43 - 45: Kelas baru dari garbage_v1i
        'Battery', 'Medical waste', 'Shoes'
    ]

    # Pemetaan Kelas dari garbage_v1i (11 kelas) ke 46 Master Classes:
    # 0: battery       -> 43 (Battery)
    # 1: biological    -> 17 (Organic)
    # 2: cardboard     -> 3  (Cardboard)
    # 3: clothes       -> 37 (Textile)
    # 4: electronics   -> 9  (Electronics)
    # 5: glass         -> 12 (Glass bottle)
    # 6: medical-waste -> 44 (Medical waste)
    # 7: metal         -> 34 (Scrap metal)
    # 8: paper         -> 18 (Paper)
    # 9: plastic       -> 39 (Unknown plastic)
    # 10: shoes        -> 45 (Shoes)
    GARBAGE_REMAP = {
        0: 43,  # battery -> Battery
        1: 17,  # biological -> Organic
        2: 3,   # cardboard -> Cardboard
        3: 37,  # clothes -> Textile
        4: 9,   # electronics -> Electronics
        5: 12,  # glass -> Glass bottle
        6: 44,  # medical-waste -> Medical waste
        7: 34,  # metal -> Scrap metal
        8: 18,  # paper -> Paper
        9: 39,  # plastic -> Unknown plastic
        10: 45  # shoes -> Shoes
    }

    # 1. Salin Dataset A (yolo_waste_merged: 15.631 gambar)
    print("\n[1/3] Menyalin Dataset A (yolo_waste_merged: ~15.631 gambar)...")
    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(ds_merged_43, split, "images")
        src_lbl_dir = os.path.join(ds_merged_43, split, "labels")
        dst_img_dir = os.path.join(target_dir, split, "images")
        dst_lbl_dir = os.path.join(target_dir, split, "labels")

        if not os.path.exists(src_img_dir):
            continue

        files = os.listdir(src_img_dir)
        print(f"      - {split}: menyalin {len(files):,} gambar...")
        for f in files:
            src_f = os.path.join(src_img_dir, f)
            dst_f = os.path.join(dst_img_dir, f)
            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)  # Hardlink instan (0 storage ekstra)
                except Exception:
                    shutil.copy2(src_f, dst_f)

            lbl_f = os.path.splitext(f)[0] + ".txt"
            src_lbl = os.path.join(src_lbl_dir, lbl_f)
            dst_lbl = os.path.join(dst_lbl_dir, lbl_f)
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                try:
                    os.link(src_lbl, dst_lbl)
                except Exception:
                    shutil.copy2(src_lbl, dst_lbl)

    # 2. Salin dan Remap Dataset B (garbage_v1i: 36.020 gambar)
    print("\n[2/3] Menyalin dan mere-mapping Dataset B (garbage_v1i: ~36.020 gambar)...")
    split_map = {"train": "train", "val": "valid", "test": "test"}
    for b_split, target_split in split_map.items():
        src_img_dir = os.path.join(ds_garbage_11, b_split, "images")
        src_lbl_dir = os.path.join(ds_garbage_11, b_split, "labels")
        dst_img_dir = os.path.join(target_dir, target_split, "images")
        dst_lbl_dir = os.path.join(target_dir, target_split, "labels")

        if not os.path.exists(src_img_dir):
            continue

        files = os.listdir(src_img_dir)
        print(f"      - {b_split} -> {target_split}: memproses {len(files):,} gambar...")
        for f in files:
            unique_img_name = f"garb_{f}"
            src_f = os.path.join(src_img_dir, f)
            dst_f = os.path.join(dst_img_dir, unique_img_name)

            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)
                except Exception:
                    shutil.copy2(src_f, dst_f)

            # Remap labels
            lbl_f = os.path.splitext(f)[0] + ".txt"
            src_lbl = os.path.join(src_lbl_dir, lbl_f)
            dst_lbl = os.path.join(dst_lbl_dir, f"garb_{lbl_f}")

            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                with open(src_lbl, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        old_id = int(parts[0])
                        new_id = GARBAGE_REMAP.get(old_id, 42)
                        new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")

                with open(dst_lbl, "w", encoding="utf-8") as out_lf:
                    out_lf.writelines(new_lines)

    # 3. Buat data.yaml
    print("\n[3/3] Membuat data.yaml untuk master dataset...")
    yaml_content = f"""# IoTong Master Waste Dataset (Merged 46 Classes: TrashNet + Roboflow 42 + Garbage 11)
path: {os.path.relpath(target_dir, os.getcwd()).replace(os.sep, '/')}
train: train/images
val: valid/images
test: test/images

nc: {len(master_classes)}
names: {master_classes}
"""
    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"      data.yaml berhasil dibuat: {yaml_path}")
    if check_det_dataset:
        print("\nMelakukan verifikasi dataset...")
        try:
            check_det_dataset(yaml_path)
        except Exception as e:
            print(f"[WARN] Verifikasi: {e}")

    train_c = len(os.listdir(os.path.join(target_dir, "train", "images")))
    val_c = len(os.listdir(os.path.join(target_dir, "valid", "images")))
    test_c = len(os.listdir(os.path.join(target_dir, "test", "images")))
    total_c = train_c + val_c + test_c

    print("=" * 65)
    print("  SUCCESS! MASTER DATASET BERHASIL DIGABUNGKAN!")
    print(f"  Total Kelas  : {len(master_classes)} kelas")
    print(f"  Train Images : {train_c:,} gambar")
    print(f"  Val Images   : {val_c:,} gambar")
    print(f"  Test Images  : {test_c:,} gambar")
    print(f"  TOTAL GAMBAR : {total_c:,} GAMBAR!")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all waste datasets into unified master dataset.")
    parser.add_argument("--output", default="dataset_raw/yolo_waste_master", help="Output directory")
    args = parser.parse_args()
    merge_all_datasets(args.output)
