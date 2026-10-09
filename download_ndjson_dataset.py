import os
import sys
import json
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def download_file(item, out_dir, max_retries=3):
    img_filename = item["file"]
    split = item.get("split", "train")
    if split == "valid":
        split = "val"
        
    img_dest = os.path.join(out_dir, split, "images", img_filename)
    lbl_filename = os.path.splitext(img_filename)[0] + ".txt"
    lbl_dest = os.path.join(out_dir, split, "labels", lbl_filename)
    
    # 1. Simpan label anotasi YOLO
    if not os.path.exists(lbl_dest):
        lines = []
        if "annotations" in item and "boxes" in item["annotations"]:
            for box in item["annotations"]["boxes"]:
                # box format: [class_id, x_center, y_center, width, height]
                lines.append(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}")
        with open(lbl_dest, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + ("\n" if lines else ""))

    # 2. Skip jika gambar sudah selesai didownload
    if os.path.exists(img_dest) and os.path.getsize(img_dest) > 100:
        return True, img_filename

    # 3. Download gambar dari CDN dengan retry
    url = item["url"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IoTongDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 0:
                    temp_dest = img_dest + ".tmp"
                    with open(temp_dest, "wb") as f:
                        f.write(data)
                    os.replace(temp_dest, img_dest)
                    return True, img_filename
        except Exception:
            if attempt == max_retries - 1:
                return False, img_filename
    return False, img_filename


def process_ndjson(ndjson_path, out_dir, num_workers=24):
    if not os.path.exists(ndjson_path):
        print(f"[ERROR] File NDJSON tidak ditemukan: {ndjson_path}")
        sys.exit(1)

    out_dir = os.path.abspath(out_dir)
    print("=" * 60)
    print("  IoTong Ultralytics NDJSON Dataset Downloader & Converter")
    print("=" * 60)
    print(f"Sumber NDJSON : {ndjson_path}")
    print(f"Target Folder : {out_dir}")
    print(f"Worker Thread : {num_workers}")
    print("-" * 60)

    # Siapkan struktur direktori YOLO
    for sp in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, sp, "labels"), exist_ok=True)

    items = []
    header = None

    print("[1/3] Membaca metadata NDJSON...")
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if idx == 0 and data.get("type") == "dataset":
                header = data
            elif data.get("type") == "image":
                items.append(data)

    total_images = len(items)
    print(f"      Total gambar terdaftar: {total_images:,}")

    # Generate data.yaml dari header
    if header and "class_names" in header:
        class_names = header["class_names"]
        # Convert string keys to int sorted
        sorted_keys = sorted([int(k) for k in class_names.keys()])
        yaml_lines = [
            f"# Ultralytics Dataset: {header.get('name', 'Garbage Dataset')}",
            f"path: {os.path.relpath(out_dir, os.getcwd()).replace(os.sep, '/')}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "",
            f"nc: {len(sorted_keys)}",
            "names:"
        ]
        for k in sorted_keys:
            yaml_lines.append(f"  {k}: {class_names[str(k)]}")
            
        yaml_path = os.path.join(out_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write("\n".join(yaml_lines) + "\n")
        print(f"      data.yaml berhasil dibuat: {yaml_path}")
        print(f"      Daftar kelas ({len(sorted_keys)}): {[class_names[str(k)] for k in sorted_keys]}")

    print(f"\n[2/3] Memulai download paralel ({num_workers} threads)...")
    print("      (Jika koneksi terputus, skrip bisa dijalankan ulang dan otomatis resume)")
    
    success_count = 0
    fail_count = 0
    failed_files = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_images, unit="img", ncols=90)
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_file, it, out_dir): it["file"] for it in items}
        for future in as_completed(futures):
            ok, fname = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(fname)
            
            if pbar:
                pbar.update(1)
            elif (success_count + fail_count) % 500 == 0:
                print(f"      Progres: {success_count + fail_count:,}/{total_images:,} ({((success_count + fail_count)/total_images)*100:.1f}%)")

    if pbar:
        pbar.close()

    print("\n[3/3] Selesai!")
    print(f"      Berhasil: {success_count:,} gambar")
    if fail_count > 0:
        print(f"      Gagal   : {fail_count:,} gambar")
        fail_log = os.path.join(out_dir, "failed_downloads.txt")
        with open(fail_log, "w", encoding="utf-8") as ff:
            ff.write("\n".join(failed_files) + "\n")
        print(f"      Daftar gagal disimpan di: {fail_log}")
    print("=" * 60)
    print("Dataset siap digunakan untuk training YOLO!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert Ultralytics NDJSON dataset to YOLO format.")
    parser.add_argument("--ndjson", default="garbagev1iyolov11.ndjson", help="Path ke file .ndjson")
    parser.add_argument("--output", default="dataset_raw/garbage_v1i", help="Path direktori output dataset")
    parser.add_argument("--workers", type=int, default=24, help="Jumlah thread paralel (default: 24)")
    args = parser.parse_args()

    process_ndjson(args.ndjson, args.output, args.workers)
