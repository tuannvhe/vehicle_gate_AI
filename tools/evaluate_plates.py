"""
Đo độ chính xác đọc biển trên bộ ảnh ĐÃ GÁN NHÃN - chạy ĐÚNG luồng production
(main._process_single_image + ngưỡng MIN_SUCCESS_CONFIDENCE). Dùng mỗi khi đổi
camera / đổi ngưỡng trong config.py / sửa plate_reader.py, để biết chắc thay
đổi tốt lên hay xấu đi trước khi đưa vào chạy thật.

File nhãn (JSON): { "<đường dẫn ảnh, tương đối so với --root>": <nhãn> }
  - "12B1 05972"  : ảnh có biển, đọc đúng phải ra chuỗi này (dòng trên + dãy số)
  - null          : ảnh KHÔNG có biển (phải "không đọc được")
  - "PARTIAL"     : biển bị mép khung cắt mất ký tự (không bắt buộc đọc được)

Ví dụ:
  venv311\\Scripts\\python.exe tools\\evaluate_plates.py training\\eval\\labels_2026-09.json
  venv311\\Scripts\\python.exe tools\\evaluate_plates.py labels.json --degrade ir --csv ket_qua.csv

Chỉ số quan trọng nhất: "ĐỌC SAI mà vẫn chấp nhận" và "ảnh không có biển mà
đọc ra biển" - 2 loại này sinh cảnh báo lệch biển giả hoặc cho qua nhầm. "Từ
chối" (không đọc được) chỉ làm bảo vệ phải xem ảnh thủ công, an toàn hơn.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_PHOTO_ROOT = r"C:\Vehicle_Gate_Management\Vehicle_Gate_Management\wwwroot\photos"


def degrade(img, mode, rng):
    """Giả lập điều kiện xấu trên ảnh ban ngày: dark = thiếu sáng + nhiễu,
    ir = như dark nhưng ảnh xám (camera hồng ngoại ban đêm), blur = mờ rung."""
    if mode in ("dark", "ir"):
        f = (img.astype(np.float32) / 255.0) ** 2.2 * 0.45
        f += rng.normal(0, 0.03, f.shape)
        img = np.clip(f * 255, 0, 255).astype(np.uint8)
    if mode == "ir":
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if mode == "blur":
        img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="file JSON nhãn")
    ap.add_argument("--root", default=DEFAULT_PHOTO_ROOT, help="thư mục gốc của đường dẫn ảnh trong file nhãn")
    ap.add_argument("--degrade", choices=["dark", "ir", "blur"], help="giả lập điều kiện xấu")
    ap.add_argument("--csv", help="ghi chi tiết từng ảnh ra file CSV")
    args = ap.parse_args()

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    import main as service
    from config import config

    labels = json.load(open(args.labels, encoding="utf-8"))
    rng = np.random.default_rng(0)
    rows, total_time = [], 0.0
    for i, (rel, truth) in enumerate(labels.items(), 1):
        path = rel if os.path.isabs(rel) else os.path.join(args.root, rel)
        img = cv2.imread(path)
        if img is None:
            print(f"  ! không đọc được ảnh: {path}")
            continue
        if args.degrade:
            img = degrade(img, args.degrade, rng)
        t0 = time.perf_counter()
        plate, conf, err = service._process_single_image(img)
        total_time += time.perf_counter() - t0
        ok = bool(plate) and conf >= config.MIN_SUCCESS_CONFIDENCE
        rows.append({"image": rel, "truth": truth, "plate": plate if ok else "", "conf": round(conf, 3), "error": err or ""})
        if i % 50 == 0:
            print(f"  ... {i}/{len(labels)}")

    def digits(p):
        return (p or "").split(" ")[-1]

    plates = [r for r in rows if r["truth"] not in (None, "PARTIAL")]
    others = [r for r in rows if r["truth"] in (None, "PARTIAL")]
    full = sum(1 for r in plates if r["plate"] == r["truth"])
    dig = sum(1 for r in plates if r["plate"] and digits(r["plate"]) == digits(r["truth"]))
    wrong = [r for r in plates if r["plate"] and digits(r["plate"]) != digits(r["truth"])]
    rejected = sum(1 for r in plates if not r["plate"])
    false_accept = [r for r in others if r["plate"] and r["truth"] is None]

    print()
    print(f"Ảnh có biển: {len(plates)}")
    print(f"  Đọc đúng dãy số            : {dig} ({dig / max(1, len(plates)):.1%})")
    print(f"  Đọc đúng cả biển           : {full}")
    print(f"  ĐỌC SAI mà vẫn chấp nhận   : {len(wrong)}  {Counter(r['plate'] for r in wrong).most_common(8)}")
    print(f"  Từ chối (không đọc được)   : {rejected}")
    print(f"Ảnh không có biển: {sum(1 for r in others if r['truth'] is None)}")
    print(f"  ĐỌC RA BIỂN (nhận nhầm)    : {len(false_accept)}  {[(r['image'], r['plate']) for r in false_accept][:8]}")
    print(f"Thời gian trung bình: {1000 * total_time / max(1, len(rows)):.0f} ms/ảnh")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Đã ghi chi tiết: {args.csv}")


if __name__ == "__main__":
    main()
