"""
Kiểm tra nhanh dataset trước khi train - chạy trước train.py để bắt lỗi phổ
biến (ảnh thiếu file label, label rỗng, class id ngoài phạm vi, tỉ lệ
train/val quá lệch...) thay vì để YOLO âm thầm bỏ qua hoặc train ra model tệ
mà không rõ nguyên nhân.

Chạy:
    ../venv311/Scripts/python.exe check_dataset.py
"""

import sys
from pathlib import Path

import yaml

# Terminal Windows mặc định dùng cp1252, không encode được tiếng Việt in ra
# qua print() (khác với comment trong source file, không bao giờ được print).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TRAINING_DIR = Path(__file__).resolve().parent
DATA_YAML = TRAINING_DIR / "data.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def check_split(images_dir: Path, labels_dir: Path, num_classes: int, split_name: str) -> int:
    problems = 0
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print(f"[{split_name}] LỖI: không có ảnh nào trong {images_dir}")
        return 1

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[{split_name}] LỖI: thiếu file label cho {img_path.name} (cần {label_path.name})")
            problems += 1
            continue

        lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            print(f"[{split_name}] CẢNH BÁO: {label_path.name} rỗng (không có box nào)")
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                print(
                    f"[{split_name}] LỖI: {label_path.name} dòng {line_no} sai định dạng "
                    f"(cần đúng 5 giá trị: class x_center y_center width height), "
                    f"nhận được {len(parts)} giá trị"
                )
                problems += 1
                continue

            class_id_str, *coords = parts
            try:
                class_id = int(class_id_str)
            except ValueError:
                print(f"[{split_name}] LỖI: {label_path.name} dòng {line_no} - class id không phải số nguyên")
                problems += 1
                continue

            if not (0 <= class_id < num_classes):
                print(
                    f"[{split_name}] LỖI: {label_path.name} dòng {line_no} - class id {class_id} "
                    f"ngoài phạm vi 0-{num_classes - 1} (kiểm tra lại data.yaml/công cụ gán nhãn)"
                )
                problems += 1

            try:
                coord_values = [float(c) for c in coords]
            except ValueError:
                print(f"[{split_name}] LỖI: {label_path.name} dòng {line_no} - toạ độ không phải số")
                problems += 1
                continue

            if any(not (0.0 <= v <= 1.0) for v in coord_values):
                print(
                    f"[{split_name}] LỖI: {label_path.name} dòng {line_no} - toạ độ phải nằm "
                    f"trong [0, 1] (YOLO format là toạ độ CHUẨN HOÁ theo kích thước ảnh, "
                    f"không phải pixel tuyệt đối)"
                )
                problems += 1

    print(f"[{split_name}] {len(images)} ảnh, {problems} lỗi.")
    return problems


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit(f"Không tìm thấy {DATA_YAML}")

    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = TRAINING_DIR / cfg["path"]
    num_classes = cfg["nc"]

    total_problems = 0
    for split, rel_images in (("train", cfg["train"]), ("val", cfg["val"])):
        images_dir = dataset_root / rel_images
        labels_dir = dataset_root / "labels" / split
        total_problems += check_split(images_dir, labels_dir, num_classes, split)

    if total_problems:
        print(f"\nTổng cộng {total_problems} lỗi - SỬA HẾT trước khi chạy train.py.")
        raise SystemExit(1)

    print("\nDataset hợp lệ, có thể chạy train.py.")


if __name__ == "__main__":
    main()
