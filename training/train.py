"""
Fine-tune character_detector.pt bằng dữ liệu ảnh biển số mới (đặc biệt ảnh
ban đêm/thiếu sáng/lóa đèn) để cải thiện độ chính xác đọc ký tự trong điều
kiện hiện tại model còn yếu.

ĐÂY LÀ FINE-TUNE, KHÔNG PHẢI TRAIN TỪ ĐẦU: mặc định load thẳng weight
models/character_detector.pt đang dùng trong production làm điểm khởi đầu,
rồi học tiếp trên dữ liệu mới - tận dụng những gì model đã học tốt (ảnh
sáng đủ), chỉ cần dạy thêm phần còn yếu (ảnh tối), thay vì học lại từ số 0
(vừa lâu vừa cần dataset lớn hơn nhiều mới đạt lại chất lượng hiện tại).

Cách chạy cơ bản (xem thêm hướng dẫn đầy đủ ở training/README.md):

    cd training
    ../venv311/Scripts/python.exe train.py --epochs 100 --device cpu

Sau khi train xong, kiểm tra kỹ ở runs/<name>/weights/best.pt bằng ảnh thật
TRƯỚC KHI thay vào production (xem hướng dẫn --replace bên dưới).
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Terminal Windows mặc định dùng cp1252, không encode được tiếng Việt in ra
# qua print() (khác với comment trong source file, không bao giờ được print).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TRAINING_DIR = Path(__file__).resolve().parent
AI_SERVICE_DIR = TRAINING_DIR.parent
DEFAULT_WEIGHTS = AI_SERVICE_DIR / "models" / "character_detector.pt"
DEFAULT_DATA_YAML = TRAINING_DIR / "data.yaml"
PRODUCTION_MODEL_PATH = AI_SERVICE_DIR / "models" / "character_detector.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Weight khởi điểm để fine-tune tiếp (mặc định: model đang chạy production).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="File data.yaml mô tả dataset (mặc định: training/data.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Số epoch train.")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Giảm xuống (vd 4-8) nếu máy CPU/GPU yếu bị hết RAM/VRAM.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đưa vào train - PHẢI khớp YOLO_IMAGE_SIZE bên config.py "
        "để việc train phản ánh đúng điều kiện lúc infer thực tế.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="'cpu', '0' (GPU đầu tiên), hoặc 'cpu,0' v.v. Train trên CPU sẽ RẤT chậm "
        "(có thể vài giờ tới cả ngày tùy dataset) - ưu tiên dùng máy có GPU NVIDIA nếu có.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Dừng sớm (early stopping) nếu sau N epoch không cải thiện thêm - tránh "
        "train dư thời gian vô ích và tránh overfit khi dataset nhỏ.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Đóng băng N layer đầu (backbone) không cho học lại - dataset fine-tune "
        "thường NHỎ hơn nhiều so với dataset gốc đã train model, đóng băng backbone "
        "giúp giữ lại đặc trưng tổng quát đã học, chỉ học lại phần nhận diện "
        "riêng cho đặc điểm ảnh mới (tối/lóa), giảm nguy cơ overfit/quên kiến thức cũ. "
        "Đặt 0 để học lại toàn bộ (chỉ nên làm khi dataset đủ lớn, vài nghìn ảnh trở lên).",
    )
    parser.add_argument(
        "--name",
        default="night_finetune",
        help="Tên thư mục kết quả, lưu tại training/runs/detect/<name>/.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Sau khi train xong, TỰ ĐỘNG copy best.pt đè vào models/character_detector.pt. "
        "Mặc định KHÔNG bật - nên tự kiểm tra chất lượng bằng ảnh thật trước, tránh "
        "đưa thẳng 1 model chưa kiểm chứng vào production.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Không tìm thấy {args.data}. Xem training/README.md để biết cách chuẩn bị "
            "dataset (thư mục images/train, images/val, labels/train, labels/val)."
        )
    if not args.weights.exists():
        raise SystemExit(f"Không tìm thấy weight khởi điểm {args.weights}.")

    model = YOLO(str(args.weights))

    # augmentation nghiêng về mô phỏng điều kiện SÁNG THẤT THƯỜNG (đêm/lóa đèn)
    # thay vì bộ mặc định của ultralytics (vốn tối ưu cho ảnh chụp ban ngày đa
    # dạng góc/tỉ lệ tổng quát) - hsv_v (độ sáng) và hsv_s (độ bão hòa) được nới
    # rộng hơn mặc định (0.4/0.7) để model quen với ảnh quá tối/quá chói hơn;
    # degrees/shear/perspective giữ THẤP vì ký tự trên biển số VN không xoay/
    # nghiêng nhiều trong thực tế, xoay mạnh chỉ tạo dữ liệu giả không thực tế.
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        freeze=args.freeze if args.freeze > 0 else None,
        project=str(TRAINING_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        # --- augmentation tinh chỉnh cho ánh sáng khó ---
        hsv_h=0.015,       # mặc định - màu sắc biển số VN không đổi (nền trắng/xanh)
        hsv_s=0.5,         # nới nhẹ so với mặc định 0.7 -> tăng 0.5 để mô phỏng
                            # ảnh camera IR ám xanh/xám (bão hòa màu thấp bất thường)
        hsv_v=0.6,         # nới mạnh (mặc định 0.4) -> mô phỏng dải sáng rộng hơn:
                            # từ ảnh tối om (thiếu sáng) tới cháy sáng (lóa đèn pha)
        degrees=5.0,       # xoay nhẹ - camera lắp cố định, biển hiếm khi nghiêng nhiều
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,        # KHÔNG lật dọc - ký tự lộn ngược không tồn tại thực tế
        fliplr=0.0,        # KHÔNG lật ngang - lật sẽ biến '3' thành 'E' giả, sai hoàn toàn
        mosaic=0.5,        # giảm so với mặc định 1.0 - dataset fine-tune nhỏ, mosaic
                            # (ghép 4 ảnh) quá nhiều dễ tạo bối cảnh giả không giống
                            # ảnh crop biển số thật (vốn đã crop sát, không có nền phụ)
        close_mosaic=10,   # tắt mosaic ở N epoch cuối để model "chốt" trên ảnh thật
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTrain xong. Weight tốt nhất: {best_weights}")
    print(f"Kết quả chi tiết (mAP, PR curve, confusion matrix...): {results.save_dir}")

    if args.replace:
        if not best_weights.exists():
            raise SystemExit("Không tìm thấy best.pt - có thể train đã thất bại giữa chừng.")
        shutil.copy2(best_weights, PRODUCTION_MODEL_PATH)
        print(f"Đã copy đè vào {PRODUCTION_MODEL_PATH} - KHỞI ĐỘNG LẠI ai-service để áp dụng.")
    else:
        print(
            "\nChưa tự động thay vào production (chạy lại kèm --replace nếu muốn).\n"
            f"Kiểm tra bằng ảnh thật trước, sau đó copy thủ công:\n"
            f"  copy \"{best_weights}\" \"{PRODUCTION_MODEL_PATH}\""
        )


if __name__ == "__main__":
    main()
