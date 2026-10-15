"""
Công cụ dò hệ số khử méo ống kính (k1) BẰNG MẮT - không cần hiệu chỉnh
camera chính xác bằng bàn cờ caro.

Cách dùng:
    python tools/preview_undistort.py duong_dan_anh.jpg

Sẽ tạo ra 1 ảnh ghép nhiều mức k1 khác nhau cạnh nhau (lưu tại
<tên ảnh gốc>_undistort_preview.jpg) - anh xem thử mức nào làm đường thẳng
(hàng rào, cạnh biển số) hết cong nhất thì dùng đúng giá trị đó, set vào
biến môi trường LENS_UNDISTORT_K1 (file .env hoặc appsettings của ai-service)
rồi khởi động lại service.

Nếu ảnh CÒN cong theo chiều ngược lại (méo kiểu lõm/pincushion thay vì
phình/barrel) ở 1 mức k1 nào đó, nghĩa là đã đi quá đà - thử giá trị nhỏ
hơn hoặc đổi dấu.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessor import ImagePreprocessor  # noqa: E402

# Các mức k1 thử qua - chỉnh lại danh sách này nếu cần dò khoảng khác.
K1_VALUES = [0.0, -0.15, -0.3, -0.45, -0.6]


def main():
    if len(sys.argv) < 2:
        print("Dùng: python tools/preview_undistort.py duong_dan_anh.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Không đọc được ảnh: {img_path}")
        sys.exit(1)

    # Thu nhỏ mỗi ô xem trước để ảnh ghép không quá khổ, vẫn đủ để so sánh
    # độ cong đường thẳng bằng mắt.
    preview_w = 480
    scale = preview_w / image.shape[1]
    preview_h = int(image.shape[0] * scale)

    tiles = []
    for k1 in K1_VALUES:
        undistorted = ImagePreprocessor.undistort(image, k1)
        tile = cv2.resize(undistorted, (preview_w, preview_h))
        cv2.putText(
            tile, f"k1={k1}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        tiles.append(tile)

    # Ghép thành lưới 2 hàng cho dễ nhìn nếu nhiều hơn 3 mức.
    per_row = 3
    rows = []
    for i in range(0, len(tiles), per_row):
        row_tiles = tiles[i:i + per_row]
        while len(row_tiles) < per_row:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    grid = np.vstack(rows)

    out_path = img_path.with_name(f"{img_path.stem}_undistort_preview.jpg")
    cv2.imwrite(str(out_path), grid)
    print(f"Đã lưu ảnh so sánh tại: {out_path}")
    print("Xem thử mức k1 nào làm đường thẳng hết cong nhất, dùng đúng giá trị đó.")


if __name__ == "__main__":
    main()
