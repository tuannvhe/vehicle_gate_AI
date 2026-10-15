# Bộ ảnh đánh giá đọc biển

`labels_2026-09.json`: 416 ảnh thật (08/2026–09/2026, lane1, cả cam0 lẫn cam1)
chụp bằng **camera tạm** (camera trong nhà, góc rộng, ảnh lồi, biển nghiêng 0–60°).
Đường dẫn tương đối so với `Vehicle_Gate_Management/wwwroot/photos`.

Nhãn gán tay:
- log id ≤ 261 → biển `66F1 87656` (biển in trên giấy), log id ≥ 264 → `12B1 05972`
- `null`: ảnh không có biển (camera chĩa trời/tường, biển báo "CẤM HÚT THUỐC",
  chỉ có dòng giờ/chữ "Lan1" của camera...)
- `"PARTIAL"`: biển bị mép khung cắt mất ký tự (321_cam0, 328_cam1, 381_cam0)

Chạy đánh giá:

```
venv311\Scripts\python.exe tools\evaluate_plates.py training\eval\labels_2026-09.json
venv311\Scripts\python.exe tools\evaluate_plates.py training\eval\labels_2026-09.json --degrade ir
```

Kết quả tham chiếu (09/2026, cấu hình mặc định): đọc đúng dãy số 214/220,
đọc sai mà vẫn chấp nhận 1 (biển bị mép khung cắt), ảnh không có biển đọc
nhầm ra biển 0, ~650 ms/ảnh trên CPU.

Khi lắp camera thật: gán nhãn 1 bộ ảnh mới theo cùng định dạng rồi đo lại -
các ngưỡng trong `src/config.py` được chỉnh trên bộ ảnh camera tạm này.
