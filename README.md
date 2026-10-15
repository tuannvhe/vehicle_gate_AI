# Vehicle Gate — Dịch vụ AI đọc biển số

Dịch vụ FastAPI đọc biển số xe máy Việt Nam từ ảnh camera cho hệ thống kiểm soát xe ra vào **Vehicle Gate**.
Backend .NET gửi ảnh sau mỗi lượt quẹt thẻ; dịch vụ trả biển số đọc được (đồng bộ) và đồng thời gửi kết quả về webhook của backend.

Các repo liên quan:
- Giao diện web: [vehicle_gate_UI](https://github.com/tuannvhe/vehicle_gate_UI)
- Backend .NET: `Vehicle_Gate_Management` (gọi `POST /api/process`, nhận kết quả tại `/api/ai/webhook/result`)

## Cách hoạt động

```
Ảnh camera ──► Tìm vùng biển (YOLOv8, models/license_plate_detector.pt)
          ──► Đọc chữ (PaddleOCR PP-OCRv5, models/ocr_rec/)    ── dự phòng: YOLO nhận từng ký tự
          ──► Hậu xử lý theo cấu trúc biển VN                       (models/character_detector.pt)
          ──► Trả kết quả + gửi webhook về backend
```

- **Bộ đọc chính** là PaddleOCR (`src/plate_reader.py`), chạy offline với model để sẵn trong `models/ocr_rec/`.
  Nếu không nạp được (thiếu gói paddle, model hỏng), dịch vụ tự chuyển sang bộ nhận diện từng ký tự bằng YOLO (`src/plate_ocr.py`) — kém chính xác hơn.
- **Cấu trúc biển áp dụng:** dòng trên 4 ký tự (vị trí 1–2 là số, vị trí 3 là chữ, vị trí 4 là chữ hoặc số), dòng dưới 4–5 chữ số.
  Chữ dễ nhầm ở vị trí bắt buộc là số được sửa tự động (B→8, D→0, S→5, Z→2, G→6).
- **Nhiều ảnh mỗi lượt:** ảnh được đọc lần lượt, dừng ngay khi có ảnh đủ tin cậy; ảnh camera còn lại chỉ dùng làm dự phòng.

## Cài đặt

Yêu cầu Python 3.11 (Windows hoặc Linux), chạy trên CPU.

```bash
python -m venv venv311
venv311\Scripts\activate          # Linux: source venv311/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Linux: cp .env.example .env
python run.py
```

Dịch vụ chạy ở `http://localhost:5000`. Kiểm tra: `GET /health`. Tài liệu API tự sinh: `http://localhost:5000/docs`.

## Cấu hình (`.env`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `WEBHOOK_URL` | `https://localhost:7069/api/ai/webhook/result` | Địa chỉ backend nhận kết quả |
| `BACKEND_VERIFY_SSL` | `false` | Đặt `false` khi backend dùng chứng chỉ tự ký trong mạng nội bộ |
| `READ_FULL_PLATE` | `true` | Đọc cả 2 dòng biển (`false` = chỉ dòng số) |
| `ENABLE_TEST_ENDPOINTS` | `false` | Bật `/api/test-batch`. **Không bật trên máy chạy thật** |

Các ngưỡng tin cậy và tham số xử lý ảnh khác nằm trong `src/config.py`.

## API

### `POST /api/process`

Backend gọi sau mỗi lượt quẹt thẻ.

```json
{
  "LogId": 452,
  "CardCode": "T",
  "ImageUrls": ["https://localhost:7069/photos/2026/09/24/lane1/452_cam1.jpg"],
  "Direction": "out",
  "EntryLogId": 451,
  "Timestamp": "2026-09-24T17:37:31"
}
```

Phản hồi:

```json
{ "log_id": 452, "success": true, "plate_number": "12B1 05972", "ocr_confidence": 0.93, "error_message": null }
```

Trả `503` nếu model chưa nạp được — đây là lỗi hạ tầng, khác với trường hợp ảnh không đọc được biển (`success: false`).

### `POST /api/test-batch`

Chỉ có khi `ENABLE_TEST_ENDPOINTS=true`. Đọc toàn bộ ảnh trong một thư mục trên máy chạy dịch vụ; nếu thư mục có `labels.json` (tên file → biển đúng) thì tính luôn độ chính xác.

```json
{ "folder": "C:\\ai-service\\training\\raw_images", "read_full_plate": true }
```

## Cấu trúc thư mục

```
src/
  main.py            API FastAPI, điều phối đọc nhiều ảnh, gửi webhook
  plate_detector.py  Tìm vùng biển (YOLOv8)
  plate_reader.py    Đọc biển bằng PaddleOCR (bộ đọc chính)
  plate_ocr.py       Nhận diện từng ký tự bằng YOLO (dự phòng)
  preprocessor.py    Tiền xử lý ảnh
  postprocessor.py   Chuẩn hoá chuỗi biển số
  config.py          Toàn bộ cấu hình và ngưỡng
models/              Model đã huấn luyện (đi kèm repo)
training/            Huấn luyện bổ sung model nhận ký tự (train.py, check_dataset.py, data.yaml)
tools/               Công cụ đánh giá độ chính xác, xem trước nắn méo ảnh
```

## Huấn luyện thêm và đánh giá

- `training/train.py` huấn luyện tiếp model nhận ký tự từ `models/character_detector.pt`, có tăng cường dữ liệu cho ảnh ban đêm. Kiểm tra bộ dữ liệu trước bằng `training/check_dataset.py`.
- `tools/evaluate_plates.py <labels.json>` đo tỉ lệ đọc đúng trên một bộ ảnh đã gán nhãn (xem `training/eval/README.md`).

Ảnh thật và file nhãn chứa biển số **không** được đưa lên repo (xem `.gitignore`).
