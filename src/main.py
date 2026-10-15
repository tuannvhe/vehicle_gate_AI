import cv2
import numpy as np
import requests
import os
import threading
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger
import uvicorn
import sys
import json

from config import config
from plate_detector import PlateDetector
from plate_ocr import PlateOCR
from plate_reader import PlateReader
from postprocessor import PlatePostProcessor

# ===== Models =====
class ProcessRequest(BaseModel):
    log_id: int = Field(..., alias="LogId")
    card_code: str = Field(..., alias="CardCode")
    image_urls: List[str] = Field(..., alias="ImageUrls")
    direction: str = Field("in", alias="Direction")
    entry_log_id: Optional[int] = Field(None, alias="EntryLogId")
    timestamp: str = Field(..., alias="Timestamp")
    # None (mặc định, caller không gửi field này - luồng thật/AiDispatchService
    # không đổi gì) -> dùng config.READ_FULL_PLATE global. Chỉ set khi caller
    # CHỦ ĐỘNG muốn xem thử đọc đủ biển (tỉnh+series+số) cho 1 request, VD API
    # test bên .NET (AiProcessingService.TestProcessAsync) - không ảnh hưởng
    # hành vi mặc định của luồng camera thật.
    read_full_plate: Optional[bool] = Field(None, alias="ReadFullPlate")

    class Config:
        populate_by_name = True

class ProcessResponse(BaseModel):
    log_id: int
    success: bool
    plate_number: Optional[str] = None
    ocr_confidence: Optional[float] = None
    error_message: Optional[str] = None


class BatchTestRequest(BaseModel):
    # Đường dẫn TUYỆT ĐỐI tới thư mục ảnh trên chính máy đang chạy ai-service
    # (không phải URL) - đây là công cụ test cục bộ, tự chọn ảnh trên máy.
    folder: str = Field(..., description="Đường dẫn thư mục chứa ảnh cần test, VD: C:\\ai-service\\training\\raw_images")
    # Mặc định True (khác /api/process) - đây là API TEST để người dùng xem/
    # đối chiếu kết quả bằng mắt, nên ưu tiên hiển thị đủ biển (tỉnh+series+
    # số) thay vì chỉ dòng số như luồng production. Vẫn có thể set False nếu
    # chỉ muốn xem đúng những gì luồng thật đang đọc.
    read_full_plate: bool = Field(True, description="Đọc đủ biển (tỉnh+series+số) thay vì chỉ dòng số")


class BatchTestItem(BaseModel):
    filename: str
    plate_number: Optional[str] = None
    confidence: Optional[float] = None
    expected_plate: Optional[str] = None
    matched: Optional[bool] = None
    error: Optional[str] = None


class BatchTestResponse(BaseModel):
    folder: str
    total_images: int
    read_success: int
    # Chỉ tính accuracy trên ảnh CÓ ghi biển số đúng trong labels.json - ảnh
    # không có ground-truth vẫn được đọc và trả kết quả, chỉ không tính vào %.
    total_with_expected: int
    correct: int
    accuracy: Optional[float] = None
    results: List[BatchTestItem]

# ===== Initialize =====
app = FastAPI(
    title="Vietnam License Plate Recognition Service",
    version="1.0.0",
    description="AI Service for recognizing Vietnam license plates"
)

detector = PlateDetector()
# Bộ đọc biển CHÍNH: PaddleOCR đọc từng dòng + nắn thẳng + kiểm tra cấu trúc
# biển VN (xem plate_reader.py). Nếu không load được (thiếu gói paddle/model
# hỏng) thì RƠI VỀ model YOLO detect-từng-ký-tự cũ (plate_ocr.py) - kém chính
# xác hơn nhiều nhưng service vẫn chạy, thay vì từ chối mọi request.
ocr = PlateReader()
if ocr.model is None:
    logger.error("Không load được PlateReader (PaddleOCR) - dùng tạm character_detector.pt (kém chính xác hơn)")
    ocr = PlateOCR()
postprocessor = PlatePostProcessor()


def _read_plate(plate_img, read_full_plate=None):
    """Gọi đúng hàm đọc của bộ đọc đang dùng (PlateReader.read / PlateOCR.recognize)."""
    if isinstance(ocr, PlateReader):
        return ocr.read(plate_img, read_full_plate=read_full_plate)
    return ocr.recognize(plate_img, read_full_plate=read_full_plate)

# Dùng chung 1 Session thay vì tạo connection mới mỗi lần gọi requests.get()
# - tái sử dụng kết nối TCP tới cùng máy chủ ảnh, giảm độ trễ mỗi request.
http_session = requests.Session()

# Giới hạn ngược (backpressure) - xem giải thích ở config.MAX_CONCURRENT_INFERENCE.
# threading.Semaphore (không phải asyncio) vì /api/process chạy trong threadpool
# của FastAPI, không phải trên event loop.
inference_semaphore = threading.Semaphore(config.MAX_CONCURRENT_INFERENCE)


@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting License Plate Recognition Service")
    logger.info(f"📸 Model path: {config.MODEL_PATH}")
    logger.info(f"🔗 Webhook URL: {config.WEBHOOK_URL}")
    logger.info(f"🌐 Server: http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"⚙️  Ngưỡng: min_success={config.MIN_SUCCESS_CONFIDENCE}")

    # WARM-UP: lần inference ĐẦU TIÊN sau khi load model luôn chậm hơn hẳn
    # các lần sau (PyTorch/EasyOCR phải khởi tạo kernel, cấp phát bộ nhớ lần
    # đầu). Chạy trước 1 ảnh giả ngay lúc khởi động để "làm nóng" model, để
    # xe ĐẦU TIÊN thật sự quẹt vào giờ cao điểm không phải gánh chi phí này.
    try:
        dummy = np.zeros((200, 600, 3), dtype=np.uint8)
        detector.detect(dummy)
        _read_plate(dummy)
        logger.info("🔥 Model warm-up hoàn tất")
    except Exception as e:
        logger.warning(f"Warm-up lỗi (không nghiêm trọng, model vẫn hoạt động): {e}")


@app.get("/")
async def root():
    return {
        "service": "License Plate Recognition",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process": "/api/process (POST)"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "ocr_loaded": ocr.model is not None,
        "ocr_engine": type(ocr).__name__,
    }


def _resolve_url(url: str) -> str:
    if not url.startswith("http"):
        return config.BASE_IMAGE_URL + url
    return url


# SSRF: image_urls trong request /api/process do .NET backend gửi, nhưng
# KHÔNG có gì đảm bảo giá trị đó luôn hợp lệ/không bị thao túng (backend bug,
# hoặc chính backend bị compromise) - nếu service tự do tải BẤT KỲ URL tuyệt
# đối nào được đưa vào, kẻ tấn công có thể lợi dụng để dò/truy cập tài
# nguyên nội bộ khác (VD metadata endpoint, service nội bộ khác) rồi đọc kết
# quả qua lỗi OCR/timing. Chỉ cho phép tải ảnh từ ĐÚNG host của backend
# (BASE_IMAGE_URL) - vốn là nguồn hợp lệ duy nhất trong thiết kế hệ thống.
_ALLOWED_IMAGE_HOST = urlparse(config.BASE_IMAGE_URL).hostname


def _is_allowed_image_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host is not None and host == _ALLOWED_IMAGE_HOST


def _download_image(url: str):
    """Tải ảnh, trả về ảnh OpenCV hoặc None nếu lỗi (không raise - để vòng
    lặp thử ảnh tiếp theo thay vì làm hỏng toàn bộ request)."""
    if not _is_allowed_image_url(url):
        logger.warning(f"Từ chối tải ảnh - host không được phép (chỉ cho phép {_ALLOWED_IMAGE_HOST}): {url}")
        return None

    try:
        # stream=True + tự đếm byte tải về - Content-Length có thể bị thiếu
        # hoặc khai sai, không thể tin tưởng riêng nó để chặn ảnh quá khổ.
        response = http_session.get(
            url,
            verify=config.BACKEND_VERIFY_SSL,
            timeout=config.IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
        )
        if response.status_code != 200:
            logger.warning(f"Download failed ({url}): HTTP {response.status_code}")
            response.close()
            return None

        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > config.IMAGE_MAX_DOWNLOAD_BYTES:
            logger.warning(f"Ảnh quá lớn theo Content-Length ({content_length} bytes), từ chối: {url}")
            response.close()
            return None

        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > config.IMAGE_MAX_DOWNLOAD_BYTES:
                logger.warning(f"Ảnh vượt giới hạn {config.IMAGE_MAX_DOWNLOAD_BYTES} bytes lúc tải, từ chối: {url}")
                response.close()
                return None
            chunks.append(chunk)

        img_bytes = np.frombuffer(b"".join(chunks), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"Ảnh không hợp lệ ({url})")
            return None

        # Resize nếu vượt kích thước cho phép - vừa tiết kiệm CPU/RAM cho
        # ảnh gốc quá lớn, vừa chặn kiểu ảnh "bom giải nén" (file nén nhỏ
        # nhưng giải mã ra độ phân giải khổng lồ).
        h, w = image.shape[:2]
        if w > config.IMAGE_MAX_WIDTH or h > config.IMAGE_MAX_HEIGHT:
            scale = min(config.IMAGE_MAX_WIDTH / w, config.IMAGE_MAX_HEIGHT / h)
            image = cv2.resize(
                image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )

        return image
    except Exception as e:
        logger.warning(f"Lỗi tải/giải mã ảnh ({url}): {e}")
        return None


def _process_single_image(image, read_full_plate: Optional[bool] = None) -> tuple[Optional[str], float, Optional[str]]:
    """Xử lý 1 ảnh: detect (nhiều ứng viên) -> đọc biển -> postprocess.
    Trả về (plate_number hoặc None, confidence, error_message hoặc None).

    🆕 Thử lần lượt các vùng ứng viên (PlateDetector.detect_candidates), giữ
    kết quả đọc được tin cậy nhất - vùng không phải biển (biển báo, chữ in
    trên ảnh camera...) tự bị loại ở bước đọc vì không đúng cấu trúc biển VN.

    🆕 Confidence cuối = confidence của BƯỚC ĐỌC (đã gồm bỏ phiếu giữa các
    biến thể + kiểm tra cấu trúc), KHÔNG còn trộn 30% confidence detect như
    trước: biển thật nghiêng/xa chỉ được detect 0.1-0.4 dù đọc rất chắc, trộn
    vào làm từ chối oan; ngược lại confidence detect cao cũng không cứu được 1
    lần đọc sai, nên nó chỉ còn dùng để XẾP THỨ TỰ ứng viên."""
    try:
        candidates = detector.detect_candidates(image)
        if not candidates:
            logger.info("detect_conf=None (không tìm thấy biển số trong khung hình)")
            return None, 0.0, "No plate detected"
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None, 0.0, f"Detection error: {str(e)}"

    best_plate, best_conf, last_error = None, 0.0, "OCR failed - no text detected"
    for idx, (plate_img, detect_conf, touches_edge) in enumerate(candidates):
        try:
            plate_text, ocr_conf = _read_plate(plate_img, read_full_plate=read_full_plate)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            last_error = f"OCR error: {str(e)}"
            continue
        if not plate_text:
            logger.info(f"Ứng viên {idx + 1}/{len(candidates)}: detect_conf={detect_conf:.2f}, không đọc được biển hợp lệ")
            continue

        try:
            plate_number, final_conf = postprocessor.postprocess(plate_text, ocr_conf)
        except Exception as e:
            logger.error(f"Postprocess error: {e}")
            last_error = f"Postprocess error: {str(e)}"
            continue

        # Biển chạm mép khung hình mà chỉ đọc ra 4 số: rất có thể mép khung
        # đã cắt mất 1 số (biển mới 5 số) - đo thực tế: "0597" thay vì
        # "05972". Không đủ chắc để dùng so khớp vào/ra -> hạ confidence.
        if touches_edge and len(plate_number.split(" ")[-1]) == 4:
            logger.info(f"Biển chạm mép khung hình và chỉ có 4 số ('{plate_number}') - nghi bị cắt, hạ confidence")
            final_conf *= config.OCR_DISAGREE_PENALTY

        logger.info(
            f"Ứng viên {idx + 1}/{len(candidates)}: detect_conf={detect_conf:.2f}, "
            f"ocr_conf={ocr_conf:.2f}, postprocess_conf={final_conf:.2f}, plate='{plate_number}'"
        )
        if plate_number and final_conf > best_conf:
            best_plate, best_conf = plate_number, final_conf
            if best_conf >= config.MIN_SUCCESS_CONFIDENCE:
                break  # ứng viên xếp theo detect_conf giảm dần - đã đủ chắc, khỏi thử tiếp

    if not best_plate:
        return None, best_conf, last_error
    return best_plate, best_conf, None


def _send_webhook(result_data: dict) -> None:
    log_id = result_data.get("LogId")
    try:
        webhook_response = http_session.post(
            config.WEBHOOK_URL,
            json=result_data,
            verify=config.BACKEND_VERIFY_SSL,
            timeout=10,
        )
        if webhook_response.status_code == 200:
            logger.info(f"[{log_id}] Webhook sent successfully")
        else:
            # Backend NHẬN được request nhưng từ chối/lỗi xử lý - khác hẳn
            # trường hợp không kết nối được tới backend (xem except bên
            # dưới). .NET có cơ chế tự phát hiện log kẹt "processing" quá hạn
            # để retry (LogService.ReconcileStaleProcessingAsync), nhưng phân
            # biệt rõ 2 loại lỗi ở log giúp chẩn đoán sự cố thật nhanh hơn
            # nhiều khi cần soi lại (mất kết nối backend vs backend từ chối).
            logger.warning(f"[{log_id}] Webhook trả lỗi từ backend: HTTP {webhook_response.status_code} - {webhook_response.text[:200]}")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{log_id}] KHÔNG kết nối được tới backend để gửi webhook ({config.WEBHOOK_URL}): {e}")
    except Exception as e:
        logger.warning(f"[{log_id}] Lỗi không xác định khi gửi webhook: {e}")


# QUAN TRỌNG: khai báo "def" thường (KHÔNG "async def"). Toàn bộ code bên
# trong (requests.get, detector.detect, ocr.recognize) đều là hàm ĐỒNG BỘ
# (blocking) - nếu khai báo async def mà gọi thẳng các hàm này, sẽ CHẶN
# event loop của FastAPI trong suốt thời gian xử lý (thường 0.5-2s/ảnh).
# Với "def" thường, FastAPI tự động chạy hàm trong threadpool riêng, nên
# nhiều request tới cùng lúc (2 làn cùng gửi ảnh) được xử lý song song
# thay vì xếp hàng chờ nhau - đúng yêu cầu tối ưu tốc độ phản hồi.
@app.post("/api/process", response_model=ProcessResponse)
def process(request: ProcessRequest):
    """
    Xử lý lần lượt các ảnh trong image_urls THEO THỨ TỰ, DỪNG NGAY khi 1 ảnh
    đọc được biển đủ tin cậy (>= MIN_SUCCESS_CONFIDENCE); nếu không ảnh nào
    đủ tin cậy thì lấy kết quả có confidence cao nhất (vẫn bị từ chối ở ngưỡng).

    .NET (AiDispatchService) gửi ảnh camera phụ trách hướng đi ĐẦU TIÊN (Camera
    Trước cho lượt VÀO, Camera Sau cho lượt RA), kèm ảnh camera còn lại làm dự
    phòng phía sau (AiProcessing:FallbackToOtherCamera) - trường hợp bình
    thường chỉ tốn thời gian xử lý 1 ảnh, ảnh dự phòng chỉ được đọc khi ảnh
    chính thất bại (camera lệch hướng/bị che/biển bị cắt mép).
    """
    if not request.image_urls:
        return ProcessResponse(
            log_id=request.log_id, success=False, error_message="No image URL provided"
        )

    # Model load lỗi lúc khởi động (file .pt hỏng/thiếu) -> detector.model là
    # None mãi mãi. TRƯỚC ĐÂY: request vẫn được nhận, luôn trả "No plate
    # detected" với success=false trông y hệt 1 lượt xử lý bình thường thất
    # bại - triển khai bị hỏng kiểu này có thể chạy âm thầm rất lâu không ai
    # phát hiện. Trả 503 rõ ràng để lộ ra ngay đây là LỖI HẠ TẦNG, không phải
    # "không đọc được biển số".
    if detector.model is None or ocr.model is None:
        logger.error(f"[{request.log_id}] Model YOLO (detect biển/ký tự) chưa load được - từ chối xử lý")
        raise HTTPException(status_code=503, detail="AI model chưa sẵn sàng (load thất bại lúc khởi động)")

    # Backpressure: service này CPU-bound, không có hàng đợi nội bộ - nếu đã
    # đủ request đang inference, TỪ CHỐI NGAY (503) thay vì nhận thêm rồi làm
    # chậm đều tất cả. Caller (.NET LogProcessingWorker) sẽ coi đây là lỗi gửi
    # AI, tự backoff và thử lại sau (xem LogService.MarkProcessingFailedAsync).
    if not inference_semaphore.acquire(blocking=False):
        logger.warning(f"[{request.log_id}] AI service đang bận - từ chối request, để .NET tự retry")
        raise HTTPException(
            status_code=503,
            detail="AI service đang xử lý hết công suất, thử lại sau",
        )

    try:
        best_plate: Optional[str] = None
        best_conf = 0.0
        best_idx = 0
        last_error: Optional[str] = None

        for idx, raw_url in enumerate(request.image_urls):
            url = _resolve_url(raw_url)
            logger.info(f"[{request.log_id}] Thử ảnh {idx + 1}/{len(request.image_urls)}: {url}")

            image = _download_image(url)
            if image is None:
                last_error = last_error or f"Cannot download/decode image {idx + 1}"
                continue

            plate_number, conf, err = _process_single_image(image, read_full_plate=request.read_full_plate)
            if plate_number and conf > best_conf:
                best_plate, best_conf, best_idx = plate_number, conf, idx
            if err and not plate_number:
                last_error = last_error or err
            if best_conf >= config.MIN_SUCCESS_CONFIDENCE:
                if idx + 1 < len(request.image_urls):
                    logger.info(f"[{request.log_id}] Ảnh {idx + 1} đã đọc chắc được biển - bỏ qua ảnh dự phòng còn lại")
                break
    finally:
        inference_semaphore.release()

    success = best_plate is not None and best_conf >= config.MIN_SUCCESS_CONFIDENCE
    final_plate = best_plate if success else None
    final_conf = best_conf if success else None
    error_message = None if success else (last_error or "Invalid plate or low confidence")

    logger.info(f"[{request.log_id}] Kết quả cuối: '{final_plate}', conf={best_conf:.2f}, success={success}")

    result_data = {
        "LogId": request.log_id,
        "Success": bool(success),
        "PlateNumber": final_plate,
        "OcrConfidence": final_conf,
        "ErrorMessage": error_message,
        # Ảnh đầu tiên là camera phụ trách hướng đi, các ảnh sau là dự phòng
        # (xem docstring) - backend không báo lệch biển dựa trên ảnh dự phòng.
        "FromFallbackImage": bool(success and best_idx > 0),
    }
    _send_webhook(result_data)

    return ProcessResponse(
        log_id=request.log_id,
        success=success,
        plate_number=final_plate,
        ocr_confidence=final_conf,
        error_message=error_message,
    )


_TEST_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@app.post("/api/test-batch", response_model=BatchTestResponse)
def test_batch(request: BatchTestRequest):
    """
    Đọc thử TẤT CẢ ảnh trong 1 thư mục cục bộ trên máy đang chạy ai-service,
    trả về biển số đọc được cho từng ảnh - dùng để so sánh model cũ/mới bằng
    ảnh thật (VD sau khi fine-tune xong, xem training/README) mà không cần
    đẩy ảnh qua toàn bộ luồng camera/backend thật.

    Nếu trong thư mục có sẵn file "labels.json" (dạng {"ten_anh.jpg":
    "51H12345"}), tự động so khớp và tính accuracy; không có thì vẫn đọc và
    trả kết quả bình thường, chỉ không tính % đúng.
    """
    if not config.ENABLE_TEST_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

    folder = Path(request.folder)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Không tìm thấy thư mục: {folder}")

    labels: dict[str, str] = {}
    labels_path = folder / "labels.json"
    if labels_path.is_file():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Không đọc được labels.json ({labels_path}): {e}")

    image_paths = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _TEST_IMAGE_EXTS
    )
    if not image_paths:
        raise HTTPException(status_code=400, detail=f"Không có ảnh nào trong thư mục: {folder}")

    if detector.model is None or ocr.model is None:
        raise HTTPException(status_code=503, detail="AI model chưa sẵn sàng (load thất bại lúc khởi động)")

    results: List[BatchTestItem] = []
    read_success = 0
    total_with_expected = 0
    correct = 0

    for img_path in image_paths:
        expected = labels.get(img_path.name)

        image = cv2.imread(str(img_path))
        if image is None:
            results.append(BatchTestItem(
                filename=img_path.name, expected_plate=expected, error="Không đọc được file ảnh (hỏng/định dạng lạ)"
            ))
            continue

        with inference_semaphore:
            plate_number, conf, err = _process_single_image(image, read_full_plate=request.read_full_plate)

        matched = None
        if plate_number:
            read_success += 1
        if expected is not None:
            total_with_expected += 1
            matched = plate_number == expected
            if matched:
                correct += 1

        results.append(BatchTestItem(
            filename=img_path.name,
            plate_number=plate_number,
            confidence=round(conf, 3) if plate_number else None,
            expected_plate=expected,
            matched=matched,
            error=err,
        ))

    accuracy = round(correct / total_with_expected, 4) if total_with_expected else None

    logger.info(
        f"[test-batch] folder={folder}, {len(image_paths)} ảnh, "
        f"đọc được {read_success}, accuracy={accuracy}"
    )

    return BatchTestResponse(
        folder=str(folder),
        total_images=len(image_paths),
        read_success=read_success,
        total_with_expected=total_with_expected,
        correct=correct,
        accuracy=accuracy,
        results=results,
    )


# ===== Run =====
if __name__ == "__main__":
    # WORKERS > 1: chạy nhiều process song song, mỗi process tự load riêng
    # 1 bản YOLO + EasyOCR (tốn thêm RAM tương ứng số worker, nhưng đổi lại
    # tận dụng được nhiều CPU core thật sự - quan trọng nếu máy không có
    # GPU). Nên đặt bằng khoảng số CPU core VẬT LÝ của máy chủ (không phải
    # logical/hyperthread) trừ đi 1 (chừa lại cho hệ điều hành + backend
    # .NET nếu chạy chung máy). Ví dụ máy 4 core -> WORKERS=2 hoặc 3.
    #
    # LƯU Ý: workers > 1 sẽ ép reload=False (đã vậy sẵn) và bắt buộc chạy
    # qua chuỗi import "main:app" (đã đúng sẵn) - không cần đổi gì thêm.
    workers = int(os.getenv("WORKERS", "1"))
    logger.info(f"🧵 Khởi động với {workers} worker process")

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        workers=workers,
    )
