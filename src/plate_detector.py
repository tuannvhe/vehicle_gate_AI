import threading
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger
from config import config
from preprocessor import ImagePreprocessor


class PlateDetector:
    def __init__(self):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.model = None
        # Ultralytics YOLO tái sử dụng 1 predictor nội bộ (self.model.predictor)
        # giữ state theo từng lần gọi (results/batch) - gọi đồng thời từ nhiều
        # thread (FastAPI chạy route đồng bộ trong threadpool, cho phép tối đa
        # MAX_CONCURRENT_INFERENCE request cùng lúc) có thể làm lẫn kết quả
        # giữa các request khác nhau. Khóa lại để đảm bảo ĐÚNG kết quả - với
        # hệ thống kiểm soát ra vào, đọc sai biển số nguy hiểm hơn nhiều so
        # với việc xử lý chậm hơn 1 chút khi có 2 xe quét trùng thời điểm.
        self._infer_lock = threading.Lock()
        # Số luồng PyTorch lúc khởi tạo - PaddleOCR (plate_reader.py, chạy cùng
        # process) dùng CHUNG runtime OpenMP và tự đặt lại số luồng về 1 mỗi
        # lần đọc, khiến YOLO chạy 1 luồng (đo: ~100ms -> ~350ms/ảnh). Đặt lại
        # trước mỗi lần detect (xem _run).
        self._torch_threads = torch.get_num_threads()
        self._load_model()

    def _load_model(self):
        try:
            if self.config.MODEL_PATH.exists():
                self.model = YOLO(str(self.config.MODEL_PATH))
                logger.info(f"Loaded model from {self.config.MODEL_PATH}")
            else:
                # Fallback: dùng model pretrained
                self.model = YOLO('yolov8n.pt')
                logger.warning("Using default YOLO model (not fine-tuned)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _run(self, image, conf):
        with self._infer_lock:
            if torch.get_num_threads() != self._torch_threads:
                torch.set_num_threads(self._torch_threads)
            results = self.model(
                image,
                conf=conf,
                iou=self.config.YOLO_IOU,
                imgsz=self.config.YOLO_IMAGE_SIZE,
                # SỬA: bản gốc định nghĩa config.YOLO_DEVICE nhưng KHÔNG truyền
                # vào đây - nghĩa là dù đổi config thành "cuda" cũng không có
                # tác dụng, model luôn chạy mặc định (thường tự chọn CPU nếu
                # không chỉ định rõ). Đây là chỗ thực sự bật GPU nếu máy có.
                device=self.config.YOLO_DEVICE,
                verbose=False,  # tắt log chi tiết của ultralytics mỗi lần chạy - đỡ tốn I/O console lúc cao điểm nhiều request/giây
            )
        if not results or len(results[0].boxes) == 0:
            return []
        return [(float(b.conf[0]), list(map(int, b.xyxy[0].tolist()))) for b in results[0].boxes]

    def _crop(self, image, box):
        """Cắt vùng biển kèm lề - 🐛 FIX: trước đây đệm CỐ ĐỊNH 10px, box YOLO
        với biển nghiêng/méo thường ôm hụt 1 góc biển -> cắt mất 1 phần ký tự
        đầu/cuối (thực tế: số "8" đầu dãy "876.56" bị cắt nửa, OCR đọc ra
        "976.56"). Đệm theo TỈ LỆ kích thước box."""
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        pad_x = max(10, int((x2 - x1) * self.config.PLATE_CROP_PADDING))
        pad_y = max(10, int((y2 - y1) * self.config.PLATE_CROP_PADDING))
        return image[max(0, y1 - pad_y): min(h, y2 + pad_y), max(0, x1 - pad_x): min(w, x2 + pad_x)]

    @staticmethod
    def _rotate(image, angle):
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += new_w / 2 - w / 2
        m[1, 2] += new_h / 2 - h / 2
        return cv2.warpAffine(image, m, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    def detect_candidates(self, image):
        """Trả về list (plate_image, confidence, touches_edge) các vùng CÓ
        THỂ là biển, xếp theo confidence giảm dần (tối đa PLATE_MAX_CANDIDATES)
        - [] nếu không có. `touches_edge` = box chạm mép khung hình (biển có
        thể bị cắt mất ký tự - xem main._process_single_image).

        🆕 Trả NHIỀU ứng viên thay vì chỉ box tốt nhất, ở ngưỡng thấp hơn
        (PLATE_CANDIDATE_MIN_CONF) - để bước đọc biển (kiểm tra đúng cấu trúc
        biển VN) tự quyết vùng nào thật sự là biển. Đo trên 416 ảnh thật: model
        detect biển cho biển THẬT nghiêng/xa chỉ 0.11-0.4 (dưới ngưỡng 0.25 cũ
        -> "No plate detected" oan), trong khi biển "CẤM HÚT THUỐC", chữ "Lan1"
        và dòng giờ in trên ảnh camera lại được 0.5-0.64 - chọn box cao nhất
        dễ đọc nhầm biển báo thay vì biển số.

        🆕 Không thấy ứng viên nào -> thử lại trên ảnh XOAY các góc
        PLATE_DETECT_RETRY_ANGLES: model detect không được train với ảnh
        xoay, biển nghiêng >45° (camera tạm đặt lệch) bị bỏ sót hoàn toàn.

        🐛 FIX (giữ từ bản cũ): các nhánh "không tìm thấy biển" trả RỖNG, KHÔNG
        trả nguyên ảnh gốc - tránh OCR chạy trên toàn khung hình.
        """
        if self.model is None:
            return []

        try:
            # Khử méo ống kính TRƯỚC khi detect (no-op nếu LENS_UNDISTORT_K1=0)
            # - xem ImagePreprocessor.undistort. Làm ở đây (không phải trong
            # OCR) để cả bbox YOLO detect lẫn vùng crop sau đó đều tính trên
            # ảnh đã nắn thẳng, tránh lệch tọa độ giữa 2 bước.
            image = self.preprocessor.undistort(
                image, self.config.LENS_UNDISTORT_K1, self.config.LENS_UNDISTORT_K2
            )

            boxes = self._run(image, self.config.PLATE_CANDIDATE_MIN_CONF)
            source = image
            if not boxes:
                for angle in self.config.PLATE_DETECT_RETRY_ANGLES:
                    rotated = self._rotate(image, angle)
                    boxes = self._run(rotated, self.config.PLATE_CANDIDATE_MIN_CONF)
                    if boxes:
                        logger.info(f"Không thấy biển ở ảnh gốc - tìm thấy khi xoay {angle:+d}°")
                        source = rotated
                        break
            if not boxes:
                logger.warning("No plate detected")
                return []

            # Ultralytics không đảm bảo thứ tự box theo confidence - tự sắp.
            boxes.sort(key=lambda b: b[0], reverse=True)
            candidates = []
            sh, sw = source.shape[:2]
            for conf, box in boxes[: self.config.PLATE_MAX_CANDIDATES]:
                crop = self._crop(source, box)
                if crop.size:
                    x1, y1, x2, y2 = box
                    touches_edge = x1 <= 2 or y1 <= 2 or x2 >= sw - 3 or y2 >= sh - 3
                    candidates.append((crop, conf, touches_edge))
            logger.debug("Plate candidates: " + ", ".join(f"{c:.2f}{' (mép)' if e else ''}" for _, c, e in candidates))
            return candidates

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect(self, image):
        """Tương thích ngược: chỉ trả ứng viên tốt nhất (plate_image, conf)."""
        candidates = self.detect_candidates(image)
        return candidates[0][:2] if candidates else (None, 0.0)
