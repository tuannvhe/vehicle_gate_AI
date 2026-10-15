import cv2
import numpy as np
from loguru import logger

# Độ sáng trung bình (0-255) dưới ngưỡng này coi là ảnh thiếu sáng (ban đêm/
# ngược sáng) - Otsu threshold thẳng trên ảnh tối đều dễ chọn sai ngưỡng vì
# giả định histogram có 2 đỉnh rõ ràng (nền/chữ), ảnh tối gần như chỉ có 1
# đỉnh. CHƯA đo thực tế trên ảnh camera IR ban đêm thật - cần tinh chỉnh lại
# ngưỡng này sau khi có ảnh mẫu.
LOW_LIGHT_MEAN_THRESHOLD = 80


class ImagePreprocessor:
    @staticmethod
    def undistort(image, k1: float, k2: float = 0.0):
        """Khử méo ống kính (barrel/fisheye) bằng mô hình distortion đơn giản
        của OpenCV - dùng ma trận camera "tổng hợp" (không cần hiệu chỉnh
        chính xác bằng bàn cờ caro) vì mục tiêu chỉ là NẮN BỚT đường cong ở
        rìa khung hình cho gần thẳng, không cần tuyệt đối chính xác.

        k1=0 -> trả nguyên ảnh gốc (no-op), để gọi hàm này vô điều kiện ở nơi
        dùng mà không cần if riêng.
        """
        if not k1 and not k2:
            return image
        h, w = image.shape[:2]
        # fx=fy=w là xấp xỉ hợp lý cho ống kính góc rộng phổ biến - không cần
        # đúng tuyệt đối, chỉ ảnh hưởng tỉ lệ độ cong được nắn bao nhiêu tại
        # cùng 1 giá trị k1, nên tinh chỉnh k1 bằng mắt sẽ tự bù cho việc này.
        camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
        return cv2.undistort(image, camera_matrix, dist_coeffs)

    @staticmethod
    def preprocess(image):
        """Tiền xử lý ảnh trước khi đưa vào AI"""
        try:
            # 1. Resize nếu quá lớn
            h, w = image.shape[:2]
            if w > 1280 or h > 720:
                scale = min(1280/w, 720/h)
                new_w, new_h = int(w*scale), int(h*scale)
                image = cv2.resize(image, (new_w, new_h))
                logger.debug(f"Resized to {new_w}x{new_h}")

            # 2. Chuyển sang ảnh xám
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 2b. Ảnh thiếu sáng (ban đêm/ngược sáng đèn pha) - tăng tương phản
            # cục bộ bằng CLAHE trước khi làm nét, thay vì đưa thẳng ảnh tối
            # đều vào Otsu threshold ở bước 4 (dễ chọn sai ngưỡng, mất chữ).
            # CLAHE tốt hơn cân bằng histogram toàn cục thông thường vì xử lý
            # theo từng vùng nhỏ (tile) - không bị vùng chói đèn pha kéo lệch
            # cả ảnh.
            mean_brightness = float(np.mean(gray))
            if mean_brightness < LOW_LIGHT_MEAN_THRESHOLD:
                logger.debug(
                    f"Ảnh thiếu sáng (mean={mean_brightness:.1f} < "
                    f"{LOW_LIGHT_MEAN_THRESHOLD}) - áp CLAHE tăng tương phản"
                )
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            # 3. Làm nét ảnh (Unsharp Mask) để làm nổi rõ các cạnh chữ
            gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
            unsharp_image = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

            # 4. Chuyển về ảnh đen trắng (Thresholding) để OCR dễ đọc
            _, thresh = cv2.threshold(unsharp_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 5. Chuyển lại về 3 kênh màu để EasyOCR nhận diện
            final_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            return final_image
        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            return image