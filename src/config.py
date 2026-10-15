import os
from pathlib import Path
from dotenv import load_dotenv

# Tự động tìm đường dẫn gốc của dự án (C:\ai-service)
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()


class Config:
    # Paths
    MODEL_PATH = BASE_DIR / "models" / "license_plate_detector.pt"
    # Model YOLO phát hiện TỪNG KÝ TỰ trên biển (0-9, A-Z trừ I/J/O/Q/W -
    # đúng quy định chữ được phép dùng trên biển số VN) - THAY THẾ EasyOCR
    # (xem plate_ocr.py). Nguồn: repo cộng đồng
    # "Yolov8-Detect-Vietnamese-license-plates-and-characters", train riêng
    # cho biển số VN nên bền hơn EasyOCR (model OCR chữ viết chung chung) với
    # ảnh méo/mờ/nghiêng - vốn là nguyên nhân đọc sai lặp lại nhiều lần.
    CHARACTER_MODEL_PATH = BASE_DIR / "models" / "character_detector.pt"

    # 🆕 Model OCR dòng chữ (PaddleOCR text recognition) - ĐỌC BIỂN CHÍNH,
    # thay character_detector.pt ở trên (xem plate_reader.py để biết lý do,
    # số liệu đo). Để sẵn trong models/ để chạy KHÔNG CẦN Internet.
    OCR_REC_MODEL_NAME = os.getenv("OCR_REC_MODEL_NAME", "en_PP-OCRv5_mobile_rec")
    OCR_REC_MODEL_DIR = BASE_DIR / "models" / "ocr_rec" / OCR_REC_MODEL_NAME
    # Biển được đọc ở nhiều biến thể hình học (nắn phối cảnh / xoay thẳng /
    # crop gốc) rồi bỏ phiếu. Kết quả KHÔNG CHẮC (hòa phiếu, hoặc chỉ 1 biến
    # thể đọc ra mà score < OCR_SINGLE_VOTE_MIN_SCORE) bị nhân confidence với
    # OCR_DISAGREE_PENALTY - 0.45 đảm bảo luôn dưới MIN_SUCCESS_CONFIDENCE
    # (0.5) => bị từ chối. Đo trên 219 biển thật: 207 đúng / 1 sai được chấp
    # nhận (so với 118 đúng / 95 sai của character_detector.pt cũ).
    OCR_DISAGREE_PENALTY = float(os.getenv("OCR_DISAGREE_PENALTY", "0.45"))
    OCR_SINGLE_VOTE_MIN_SCORE = float(os.getenv("OCR_SINGLE_VOTE_MIN_SCORE", "0.9"))
    # Góc xoay mù để đọc thử khi KHÔNG tìm được hình học biển nào (xem
    # PlateReader.read) - mỗi góc = 1 lượt OCR. MẶC ĐỊNH TẮT: đo trên 416 ảnh
    # thật, bật "15,-15,30,-30" làm vùng rác (biển báo, nền) có thêm cơ hội
    # lọt kiểm tra cấu trúc -> 2 ảnh không có biển bị đọc ra biển, và chậm
    # thêm ~350ms/ảnh. Chỉ bật khi camera thật hay có biển nghiêng rất mạnh.
    OCR_FALLBACK_ANGLES = [
        int(a) for a in os.getenv("OCR_FALLBACK_ANGLES", "").split(",") if a.strip()
    ]
    # Nới tứ giác vùng biển ra ngoài (tỉ lệ) trước khi nắn thẳng - vùng nền
    # trắng tìm được nằm trong viền biển, không nới sẽ cắt mất ký tự sát mép.
    OCR_QUAD_MARGIN = float(os.getenv("OCR_QUAD_MARGIN", "0.08"))

    # YOLO
    YOLO_CONFIDENCE = 0.25
    YOLO_IOU = 0.45
    # Đệm thêm quanh box biển YOLO trước khi crop (tỉ lệ theo kích thước box).
    PLATE_CROP_PADDING = float(os.getenv("PLATE_CROP_PADDING", "0.12"))
    # Ngưỡng + số lượng ứng viên vùng biển đưa sang bước đọc (xem
    # PlateDetector.detect_candidates) - ngưỡng thấp hơn YOLO_CONFIDENCE vì
    # bước đọc biển đã tự loại vùng không đúng cấu trúc biển VN.
    PLATE_CANDIDATE_MIN_CONF = float(os.getenv("PLATE_CANDIDATE_MIN_CONF", "0.1"))
    PLATE_MAX_CANDIDATES = int(os.getenv("PLATE_MAX_CANDIDATES", "3"))
    # Góc xoay ảnh để thử detect lại khi ảnh gốc không thấy biển nào (biển
    # nghiêng mạnh). Mỗi góc = 1 lượt YOLO, chỉ tốn khi ảnh gốc không thấy.
    PLATE_DETECT_RETRY_ANGLES = [
        int(a) for a in os.getenv("PLATE_DETECT_RETRY_ANGLES", "45,-45").split(",") if a.strip()
    ]

    # Ngưỡng riêng cho model phát hiện ký tự - ký tự nhỏ hơn nhiều so với cả
    # tấm biển nên cần tinh chỉnh riêng, không dùng chung YOLO_CONFIDENCE/IOU
    # ở trên (vốn tinh chỉnh cho việc phát hiện CẢ tấm biển).
    CHARACTER_YOLO_CONFIDENCE = float(os.getenv("CHARACTER_YOLO_CONFIDENCE", "0.4"))
    CHARACTER_YOLO_IOU = float(os.getenv("CHARACTER_YOLO_IOU", "0.45"))
    # 640 là mặc định an toàn. Nếu cần nhanh hơn nữa và chấp nhận đánh đổi
    # 1 phần độ chính xác, có thể thử 480 hoặc 416 (giảm dần theo bước 32,
    # vì YOLO yêu cầu kích thước chia hết cho 32) - PHẢI đo lại độ chính xác
    # thực tế trước khi đổi mặc định, không nên đoán mò.
    YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))

    # Các góc (độ) xoay vùng biển để đọc ký tự rồi bỏ phiếu (xem
    # PlateOCR._vote) - camera góc rộng đặt lệch làm biển nghiêng ~15-20°.
    # Mỗi góc = 1 lượt inference (~150ms CPU). Đo trên 95 ảnh thật (09/2026,
    # cam0+cam1, biển 12B1 05972) - đúng dãy số / đúng đủ biển:
    #   "0"                 38 / 24  (1 lượt)
    #   "0,12,-12"          58 / 32  (3 lượt) <- mặc định
    #   "0,8,-8,16,-16"     56 / 35  (5 lượt)
    # Đặt "0" để tắt hẳn việc xoay.
    OCR_DESKEW_ANGLES = [
        int(a) for a in os.getenv("OCR_DESKEW_ANGLES", "0,12,-12").split(",") if a.strip()
    ]
    YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # đổi thành "cuda" nếu máy có GPU NVIDIA

    # Số request /api/process được xử lý ĐỒNG THỜI tối đa TRONG 1 PROCESS -
    # đây là service CPU-bound (YOLO + EasyOCR không có GPU), nhận thêm request
    # vượt quá con số này chỉ làm MỌI request đang xử lý chậm đều nhau, không
    # tăng thông lượng thực - nên thà từ chối ngay (HTTP 503) để .NET backend
    # tự đưa vào hàng đợi retry có backoff (xem LogService.MarkProcessingFailedAsync
    # bên .NET), còn hơn để dồn ứ âm thầm dẫn tới timeout. Nên đặt bằng hoặc
    # THẤP HƠN AiProcessing:MaxConcurrentRequests bên .NET appsettings.json,
    # nhân với WORKERS (mỗi worker process có giới hạn riêng của mình).
    MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "2"))

    # Phân biệt biển 1 dòng (ô tô, dẹt ngang) và 2 dòng (xe máy, gần vuông)
    # bằng tỉ lệ width/height của vùng biển sau khi YOLO detect. Biển ô tô
    # VN thực tế ~4-5 lần rộng hơn cao, biển xe máy ~1-1.4 lần - ngưỡng 2.0
    # để dư margin an toàn. NÊN TINH CHỈNH LẠI bằng cách đo thử vài chục ảnh
    # biển thật của công ty (log giá trị aspect_ratio ở mức DEBUG để đo).
    PLATE_ASPECT_RATIO_THRESHOLD = float(os.getenv("PLATE_ASPECT_RATIO_THRESHOLD", "2.0"))

    # Confidence tối thiểu RIÊNG cho DÒNG TRÊN (tỉnh+series) của biển xe máy
    # 2 dòng để được đưa vào chuỗi hiển thị - dòng trên chỉ dùng để HIỂN THỊ
    # (so khớp vào/ra vẫn luôn dựa vào dòng dưới/số, xem PlateMatcher.cs),
    # nên thà bỏ trống còn hơn hiện SAI: dòng trên thường bị chói/lóa ảnh
    # hưởng nặng hơn dòng dưới do gần mép trên biển, hướng phản chiếu ánh
    # sáng trực tiếp hơn. Khi confidence trung bình của riêng dòng trên dưới
    # ngưỡng này, bỏ hẳn dòng trên, chỉ giữ lại dòng dưới trong kết quả.
    LINE_MIN_CONFIDENCE = float(os.getenv("LINE_MIN_CONFIDENCE", "0.55"))

    # Số ký tự TỐI ĐA hợp lệ của dòng trên (tỉnh+series) biển xe máy 2 dòng -
    # thực tế chỉ có dạng "59V2" (2 số tỉnh + 1-2 chữ/số series) = tối đa 4
    # ký tự. Phát hiện ra NHIỀU HƠN số này gần như chắc chắn là nhiễu (vết
    # trầy/ốc vít/mép khung biển bị nhận nhầm thành ký tự - xem log thực tế
    # "59V2E 54411": dòng trên chỉ nên là "59V2", chữ "E" thừa là nhiễu) - tự
    # động bỏ (những) ký tự confidence thấp nhất cho tới khi còn đúng số này
    # (xem PlateOCR._trim_top_line).
    TOP_LINE_MAX_CHARS = int(os.getenv("TOP_LINE_MAX_CHARS", "4"))

    # Số ký tự TỐI THIỂU để còn coi dòng trên là "đọc được" - dưới mức này
    # (VD chỉ 1-2 ký tự lẻ tẻ) thà bỏ hẳn còn hơn hiện 1 mẩu thông tin vô
    # nghĩa (khác với thiếu ĐÚNG 1 ký tự cuối kiểu "59V" - vẫn còn giá trị
    # tham khảo, không bỏ). Dòng trên chỉ để HIỂN THỊ, không ảnh hưởng so
    # khớp vào/ra (luôn chỉ dùng dòng số - xem PlateMatcher.cs).
    TOP_LINE_MIN_CHARS = int(os.getenv("TOP_LINE_MIN_CHARS", "3"))

    # 🆕 BẬT LẠI đọc ĐỦ BIỂN (tỉnh+series+số) cho production - trước đây tắt
    # vì dòng trên hay đọc sai dưới điều kiện chói (xem log 290: "123" thay
    # vì "12B1"). Giờ bật lại vì AN TOÀN VỀ BẢN CHẤT dù dòng trên có thể vẫn
    # còn đọc sai đôi lúc:
    #   - So khớp vào/ra (PlateMatcher.cs) CHỈ VÀ LUÔN LUÔN dùng dòng số phía
    #     dưới (đã tự tách riêng đúng phần này dù PlateNumber có cả 2 dòng,
    #     xem PlateMatcher.NormalizeDigits) - dòng trên có đọc sai cũng KHÔNG
    #     BAO GIỜ gây cảnh báo lệch biển giả/bỏ sót cảnh báo thật.
    #   - Đã thêm TOP_LINE_MAX_CHARS/TOP_LINE_MIN_CHARS ở trên để loại ký tự
    #     nhiễu thừa/thiếu quá mức, cộng LINE_MIN_CONFIDENCE sẵn có - giảm hẳn
    #     tần suất dòng trên hiện SAI rành rành, dù không bắt được 100%
    #     trường hợp model tự tin nhưng đọc sai giá trị ký tự (khác với sai
    #     SỐ LƯỢNG ký tự).
    # Rủi ro còn lại CHỈ là hiển thị (dòng trên có thể thỉnh thoảng sai vài
    # ký tự), không phải rủi ro an ninh/nghiệp vụ. Tắt lại bằng
    # READ_FULL_PLATE=false qua env nếu thực tế vận hành thấy phiền.
    READ_FULL_PLATE = os.getenv("READ_FULL_PLATE", "true").lower() == "true"

    # Confidence tối thiểu để coi là "đọc được" và gửi về backend - dưới
    # ngưỡng này coi như thất bại, KHÔNG trả biển số (thà không có còn hơn
    # có nhưng sai, vì hệ thống dùng biển số để cảnh báo lệch vào/ra - mục
    # 1.3/4.6 tài liệu). Bản gốc để 0.3 - quá thấp, dễ gây cảnh báo sai.
    # Đã test xong ở 0.4 (chỉ để kiểm tra luồng cảnh báo lệch biển số) - đưa
    # về 0.5 làm mức vận hành thực tế: thấp hơn 0.55 gốc (đỡ từ chối oan biển
    # đọc đúng nhưng lệch định dạng nhẹ) nhưng vẫn đủ cao để lọc bớt kết quả
    # rác. Nên tinh chỉnh lại sau khi có đủ dữ liệu confidence thực tế từ biển
    # thật (không phải bản photocopy) qua vài ngày vận hành.
    MIN_SUCCESS_CONFIDENCE = float(os.getenv("MIN_SUCCESS_CONFIDENCE", "0.5"))

    # Timeout tải ảnh - đây là ảnh trong mạng LAN nội bộ (camera/backend
    # cùng công ty), không phải Internet, nên không cần timeout 30s như bản
    # gốc; 8s là dư nhiều rồi, tránh 1 lượt quét bị treo quá lâu nếu camera
    # phản hồi chậm bất thường.
    IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "8"))

    # Image - resize ảnh TẢI VỀ nếu vượt kích thước này (giữ tỉ lệ), áp dụng
    # trong main._download_image() trước khi đưa vào detect/OCR. Vừa tránh
    # tốn RAM/CPU vô ích với ảnh gốc quá lớn, vừa là 1 lớp phòng vệ trước ảnh
    # "bom giải nén" (file nén nhỏ nhưng giải mã ra ảnh khổng lồ).
    #
    # 🐛 FIX: bản gốc để 1280x720 - THẤP HƠN độ phân giải camera thực tế
    # đang dùng (main stream Hikvision, xem CamerasManagementPage/Stream=1),
    # nghĩa là ảnh bị cắt giảm chi tiết NGAY TRƯỚC khi detect/OCR, lãng phí
    # đúng phần độ phân giải cao mà camera cung cấp - ảnh hưởng trực tiếp
    # tới độ rõ nét vùng biển số sau khi crop (xem PlateDetector.detect(),
    # vùng crop lấy từ chính ảnh đã bị resize ở đây). Nâng lên 1920x1080
    # (2MP - độ phân giải main stream phổ biến của camera IP hiện tại).
    # Cho phép override qua env để dễ tinh chỉnh lại nếu sau này đổi camera
    # độ phân giải khác (vd 4MP=2560x1440) mà không cần sửa code.
    IMAGE_MAX_WIDTH = int(os.getenv("IMAGE_MAX_WIDTH", "1920"))
    IMAGE_MAX_HEIGHT = int(os.getenv("IMAGE_MAX_HEIGHT", "1080"))

    # Giới hạn dung lượng tải về cho 1 ảnh - chặn cả theo Content-Length khai
    # báo LẪN đếm byte thực tế trong lúc tải (phòng trường hợp header nói dối
    # hoặc không có). 15MB là dư nhiều cho ảnh camera thông thường.
    IMAGE_MAX_DOWNLOAD_BYTES = int(os.getenv("IMAGE_MAX_DOWNLOAD_BYTES", str(15 * 1024 * 1024)))

    # Khử méo ống kính (barrel/fisheye) TRƯỚC khi detect/OCR - camera góc rộng
    # (VD Camera Trước) bẻ cong đường thẳng ở rìa khung hình, làm hình dạng
    # ký tự trên biển số bị méo dù ảnh rõ nét/đủ sáng, khiến OCR đọc sai dù
    # mắt người vẫn đọc được nhờ tự bù trừ trong đầu. K1=0 (mặc định) = tắt
    # hẳn, không ảnh hưởng gì tới ảnh. Dò giá trị bằng mắt qua
    # tools/preview_undistort.py (không cần hiệu chỉnh camera chính xác bằng
    # bàn cờ caro) - K1 dương kéo méo kiểu "phình" (barrel) về thẳng.
    LENS_UNDISTORT_K1 = float(os.getenv("LENS_UNDISTORT_K1", "0.0"))
    LENS_UNDISTORT_K2 = float(os.getenv("LENS_UNDISTORT_K2", "0.0"))

    # API
    API_HOST = "0.0.0.0"
    API_PORT = 5000

    # Webhook
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://localhost:7069/api/ai/webhook/result")
    BACKEND_VERIFY_SSL = os.getenv("BACKEND_VERIFY_SSL", "false").lower() == "true"

    # Đường dẫn gốc của Backend C# - ghép vào ảnh nếu ImageUrls trả về dạng tương đối
    BASE_IMAGE_URL = os.getenv("BASE_IMAGE_URL", "https://localhost:7069")

    # Bật /api/test-batch (đọc thử 1 thư mục ảnh CỤC BỘ trên máy chạy service -
    # dùng để so sánh model cũ/mới bằng ảnh thật trước khi đưa vào production,
    # xem training/README). Endpoint này đọc TRỰC TIẾP filesystem theo đường
    # dẫn caller tự chỉ định - chỉ nên bật khi ai-service chạy ở máy dev/test,
    # KHÔNG bật trên máy đang phục vụ production (mặc định tắt, phải bật rõ
    # ràng qua env). Trả 404 (không phải 403) khi tắt để không lộ ra là có
    # endpoint này nếu ai đó dò quét.
    ENABLE_TEST_ENDPOINTS = os.getenv("ENABLE_TEST_ENDPOINTS", "false").lower() == "true"


config = Config()
