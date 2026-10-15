import threading
from ultralytics import YOLO
from loguru import logger
from config import config


class PlateOCR:
    """
    Đọc ký tự trên biển số bằng model YOLO PHÁT HIỆN TỪNG KÝ TỰ
    (character_detector.pt, train riêng cho biển số VN: 0-9, A-Z trừ
    I/J/O/Q/W - đúng quy định chữ hợp lệ trên biển số VN) - THAY THẾ EasyOCR.

    🐛 LÝ DO ĐỔI: EasyOCR là OCR chữ viết CHUNG CHUNG (không hiểu ngữ cảnh
    biển số), rất nhạy với ảnh méo/mờ/nghiêng - suốt quá trình vận hành thực
    tế liên tục gặp lỗi đọc nhầm ký tự (0/U, méo hình do góc camera...) dù đã
    vá nhiều lớp ở postprocessor.py. Model detect-từng-ký-tự bản chất chỉ là
    nhận diện HÌNH DẠNG từng ký tự như 1 object riêng (giống việc detect biển
    số ở PlateDetector) - không "đọc chữ" theo ngữ cảnh nên bền hơn nhiều với
    ảnh không hoàn hảo, và tự nhiên loại trừ nhầm lẫn chữ/số nằm ngoài phạm
    vi hợp lệ (model chỉ có 31 class: 0-9 + các chữ được phép).

    🆕 ĐỌC ĐỦ BIỂN (tỉnh+series+số) - model character_detector.pt vốn đã có
    sẵn class cho cả CHỮ (A-Z trừ I/J/O/Q/W) lẫn SỐ nên không cần train lại,
    chỉ cần ghép đủ cả 2 dòng thay vì chỉ lấy dòng số như bản trước. Việc SO
    KHỚP vào/ra (PlateMatcher.cs phía backend) vẫn CHỈ so theo số ký tự SỐ
    trùng nhau như cũ (không quan tâm phần chữ) - vì chữ vẫn dễ đọc nhầm hơn
    số, giữ nguyên rủi ro gian lận đã thống nhất; đọc đủ biển chỉ để HIỂN
    THỊ cho dễ đối chiếu thủ công.

    Interface: `recognize(plate_image) -> (plate_text, confidence)`.
    Bản EasyOCR cũ lưu lại ở plate_ocr_easyocr_legacy.py.bak để đối chiếu/
    rollback nếu cần.
    """

    def __init__(self):
        self.config = config
        self.model = None
        self.class_names = {}
        self._plate_class_id = None
        # Ultralytics YOLO không đảm bảo an toàn khi gọi đồng thời từ nhiều
        # thread trên cùng 1 model instance (giống PlateDetector._infer_lock).
        self._infer_lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(str(self.config.CHARACTER_MODEL_PATH))
            self.class_names = self.model.names
            # Model gốc có thêm 1 class "license plates" (khung biển) ngoài
            # các ký tự - phải loại trừ khi ghép text, không thì bị dính rác
            # vào chuỗi kết quả.
            self._plate_class_id = next(
                (idx for idx, name in self.class_names.items() if "plate" in name.lower()),
                None,
            )
            logger.info(
                f"Character detector loaded successfully ({len(self.class_names)} classes)"
            )
        except Exception as e:
            logger.error(f"Failed to load character detector model: {e}")
            self.model = None

    def _prepare_for_ocr(self, plate_image):
        """Phóng to nếu ảnh quá nhỏ. Trả về ảnh BGR.

        🐛 FIX: trước đây đưa ảnh qua ImagePreprocessor.preprocess (xám ->
        làm nét -> nhị phân Otsu) - di sản từ thời EasyOCR. character_detector
        .pt được train trên ẢNH MÀU THẬT, không phải ảnh đen trắng; nhị phân
        hóa còn XÓA MẤT nét chữ ở vùng chói (thực tế: biển "12B1 05972" rõ nét
        chỉ còn đọc ra "05", mất hẳn "B1"/"7"). Đưa thẳng ảnh crop gốc vào
        model: đọc đúng "12B1 05972" trên cùng ảnh đó."""
        return self._upscale_if_small(plate_image, min_w=200, min_h=150)

    @staticmethod
    def _rotate(img, angle):
        """Xoay `img` quanh tâm `angle` độ (dương = ngược chiều kim đồng hồ),
        MỞ RỘNG khung để không cắt mất ký tự ở góc biển."""
        import cv2
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += new_w / 2 - w / 2
        m[1, 2] += new_h / 2 - h / 2
        return cv2.warpAffine(img, m, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _split_result(plate_text):
        """Tách 1 kết quả đọc thành (phần đầu, dãy số, dấu nối) - biển 2 dòng
        tách theo khoảng trắng ("12B1 05972" -> "12B1", "05972", " "), biển
        1 dòng lấy dãy số liên tục ở cuối ("30A12345" -> "30A", "12345", "")."""
        if " " in plate_text:
            top, bottom = plate_text.rsplit(" ", 1)
            return top, bottom, " "
        digits = len(plate_text) - len(plate_text.rstrip("0123456789"))
        return plate_text[: len(plate_text) - digits], plate_text[len(plate_text) - digits:], ""

    @classmethod
    def _vote(cls, results):
        """Chọn kết quả cuối từ các lần đọc ở nhiều góc xoay bằng BỎ PHIẾU
        (cộng dồn confidence) RIÊNG từng phần: dãy số (chỉ tính phiếu hợp lệ
        4-5 số) rồi tới phần tỉnh/series. KHÔNG chọn theo confidence cao
        nhất: đo trên 95 ảnh thật, ở góc xoay sai model thường đọc SAI nhưng
        RẤT TỰ TIN (VD "72B 05922" conf 0.9) - chọn theo conf làm kết quả tệ
        hơn cả không xoay; đồng thuận giữa các góc bền hơn hẳn.
        Không có phiếu dãy số hợp lệ nào -> trả kết quả conf cao nhất (giữ
        hành vi cũ, để postprocessor/backend tự đánh giá)."""
        results = [(t, c) for t, c in results if t]
        if not results:
            return "", 0.0

        bottom_votes = {}
        for text, conf in results:
            _, bottom, _ = cls._split_result(text)
            if 4 <= len(bottom) <= 5:
                bottom_votes[bottom] = bottom_votes.get(bottom, 0.0) + conf
        if not bottom_votes:
            return max(results, key=lambda r: r[1])

        bottom = max(bottom_votes, key=bottom_votes.get)
        agreeing = [(t, c) for t, c in results if cls._split_result(t)[1] == bottom]
        top_votes = {}
        for text, conf in results:
            top, _, _ = cls._split_result(text)
            if top:
                top_votes[top] = top_votes.get(top, 0.0) + conf
        top = max(top_votes, key=top_votes.get) if top_votes else ""
        # Phần đầu có thể lấy từ lần đọc ở góc khác với dãy số - biển 2 dòng
        # (có lần đọc nào tách dòng) thì luôn nối bằng khoảng trắng.
        sep = " " if any(" " in t for t, _ in results) else ""
        conf = sum(c for _, c in agreeing) / len(agreeing)
        return (f"{top}{sep}{bottom}" if top else bottom), conf

    @staticmethod
    def _upscale_if_small(img, min_w, min_h):
        """Phóng to `img` nếu nhỏ hơn (min_w, min_h)."""
        h, w = img.shape[:2]
        if w < min_w or h < min_h:
            scale = max(3.0, (min_w + 200) / max(w, 1), (min_h + 100) / max(h, 1))
            new_w, new_h = int(w * scale), int(h * scale)
            import cv2
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _detect_characters(self, image):
        """Chạy YOLO character detector trên `image`, trả về list ký tự đã
        phát hiện dạng dict {cx, cy, label, conf}, ĐÃ loại bỏ class "khung
        biển" nếu có, KHÔNG sắp xếp thứ tự (việc sắp xếp trái->phải/dòng
        trên-dưới làm ở recognize())."""
        with self._infer_lock:
            results = self.model(
                image,
                conf=self.config.CHARACTER_YOLO_CONFIDENCE,
                iou=self.config.CHARACTER_YOLO_IOU,
                # 🐛 FIX: NMS mặc định chỉ lọc trùng khung CHỒNG LẤN trong
                # CÙNG 1 class - 1 ký tự vật lý duy nhất (VD "9") có nét cong
                # dễ bị model detect ra 2 khung gần trùng nhau nhưng khác
                # class (VD 1 khung "9", 1 khung "2") vẫn lọt qua NMS mặc
                # định vì thuộc 2 class khác nhau, sinh ra ký tự thừa trong
                # kết quả (đã thấy thực tế: "05972" ra thành "059272" -
                # thừa 1 số "2" chồng lên đúng vị trí số "9"/"7"). agnostic_nms
                # lọc trùng BẤT KỂ class, chỉ giữ khung tin cậy cao nhất tại
                # mỗi vị trí - đúng bản chất vật lý: 2 ký tự thật KHÔNG BAO
                # GIỜ chồng lấn đáng kể lên nhau trên biển số.
                agnostic_nms=True,
                verbose=False,
            )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        chars = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id == self._plate_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            chars.append({
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "label": self.class_names.get(class_id, "?"),
                "conf": float(box.conf[0]),
            })

        # Log confidence TỪNG ký tự (không chỉ trung bình) - để có số liệu
        # thật khi cần tinh chỉnh lại CHARACTER_YOLO_CONFIDENCE (0.4 hiện chỉ
        # là ước lượng ban đầu, chưa có dữ liệu thực tế để chốt).
        logger.debug(
            "Ký tự phát hiện: "
            + ", ".join(f"{c['label']}({c['conf']:.2f})" for c in sorted(chars, key=lambda c: c["cx"]))
        )
        return chars

    @staticmethod
    def _chars_to_text(chars):
        """Sắp ký tự theo trục X (trái->phải) rồi ghép thành chuỗi."""
        ordered = sorted(chars, key=lambda c: c["cx"])
        text = "".join(c["label"] for c in ordered)
        avg_conf = sum(c["conf"] for c in ordered) / len(ordered) if ordered else 0.0
        return text, avg_conf

    @staticmethod
    def _split_two_lines(chars):
        """Tách ký tự đã phát hiện thành 2 nhóm dòng trên/dòng dưới bằng
        phân cụm 2 nhóm (K-Means 1 chiều trên tọa độ Y).

        🐛 FIX (lần 1 - đã bỏ): so mỗi ký tự với TRUNG VỊ (median) Y của
        toàn bộ ký tự - nếu biển nghiêng do góc camera, 1 ký tự rìa dòng
        dưới có thể bị đẩy cao hơn các ký tự CÙNG dòng, rơi nhầm qua "phe"
        dòng trên theo median (thực tế: số "0" đầu dòng dưới "05972" bị
        tính nhầm sang dòng trên, ra "012B1 5972" thay vì "12B1 05972").

        🐛 FIX (lần 2 - đã bỏ): thử tìm KHOẢNG CÁCH Y LỚN NHẤT giữa 2 ký tự
        liền kề (sau khi sắp theo Y) để làm ranh giới - nhưng cách này lại
        quá NHẠY VỚI 1 ĐIỂM DUY NHẤT: chỉ 1 ký tự bị lệch Y bất thường (do
        detect sai/nhiễu) có thể tạo ra 1 khoảng cách giả trông "lớn nhất"
        ở vị trí sai, làm tách lệch hẳn (quan sát trên ảnh mẫu: có ảnh tách
        1 ký tự dòng trên / 7 ký tự dòng dưới - rõ ràng vô lý).

        Sửa đúng: phân cụm 2 nhóm dựa trên TOÀN BỘ điểm dữ liệu (K-Means 1
        chiều, vài vòng lặp) - bền hơn hẳn với nhiễu/nghiêng vì mỗi điểm chỉ
        đóng góp 1 phần nhỏ vào việc xác định 2 tâm cụm, không có 1 điểm lẻ
        nào đủ sức tự quyết định toàn bộ ranh giới.
        """
        if len(chars) < 2:
            return chars, []

        cys = sorted(c["cy"] for c in chars)
        center_top, center_bottom = cys[0], cys[-1]
        top_group, bottom_group = chars, []

        for _ in range(10):
            top_group = [c for c in chars if abs(c["cy"] - center_top) <= abs(c["cy"] - center_bottom)]
            bottom_group = [c for c in chars if abs(c["cy"] - center_top) > abs(c["cy"] - center_bottom)]
            if not top_group or not bottom_group:
                break
            new_top = sum(c["cy"] for c in top_group) / len(top_group)
            new_bottom = sum(c["cy"] for c in bottom_group) / len(bottom_group)
            if new_top == center_top and new_bottom == center_bottom:
                break
            center_top, center_bottom = new_top, new_bottom

        # Đảm bảo "top_group" luôn là nhóm Y NHỎ HƠN (dòng trên thật sự) -
        # vòng lặp trên không tự đảm bảo thứ tự này.
        if center_top > center_bottom:
            top_group, bottom_group = bottom_group, top_group

        return top_group, bottom_group

    @staticmethod
    def _trim_top_line(chars, max_chars):
        """Cắt bớt dòng trên (tỉnh+series) nếu phát hiện NHIỀU KÝ TỰ HƠN mức
        hợp lệ tối đa (VD "59V2E" trong khi biển thật chỉ có "59V2" 4 ký tự -
        chữ "E" thừa là nhiễu: vết trầy/ốc vít/khung biển bị nhận nhầm thành
        ký tự). Giữ lại đúng `max_chars` ký tự có CONFIDENCE CAO NHẤT (ký tự
        nhiễu thường mờ/không rõ nét hơn ký tự thật nên tự nhiên có confidence
        thấp hơn), rồi sắp lại theo trục X như bình thường."""
        if len(chars) <= max_chars:
            return chars
        # so bằng id(), KHÔNG bằng "in" (so giá trị dict) - 2 ký tự khác nhau
        # tình cờ có cùng label/conf/tọa độ (hiếm nhưng không phải không thể)
        # sẽ khiến "in" so trùng nhầm, làm dropped tính sai/tính thiếu.
        kept = sorted(chars, key=lambda c: c["conf"], reverse=True)[:max_chars]
        kept_ids = {id(c) for c in kept}
        dropped = [c for c in chars if id(c) not in kept_ids]
        logger.info(
            f"Dòng trên có {len(chars)} ký tự > {max_chars} hợp lệ - bỏ ký tự nghi nhiễu: "
            + ", ".join(f"{c['label']}({c['conf']:.2f})" for c in dropped)
        )
        return kept

    @classmethod
    def _apply_top_line_structure(cls, top_chars):
        """Ép cấu trúc CỐ ĐỊNH của dòng trên biển xe máy (tỉnh+series) theo
        đúng quy định biển số VN:
          - 2 ký tự ĐẦU (mã tỉnh) LUÔN LÀ SỐ.
          - Ký tự THỨ 3 LUÔN LÀ CHỮ - series chỉ có dạng "chữ+chữ" (VD "AA",
            "AB") hoặc "chữ+số" (VD "B1"), KHÔNG BAO GIỜ "số+chữ" (VD "1B"
            là sai cấu trúc, không tồn tại thực tế).
          - Ký tự thứ 4 (nếu có) KHÔNG ràng buộc - chữ hoặc số đều hợp lệ.
        SỬA được khi ký tự tại đúng vị trí đó thuộc dạng dễ nhầm hình dạng
        (DIGIT_CONFUSION_MAP/ngược lại). Trả về (chars_đã_sửa, is_valid) -
        `is_valid=False` khi CÓ vị trí bắt buộc mà ký tự hoàn toàn KHÔNG LIÊN
        QUAN gì tới loại cần (VD chữ "E" ở vị trí mã tỉnh - không có số nào
        trông giống "E" để mà đoán) - lúc đó COI CẢ DÒNG TRÊN LÀ KHÔNG ĐÁNG
        TIN (model đã detect sai hẳn loại ký tự, không phải chỉ nhầm hình
        dạng gần giống), nên bỏ hẳn ở recognize() thay vì hiện 1 chuỗi rõ
        ràng sai cấu trúc biển VN (VD "E5L2" - "E" không thể là mã tỉnh)."""
        ordered = sorted(top_chars, key=lambda c: c["cx"])
        fixed = []
        valid = True
        for i, c in enumerate(ordered):
            if i in (0, 1):
                new_label = cls._as_digit_label(c["label"])
            elif i == 2:
                new_label = cls._as_letter_label(c["label"])
            else:
                new_label = c["label"]
            if new_label is None:
                valid = False
                new_label = c["label"]
            fixed.append({**c, "label": new_label})
        return fixed, valid

    # Chữ cái HÌNH DẠNG giống hệt 1 số nào đó trên mặt biển phản quang/dập
    # nổi - dùng để TỰ SỬA khi 1 vị trí ĐÃ BIẾT CHẮC THEO CẤU TRÚC BIỂN VN
    # phải là số (dòng dưới biển xe máy, dãy số cuối biển ô tô) mà model lại
    # phân loại nhầm sang chữ. VD thực tế: "8" đọc nhầm ra "B", "2" đọc nhầm
    # ra "Z" (và ngược lại). CHỈ dùng ở những vị trí chắc chắn là số - tuyệt
    # đối KHÔNG áp dụng cho dòng trên/biển ô tô đọc đủ (ở đó chữ là hợp lệ,
    # ép về số sẽ sai theo chiều ngược lại).
    DIGIT_CONFUSION_MAP = {
        "B": "8",
        "D": "0",
        "S": "5",
        "Z": "2",
        "G": "6",
    }

    @classmethod
    def _as_digit_label(cls, label):
        """Trả về ký tự SỐ tương ứng nếu `label` vốn đã là số hoặc là 1 chữ
        cái hay bị nhận nhầm với số (DIGIT_CONFUSION_MAP) - None nếu không
        map được (chữ cái không liên quan gì tới số, VD "V", "N")."""
        if label.isdigit():
            return label
        return cls.DIGIT_CONFUSION_MAP.get(label)

    @classmethod
    def _as_letter_label(cls, label):
        """Chiều NGƯỢC LẠI của _as_digit_label - trả về CHỮ tương ứng nếu
        `label` vốn đã là chữ hoặc là 1 SỐ hay bị nhận nhầm với chữ (dùng
        đúng DIGIT_CONFUSION_MAP nhưng đảo chiều: "8"->"B", "0"->"D",...) -
        None nếu là số không liên quan gì tới chữ nào (VD "3", "7")."""
        if not label.isdigit():
            return label
        for letter, digit in cls.DIGIT_CONFUSION_MAP.items():
            if digit == label:
                return letter
        return None

    @classmethod
    def _trailing_digits(cls, chars):
        """Lấy DÃY SỐ LIÊN TỤC ở CUỐI (sắp theo trục X) - dùng cho biển 1
        dòng (ô tô) khi READ_FULL_PLATE=False: "30A12345" chỉ lấy "12345"
        (serial), bỏ "30A" (tỉnh+series). Duyệt ngược từ ký tự cuối, dừng
        ngay khi gặp ký tự KHÔNG THỂ là số (kể cả sau khi đã thử sửa theo
        DIGIT_CONFUSION_MAP) - ký tự sửa được thì lấy label đã sửa, không
        giữ nguyên chữ sai."""
        ordered = sorted(chars, key=lambda c: c["cx"])
        trailing = []
        for c in reversed(ordered):
            digit_label = cls._as_digit_label(c["label"])
            if digit_label is None:
                break
            trailing.append({**c, "label": digit_label})
        trailing.reverse()
        return trailing

    @classmethod
    def _force_digits_line(cls, chars):
        """Ép 1 dòng ĐÃ BIẾT CHẮC toàn số (dòng dưới biển xe máy) về đúng ký
        tự số - chữ cái hình dạng giống số (DIGIT_CONFUSION_MAP) được SỬA
        LẠI thành số tương ứng, thay vì giữ nguyên chữ sai (sẽ bị tầng so
        khớp phía backend LỌC BỎ hẳn ký tự đó - PlateMatcher.NormalizeDigits
        chỉ giữ char.IsDigit - làm THIẾU 1 ký tự thay vì đúng nhưng khác vị
        trí, dễ trượt ngưỡng độ dài 4-5 hoặc khớp tuyệt đối mới).

        🐛 FIX: ký tự KHÔNG map được về số (không phải số, cũng không thuộc
        DIGIT_CONFUSION_MAP - VD nhiễu từ logo/watermark/vết trầy lọt vào
        vùng crop) trước đây bị GIỮ NGUYÊN, làm rác lẫn vào chuỗi cuối (VD
        thực tế: "82758E" thay vì đúng "82758" - ký tự "E" nhiễu không thuộc
        map). Vì dòng này ĐÃ CHẮC CHẮN chỉ toàn số theo cấu trúc biển VN, 1
        ký tự hoàn toàn không liên quan gì tới số chỉ có thể là nhiễu - BỎ
        HẲN thay vì giữ, giống cách dòng trên đã lọc nhiễu qua _trim_top_line."""
        result = []
        dropped = []
        for c in chars:
            digit_label = cls._as_digit_label(c["label"])
            if digit_label is not None:
                result.append({**c, "label": digit_label})
            else:
                dropped.append(c)
        if dropped:
            logger.info(
                "Dòng dưới có ký tự không thể là số (nghi nhiễu) - bỏ: "
                + ", ".join(f"{c['label']}({c['conf']:.2f})" for c in dropped)
            )
        return result

    def _assemble(self, chars, aspect_ratio, effective_full_plate):
        """Ghép list ký tự đã phát hiện thành (plate_text, confidence) theo
        bố cục biển 1 dòng (ô tô) / 2 dòng (xe máy) - xem recognize()."""
        if aspect_ratio >= self.config.PLATE_ASPECT_RATIO_THRESHOLD:
            # ---- Biển 1 dòng (ô tô) ----
            if effective_full_plate:
                logger.debug(
                    f"aspect_ratio={aspect_ratio:.2f} >= "
                    f"{self.config.PLATE_ASPECT_RATIO_THRESHOLD} -> biển 1 dòng (ô tô), ghép toàn bộ"
                )
                # 🐛 FIX: trước đây ghép toàn bộ ký tự NGUYÊN VĂN không qua
                # sửa lỗi nhầm hình dạng (VD "8"->"B") - khác biệt xử lý so
                # với dòng dưới biển xe máy (luôn ép về số qua
                # _force_digits_line). Vì PlateMatcher.NormalizeDigits phía
                # backend chỉ giữ ký tự SỐ khi so khớp, 1 ký tự chữ lẫn
                # trong dãy serial cuối (vốn PHẢI là số) sẽ bị LỌC BỎ hẳn -
                # làm thiếu 1 ký tự, dễ trượt so khớp vào/ra. Sửa: vẫn ép
                # dãy số cuối (serial) về đúng ký tự số như nhánh
                # read_full_plate=False, chỉ khác là GIỮ NGUYÊN phần đầu
                # (tỉnh+series, có thể là chữ) thay vì bỏ hẳn.
                trailing = self._trailing_digits(chars)
                trailing_ids = {id(c) for c in trailing}
                prefix = [c for c in chars if id(c) not in trailing_ids]
                plate_text, conf = self._chars_to_text(prefix + trailing)
            else:
                logger.debug(
                    f"aspect_ratio={aspect_ratio:.2f} >= "
                    f"{self.config.PLATE_ASPECT_RATIO_THRESHOLD} -> biển 1 dòng (ô tô), lấy serial cuối"
                )
                plate_text, conf = self._chars_to_text(self._trailing_digits(chars))
        else:
            # ---- Biển 2 dòng (xe máy) ----
            logger.debug(
                f"aspect_ratio={aspect_ratio:.2f} < "
                f"{self.config.PLATE_ASPECT_RATIO_THRESHOLD} -> biển 2 dòng (xe máy)"
            )
            top_chars, bottom_chars = self._split_two_lines(chars)
            bottom_chars = self._force_digits_line(bottom_chars)
            bottom_text, bottom_conf = self._chars_to_text(bottom_chars)

            if not effective_full_plate:
                plate_text, conf = bottom_text, bottom_conf
            else:
                top_chars = self._trim_top_line(top_chars, self.config.TOP_LINE_MAX_CHARS)
                top_chars, top_structure_ok = self._apply_top_line_structure(top_chars)
                if not top_structure_ok:
                    logger.info(
                        "Dòng trên sai cấu trúc biển VN (có ký tự không thể là "
                        f"số/chữ theo đúng vị trí) - bỏ dòng trên: "
                        + "".join(c["label"] for c in sorted(top_chars, key=lambda c: c["cx"]))
                    )
                    top_chars = []
                top_text, top_conf = self._chars_to_text(top_chars)

                # 🐛 FIX: dòng trên (tỉnh+series) nằm sát mép trên biển,
                # thực tế hứng chói/lóa nặng hơn dòng dưới (số serial)
                # rất nhiều (VD ảnh trời nắng gắt) - trước đây tính
                # CHUNG 1 confidence trung bình cho cả biển, dòng trên
                # đọc sai/rác vẫn lọt qua vì dòng dưới kéo trung bình
                # lên cao. Đánh giá RIÊNG dòng trên: nếu confidence
                # trung bình của chính nó quá thấp, bỏ hẳn (chỉ giữ dòng
                # dưới) thay vì hiện chữ sai gây hiểu nhầm - dòng dưới
                # vẫn luôn là phần quyết định SO KHỚP vào/ra nên không
                # bị ảnh hưởng gì khi bỏ dòng trên.
                if top_chars and top_conf < self.config.LINE_MIN_CONFIDENCE:
                    logger.info(
                        f"Dòng trên confidence thấp ({top_conf:.2f} < "
                        f"{self.config.LINE_MIN_CONFIDENCE}) - bỏ dòng trên ('{top_text}'), chỉ giữ dòng dưới"
                    )
                    top_text = ""
                elif top_chars and len(top_chars) < self.config.TOP_LINE_MIN_CHARS:
                    # Quá ít ký tự (VD chỉ 1-2 ký tự lẻ tẻ) để còn là 1
                    # dòng biển có nghĩa - khác với thiếu ĐÚNG 1 ký tự
                    # cuối (VD "59V" thay vì "59V2") vẫn còn giá trị tham
                    # khảo nên KHÔNG bị bỏ ở đây.
                    logger.info(
                        f"Dòng trên chỉ có {len(top_chars)} ký tự < "
                        f"{self.config.TOP_LINE_MIN_CHARS} - bỏ dòng trên ('{top_text}'), chỉ giữ dòng dưới"
                    )
                    top_text = ""

                plate_text = f"{top_text} {bottom_text}".strip()
                kept_chars = bottom_chars if not top_text else top_chars + bottom_chars
                conf = sum(c["conf"] for c in kept_chars) / len(kept_chars) if kept_chars else 0.0

        return plate_text, conf

    def recognize(self, plate_image, read_full_plate=None):
        """
        Nhận dạng biển số - trả về (plate_text, confidence).

        `read_full_plate`: ghi đè `config.READ_FULL_PLATE` CHO ĐÚNG 1 LẦN GỌI
        NÀY - None (mặc định) thì dùng nguyên giá trị global trong config
        (đúng hành vi production hiện tại). Dùng để API test (VD
        /api/test-batch, hoặc /api/process khi caller gửi kèm ReadFullPlate)
        có thể xem thử đọc ĐỦ BIỂN mà KHÔNG cần đổi config.READ_FULL_PLATE
        global - vốn đang cố ý để False cho luồng thật vì dòng trên còn đọc
        sai nhiều dưới điều kiện chói (xem giải thích ở config.py).

        Tùy giá trị hiệu lực (`read_full_plate` nếu có, không thì
        `config.READ_FULL_PLATE`):
        - True (mặc định hiện tại, xem config.READ_FULL_PLATE): đọc ĐỦ BIỂN
          (tỉnh+series+số).
        - False: chỉ đọc PHẦN SỐ (dòng dưới của biển 2 dòng, hoặc dãy số
          cuối của biển 1 dòng) - ổn định hơn phần chữ dưới điều kiện chói
          thực tế, dùng khi cần tắt lại đọc đủ biển qua env.

        QUAN TRỌNG: biển số Việt Nam có 2 dạng bố cục khác hẳn nhau - biển ô
        tô 1 DÒNG (vd "30A-123.45") và biển xe máy 2 DÒNG (dòng trên "66F1",
        dòng dưới "876.90"). Model YOLO phát hiện biển (PlateDetector) chỉ có
        1 class "license_plate", KHÔNG phân biệt được 2 dạng này khi detect -
        nên đo tỉ lệ khung hình (aspect ratio) của vùng biển đã detect để tự
        nhận biết 1 dòng hay 2 dòng - biển ô tô dẹt ngang (~4-5 lần rộng hơn
        cao), biển xe máy gần vuông (~1-1.4 lần).

        Detect ký tự trên cả tấm biển (thử xoay theo OCR_DESKEW_ANGLES, giữ
        góc cho kết quả tốt nhất), sau đó PHÂN CỤM theo tọa độ Y thực tế
        (K-Means) để tách dòng trên/dưới cho biển 2 dòng.
        """
        if self.model is None:
            return "", 0.0

        effective_full_plate = (
            self.config.READ_FULL_PLATE if read_full_plate is None else read_full_plate
        )

        try:
            processed = self._prepare_for_ocr(plate_image)
            h, w = processed.shape[:2]
            aspect_ratio = w / h if h > 0 else 0

            # 🆕 Camera góc rộng đặt lệch -> biển xe máy trong ảnh thường
            # NGHIÊNG 15-20°, model ký tự đọc kém hẳn với chữ nghiêng (thực
            # tế: không xoay ra "32", xoay 20° ra "12B1 05972"). Đọc ở từng
            # góc trong OCR_DESKEW_ANGLES rồi BỎ PHIẾU (xem _vote).
            results = []
            for angle in self.config.OCR_DESKEW_ANGLES:
                image = processed if angle == 0 else self._rotate(processed, angle)
                chars = self._detect_characters(image)
                if not chars:
                    continue
                text, conf = self._assemble(chars, aspect_ratio, effective_full_plate)
                logger.debug(f"Góc {angle:+d}°: '{text}', conf={conf:.2f}")
                results.append((text, conf))

            if not results:
                logger.info("Character detector không phát hiện được ký tự nào")
                return "", 0.0

            plate_text, conf = self._vote(results)
            logger.debug(f"Character detect: '{plate_text}', conf={conf:.2f}")
            return plate_text, conf

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0
