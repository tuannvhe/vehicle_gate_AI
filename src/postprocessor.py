from loguru import logger


class PlatePostProcessor:
    """
    Chuẩn hóa chuỗi biển số ĐỦ (tỉnh+series+số) - dọn ký tự lạ/khoảng trắng
    thừa, kiểm tra độ dài hợp lý.

    🆕 ĐỌC LẠI ĐỦ BIỂN: character_detector.pt tự phân loại đúng từng ký tự
    (số/chữ hợp lệ theo quy định biển VN) ngay từ đầu nên không cần các bảng
    ánh xạ chữ<->số như OCR chữ chung chung trước đây - chỉ cần lọc ký tự lạ
    (phòng hờ nhiễu) và validate độ dài. Việc SO KHỚP vào/ra (PlateMatcher.cs
    phía backend) vẫn chỉ dùng phần SỐ trong chuỗi này (đếm số trùng nhau),
    không quan tâm phần chữ - chuỗi đủ biển ở đây chỉ để HIỂN THỊ.
    """

    # Biển 1 dòng ngắn nhất kiểu "30A12345" (8 ký tự) tới biển 2 dòng dài
    # nhất kiểu "12B1 05972" (10 ký tự kể cả khoảng trắng).
    MIN_LENGTH = 7
    MAX_LENGTH = 11
    # Dòng trên bị bỏ do confidence thấp (xem plate_ocr.py) -> kết quả chỉ
    # còn lại dòng số (4-5 ký tự) - vẫn là kết quả HỢP LỆ (chủ động, không
    # phải lỗi), không nên bị phạt confidence như 1 chuỗi thiếu ký tự thật.
    FALLBACK_DIGITS_LENGTHS = (4, 5)

    @classmethod
    def postprocess(cls, plate_text: str, confidence: float):
        """Trả về (plate, confidence) - plate là chuỗi biển đã dọn ký tự lạ,
        confidence bị giảm 0.7x nếu độ dài bất thường (không loại bỏ hẳn -
        vẫn để PlateMatcher phía backend tự quyết có đủ số ký tự SỐ trùng để
        khớp hay không)."""
        try:
            plate = "".join(c for c in (plate_text or "") if c.isalnum() or c == " ").strip().upper()

            length_ok = (
                cls.MIN_LENGTH <= len(plate) <= cls.MAX_LENGTH
                or (plate.isdigit() and len(plate) in cls.FALLBACK_DIGITS_LENGTHS)
            )
            if not length_ok:
                logger.warning(f"Độ dài biển bất thường ({len(plate)} ký tự): '{plate}'")
                confidence = confidence * 0.7

            final_conf = min(confidence, 1.0)
            logger.debug(f"Final plate: '{plate}', conf={final_conf:.2f}")
            return plate, final_conf

        except Exception as e:
            logger.error(f"Postprocess error: {e}")
            return plate_text, confidence
