"""
Đọc biển số từ ảnh vùng biển đã crop (output của PlateDetector) - THAY THẾ
model YOLO detect-từng-ký-tự (character_detector.pt, xem plate_ocr.py).

🐛 LÝ DO ĐỔI (đo trên 220 ảnh biển thật đã gán nhãn tay, 09/2026):
character_detector.pt (model cộng đồng, train với degrees=0 - không hề thấy
chữ nghiêng lúc train) đọc đúng dãy số chỉ 118/220 ảnh và CHẤP NHẬN SAI 95/220
ảnh ("8756" thay vì "87656", "05922" thay vì "05972"...) - ngay cả trên biển
rõ nét, nó bỏ sót hẳn ký tự (không detect ra dù hạ conf xuống 0.1). Đây là
giới hạn của chính model, không sửa được bằng hậu xử lý.

Luồng mới:
  1. NẮN THẲNG biển (_rectify, _text_block_crop): tìm 4 góc vùng nền trắng
     của biển -> kéo phối cảnh về hình chữ nhật đứng thẳng; hoặc xoay thẳng
     theo khối ký tự. Xử lý được biển nghiêng/xoay (camera góc rộng đặt lệch,
     biển nghiêng 15-60°) và méo phối cảnh - thay cho việc thử xoay mù nhiều
     góc cố định như bản trước.
  2. TÁCH DÒNG (_split_lines): biển xe máy 2 dòng tách theo khe trắng giữa
     2 dòng chữ; biển ô tô 1 dòng giữ nguyên.
  3. ĐỌC TỪNG DÒNG bằng PaddleOCR text recognition (en_PP-OCRv5_mobile_rec,
     ~65ms/biển trên CPU) - model OCR dòng chữ train trên dữ liệu rất lớn, bền
     với mờ/chói/nghiêng nhẹ hơn hẳn detect-từng-ký-tự.
  4. KIỂM TRA CẤU TRÚC BIỂN VN (_parse_*): mã tỉnh 2 số + series + dãy số 4-5
     số; tự sửa nhầm lẫn hình dạng theo VỊ TRÍ (O->0, B->8 ở vị trí phải là
     số...). Kết quả SAI cấu trúc bị loại hẳn (thà "không đọc được" còn hơn
     trả biển sai - hệ thống dùng biển để cảnh báo lệch vào/ra).

Mỗi biển được đọc ở 3 biến thể hình học (nắn phối cảnh theo 4 góc / xoay
thẳng theo góc 4 góc / xoay thẳng theo khối ký tự) rồi BỎ PHIẾU dãy số; kết
quả không đồng thuận bị hạ confidence (từ chối) thay vì tự chọn bừa. Bộ biến
thể + luật chấp nhận được CHỌN BẰNG ĐO trên 219 ảnh biển thật đã gán nhãn
(camera tạm góc rộng, biển nghiêng 0-60°): 212 đúng dãy số / 2 sai được chấp
nhận (cả 2 là biển bị mép khung hình cắt mất ký tự cuối) - so với 118 đúng /
95 sai của character_detector.pt.

Định dạng trả về LUÔN là "<dòng trên> <dãy số>" (VD "12B1 05972", ô tô
"30A 12345") - khớp với PlateMatcher.cs phía backend (tách theo khoảng trắng
cuối cùng). Trước đây biển ô tô trả liền "30A12345" khiến backend lấy lẫn cả
mã tỉnh vào dãy số (7 số) -> không bao giờ so khớp được.
"""
import os
import re
import threading

import cv2
import numpy as np
from loguru import logger

from config import config

# Chữ cái hình dạng giống số - dùng ở vị trí CHẮC CHẮN là số (mã tỉnh, dãy số).
_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4",
}
# Chiều ngược lại - dùng ở vị trí CHẮC CHẮN là chữ (ký tự series đầu tiên).
_TO_LETTER = {"0": "D", "8": "B", "6": "G", "5": "S", "2": "Z", "4": "A", "7": "T", "1": "T"}
# Chữ KHÔNG dùng trên biển VN (I, J, O, Q, W) -> chữ hợp lệ gần giống nhất.
_INVALID_LETTER_FIX = {"O": "D", "Q": "D", "I": "T", "J": "U", "W": "V"}

_TOP_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d?$")
_BOTTOM_RE = re.compile(r"^\d{4,5}$")


def _clean(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _fix_top(raw, strict_series=False):
    """Ép dòng trên (mã tỉnh + series, VD "12B1", "29AA") về đúng cấu trúc
    theo vị trí. Trả về chuỗi đã sửa hoặc None nếu không thể là dòng trên.

    `strict_series`: ký tự series đầu PHẢI được đọc ra là chữ thật (không
    sửa số->chữ) - dùng cho biển 1 dòng, nơi chỉ có ký tự này phân biệt biển
    với 1 dãy số bất kỳ (VD dòng ngày giờ "25-08-2026" in trên ảnh camera sẽ
    thành biển "25D 82026" nếu cho sửa "0"->"D")."""
    s = _clean(raw)
    if not 3 <= len(s) <= 5:
        return None
    out = []
    for i, ch in enumerate(s):
        if i < 2:  # mã tỉnh: luôn là số
            ch = ch if ch.isdigit() else _TO_DIGIT.get(ch)
        elif i == 2:  # ký tự series đầu: luôn là chữ
            if ch.isalpha():
                ch = _INVALID_LETTER_FIX.get(ch, ch)
            else:
                ch = None if strict_series else _TO_LETTER.get(ch)
        else:  # ký tự sau: chữ (series 2 chữ) hoặc số - chỉ sửa chữ không hợp lệ
            # Ở vị trí này số hợp lệ nên chữ KHÔNG hợp lệ gần giống số được
            # sửa về SỐ (đo thực tế: "12-BI" là "12B1", không phải "12BT").
            ch = {"I": "1", "J": "1", "O": "0", "Q": "0"}.get(ch) or _INVALID_LETTER_FIX.get(ch, ch)
        if ch is None:
            return None
        out.append(ch)
    fixed = "".join(out)
    return fixed if _TOP_RE.match(fixed) else None


def _fix_bottom(raw):
    """Ép dãy số (4-5 số) - chữ giống số được sửa về số. None nếu sai.

    Bỏ rác ở 2 đầu trước khi kiểm tra: ký tự không thể là số (VD "V059.72")
    và viền dọc của biển bị đọc thành "1" (VD "1059.72" - dãy số biển VN tối
    đa 5 số nên 6 số có "1" ở đầu/cuối gần như chắc chắn là viền)."""
    s = _clean(raw)
    fixed = "".join(ch if ch.isdigit() else _TO_DIGIT.get(ch, "?") for ch in s).strip("?")
    if _BOTTOM_RE.match(fixed):
        return fixed
    if len(fixed) == 6 and fixed[0] == "1" and fixed[1:].isdigit():
        return fixed[1:]
    if len(fixed) == 6 and fixed[-1] == "1" and fixed[:-1].isdigit():
        return fixed[:-1]
    return None


_DATETIME_RE = re.compile(r"\d{1,2}[-/:]\d{1,2}[-/:]\d{2,4}")


def _parse_one_line(raw):
    """Biển ô tô 1 dòng "30A12345" -> ("30A", "12345"). Thử mọi điểm cắt
    series/dãy số, ưu tiên dãy số 5 số (biển mới) rồi 4 số. Loại hẳn chuỗi
    dạng ngày/giờ (dòng thời gian camera in đè lên ảnh hay bị detect nhầm)."""
    if _DATETIME_RE.search(raw or ""):
        return None, None
    s = _clean(raw)
    for bottom_len in (5, 4):
        if len(s) <= bottom_len + 2:
            continue
        top, bottom = _fix_top(s[:-bottom_len], strict_series=True), _fix_bottom(s[-bottom_len:])
        if top and bottom:
            return top, bottom
    return None, None


def _order_points(pts):
    """Sắp 4 góc biển theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái.

    🐛 FIX: bản đầu dùng tổng/hiệu tọa độ (cách phổ biến) - SAI khi biển
    nghiêng >= ~35°: chọn nhầm góc, biển bị nắn XOAY 90° (đọc ra "1234 | 29C"
    thay vì "29C-1234"). Sửa: sắp 4 điểm theo chiều kim đồng hồ quanh tâm,
    rồi chọn cạnh "trên" là 1 trong 2 cạnh DÀI (biển VN luôn rộng hơn cao, cả
    ô tô lẫn xe máy) có hướng gần với hướng sang phải nhất - đúng với mọi góc
    nghiêng dưới 90°."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # chiều kim đồng hồ (trục y hướng xuống)
    best, best_key = None, None
    for k in range(4):
        q = np.roll(pts, -k, axis=0)
        top, side = q[1] - q[0], q[2] - q[1]
        is_long = np.linalg.norm(top) >= np.linalg.norm(side)
        key = (is_long, top[0] / (np.linalg.norm(top) + 1e-6))  # ưu tiên cạnh dài, hướng sang phải
        if best_key is None or key > best_key:
            best, best_key = q, key
    return best.astype(np.float32)


class PlateReader:
    def __init__(self):
        self.config = config
        self.model = None
        self._infer_lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        try:
            # Không cho paddlex tự dò/tải model từ Internet lúc khởi động -
            # model đã để sẵn trong models/ (máy chạy thật có thể không có mạng).
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import TextRecognition

            self.model = TextRecognition(
                model_name=self.config.OCR_REC_MODEL_NAME,
                model_dir=str(self.config.OCR_REC_MODEL_DIR),
            )
            logger.info(f"Plate reader loaded ({self.config.OCR_REC_MODEL_NAME})")
        except Exception as e:
            logger.error(f"Failed to load plate reader OCR model: {e}")
            self.model = None

    # ---------- Hình học: nắn thẳng + tách dòng ----------

    @staticmethod
    def _find_plate_quad(crop):
        """Tìm 4 góc vùng nền sáng của biển trong ảnh crop. Trả về (4 góc,
        bao lồi vùng nền) hoặc None nếu không tìm được vùng đủ lớn/đủ giống
        hình chữ nhật."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Mở (erode->dilate) cắt các dải nối mảnh giữa nền biển và nền sáng
        # xung quanh (tường/sàn bê tông sáng dính sát mép biển).
        k = max(3, int(min(h, w) * 0.03)) | 1
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Ưu tiên vùng chứa TÂM ảnh (detector crop quanh biển, biển nằm giữa),
        # rồi mới tới vùng lớn nhất.
        center = (w / 2, h / 2)
        best = max(
            contours,
            key=lambda c: (cv2.pointPolygonTest(c, center, False) >= 0, cv2.contourArea(c)),
        )
        area = cv2.contourArea(best)
        if area < 0.08 * h * w:
            return None

        hull = cv2.convexHull(best)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull))
        quad = _order_points(quad)

        # Vùng tìm được phải "đầy" gần như cả tứ giác - tránh nhận nhầm 1 mảng
        # nền lồi lõm bất kỳ thành biển.
        if cv2.contourArea(quad) <= 0 or area / cv2.contourArea(quad) < 0.75:
            return None
        # Vùng nhỏ (< 20% ảnh crop) chỉ được nhận khi DÀI như biển ô tô: biển ô
        # tô nghiêng mạnh chỉ chiếm ~15% khung bao thẳng của nó, còn vùng nhỏ
        # gần vuông thường là mảng nền/rác (đo thực tế: hạ ngưỡng cho mọi hình
        # dạng làm tăng đọc nhầm vật không phải biển).
        tl, tr, br, bl = quad
        ratio = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(
            1e-6, max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        )
        if area < 0.2 * h * w and ratio < 2.5:
            return None
        return quad, hull

    def _rectify(self, crop):
        """Kéo phối cảnh vùng biển về hình chữ nhật đứng thẳng. Trả về
        (ảnh_đã_nắn, tỉ_lệ_w/h, góc_nghiêng_độ) hoặc (None, None, None) nếu
        không tìm được biển."""
        found = self._find_plate_quad(crop)
        if found is None:
            return None, None, None
        quad = found[0]

        # 🐛 Nới tứ giác ra ngoài 1 chút: vùng nền trắng tìm được nằm BÊN
        # TRONG viền biển, ký tự sát mép (VD số "0" đầu dãy số) bị cắt mất 1
        # phần -> OCR đọc nhầm "0" thành "7"/"1" (đo thực tế: "759.72" thay
        # vì "059.72"). Đệm ảnh trước để phần nới ra ngoài crop không bị đen.
        center = quad.mean(axis=0)
        quad = center + (quad - center) * (1.0 + self.config.OCR_QUAD_MARGIN)
        pad = int(max(crop.shape[:2]) * self.config.OCR_QUAD_MARGIN) + 2
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        quad = (quad + pad).astype(np.float32)

        tl, tr, br, bl = quad
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        if width < 20 or height < 20:
            return None, None, None
        ratio = width / height
        angle = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
        # Chuẩn hóa chiều cao để ký tự đủ lớn cho OCR (model rec resize về
        # cao 48px mỗi dòng - biển 2 dòng cần ~200px để mỗi dòng ~90px).
        out_h = 200 if ratio < self.config.PLATE_ASPECT_RATIO_THRESHOLD else 100
        out_w = int(round(out_h * ratio))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        warped = cv2.warpPerspective(padded, cv2.getPerspectiveTransform(quad, dst), (out_w, out_h))
        return warped, ratio, angle

    @staticmethod
    def _text_block_crop(crop):
        """Biến thể xoay thẳng theo KHỐI KÝ TỰ: lấy các vệt tối cỡ ký tự, hình
        chữ nhật xoay nhỏ nhất bao trọn chúng cho ra góc nghiêng + phạm vi
        chữ trên biển. Bền hơn _rectify khi nền quanh biển cũng sáng (sàn bê
        tông/tường trắng dính liền nền biển - vùng nền trắng khi đó lan ra
        ngoài biển, không tìm được 4 góc) vì chỉ dựa vào CHÍNH các ký tự."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        block = max(15, (min(h, w) // 4) | 1)
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        comps = []
        for i in range(1, n):
            x, y, cw, ch, _ = stats[i]
            if not (0.12 * h <= ch <= 0.5 * h and 0.02 * w <= cw <= 0.35 * w):
                continue  # không cỡ ký tự
            if not (0.9 <= ch / max(cw, 1) <= 6):
                continue  # ký tự biển số luôn cao hơn rộng
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue  # chạm mép crop - thường là nền/xe, không phải ký tự
            comps.append(i)
        if len(comps) < 4:
            return None, None
        median_h = np.median([stats[i][3] for i in comps])
        comps = [i for i in comps if 0.6 * median_h <= stats[i][3] <= 1.5 * median_h]
        if len(comps) < 4:
            return None, None

        pts = np.column_stack(np.where(np.isin(labels, comps)))[:, ::-1].astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        # Chuẩn hóa để rw là chiều ngang của khối chữ, góc xoay trong ±45°.
        if ang > 45:
            rw, rh, ang = rh, rw, ang - 90
        elif ang < -45:
            rw, rh, ang = rh, rw, ang + 90
        # Nới ra ngoài khối chữ (lề biển) - dọc nhiều hơn ngang vì khối chữ
        # sát mép trên/dưới của ký tự, còn OCR cần chút nền quanh chữ.
        rw *= 1.2
        rh *= 1.36
        pad = int(max(h, w) * 0.3)
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        m = cv2.getRotationMatrix2D((cx + pad, cy + pad), ang, 1.0)
        rotated = cv2.warpAffine(padded, m, (padded.shape[1], padded.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        x0, y0 = int(cx + pad - rw / 2), int(cy + pad - rh / 2)
        out = rotated[max(0, y0): y0 + int(rh), max(0, x0): x0 + int(rw)]
        if out.shape[0] < 20 or out.shape[1] < 20:
            return None, None
        return out, rw / rh

    @staticmethod
    def _rotate(img, angle):
        """Xoay ảnh `angle` độ quanh tâm, mở rộng khung để không mất góc."""
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += new_w / 2 - w / 2
        m[1, 2] += new_h / 2 - h / 2
        return cv2.warpAffine(img, m, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _split_lines(plate):
        """Tách biển 2 dòng thành [dòng trên, dòng dưới] tại hàng ÍT NÉT CHỮ
        nhất trong khoảng giữa biển (khe trắng giữa 2 dòng)."""
        h = plate.shape[0]
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        profile = cv2.blur(ink.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, max(3, h // 20))).ravel()
        lo, hi = int(h * 0.35), int(h * 0.65)
        split = lo + int(np.argmin(profile[lo:hi])) if hi > lo else h // 2
        return [plate[:split], plate[split:]]

    # ---------- OCR ----------

    def _recognize_lines(self, lines):
        lines = [ln for ln in lines if ln.shape[0] >= 8 and ln.shape[1] >= 8]
        if not lines:
            return []
        with self._infer_lock:
            outs = self.model.predict(lines)
        return [(o["rec_text"], float(o["rec_score"])) for o in outs]

    def _read_variant(self, plate, two_line):
        """Đọc 1 ảnh biển (đã/chưa nắn) -> dict kết quả đã kiểm tra cấu trúc.
        `two_line=None`: chưa biết bố cục (ảnh xoay mù - khung bao của biển ô
        tô 1 dòng nghiêng mạnh gần vuông như biển xe máy) -> đọc thử CẢ 2 bố
        cục, giữ kết quả đúng cấu trúc hơn."""
        if two_line is None:
            candidates = [r for r in (self._read_variant(plate, True), self._read_variant(plate, False)) if r]
            if not candidates:
                return None
            return max(candidates, key=lambda r: (r["bottom"] is not None, r["top"] is not None, r["bottom_score"]))
        if two_line:
            recs = self._recognize_lines(self._split_lines(plate))
            if len(recs) != 2:
                return None
            (top_raw, top_score), (bot_raw, bot_score) = recs
            top, bottom = _fix_top(top_raw), _fix_bottom(bot_raw)
            raw = f"{top_raw} | {bot_raw}"
        else:
            recs = self._recognize_lines([plate])
            if not recs:
                return None
            raw, score = recs[0]
            top, bottom = _parse_one_line(raw)
            top_score = bot_score = score
        return {
            "raw": raw,
            "top": top,
            "bottom": bottom,
            "top_score": top_score,
            "bottom_score": bot_score,
        }

    @staticmethod
    def _vote_bottom(values):
        """Bỏ phiếu dãy số: values = [(dãy_số, điểm)]. Lần đọc THIẾU đúng 1 ký
        tự so với 1 dãy dài hơn (VD "0597" so với "05972") được tính là ỦNG HỘ
        dãy dài hơn thay vì chống lại. Trả về (dãy thắng, số phiếu ủng hộ, điểm cao nhất của phiếu
        khớp đúng dãy thắng, có đối thủ ngang phiếu không tương thích)."""
        if not values:
            return None, 0, 0.0, False
        distinct = {v for v, _ in values}

        def supports(read, cand):
            # Đọc THIẾU đúng 1 ký tự bất kỳ (sát mép bị cắt: "0597"/"05972";
            # 2 số giống nhau liền nhau bị gộp: "5411"/"54411") vẫn là CÙNG 1
            # biển -> ủng hộ bản đầy đủ hơn.
            if read == cand:
                return True
            if len(read) != len(cand) - 1:
                return False
            return any(cand[:i] + cand[i + 1:] == read for i in range(len(cand)))

        # Phiếu "đọc thiếu" chỉ được tính khi dãy đầy đủ có ít nhất 1 lần đọc
        # ĐÚNG Y HỆT đủ chắc (>= 0.8) - không thì 2 lần đọc yếu/sai (VD "0597"
        # score 0.48 + "05977" score 0.69) tự "ủng hộ" nhau thành kết quả sai.
        exact_best = {c: max(sc for v, sc in values if v == c) for c in distinct}
        support = {
            c: sum(1 for v, _ in values if v == c or (exact_best[c] >= 0.8 and supports(v, c)))
            for c in distinct
        }
        weight = {c: sum(sc for v, sc in values if v == c or (exact_best[c] >= 0.8 and supports(v, c))) for c in distinct}
        winner = max(distinct, key=lambda c: (support[c], weight[c], len(c)))
        rival = any(c != winner and not supports(c, winner) and support[c] == support[winner] for c in distinct)
        best = max(sc for v, sc in values if v == winner)
        return winner, support[winner], best, rival

    def _vote_top(self, values):
        """Bỏ phiếu dòng trên (tỉnh+series, CHỈ HIỂN THỊ) - trả về (dòng trên,
        điểm) hoặc (None, 0.0) nếu KHÔNG ĐỦ CHẮC. Nhiều phiếu nhất thắng; hòa
        phiếu thì lấy cách đọc score cao nhất. Chỉ nhận khi >= 2 biến thể đọc
        giống nhau hoặc score >= OCR_SINGLE_VOTE_MIN_SCORE - thà chỉ hiện dãy
        số còn hơn hiện sai dòng trên."""
        support, best = {}, {}
        for v, sc in values:
            support[v] = support.get(v, 0) + 1
            best[v] = max(best.get(v, 0.0), sc)
        if not support:
            return None, 0.0
        winner = max(support, key=lambda v: (support[v], best[v]))
        if support[winner] < 2 and best[winner] < self.config.OCR_SINGLE_VOTE_MIN_SCORE:
            logger.debug(f"Dòng trên không đủ chắc ({dict(support)}) - chỉ trả dãy số")
            return None, 0.0
        return winner, best[winner]

    def read(self, crop, read_full_plate=None):
        """Đọc biển số từ ảnh crop. Trả về (plate_text, confidence) - chuỗi
        rỗng + 0.0 nếu không đọc ra được biển ĐÚNG CẤU TRÚC."""
        if self.model is None or crop is None or crop.size == 0:
            return "", 0.0
        full_plate = self.config.READ_FULL_PLATE if read_full_plate is None else read_full_plate
        two_line_ratio = self.config.PLATE_ASPECT_RATIO_THRESHOLD

        try:
            # Đọc cùng 1 biển ở nhiều biến thể hình học rồi BỎ PHIẾU - mỗi
            # biến thể có điểm yếu riêng (nắn phối cảnh sai khi 4 góc lệch;
            # xoay theo khối chữ không sửa được phối cảnh...), đồng thuận giữa
            # chúng đáng tin hơn hẳn 1 lần đọc đơn lẻ dù score cao. Bộ 3 biến
            # thể này được chọn bằng đo thực tế (xem docstring đầu file).
            variants = []
            warped, ratio, angle = self._rectify(crop)
            if warped is not None:
                two_line = ratio < two_line_ratio
                variants.append(("rectified", warped, two_line))
                if abs(angle) > 3:
                    variants.append(("deskewed", self._rotate(crop, angle), two_line))
            block, block_ratio = self._text_block_crop(crop)
            if block is not None:
                variants.append(("text_block", block, block_ratio < two_line_ratio))
            if not variants:
                # Không tìm được hình học biển (nền quanh biển sáng như biển,
                # biển nghiêng quá mạnh...) -> đọc ảnh gốc + xoay mù vài góc cố
                # định rồi bỏ phiếu như bình thường. Chỉ tốn thêm ở ca khó này.
                # Bố cục 1/2 dòng chưa biết (two_line=None) - xem _read_variant.
                variants.append(("raw", crop, None))
                for fallback_angle in self.config.OCR_FALLBACK_ANGLES:
                    variants.append((f"rot{fallback_angle:+d}", self._rotate(crop, fallback_angle), None))

            results = []
            for name, img, two_line in variants:
                res = self._read_variant(img, two_line)
                if res:
                    logger.debug(
                        f"[{name}] raw='{res['raw']}' -> top={res['top']} bottom={res['bottom']} "
                        f"score={res['top_score']:.2f}/{res['bottom_score']:.2f}"
                    )
                    results.append(res)
                # Dừng sớm: 2 biến thể đã đọc ra CÙNG dãy số, score cao -> đã
                # đủ điều kiện chấp nhận, biến thể thứ 3 không đổi được kết quả.
                good = [r["bottom"] for r in results if r["bottom"] and r["bottom_score"] >= self.config.OCR_SINGLE_VOTE_MIN_SCORE]
                if len(good) >= 2 and len(set(good)) == 1 and len({r["bottom"] for r in results if r["bottom"]}) == 1:
                    break

            bottom, support, conf, rival = self._vote_bottom(
                [(r["bottom"], r["bottom_score"]) for r in results if r["bottom"]]
            )
            if bottom is None:
                logger.info("Không đọc được dãy số biển đúng cấu trúc (4-5 số)")
                return "", 0.0

            # LUẬT CHẤP NHẬN: có dãy số KHÁC (không phải đọc thiếu) ngang phiếu,
            # hoặc chỉ 1 biến thể đọc ra mà score chưa thật cao -> KHÔNG CHẮC,
            # hạ confidence để main.py từ chối (thà "không đọc được" còn hơn trả
            # sai rồi báo lệch biển giả / cho qua nhầm).
            if rival or (support < 2 and conf < self.config.OCR_SINGLE_VOTE_MIN_SCORE):
                valid_bottoms = sorted({r["bottom"] for r in results if r["bottom"]})
                logger.info(
                    f"Kết quả không chắc chắn: '{bottom}' {support}/{len(results)} phiếu, "
                    f"score={conf:.2f}, các dãy số đọc được={valid_bottoms} - hạ confidence"
                )
                conf *= self.config.OCR_DISAGREE_PENALTY

            top = None
            if full_plate:
                top, top_score = self._vote_top([(r["top"], r["top_score"]) for r in results if r["top"]])
                if top and top_score < self.config.LINE_MIN_CONFIDENCE:
                    logger.info(f"Dòng trên confidence thấp ({top_score:.2f}) - bỏ dòng trên '{top}'")
                    top = None

            plate_text = f"{top} {bottom}" if top else bottom
            logger.debug(f"Plate reader: '{plate_text}', conf={conf:.2f} ({support}/{len(results)} phiếu)")
            return plate_text, float(conf)
        except Exception as e:
            logger.error(f"Plate reader error: {e}")
            return "", 0.0
