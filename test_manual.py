# test_manual.py
import cv2
import numpy as np
import requests
from src.plate_detector import PlateDetector
from src.plate_ocr import PlateOCR
from src.postprocessor import PlatePostProcessor

# Khởi tạo các model
detector = PlateDetector()
ocr = PlateOCR()
postprocessor = PlatePostProcessor()

# Đường dẫn ảnh tuyệt đối
image_path = "C:/Vehicle_Gate_Management/Vehicle_Gate_Management/wwwroot/photos/2026/08/06/lane1/121_cam0_639216032762316759.jpg"

# Đọc ảnh bằng OpenCV
image = cv2.imread(image_path)

# 1. Detect plate
plate_img, detect_conf = detector.detect(image)
print(f"Detect conf: {detect_conf}")

if plate_img is not None:
    # 2. OCR
    ocr_text, ocr_conf = ocr.recognize(plate_img)
    print(f"OCR text: {ocr_text}, conf: {ocr_conf}")

    # 3. Postprocess
    plate_number, final_conf = postprocessor.postprocess(ocr_text, ocr_conf)
    print(f"Final plate: {plate_number}, conf: {final_conf}")