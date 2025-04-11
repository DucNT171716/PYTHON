import cv2
from PIL import Image
import numpy as np
import easyocr
import os

reader = easyocr.Reader(['vi', 'en'])

def enhance_image_for_vietnamese(image_path, output_path=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)

    white_pixels = np.sum(dilated == 255)
    black_pixels = np.sum(dilated == 0)
    if white_pixels < black_pixels:
        dilated = cv2.bitwise_not(dilated)
    if output_path:
        cv2.imwrite(output_path, dilated)
    return dilated

def extract_vietnamese_text(image_path):
    processed_img = enhance_image_for_vietnamese(image_path)
    results = reader.readtext(processed_img)
    
    full_text = ''
    confidences = []

    for (bbox, text, prob) in results:
        full_text += text + '\n'
        confidences.append(prob)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    print(f"EasyOCR Confidence: {avg_conf * 100:.2f}%")

    return full_text.strip()

def extract_text_by_zones(image_path):
    import re

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Có thể chỉnh các zone này một chút để cải thiện kết quả
    zones = {
        "student_id": (0.13, 0.23, 0.10, 0.05),  # x, y, w, h
        "score": (0.05, 0.60, 0.15, 0.15)
    }

    results = {}

    for zone_name, (x_pct, y_pct, w_pct, h_pct) in zones.items():
        x = int(width * x_pct)
        y = int(height * y_pct)
        w = int(width * w_pct)
        h = int(height * h_pct)

        zone_img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        zone_results = reader.readtext(enhanced)
        zone_text = zone_results[0][1] if zone_results else ''

        # 💡 Làm sạch text
        if zone_name == "student_id":
            cleaned = re.sub(r'[^0-9]', '', zone_text)  # Giữ lại số
            results[zone_name] = cleaned.zfill(3)  # Đảm bảo có đủ 3 số

        elif zone_name == "score":
            match = re.search(r'\d+[.,]?\d*', zone_text)
            if match:
                score = match.group(0).replace('.', ',')  # Đổi dấu chấm thành phẩy nếu cần
                results[zone_name] = score
            else:
                results[zone_name] = ''

        # Vẽ vùng zone lên ảnh nếu cần debug
        cv2.imwrite(f"zone_{zone_name}.png", zone_img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("annotated_zones_easyocr.png", img)
    return results

# Example usage
if __name__ == "__main__":
    image_path = "SourceCode/PythonModule/StudentInfo/hocSinh2.jpg"

    print("Extracting full text with EasyOCR...")
    text = extract_vietnamese_text(image_path)
    print("\nExtracted Text:")
    print(text)

    print("\nExtracting text by zones with EasyOCR...")
    zone_results = extract_text_by_zones(image_path)
    print("\nZone Results:")
    for zone, text in zone_results.items():
        print(f"{zone}: {text}")
