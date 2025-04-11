# Filename: scan_grade_50mcq.py

import cv2
import numpy as np

def find_corners_and_warp(image_path, output_size=(800, 1120)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_centers = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect_ratio = w / float(h)

            if area > 1000 and 0.8 <= aspect_ratio <= 1.2:
                cx, cy = x + w // 2, y + h // 2
                square_centers.append((cx, cy))

    if len(square_centers) < 4:
        print("❌ Không tìm đủ 4 ô vuông ở 4 góc.")
        return None

    h_img, w_img = gray.shape

    def nearest_to(x_ref, y_ref):
        return min(square_centers, key=lambda p: (p[0] - x_ref)**2 + (p[1] - y_ref)**2)

    corners = np.array([
        nearest_to(0, 0),
        nearest_to(w_img, 0),
        nearest_to(w_img, h_img),
        nearest_to(0, h_img)
    ], dtype="float32")

    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, output_size)
    print("✅ Đã căn chỉnh thành công.")
    return warped

def grade_mcq(warped_image, answer_key):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    roi = thresh[300:1100, 50:750]
    rows, cols = 50, 4
    box_h, box_w = roi.shape[0] // rows, roi.shape[1] // cols

    selected = []
    for r in range(rows):
        marks = []
        for c in range(cols):
            box = roi[r*box_h:(r+1)*box_h, c*box_w:(c+1)*box_w]
            count = cv2.countNonZero(box)
            marks.append(count)

        max_count = max(marks)
        if max_count < 100:  # Ngưỡng pixel đen để xác định ô tô (có thể điều chỉnh tùy thực tế)
            selected.append(-1)  # chưa tô
        else:
            selected.append(np.argmax(marks))

    # Đếm số đáp án đúng và loại bỏ câu chưa tô (-1)
    score = sum(1 for i in range(len(answer_key)) if selected[i] != -1 and selected[i] == answer_key[i])
    percent = (score / len(answer_key)) * 100
    num_blank = selected.count(-1)
    print(f"✅ Score: {score}/{len(answer_key)} ({percent}%)")
    print(f"📝 Số câu chưa tô: {num_blank}/{len(answer_key)}")
    return score, percent


# Import và hàm find_corners_and_warp, grade_mcq (giữ nguyên)...

def extract_digits(roi, num_digits, options=10):
    h, w = roi.shape
    box_h, box_w = h // options, w // num_digits
    digits = ""
    for i in range(num_digits):
        col = roi[:, i*box_w:(i+1)*box_w]
        counts = [cv2.countNonZero(col[j*box_h:(j+1)*box_h, :]) for j in range(options)]
        max_count = max(counts)
        if max_count < 50:
            digits += "X"
        else:
            digit = np.argmax(counts)
            digits += str(digit)
    return digits

def extract_id_and_code(warped_image):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    roi_sbd = thresh[100:300, 600:750]  # Căn chỉnh lại vùng này nếu cần thiết
    roi_code = thresh[100:300, 750:800]  # Căn chỉnh lại vùng này nếu cần thiết

    sbd = extract_digits(roi_sbd, 6)
    code = extract_digits(roi_code, 3)

    print(f"🆔 Số báo danh: {sbd}")
    print(f"📄 Mã đề: {code}")

# Main code (giữ nguyên...)


if __name__ == "__main__":
    image_path = "StudentInfo/50MCQ.jpg"
    answer_key = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    warped_image = find_corners_and_warp(image_path)

    if warped_image is not None:
        grade_mcq(warped_image, answer_key)
        extract_id_and_code(warped_image)
