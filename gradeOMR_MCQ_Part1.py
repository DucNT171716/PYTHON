from imutils.perspective import four_point_transform
import imutils
from scipy.spatial import distance as dist
import cv2
import numpy as np
from click_markers import remove_black_squares

# PHẦN I: Đáp án đúng mẫu (A=0, B=1, C=2, D=3)
answer_key_part1 = [
    1, 0, 2, 3, 1, 0, 2, 2, 3, 1,
    1, 2, 3, 1, 0, 2, 1, 3, 2, 0,
    0, 1, 3, 2, 1, 0, 3, 2, 2, 0,
    1, 1, 3, 1, 2, 0, 3, 1, 2, 0
]

answer_key_part2 = ["Đúng", "Sai", "Đúng", "Đúng", "Sai", "Sai", "Đúng", "Sai"]
answer_key_part3 = ['1,23', '3,14', '0,50', '2,35', '5,00', '0,75']

def find_corners_and_warp(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docCnt = None

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        return None, "❌ Không tìm thấy contour tờ giấy."

    warped = four_point_transform(orig, docCnt.reshape(4, 2))
    return warped, "✅ Đã căn chỉnh chính xác theo contour tờ giấy."





def extract_digits(thresh, roi, num_digits=6, num_options=10):
    x, y, w, h = roi
    region = thresh[y:y+h, x:x+w]
    digit_width = w // num_digits
    extracted_number = ""
    for i in range(num_digits):
        digit_img = region[:, i * digit_width:(i + 1) * digit_width]
        cell_height = h // num_options
        detected_digit = None
        for j in range(num_options):
            cell = digit_img[j * cell_height:(j + 1) * cell_height, :]
            if np.count_nonzero(cell == 0) > 200:
                detected_digit = str(j)
                break
        extracted_number += detected_digit if detected_digit else "X"
    return extracted_number

def extract_sbd(thresh):
    # Cột bên trái (7) trong ảnh: tọa độ x ~ 100, width ~ 160
    # Dòng ~ 90–170
    roi_sbd = (100, 90, 160, 80)
    return extract_digits(thresh, roi_sbd, num_digits=6)

def extract_ma_de(thresh):
    # Cột bên phải (8): tọa độ x ~ 600, width ~ 80
    roi_ma_de = (600, 90, 80, 80)
    return extract_digits(thresh, roi_ma_de, num_digits=3)
def get_x_area(c):
    x, y, w, h = cv2.boundingRect(c)
    return x * y

def extract_answer_blocks_part1(thresh_img):
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=get_x_area)
    
    blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 50000:  # chỉ lấy những khối đủ lớn
            check_min = x * y - x_old * y_old
            check_max = (x + w) * (y + h) - (x_old + w_old) * (y_old + h_old)
            if len(blocks) == 0 or (check_min > 20000 and check_max > 20000):
                roi = thresh_img[y:y+h, x:x+w]
                blocks.append((roi, (x, y, w, h)))
                x_old, y_old, w_old, h_old = x, y, w, h

    return sorted(blocks, key=lambda b: b[1][0])

def grade_part1_fixed(thresh_img):
    h, w = thresh_img.shape
    # Cắt phần PHẦN I từ khoảng 23%–64% chiều cao và 10%–90% chiều rộng
    roi = thresh_img[420:1150, 130:990]
    
    rows, cols = 40, 4
    box_h, box_w = roi.shape[0] // rows, roi.shape[1] // cols

    answers = []
    print("🔍 Bắt đầu chấm PHẦN I (40 câu):")
    for r in range(rows):
        marks = []
        for c in range(cols):
            x1, y1 = c * box_w, r * box_h
            x2, y2 = (c + 1) * box_w, (r + 1) * box_h
            cell = roi[y1:y2, x1:x2]
            val = cv2.countNonZero(cell)
            marks.append(val)
        max_val = max(marks)
        max_idx = marks.index(max_val)
        avg_other = (sum(marks) - max_val) / 3
        if max_val < 5 or max_val < avg_other * 1.05:
            answers.append(-1)
            print(f"Câu {r+1:02}: Không rõ")
        else:
            answers.append(max_idx)
            print(f"Câu {r+1:02}: {'ABCD'[max_idx]}")

    score = sum(1 for a, b in zip(answers, answer_key_part1) if a != -1 and a == b)
    percent = int(score * 100 / 40)
    print(f"\n✅ PHẦN I Score: {percent}%")
    return percent, answers




def grade_part2(thresh):
    h, w = thresh.shape
    roi = thresh[int(h * 0.82):int(h * 0.95), int(w * 0.12):int(w * 0.88)]
    rows, cols = 8, 2
    box_h, box_w = roi.shape[0] // rows, roi.shape[1] // cols
    result = []
    for r in range(rows):
        d_box = roi[r*box_h:(r+1)*box_h, 0:box_w]
        s_box = roi[r*box_h:(r+1)*box_h, box_w:2*box_w]
        d_val = cv2.countNonZero(d_box)
        s_val = cv2.countNonZero(s_box)
        result.append("Đúng" if d_val > s_val else "Sai")
    score = sum([1 for a, b in zip(result, answer_key_part2) if a == b])
    percent = int(score * 100 / len(answer_key_part2))
    print(f"✅ PHẦN II Score: {percent}%")
    print(f"📝 Kết quả: {result}")
    return percent, result


def grade_part3(thresh):
    h, w = thresh.shape
    roi = thresh[int(h * 0.95):int(h * 1.00), int(w * 0.10):int(w * 0.92)]
    rows = 11
    digits = 4
    cols = 6
    box_h = roi.shape[0] // rows
    box_w = roi.shape[1] // (cols * digits)
    answers = []
    for c in range(cols):
        number = ""
        for d in range(digits):
            max_val = 0
            selected_digit = ''
            for r in range(rows):
                x = (c * digits + d) * box_w
                y = r * box_h
                box = roi[y:y+box_h, x:x+box_w]
                val = cv2.countNonZero(box)
                if val > max_val:
                    max_val = val
                    selected_digit = ',' if r == 0 else str(r - 1)
            number += selected_digit
        answers.append(number)
    score = sum([1 for a, b in zip(answers, answer_key_part3) if a == b])
    percent = int(score * 100 / len(answer_key_part3))
    print(f"✅ PHẦN III Score: {percent}%")
    print("📝 Số đã tô:", answers)
    return percent, answers


def grade_omr(image_path):
    cleaned = remove_black_squares(image_path)
    cv2.imwrite("cleaned_temp.jpg", cleaned)
    warped, msg = find_corners_and_warp("cleaned_temp.jpg")
    if warped is None:
        print("❌", msg)
        return
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    sbd = extract_sbd(thresh)
    made = extract_ma_de(thresh)
    print(f"🆔 Số báo danh: {sbd}")
    print(f"📄 Mã đề: {made}")

    print("📘 Đang chấm bài sau khi đã căn chỉnh...\n")
    p1_score, answers = grade_part1_fixed(thresh)
    p2_score, _ = grade_part2(thresh)
    p3_score, _ = grade_part3(thresh)

    print("\n📊 TỔNG KẾT ĐIỂM:")
    print("📝 Đáp án:", answers)
    print(f"🧠 PHẦN I: {p1_score}%")
    print(f"✅ PHẦN II: {p2_score}%")
    print(f"🔢 PHẦN III: {p3_score}%")
    print(f"\n🎯 **Tổng điểm trung bình**: {(p1_score + p2_score + p3_score) // 3}%")

if __name__ == "__main__":
    image_path = "StudentInfo/scanbanchuan.jpg"
    grade_omr(image_path)
