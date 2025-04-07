# gradeOMR_MCQ.py
import cv2
import numpy as np

# PHẦN I: Đáp án đúng mẫu (A=0, B=1, C=2, D=3)
answer_key_part1 = [
    1, 0, 2, 3, 1, 0, 2, 2, 3, 1,
    1, 2, 3, 1, 0, 2, 1, 3, 2, 0,
    0, 1, 3, 2, 1, 0, 3, 2, 2, 0,
    1, 1, 3, 1, 2, 0, 3, 1, 2, 0
]

# PHẦN II: Đáp án đúng dạng ["Đúng", "Sai", ...]
answer_key_part2 = ["Đúng", "Sai", "Đúng", "Đúng", "Sai", "Sai", "Đúng", "Sai"]

# PHẦN III: Đáp án đúng (số)
answer_key_part3 = ['1,23', '3,14', '0,50', '2,35', '5,00', '0,75']

def extract_digits(thresh, roi, num_digits=6, num_options=10):
    """
    Hàm trích xuất số từ vùng SBD hoặc Mã đề.
    - `thresh`: ảnh đã nhị phân hóa.
    - `roi`: vùng chứa SBD hoặc Mã đề thi.
    - `num_digits`: số lượng chữ số (SBD thường có 6 chữ số).
    - `num_options`: số lượng lựa chọn (0-9).
    """
    extracted_number = ""

    # Cắt ảnh để lấy vùng SBD/Mã đề
    x, y, w, h = roi
    region = thresh[y:y+h, x:x+w]

    # Chia thành từng cột (ứng với từng chữ số)
    digit_width = w // num_digits

    for i in range(num_digits):
        digit_img = region[:, i * digit_width:(i + 1) * digit_width]

        # Chia thành từng hàng để tìm số nào được tô
        cell_height = h // num_options
        detected_digit = None

        for j in range(num_options):
            cell = digit_img[j * cell_height:(j + 1) * cell_height, :]

            # Kiểm tra ô nào được tô
            if np.count_nonzero(cell == 0) > 200:  # Ngưỡng để xác định ô tô
                detected_digit = str(j)
                break
        
        if detected_digit is not None:
            extracted_number += detected_digit
        else:
            extracted_number += "X"  # Nếu không phát hiện được, đánh dấu là X

    return extracted_number


def extract_sbd(thresh):
    """
    Lấy số báo danh từ ảnh.
    """
    roi_sbd = (50, 100, 180, 80)  # Điều chỉnh tọa độ tùy ảnh
    return extract_digits(thresh, roi_sbd, num_digits=6)


def extract_ma_de(thresh):
    """
    Lấy mã đề thi từ ảnh.
    """
    roi_ma_de = (250, 100, 90, 80)  # Điều chỉnh tọa độ tùy ảnh
    return extract_digits(thresh, roi_ma_de, num_digits=3)


# Ví dụ sử dụng
image = cv2.imread("StudentInfo/abcd.jpg", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

sbd = extract_sbd(thresh)
ma_de = extract_ma_de(thresh)

print(f"Số báo danh: {sbd}")
print(f"Mã đề thi: {ma_de}")
def grade_part1_debug(thresh):
    # Cắt lại vùng đúng với PHẦN I theo hình ảnh bạn gửi
    roi = thresh[250:980, 100:810]  # Chiều dọc và chiều ngang đảm bảo đầy đủ 40 câu và 4 cột A-D

    rows, cols = 40, 4
    h, w = roi.shape
    box_h, box_w = h // rows, w // cols

    selected = []

    print("🔍 Bắt đầu chấm PHẦN I (40 câu trắc nghiệm):\n")

    for r in range(rows):
        marks = []
        for c in range(cols):
            x1, y1 = c * box_w, r * box_h
            x2, y2 = (c + 1) * box_w, (r + 1) * box_h
            box = roi[y1:y2, x1:x2]

            count = cv2.countNonZero(box)
            marks.append(count)

            # Debug vẽ hình từng ô đã cắt
            cv2.rectangle(roi, (x1, y1), (x2, y2), (255, 255, 255), 1)

        max_val = max(marks)
        max_idx = marks.index(max_val)

        # Tính trung bình các ô còn lại
        average_others = (sum(marks) - max_val) / (len(marks) - 1)

        print(f"Câu {r + 1:02}: A={marks[0]:3} | B={marks[1]:3} | C={marks[2]:3} | D={marks[3]:3} -> ", end="")

        if max_val < 80 or max_val < average_others * 1.25:
            selected.append(-1)
            print("Không rõ hoặc tô không đủ đậm")
        else:
            selected.append(max_idx)
            print(f"Chọn: {'ABCD'[max_idx]}")

    # So sánh với đáp án mẫu
    score = sum([1 for a, b in zip(selected, answer_key_part1) if a != -1 and a == b])
    percent = int(score * 100 / len(answer_key_part1))

    print(f"\n✅ PHẦN I Score: {percent}%")
    print(f"📝 Đáp án đã chọn: {selected}")
    print(f"ℹ️ Số câu không tô rõ: {selected.count(-1)}")

    # Hiển thị vùng ROI để debug
    cv2.imshow("PHẦN I - Vùng đáp án", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return percent, selected



def grade_part2(thresh):
    roi = thresh[1070:1300, 110:610]
    rows, cols = 8, 2
    h, w = roi.shape
    box_h, box_w = h // rows, w // cols

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
    roi3 = thresh[1330:1750, 100:730]
    rows = 11
    digits = 4
    cols = 6

    h, w = roi3.shape
    box_h = h // rows
    box_w = w // (cols * digits)

    answers = []
    for c in range(cols):
        number = ""
        for d in range(digits):
            max_val = 0
            selected_digit = ''
            for r in range(rows):
                x = (c * digits + d) * box_w
                y = r * box_h
                box = roi3[y:y+box_h, x:x+box_w]
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
    image = cv2.imread(image_path)
    if image is None:
        print("Không đọc được ảnh.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    print("📘 Đang chấm bài...\n")

    p1_score, p1_selected = grade_part1_debug(thresh)
    p2_score, p2_result = grade_part2(thresh)
    p3_score, p3_result = grade_part3(thresh)

    print("\n📊 TỔNG KẾT ĐIỂM:")
    print(f"🧠 PHẦN I: {p1_score}%")
    print(f"✅ PHẦN II: {p2_score}%")
    print(f"🔢 PHẦN III: {p3_score}%")
    print(f"\n🎯 **Tổng điểm trung bình**: {(p1_score + p2_score + p3_score) // 3}%")


# --------- CHẠY THỬ ---------
if __name__ == "__main__":
    image_path = "StudentInfo/abcd.jpg"
    grade_omr(image_path)