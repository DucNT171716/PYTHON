import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_answer_grid(image_path):
    # Đọc hình ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc hình ảnh")
        return None
    
    # Tạo bản sao để hiển thị kết quả
    output = image.copy()
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Áp dụng ngưỡng để tách nền và foreground
    _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)
    
    # Tìm các contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contour hình tròn (các ô trả lời)
    circular_contours = []
    for contour in contours:
        # Tính toán các thuộc tính hình dạng
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
            
        # Hệ số tròn = 4π(diện tích)/(chu vi)²
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Lấy hình chữ nhật bao quanh contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Lọc dựa vào kích thước, độ tròn và tỉ lệ khung hình
        if 10 < w < 40 and 10 < h < 40 and 0.7 < circularity < 1.3 and 0.7 < aspect_ratio < 1.3:
            circular_contours.append((x, y, w, h))
    
    # Sắp xếp các contour theo tọa độ y để tìm các hàng
    circular_contours.sort(key=lambda c: c[1])
    
    # Phát hiện các hàng dựa trên khoảng cách y
    rows = []
    current_row = [circular_contours[0]]
    y_threshold = 15  # Ngưỡng khoảng cách y giữa các hàng
    
    for i in range(1, len(circular_contours)):
        current_contour = circular_contours[i]
        prev_contour = circular_contours[i-1]
        
        # Nếu khoảng cách y nhỏ, thì thuộc cùng một hàng
        if abs(current_contour[1] - prev_contour[1]) < y_threshold:
            current_row.append(current_contour)
        else:
            # Nếu không, tạo hàng mới
            rows.append(current_row)
            current_row = [current_contour]
    
    # Thêm hàng cuối cùng
    if current_row:
        rows.append(current_row)
    
    # Sắp xếp các contour trong mỗi hàng theo tọa độ x
    for i in range(len(rows)):
        rows[i].sort(key=lambda c: c[0])
    
    # Tìm các vùng "Số báo danh" và "Mã đề thi" dựa trên cấu trúc
    # Thông thường, "Số báo danh" có nhiều cột hơn "Mã đề thi"
    
    # Phát hiện sự phân cách giữa hai vùng
    if len(rows) > 0 and len(rows[0]) > 3:
        # Tìm khoảng cách lớn giữa các cột để xác định ranh giới giữa hai vùng
        max_x_gap = 0
        split_index = 0
        
        for i in range(1, len(rows[0])):
            x_gap = rows[0][i][0] - (rows[0][i-1][0] + rows[0][i-1][2])
            if x_gap > max_x_gap:
                max_x_gap = x_gap
                split_index = i
        
        # Tách thành hai vùng
        if split_index > 0:
            # Tìm tọa độ cho vùng "Số báo danh"
            sbd_min_x = min([cell[0] for row in rows for cell in row[:split_index] if len(row) > split_index])
            sbd_min_y = min([cell[1] for row in rows for cell in row[:split_index] if len(row) > split_index])
            sbd_max_x = max([cell[0] + cell[2] for row in rows for cell in row[:split_index] if len(row) > split_index])
            sbd_max_y = max([cell[1] + cell[3] for row in rows for cell in row[:split_index] if len(row) > split_index])
            
            # Tìm tọa độ cho vùng "Mã đề thi"
            mdt_min_x = min([cell[0] for row in rows for cell in row[split_index:] if len(row) > split_index])
            mdt_min_y = min([cell[1] for row in rows for cell in row[split_index:] if len(row) > split_index])
            mdt_max_x = max([cell[0] + cell[2] for row in rows for cell in row[split_index:] if len(row) > split_index])
            mdt_max_y = max([cell[1] + cell[3] for row in rows for cell in row[split_index:] if len(row) > split_index])
            
            # Thêm padding
            padding = 10
            sbd_region = (sbd_min_x - padding, sbd_min_y - padding, 
                         sbd_max_x - sbd_min_x + 2*padding, sbd_max_y - sbd_min_y + 2*padding)
            mdt_region = (mdt_min_x - padding, mdt_min_y - padding, 
                         mdt_max_x - mdt_min_x + 2*padding, mdt_max_y - mdt_min_y + 2*padding)
            
            # Vẽ hình chữ nhật bao quanh vùng "Số báo danh"
            cv2.rectangle(output, (sbd_region[0], sbd_region[1]), 
                         (sbd_region[0] + sbd_region[2], sbd_region[1] + sbd_region[3]), 
                         (0, 255, 0), 2)
            cv2.putText(output, "So bao danh", (sbd_region[0], sbd_region[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Vẽ hình chữ nhật bao quanh vùng "Mã đề thi"
            cv2.rectangle(output, (mdt_region[0], mdt_region[1]), 
                         (mdt_region[0] + mdt_region[2], mdt_region[1] + mdt_region[3]), 
                         (0, 0, 255), 2)
            cv2.putText(output, "Ma de thi", (mdt_region[0], mdt_region[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Phát hiện các ô được đánh dấu
            for row in rows:
                for x, y, w, h in row:
                    # Cắt vùng của ô tròn
                    cell_roi = thresh[y:y+h, x:x+w]
                    # Đếm số pixel đen (1) trong ô
                    black_pixel_count = np.sum(cell_roi == 255)
                    # Nếu số pixel đen vượt ngưỡng, coi là đã đánh dấu
                    if black_pixel_count > 0.3 * w * h:
                        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Hiển thị kết quả
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title('Kết quả phát hiện vùng')
            plt.axis('off')
            plt.show()
            
            return {
                'so_bao_danh': sbd_region,
                'ma_de_thi': mdt_region,
                'output_image': output
            }
    
    print("Không thể phát hiện đủ các vùng")
    return None

# Sử dụng hàm
result = detect_answer_grid('StudentInfo/scanbdmade.jpg')
if result:
    print(f"Tọa độ vùng 'Số báo danh': {result['so_bao_danh']}")
    print(f"Tọa độ vùng 'Mã đề thi': {result['ma_de_thi']}")