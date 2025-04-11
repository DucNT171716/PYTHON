from PIL import Image
import cv2
import numpy as np

def remove_black_squares(image_path):
    # Đọc ảnh màu và chuyển sang grayscale
    img = Image.open(image_path)
    bw_img = img.convert('L')
    bw_img_np = np.array(bw_img)

    # Chuyển sang ảnh nhị phân trắng/đen
    _, thresh = cv2.threshold(bw_img_np, 50, 255, cv2.THRESH_BINARY)

    # Tìm contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2  # Tâm contour

        # Chỉ xóa nếu là hình vuông đen nhỏ và nằm gần góc
        if 400 < area < 10000 and abs(w - h) < 10:
            if (
                (cx < width * 0.2 and cy < height * 0.2) or  # Góc trên trái
                (cx > width * 0.8 and cy < height * 0.2) or  # Góc trên phải
                (cx < width * 0.2 and cy > height * 0.8) or  # Góc dưới trái
                (cx > width * 0.8 and cy > height * 0.8)     # Góc dưới phải
            ):
                cv2.drawContours(thresh, [c], -1, 255, -1)  # Tô trắng

    return thresh
