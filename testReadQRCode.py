import cv2
from pyzbar.pyzbar import decode
import numpy as np
def safe_decode_qr(data_bytes):
    try:
        return data_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return data_bytes.decode('latin1').encode('utf-8').decode('utf-8')
        except:
            return data_bytes.decode('utf-8', errors='replace')


def extract_qr_codes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return ["Error: Could not read image file"]
    
    # Store results
    results = []
    
    # Try multiple preprocessing techniques to improve QR detection
    
    # First attempt: Original image with resizing
    resized = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    decoded = decode(gray)
    if decoded:
        for obj in decoded:
            results.append(obj.data.decode('utf-8', errors='ignore'))
    
    # Second attempt: Try right side crop (where QR codes typically are in your image)
    if not results:
        right_half = image[:, image.shape[1]//2:]
        right_resized = cv2.resize(right_half, (right_half.shape[1]*3, right_half.shape[0]*3))
        right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
        decoded = decode(right_gray)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    # Third attempt: Try with contrast enhancement
    if not results:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        decoded = decode(enhanced)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    # Fourth attempt: Try with adaptive thresholding
    if not results:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 10)
        decoded = decode(thresh)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    # Fifth attempt: Try with different thresholds
    if not results:
        for threshold in range(50, 200, 25):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            decoded = decode(binary)
            if decoded:
                for obj in decoded:
                    results.append(obj.data.decode('utf-8', errors='replace'))
                break
    
    # Create individual crops for each QR code region
    if not results:
        # Top right QR code
        if image.shape[1] > 200 and image.shape[0] > 200:
            top_right = image[0:image.shape[0]//2, image.shape[1]//2:]
            top_right_resized = cv2.resize(top_right, (top_right.shape[1]*3, top_right.shape[0]*3))
            top_right_gray = cv2.cvtColor(top_right_resized, cv2.COLOR_BGR2GRAY)
            decoded = decode(top_right_gray)
            if decoded:
                for obj in decoded:
                    results.append(obj.data.decode('utf-8', errors='replace'))
        
            # Bottom right QR code
            bottom_right = image[image.shape[0]//2:, image.shape[1]//2:]
            bottom_right_resized = cv2.resize(bottom_right, (bottom_right.shape[1]*3, bottom_right.shape[0]*3))
            bottom_right_gray = cv2.cvtColor(bottom_right_resized, cv2.COLOR_BGR2GRAY)
            decoded = decode(bottom_right_gray)
            if decoded:
                for obj in decoded:
                    results.append(obj.data.decode('utf-8', errors='replace'))
    
    if not results:
        return ["No QR codes detected in the image"]
    
    return results

# Example usage
qr_results = extract_qr_codes("SourceCode/PythonModule/StudentInfo/hocSinh2.jpg")
print("Detected QR codes:")
for i, result in enumerate(qr_results):
    print(f"QR Code #{i+1}: {result}")