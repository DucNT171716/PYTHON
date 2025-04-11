from flask import Flask, request, jsonify
from flasgger import Swagger
import os
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from unidecode import unidecode
import re
import time
from dotenv import load_dotenv

load_dotenv() 
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

app = Flask(__name__)
swagger = Swagger(app)
@app.route('/')
def home():
    return 'Welcome to the Flask App!'

def extract_student_info(text):
    student_id = None
    score = None
    lines = text.splitlines()

    for line in lines:
        if "SBD" in line.upper():
            match_sbd = re.search(r"SBD[:\-]?\s*(\d+)", line, re.IGNORECASE)
            if match_sbd:
                student_id = match_sbd.group(1)
                break

    for i, line in enumerate(lines):
        if any(keyword in unidecode(line).lower() for keyword in ["diem", "diểm", "score", "giam khao"]):
            for j in range(i, min(i + 3, len(lines))):
                score_match = re.search(r"\d{1,2},\d{1,2}", lines[j])
                if score_match:
                    score = score_match.group(0)
                    break
            if score:
                break

    return student_id, score

def safe_decode_qr(data_bytes):
    try:
        return data_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return data_bytes.decode('latin1').encode('utf-8').decode('utf-8')
        except:
            return data_bytes.decode('utf-8', errors='replace')


def extract_qr_codes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return ["Error: Could not read image file"]
    
    results = []

    resized = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    decoded = decode(gray)
    if decoded:
        for obj in decoded:
            results.append(obj.data.decode('utf-8', errors='ignore'))
    
    if not results:
        right_half = image[:, image.shape[1]//2:]
        right_resized = cv2.resize(right_half, (right_half.shape[1]*3, right_half.shape[0]*3))
        right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
        decoded = decode(right_gray)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    if not results:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        decoded = decode(enhanced)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    if not results:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 10)
        decoded = decode(thresh)
        if decoded:
            for obj in decoded:
                results.append(obj.data.decode('utf-8', errors='replace'))
    
    if not results:
        for threshold in range(50, 200, 25):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            decoded = decode(binary)
            if decoded:
                for obj in decoded:
                    results.append(obj.data.decode('utf-8', errors='replace'))
                break
    
    if not results:
        if image.shape[1] > 200 and image.shape[0] > 200:
            top_right = image[0:image.shape[0]//2, image.shape[1]//2:]
            top_right_resized = cv2.resize(top_right, (top_right.shape[1]*3, top_right.shape[0]*3))
            top_right_gray = cv2.cvtColor(top_right_resized, cv2.COLOR_BGR2GRAY)
            decoded = decode(top_right_gray)
            if decoded:
                for obj in decoded:
                    results.append(obj.data.decode('utf-8', errors='replace'))
        
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

@app.route("/scan-essay", methods=["POST"])
def analyze_image():
    """
    Phân tích ảnh để trích xuất SBD, điểm và QR code
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Ảnh cần phân tích (JPG, PNG)
    responses:
      200:
        description: Kết quả phân tích
        examples:
          application/json: {
            "student_code": "student_code",
            "score": "student_score",
            "qr_content": ["QR Student"]
          }
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join("temp.jpg")
    file.save(filepath)

    result_raw = ""
    with open(filepath, "rb") as image_stream:
        ocr_result = client.read_in_stream(image_stream, raw=True)

    operation_location = ocr_result.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    if result.status == OperationStatusCodes.succeeded:
        for page in result.analyze_result.read_results:
            for line in page.lines:
                result_raw += line.text + "\n"

    qr_results = extract_qr_codes(filepath)
    for qr in qr_results:
        result_raw += "QR content: " + qr + "\n"

    student_id, score = extract_student_info(result_raw)

    result_scan = {
        "student_code": student_id,
        "score": score,
        "qr_content": qr_results
    }

    return jsonify(result_scan)

if __name__ == "__main__":
    app.run(debug=True)
