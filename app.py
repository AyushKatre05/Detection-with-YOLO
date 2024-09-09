# app.py

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import safe_join
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import base64
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


model_path = os.path.join(os.path.dirname(__file__), "runs/detect/train/weights/last.pt")
model = YOLO(model_path)

def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_frame(frame):
    height, width = frame.shape[:2]
    new_size = (int(width * 0.5), int(height * 0.5))
    resized_frame = cv2.resize(frame, new_size)
    r_img = cv2.resize(resized_frame, (640, 640))
    results = model(r_img)
    area = 0
    if results and results[0].boxes is not None:
        boxes_list = results[0].boxes.data.tolist()
        for box in boxes_list:
            x1, y1, x2, y2, score, class_id = box
            area += area_calc(x1, y1, x2, y2)
            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return r_img, area

@app.route('/')
def index():
    return "Welcome to the YOLO video processing API. Use /process_image or /process_video to upload files."

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    processed_img, total_area = process_frame(img)
    image_area = 640 * 640
    percentage_waste = round((total_area / image_area) * 100)
    
    encoded_img = encode_image(processed_img)
    
    result = {
        'total_area': total_area,
        'image_area': image_area,
        'percentage_waste': percentage_waste,
        'processed_image': encoded_img
    }
    
    return jsonify(result)

if __name__ == '__main__':
    # Run the app on port 5000, regardless of environment variables
    app.run(host='0.0.0.0', port=5000)
