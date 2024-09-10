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
CORS(app)


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
@cross_origin()
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

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input_path = temp_input_file.name
    video_file.save(temp_input_path)

    try:
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Error opening video file'}), 500

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a temporary file for the processed video
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_path = temp_output_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        total_area = 0
        total_frames = 0
        frame_count = 0
        frame_interval = 5 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                processed_frame, area = process_frame(frame)
                total_area += area
                total_frames += 1
                out.write(processed_frame)  # Write processed frame to output video file

            frame_count += 1

        cap.release()
        out.release()

        # Add a delay before attempting to delete files
        time.sleep(1)  

        image_area = 640 * 640
        if total_frames > 0:
            average_area = total_area / total_frames
            percentage_waste = round((average_area / image_area) * 100)
            
            # Serve the processed video file
            video_url = f"http://127.0.0.1:5000/download_video/{os.path.basename(temp_output_path)}"
            result = {
                'average_area': average_area,
                'percentage_waste': percentage_waste,
                'video_url': video_url
            }
        else:
            result = {
                'error': 'No frames processed'
            }
    finally:
        # Ensure files are deleted even if an error occurs
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary input file {temp_input_path}. It might still be in use.")
        
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except PermissionError:
                print(f"Warning: Could not delete temporary output file {temp_output_path}. It might still be in use.")

    return jsonify(result)

@app.route('/download_video/<filename>', methods=['GET'])
def download_video(filename):
    # Ensure the filename is safe by joining with the output directory securely
    output_dir = os.path.join(os.path.dirname(_file_), 'output_videos')
    video_path = safe_join(output_dir, filename)  # Use safe_join to prevent directory traversal attacks
    
    # Check if the file exists and return it
    if os.path.exists(video_path):
        try:
            return send_file(video_path, mimetype='video/mp4', as_attachment=True)
        except Exception as e:
            return jsonify({'error': f'Failed to send file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    # Run the app on port 5000, regardless of environment variables
    app.run(host='0.0.0.0', port=5000)