from flask import Flask, request, render_template, redirect, url_for
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import os
import logging
import time

# Suppress warnings and logging from the YOLO model
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Define the model path
model_path = "C:/Users/ayush/OneDrive/Desktop/detection/runs/detect/train/weights/last.pt"
model = YOLO(model_path)

# Define Flask app
app = Flask(__name__)

# Define function to calculate area of a bounding box
def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Upload route hit")
        if 'file' not in request.files:
            print("No file part")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        
        print(f"Received file: {file.filename}")

        if file.filename.endswith('.jpg') or file.filename.endswith('.jpeg') or file.filename.endswith('.png'):
            # Process image
            image = Image.open(file)
            image = np.array(image)
            r_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Resize image
            r_img = cv2.resize(r_img, (640, 640))

            results = model(r_img)
            area = 0

            # Draw bounding boxes and calculate areas
            for result in results:
                boxes = result.boxes
                boxes_list = boxes.data.tolist()
                for o in boxes_list:
                    x1, y1, x2, y2, score, class_id = o
                    pred_img = cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    x = area_calc(x1, y1, x2, y2)
                    area += x

            # Calculate percentage of waste detected
            image_area = 640 * 640
            percentage_waste = round((area / image_area) * 100)

            if not os.path.exists('static'):
                os.makedirs('static')
            cv2.imwrite('static/processed_image.jpg', cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))

            return render_template('result.html', area=area, image_area=image_area, percentage_waste=percentage_waste, image_url='static/processed_image.jpg')

        elif file.filename.endswith('.mp4'):
            # Process video
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(file.read())
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)
            frame_interval = 5  # Process every 5th frame
            resize_factor = 0.5  # Resize frame to 50% of original size

            total_area = 0
            total_frames = 0
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Resize frame
                    height, width = frame.shape[:2]
                    new_size = (int(width * resize_factor), int(height * resize_factor))
                    resized_frame = cv2.resize(frame, new_size)

                    # Process frame
                    r_img = cv2.resize(resized_frame, (640, 640))
                    results = model(r_img)
                    area = 0

                    # Draw bounding boxes and calculate areas
                    for result in results:
                        boxes = result.boxes
                        boxes_list = boxes.data.tolist()
                        for o in boxes_list:
                            x1, y1, x2, y2, score, class_id = o
                            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            x = area_calc(x1, y1, x2, y2)
                            area += x

                    total_area += area
                    total_frames += 1

                    if not os.path.exists('static'):
                        os.makedirs('static')
                    cv2.imwrite('static/processed_frame.jpg', cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
            os.remove(temp_file.name)

            if total_frames > 0:
                image_area = 640 * 640
                average_area = total_area / total_frames
                percentage_waste = round((average_area / image_area) * 100)

                return render_template('result.html', area=total_area, image_area=image_area, percentage_waste=percentage_waste, image_url='static/processed_frame.jpg')

        return redirect(request.url)

    except Exception as e:
        print(f"Error: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
