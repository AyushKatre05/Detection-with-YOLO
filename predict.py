from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model once at the start
model_path = os.path.join("models", "last.pt")

# Check if the model path exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = YOLO(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Function to calculate area of a bounding box
def calculate_area(box):
    x1, y1, x2, y2 = box[:4]
    return abs(x2 - x1) * abs(y2 - y1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                # Save the uploaded file
                file_path = os.path.join('static', 'uploaded_image.jpg')
                file.save(file_path)

                # Read image and convert to numpy array
                image = Image.open(file_path)
                image_np = np.array(image)

                # Convert image to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                resized_img = cv2.resize(image_bgr, (640, 640))

                # Perform detection
                results = model(resized_img)

                total_area = 0
                for result in results:
                    boxes = result.boxes
                    for box in boxes.data:
                        x1, y1, x2, y2, score, class_id = box.tolist()
                        total_area += calculate_area((x1, y1, x2, y2))
                        cv2.rectangle(resized_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Calculate percentage of waste detected
                image_area = 640 * 640
                percentage_waste = round((total_area / image_area) * 100, 2)

                # Save processed image
                output_image_path = os.path.join('static', 'processed_image.jpg')
                output_image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
                output_image.save(output_image_path)

                return render_template(
                    'index.html',
                    image_url=url_for('static', filename='processed_image.jpg'),
                    total_area=total_area,
                    percentage_waste=percentage_waste
                )
            except Exception as e:
                return f"Error processing image: {str(e)}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
