from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
model_path = os.path.join("models", "last.pt")
model = YOLO(model_path)

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
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Read image and convert to numpy array
        image = Image.open(file)
        image = np.array(image)

        # Preprocess image
        r_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        r_img = cv2.resize(r_img, (640, 640))

        # Perform detection
        results = model(r_img)
        area = 0

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

        # Save the processed image for display
        output_image_path = os.path.join('static', 'output.jpg')
        cv2.imwrite(output_image_path, pred_img)

        return render_template('result.html', area=area, percentage_waste=percentage_waste, output_image=output_image_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
