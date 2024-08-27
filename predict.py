import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os

# Load YOLO model once at the start
model_path = os.path.join("models", "last.pt")

# Check if the model path exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Function to calculate area of a bounding box
def calculate_area(box):
    x1, y1, x2, y2 = box[:4]
    return abs(x2 - x1) * abs(y2 - y1)

st.title("Waste Detection with YOLO")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image and convert to numpy array
        image = Image.open(uploaded_file)
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

        # Convert processed image to display in Streamlit
        output_image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

        st.image(output_image, caption='Processed Image', use_column_width=True)
        st.write(f"Total Area of Waste Detected: {total_area} pixels")
        st.write(f"Percentage of Waste Detected: {percentage_waste}%")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
