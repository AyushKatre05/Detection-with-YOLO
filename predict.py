import streamlit as st
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

# Define function to calculate area of a bounding box
def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

# Streamlit app
st.title('Waste Detection with YOLO')

# File uploader for image and video
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    if uploaded_file.type in ['image/jpeg', 'image/png']:
        # Process image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image = Image.open(uploaded_file)
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

        st.image(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB), caption='Processed Image with Detection')
        st.write(f"Area of waste detected: {area} unit sq")
        st.write(f"Area of the image: {image_area} unit sq")
        st.write(f"The Percentage of Waste detected in the image is: {percentage_waste}%")

    elif uploaded_file.type == 'video/mp4':
        # Process video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        total_area = 0
        total_frames = 0

        # Define processing interval and resize factor
        frame_interval = 5  # Process every 5th frame
        resize_factor = 0.5  # Resize frame to 50% of original size

        frame_count = 0
        start_time = time.time()

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

                # Display results frame-by-frame
                stframe.image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB), caption='Processed Video Frame with Detection', use_column_width=True)

            frame_count += 1

        cap.release()
        os.remove(temp_file.name)

        if total_frames > 0:
            image_area = 640 * 640
            average_area = total_area / total_frames
            percentage_waste = round((average_area / image_area) * 100)

            st.write(f"Total Area of waste detected in video: {total_area} unit sq")
            st.write(f"Area of the image frame: {image_area} unit sq")