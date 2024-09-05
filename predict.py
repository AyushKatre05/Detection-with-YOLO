import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import logging
import time
import numpy as np  

logging.getLogger('ultralytics').setLevel(logging.ERROR)

model_path = "C:/Users/ayush/OneDrive/Desktop/detection/runs/detect/train/weights/last.pt"
model = YOLO(model_path)

def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

st.title('Waste Detection Model')

option = st.selectbox('Select Input Type:', ('Image', 'Video'))

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
            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            area += area_calc(x1, y1, x2, y2)
    return r_img, area

if option == 'Image':
    uploaded_image = st.file_uploader("Choose an image file...", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        processed_img, total_area = process_frame(img)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption='Processed Image with Detection', use_column_width=True)
        image_area = 640 * 640
        percentage_waste = round((total_area / image_area) * 100)
        st.write(f"Total Area of waste detected: {total_area} unit sq")
        st.write(f"Area of the image frame: {image_area} unit sq")
        st.write(f"The Percentage of Waste detected in the image is: {percentage_waste}%")

elif option == 'Video':
    uploaded_video = st.file_uploader("Choose a video file...", type='mp4')

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

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
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Video Frame with Detection', use_column_width=True)

            frame_count += 1

        cap.release()
        os.remove(temp_file.name)
        if total_frames > 0:
            image_area = 640 * 640
            average_area = total_area / total_frames
            percentage_waste = round((average_area / image_area) * 100)

            st.write(f"Total Area of waste detected in video: {total_area} unit sq")
            st.write(f"Area of the image frame: {image_area} unit sq")
            st.write(f"The Percentage of Waste detected in the video is: {percentage_waste}%")
