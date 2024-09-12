import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import logging
import time
import numpy as np  

# Suppress warnings and logging from the YOLO model
logging.getLogger('ultralytics').setLevel(logging.ERROR)
# Define the model path
model_path = "runs/detect/train/weights/last.pt"
model = YOLO(model_path)
# Define function to calculate area of a bounding box
def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

# Streamlit app
st.title('Waste Detection in Image/Video')

# Allow user to select between image and video
option = st.selectbox('Select Input Type:', ('Image', 'Video'))

# Common code for bounding box detection and processing
def process_frame(frame):
    # Resize frame
    height, width = frame.shape[:2]
    new_size = (int(width * 0.5), int(height * 0.5))
    resized_frame = cv2.resize(frame, new_size)

    # Resize to YOLO input size
    r_img = cv2.resize(resized_frame, (640, 640))

    # Process frame using YOLO model
    results = model(r_img)
    area = 0

    # Extract bounding boxes and process them
    if results and results[0].boxes is not None:
        boxes_list = results[0].boxes.data.tolist()
        for box in boxes_list:
            x1, y1, x2, y2, score, class_id = box
            # Draw bounding box
            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Calculate area
            area += area_calc(x1, y1, x2, y2)

    return r_img, area

# If the user selects 'Image'
if option == 'Image':
    uploaded_image = st.file_uploader("Choose an image file...", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Process the image
        processed_img, total_area = process_frame(img)

        # Display processed image with bounding boxes
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption='Processed Image with Detection', use_column_width=True)

        # Calculate and display the waste percentage
        image_area = 640 * 640
        percentage_waste = round((total_area / image_area) * 100)
        st.write(f"Total Area of waste detected: {total_area} unit sq")
        st.write(f"Area of the image frame: {image_area} unit sq")
        st.write(f"The Percentage of Waste detected in the image is: {percentage_waste}%")

# If the user selects 'Video'
elif option == 'Video':
    uploaded_video = st.file_uploader("Choose a video file...", type='mp4')

    if uploaded_video is not None:
        # Process video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        total_area = 0
        total_frames = 0
        frame_count = 0
        frame_interval = 5  # Process every 5th frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                processed_frame, area = process_frame(frame)
                total_area += area
                total_frames += 1

                # Display video frame with bounding boxes
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Video Frame with Detection', use_column_width=True)

            frame_count += 1

        cap.release()
        os.remove(temp_file.name)

        # Calculate and display waste percentage for video
        if total_frames > 0:
            image_area = 640 * 640
            average_area = total_area / total_frames
            percentage_waste = round((average_area / image_area) * 100)

            st.write(f"Total Area of waste detected in video: {total_area} unit sq")
            st.write(f"Area of the image frame: {image_area} unit sq")
            st.write(f"The Percentage of Waste detected in the video is: {percentage_waste}%")