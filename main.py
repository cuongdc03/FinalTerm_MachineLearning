import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
import tempfile
import os

# Load model and haarcascades
model = model_from_json(open("model.json", "r").read())
model.load_weights('model.weights.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Function to detect emotions from the webcam feed
def detect_emotions(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces_detected = face_haar_cascade.detectMultiScale(gray_image)  # Detect faces

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face

        roi_gray = gray_image[y:y + h, x:x + w]  # Get face area
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model input size
        image_pixels = img_to_array(roi_gray)  # Convert to array
        image_pixels = np.expand_dims(image_pixels, axis=0)  # Expand dims for batch input

        # Model prediction
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_prediction = emotion_labels[max_index]

        # Add text label on the image
        cv2.putText(frame, emotion_prediction, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Streamlit UI
st.title("Real-time Emotion Detection")
st.write("Click on the button below to start or stop the webcam feed.")

# Create a toggle button for webcam
run_button = st.button("Start Webcam")

# Start the webcam and process the frames
if run_button:
    cap = cv2.VideoCapture(0)  # Open webcam
    stframe = st.empty()  # Create empty placeholder for the webcam stream

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Emotion detection on frame
        frame_with_emotion = detect_emotions(frame)

        # Convert frame to RGB (Streamlit expects RGB)
        frame_rgb = cv2.cvtColor(frame_with_emotion, cv2.COLOR_BGR2RGB)

        # Display the frame with detected emotion
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Stop if 'b' is pressed in the webcam window
        if cv2.waitKey(1) == ord('b'):
            break

    cap.release()  # Release the webcam

# Display instructions to stop the webcam
if not run_button:
    st.write("Webcam is not active. Click 'Start Webcam' to begin.")
