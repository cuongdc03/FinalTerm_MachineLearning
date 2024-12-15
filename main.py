import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import img_to_array

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Function to load model
@st.cache_resource
def load_selected_model(model_name):
    """
    Load the specified model
    :param model_name: Name of the model file
    :return: Loaded model
    """
    if model_name == "little_vgg":
        model = load_model("emotion_little_vgg_3.h5")  # Load entire model in h5 format
    else:
        model = model_from_json(open("model.json", "r").read())
        model.load_weights('model.weights.h5')
    return model

# Function to detect emotions
def detect_emotions(frame, model):
    """
    Detect emotions in the frame
    :param frame:  Frame detected by webcam
    :param model: Model to predict emotions
    :return: Frame with emotions detected
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)

        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_prediction = emotion_labels[max_index]
        cv2.putText(frame, emotion_prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Streamlit UI
st.title("Real-time Emotion Detection")
st.write("Choose a model to load and start the webcam feed.")

# Dropdown for selecting model
model_option = st.selectbox("Select Model", ["vgg16", "little_vgg"])

# Load the selected model
try:
    model = load_selected_model(model_option)
    st.success(f"Loaded {model_option} model successfully!")
except Exception as e:
    st.error(f"Error loading {model_option}: {str(e)}")

# Webcam button
run_button = st.button("Start Webcam")

# Webcam Stream
if run_button:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        frame_with_emotion = detect_emotions(frame, model)
        frame_rgb = cv2.cvtColor(frame_with_emotion, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        if cv2.waitKey(1) == ord('b'):
            break

    cap.release()

if not run_button:
    st.write("Webcam is not active. Click 'Start Webcam' to begin.")
