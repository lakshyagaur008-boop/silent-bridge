import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# ---------------- CONFIG ----------------
SEQUENCE_LENGTH = 10
ACTIONS = ["goodbye", "hello", "no", "please", "yes", "thanks", "sorry"]

st.set_page_config(page_title="Silent Bridge", layout="centered")
st.title("🤟 Silent Bridge – Sign Language Recognition")

model = load_model("sign_language_model.h5")

# ---------------- MEDIAPIPE ----------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])  # 225

# ---------------- SESSION STATE ----------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run_camera = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run_camera = False

frame_window = st.image([])
result_text = st.empty()

# ---------------- CAMERA LOOP ----------------
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    sequence = []

    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            result_text.error("Camera not accessible")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) < SEQUENCE_LENGTH:
            result_text.info(f"Collecting frames... {len(sequence)}/{SEQUENCE_LENGTH}")
        else:
            prediction = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )[0]

            idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            action = ACTIONS[idx]

            result_text.success(f"🖐️ Predicted: {action} ({confidence:.2f})")

        frame_window.image(frame)
        time.sleep(0.03)

    cap.release()
