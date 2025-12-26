import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

VIDEO_PATH = 'videos'
DATA_PATH = 'data'
actions = ['goodbye', 'hello', 'no', 'please', 'yes', 'thanks', 'sorry']
sequence_length = 10

os.makedirs(DATA_PATH, exist_ok=True)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
    for video_file in os.listdir(os.path.join(VIDEO_PATH, action)):
        cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, action, video_file))
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frames.append(extract_keypoints(results))

            if len(frames) == sequence_length:
                break

        cap.release()
        if len(frames) == sequence_length:
            np.save(os.path.join(DATA_PATH, action, video_file.replace('.mp4', '.npy')), frames)
