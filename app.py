import streamlit as st
import cv2
import numpy as np
from fastai.vision.all import load_learner, PILImage, Resize, aug_transforms, Normalize, imagenet_stats
from PIL import Image
import mediapipe as mp

label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

st.title("Webcam Live Feed for Sign Language Recognition")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
CROPPED_HAND_WINDOW = st.image([]) 
PREDICTION_TEXT = st.empty()

model_path = './models/sign_language_model.pkl'
learner = load_learner(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
        
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            
            width = x_max - x_min
            height = y_max - y_min
            margin_x = 0.1 * width
            margin_y = 0.1 * height
            
            x_min = int(max(0, x_min - margin_x))
            x_max = int(min(frame.shape[1], x_max + margin_x))
            y_min = int(max(0, y_min - margin_y))
            y_max = int(min(frame.shape[0], y_max + margin_y))
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            hand_img = frame[y_min:y_max, x_min:x_max]
            
            if hand_img.size > 0:
                hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)).resize((128, 128))
                
                hand_img_fastai = PILImage.create(hand_img_pil)
                
                pred, _, probs = learner.predict(hand_img_fastai)
                predicted_letter = label_to_letter.get(int(pred), 'Unknown')
                
                FRAME_WINDOW.image(frame)
                CROPPED_HAND_WINDOW.image(hand_img_pil)  
                PREDICTION_TEXT.text(f"Predicted Letter: {predicted_letter}")
            else:
                FRAME_WINDOW.image(frame)
                PREDICTION_TEXT.text("Hand cropped image is empty")
    else:
        FRAME_WINDOW.image(frame)
        PREDICTION_TEXT.text("Hand not detected")

cap.release()
hands.close()