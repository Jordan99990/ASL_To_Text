import streamlit as st
import cv2
import numpy as np
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import matplotlib.pyplot as plt

# Define label to letter mapping
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# Streamlit UI setup
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
PREDICTION_TEXT = st.empty()

# Load the model
model_path = './models/sign_language_model.pkl'
learner = load_learner(model_path)

# OpenCV video capture setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Define HSV color range for hand detection (adjust as necessary)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        hand_img = frame[y:y+h, x:x+w]
        
        hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
        hand_img_pil = hand_img_pil.resize((224, 224))
        hand_img_fastai = PILImage.create(hand_img_pil)
        
        pred, _, probs = learner.predict(hand_img_fastai)
        predicted_letter = label_to_letter.get(int(pred), 'Unknown')
        
        FRAME_WINDOW.image(frame)
        PREDICTION_TEXT.text(f"Predicted Letter: {predicted_letter}")
    else:
        FRAME_WINDOW.image(frame)
        PREDICTION_TEXT.text("Hand not detected")

cap.release()
