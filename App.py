import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from tensorflow.keras.models import load_model

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="ISL Translator", layout="wide")

@st.cache_resource
def load_resources():
    try:
        model = load_model('mlp.h5', compile=False)
        labels = np.load('label_classes.npy')
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_resources()

# Session state for tracking the sentence and stabilization
if 'translated_sentence' not in st.session_state:
    st.session_state.translated_sentence = ""
if 'last_hand_time' not in st.session_state:
    st.session_state.last_hand_time = time.time()
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# --- NEW: History Tracking Variables ---
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = ""
if 'stable_frames' not in st.session_state:
    st.session_state.stable_frames = 0
if 'last_appended_sign' not in st.session_state:
    st.session_state.last_appended_sign = ""

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- 2. UI LAYOUT ---
st.title(" Indian Sign Language Translator")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Controls")
    if st.button("Start Camera"):
        st.session_state.run_camera = True
    if st.button("Stop Camera"):
        st.session_state.run_camera = False
        st.rerun()
    if st.button("Clear Text"):
        st.session_state.translated_sentence = ""
        st.session_state.last_appended_sign = ""
        st.rerun()
    
    st.write("### Translation History:")
    text_area = st.empty()
    text_area.info(st.session_state.translated_sentence or "Waiting for signs...")
    
    # --- REFERENCE IMAGE ---
    st.write("### Sign Reference:")
    image_name = "sign_alphabet_and_numbers.png" 
    
    if os.path.exists(image_name):
        st.image(image_name, use_container_width=True)
    else:
        st.warning(f" Image not found. Looking for: '{image_name}'")

with col1:
    image_placeholder = st.empty()

# --- 3. CAMERA & INFERENCE LOOP ---
if st.session_state.run_camera:
    if model is None:
        st.error(" Cannot start camera: The model failed to load.")
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error(" Could not open webcam.")
        else:
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                data_aux = []
                
                if results.multi_hand_landmarks:
                    st.session_state.last_hand_time = time.time()
                    
                    # --- RELATIVE TO WRIST NORMALIZATION ---
                    for hand_landmarks in results.multi_hand_landmarks:
                        base_x = hand_landmarks.landmark[0].x
                        base_y = hand_landmarks.landmark[0].y
                        base_z = hand_landmarks.landmark[0].z
                        
                        for lm in hand_landmarks.landmark:
                            data_aux.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                        
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Pad or Trim to 126
                    if len(data_aux) == 63:
                        data_aux.extend(list(np.zeros(63)))
                    elif len(data_aux) > 126:
                        data_aux = data_aux[:126]
                    
                    # Prediction
                    if len(data_aux) == 126:
                        data_input = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
                        prediction = model.predict(data_input, verbose=0)
                        
                        predicted_label = labels[np.argmax(prediction)]
                        
                        # --- NEW: HISTORY BUILDER LOGIC ---
                        # 1. Check if the sign is stable
                        if predicted_label == st.session_state.current_prediction:
                            st.session_state.stable_frames += 1
                        else:
                            st.session_state.current_prediction = predicted_label
                            st.session_state.stable_frames = 0
                            
                        # 2. If held steady for ~15 frames, add it to history!
                        if st.session_state.stable_frames == 15:
                            # Prevent adding the exact same letter twice in a row accidentally
                            if predicted_label != st.session_state.last_appended_sign:
                                st.session_state.translated_sentence += predicted_label
                                st.session_state.last_appended_sign = predicted_label
                                text_area.info(st.session_state.translated_sentence)
                        
                        # Show current live prediction on the video feed
                        cv2.putText(frame, f"Live: {predicted_label}", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    # 3-Second Space Rule: If no hands are seen for 3 seconds, add a space
                    if time.time() - st.session_state.last_hand_time > 3.0:
                        if st.session_state.translated_sentence and not st.session_state.translated_sentence.endswith(" "):
                            st.session_state.translated_sentence += " "
                            st.session_state.last_appended_sign = " " # Reset so you can type the same letter again for the next word
                            text_area.info(st.session_state.translated_sentence)

                image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()