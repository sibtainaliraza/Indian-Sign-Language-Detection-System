import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. ROBUST INITIALIZATION ---
# This must be the very first thing after imports to stabilize threads
if 'translated_sentence' not in st.session_state:
    st.session_state['translated_sentence'] = ""

# --- 2. RESOURCE LOADING ---
st.set_page_config(page_title="ISL Translator", layout="wide")

@st.cache_resource
def load_resources():
    try:
        # Loading your 126-feature model and labels
        model = load_model('mlp.h5', compile=False)
        labels = np.load('label_classes.npy')
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_resources()

# MediaPipe Setup (Moved inside a cached function for stability)
@st.cache_resource
def get_mediapipe_tools():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_draw, mp_hands

hands, mp_draw, mp_hands = get_mediapipe_tools()

# --- 3. VIDEO PROCESSING CLASS ---
class ISLProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_prediction = ""
        self.stable_frames = 0
        self.last_appended_sign = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        data_aux = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Wrist-normalization logic (0,0,0) from your Indian Sign Language project
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Padding for 126 features
            if len(data_aux) == 63: data_aux.extend(list(np.zeros(63)))
            elif len(data_aux) > 126: data_aux = data_aux[:126]
            
            if len(data_aux) == 126 and model is not None:
                data_input = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
                prediction = model.predict(data_input, verbose=0)
                predicted_label = labels[np.argmax(prediction)]
                
                # Stability Logic: 15 frames for accuracy
                if predicted_label == self.current_prediction:
                    self.stable_frames += 1
                else:
                    self.current_prediction = predicted_label
                    self.stable_frames = 0
                
                if self.stable_frames == 15:
                    if predicted_label != self.last_appended_sign:
                        # Direct state update from the receiver thread
                        st.session_state.translated_sentence += predicted_label
                        self.last_appended_sign = predicted_label
                
                cv2.putText(img, f"Sign: {predicted_label}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return frame.from_ndarray(img, format="bgr24")

# --- 4. UI LAYOUT ---
st.title(" Indian Sign Language Translator")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Controls & Output")
    
    if st.button("Clear Translation History"):
        st.session_state['translated_sentence'] = ""
        st.rerun()
    
    # Using a placeholder to avoid reading state too early
    output_area = st.empty()
    output_area.success(f"**Current Translation:** {st.session_state.translated_sentence or 'Waiting for signs...'}")
    
    # Reference guide for ISL
    image_name = "sign_alphabet_and_numbers.png"
    if os.path.exists(image_name):
        st.image(image_name, caption="ISL Reference Guide", use_container_width=True)

with col1:
    webrtc_streamer(
        key="isl-translator", 
        video_processor_factory=ISLProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True  # Helps prevent the polling_thread error
    )