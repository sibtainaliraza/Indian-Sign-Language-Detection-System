#  Indian Sign Language (ISL) Real-Time Translator

An AI-powered web application designed to bridge the communication gap for the hearing and speech impaired. This system uses Computer Vision and Deep Learning to translate Indian Sign Language (ISL) gestures into text in real-time.



##  Features
* **Dual-Hand Tracking:** Utilizes MediaPipe to extract 126 key features (21 landmarks per hand, 3 coordinates each).
* **Real-Time Inference:** A custom Keras/TensorFlow MLP model classifies signs with low latency.
* **Smart History:** Includes a stabilization buffer and a 3-second "Auto-Space" rule for natural sentence building.
* **Reference UI:** Integrated sign chart for users to learn and verify gestures live.

##  Tech Stack
* **Language:** Python
* **Framework:** Streamlit (Frontend & Deployment)
* **AI/ML:** TensorFlow, Keras, NumPy
* **Computer Vision:** MediaPipe, OpenCV

##  How it Works
1. **Data Capture:** Video frames are processed via OpenCV.
2. **Feature Extraction:** MediaPipe identifies hand landmarks and calculates coordinates relative to the wrist for position-independent accuracy.
3. **Classification:** The 126-feature vector is fed into a Deep Neural Network to predict the corresponding ISL character.
4. **Display:** Predicted text is stabilized and appended to a translation history box.

---
Developed by [Sibtain Ali Raza](https://sibtainaliraza.com)