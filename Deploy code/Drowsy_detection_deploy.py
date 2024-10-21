import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import pygame
import av

# Load pre_trained Keras model
model = tf.keras.models.load_model('transferlearning_model.keras')

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('./alertsound.wav')

def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (94, 94))  
    eye_img = eye_img.astype('float32') / 255.0  
    eye_img = np.expand_dims(eye_img, axis=0)  
    return eye_img

thresh = 0.5  
frame_check = 20

# Load Haar cascade classifiers
face_cascade_path = "C:\\Users\\User\\FYP final\\haarcascade_frontalface_default.xml"
eye_cascade_path = "C:\\Users\\User\\FYP final\\haarcascade_eye.xml"
left_eye_cascade_path = "C:\\Users\\User\\FYP final\\haarcascade_lefteye_2splits.xml"
right_eye_cascade_path = "C:\\Users\\User\\FYP final\\haarcascade_righteye_2splits.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)



class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.flag = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_color)
            left_eyes = left_eye_cascade.detectMultiScale(roi_color)
            right_eyes = right_eye_cascade.detectMultiScale(roi_color)

            eye_status = []
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                eye_img = preprocess_eye(eye_img)
                prediction = model.predict(eye_img)
                eye_open = prediction[0][0] > thresh
                eye_status.append(eye_open)
                color = (0, 255, 0) if eye_open else (0, 0, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)

            for (lex, ley, lew, leh) in left_eyes:
                eye_img = roi_color[ley:ley+leh, lex:lex+lew]
                eye_img = preprocess_eye(eye_img)
                prediction = model.predict(eye_img)
                eye_open = prediction[0][0] > thresh
                eye_status.append(eye_open)
                color = (0, 255, 0) if eye_open else (0, 0, 255)
                cv2.rectangle(roi_color, (lex, ley), (lex+lew, ley+leh), color, 2)

            for (rex, rey, rew, reh) in right_eyes:
                eye_img = roi_color[rey:rey+reh, rex:rex+rew]
                eye_img = preprocess_eye(eye_img)
                prediction = model.predict(eye_img)
                eye_open = prediction[0][0] > thresh
                eye_status.append(eye_open)
                color = (0, 255, 0) if eye_open else (0, 0, 255)
                cv2.rectangle(roi_color, (rex, rey), (rex+rew, rey+reh), color, 2)

            if len(eye_status) > 0 and not all(eye_status):
                self.flag += 1
                if self.flag >= frame_check:
                    cv2.putText(img, "*****WARNING!*****", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, "*****WAKE UP & FOCUS DURING DRIVING!*****", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not pygame.mixer.music.get_busy():  # Check if music is already playing
                        pygame.mixer.music.play()
            else:
                self.flag = 0
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit application layout
st.title("Drowsy Detection Web App")
st.header("Real-time drowsiness detection using webcam")

# WebRTC streamer configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_ctx = webrtc_streamer(
    key="drowsy-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    st.markdown("### Status")
    status_text = "Monitoring for drowsiness..."
    if webrtc_ctx.video_processor.flag >= frame_check:
        status_text = "*ALERT!* Drowsiness detected!"
    st.write(status_text)
