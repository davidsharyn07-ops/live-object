import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Neon Vision AI",
    page_icon="🤖",
    layout="wide"
)

# ---------- MODERN UI ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
    overflow: hidden;
}

/* Animated glow */
@keyframes glow {
    0% { box-shadow: 0 0 5px #06b6d4; }
    50% { box-shadow: 0 0 25px #06b6d4; }
    100% { box-shadow: 0 0 5px #06b6d4; }
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #67e8f9;
    margin-top: 10px;
    text-shadow: 0 0 15px #06b6d4;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Glass Card */
.glass {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 20px;
    animation: glow 3s infinite;
}

/* Sidebar card */
.setting-card {
    background: rgba(15, 23, 42, 0.7);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(6,182,212,0.3);
}

/* Text */
h1,h2,h3,h4,p,label {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: bold;
}

/* Slider */
.stSlider > div > div {
    color: cyan !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    '<div class="main-title">🤖 NEON VISION AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Real-Time AI Object Detection System</div>',
    unsafe_allow_html=True
)

# ---------- LAYOUT ----------
left, right = st.columns([4,1])

# ---------- SETTINGS PANEL ----------
with right:

    st.markdown('<div class="setting-card">', unsafe_allow_html=True)

    st.markdown("## ⚙️ Settings")

    confidence = st.slider(
        "Detection Confidence",
        0.1,
        0.9,
        0.30
    )

    frame_skip = st.slider(
        "Performance Mode",
        1,
        5,
        2
    )

    box_color = st.selectbox(
        "Bounding Box Color",
        ["Cyan", "Green", "Red", "Yellow"]
    )

    st.success("🟢 AI Camera Active")
    st.info("🚀 Neural Detection Running")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- COLOR ----------
colors = {
    "Cyan": (255,255,0),
    "Green": (0,255,0),
    "Red": (0,0,255),
    "Yellow": (0,255,255)
}

selected_color = colors[box_color]

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- TWILIO ----------
account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

client = Client(account_sid, auth_token)
token = client.tokens.create()

# ---------- VIDEO PROCESSOR ----------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        self.frame_count += 1

        if self.frame_count % frame_skip == 0:
            results = model.predict(
                img,
                conf=confidence,
                verbose=False
            )
            self.last_result = results
        else:
            results = self.last_result

        annotated = img.copy()

        if results and results[0].boxes is not None:

            names = model.names

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])

                label = names[cls_id]

                # Glow rectangle
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    selected_color,
                    2
                )

                # Label background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - 35),
                    (x1 + 170, y1),
                    selected_color,
                    -1
                )

                # Label text
                cv2.putText(
                    annotated,
                    f"{label} {conf_val:.2f}",
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# ---------- CAMERA ----------
with left:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    webrtc_streamer(
        key="neon-ai",

        video_processor_factory=VideoProcessor,

        async_processing=True,

        rtc_configuration={
            "iceServers": token.ice_servers
        },

        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480,
                "frameRate": 20
            },
            "audio": False
        }
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<br>
<div style='text-align:center;color:#64748b;font-size:14px;'>
⚡ Powered by YOLOv8 + Streamlit WebRTC
</div>
""", unsafe_allow_html=True)
