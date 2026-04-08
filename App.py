import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🔥 AI Crowd Density Analyzer (Advanced)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
stop_btn = st.button("🛑 Stop Processing")

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    frame_skip = 3
    frame_count = 0

    while cap.isOpened():

        if stop_btn:
            st.warning("Processing Stopped")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        h, w, _ = frame.shape

        # ---------------- DETECTION ----------------
        results = model(frame, conf=0.3)

        people_count = 0
        crowd_area = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Area calc
                    area = (x2 - x1) * (y2 - y1)
                    crowd_area += area

        # ---------------- DENSITY CALC ----------------
        frame_area = h * w
        density_ratio = crowd_area / frame_area

        if density_ratio < 0.05:
            density = "LOW"
            color = (0,255,0)
        elif density_ratio < 0.15:
            density = "MEDIUM"
            color = (0,165,255)
        else:
            density = "HIGH"
            color = (0,0,255)

        # ---------------- ALERT ----------------
        if density == "HIGH":
            cv2.rectangle(frame, (0,0), (w,80), (0,0,255), -1)
            cv2.putText(frame, "⚠ HIGH CROWD ALERT ⚠",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255,255,255), 3)

        # ---------------- TEXT ----------------
        cv2.putText(frame, f"Total People: {people_count}",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

        cv2.putText(frame, f"Density: {density}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        # ---------------- DISPLAY ----------------
        stframe.image(frame, channels="BGR")

    cap.release()
