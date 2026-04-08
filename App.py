import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("AI Crowd Density Analyzer")

# ✅ Load model once (IMPORTANT)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    frame_skip = 3   # ✅ PERFORMANCE BOOST
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # ✅ Faster inference
        results = model(frame, conf=0.3)

        people_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # ✅ Density logic
        if people_count < 10:
            density = "Low"
        elif people_count < 25:
            density = "Medium"
        else:
            density = "High"

        cv2.putText(frame, f"Count: {people_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, f"Density: {density}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

    # ✅ Clean temp file (IMPORTANT)
    os.remove(tfile.name)
