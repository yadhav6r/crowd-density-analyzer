import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time
import numpy as np

st.title("🔥 AI Crowd Density Analyzer - Advanced")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
stop_button = st.button("Stop Processing")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    frame_skip = 3
    frame_count = 0
    prev_time = 0

    # Grid size (3x3 zones)
    rows, cols = 3, 3

    while cap.isOpened():

        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        h, w, _ = frame.shape
        zone_h = h // rows
        zone_w = w // cols

        # Initialize grid counts
        zone_counts = np.zeros((rows, cols))

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        results = model(frame, conf=0.3, verbose=False)

        total_people = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    total_people += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Center of box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Find zone
                    col_idx = min(cx // zone_w, cols - 1)
                    row_idx = min(cy // zone_h, rows - 1)

                    zone_counts[row_idx][col_idx] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # 🔥 Heatmap overlay
        for i in range(rows):
            for j in range(cols):
                count = zone_counts[i][j]

                # Density color
                if count < 5:
                    color = (0, 255, 0)  # green
                elif count < 10:
                    color = (0, 165, 255)  # orange
                else:
                    color = (0, 0, 255)  # red

                x1 = j * zone_w
                y1 = i * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
                alpha = 0.2
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.putText(frame, str(int(count)), (x1+10, y1+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 🔥 Alert system
        if total_people > 30:
            cv2.putText(frame, "⚠ HIGH CROWD ALERT", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # UI Info
        cv2.putText(frame, f"Total: {total_people}", (10, h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    os.remove(tfile.name)
