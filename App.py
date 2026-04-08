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

        # ---------------- GRID SETUP ----------------
        grid_size = 3
        cell_h = h // grid_size
        cell_w = w // grid_size

        grid_counts = np.zeros((grid_size, grid_size))

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    row = min(cy // cell_h, grid_size - 1)
                    col = min(cx // cell_w, grid_size - 1)

                    grid_counts[row][col] += 1

        # ---------------- HEATMAP ----------------
        overlay = frame.copy()

        for i in range(grid_size):
            for j in range(grid_size):

                count = grid_counts[i][j]

                if count == 0:
                    continue
                elif count < 3:
                    color_zone = (0, 200, 0)       # Green
                elif count < 6:
                    color_zone = (0, 140, 255)     # Orange
                else:
                    color_zone = (0, 0, 255)       # Red

                x1 = j * cell_w
                y1 = i * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_zone, -1)

                # Zone count text
                cv2.putText(frame, f"{int(count)}",
                            (x1 + 20, y1 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255,255,255), 2)

        # Blend overlay
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # ---------------- GRID LINES ----------------
        for i in range(1, grid_size):
            cv2.line(frame, (0, i * cell_h), (w, i * cell_h), (255,255,255), 2)

        for j in range(1, grid_size):
            cv2.line(frame, (j * cell_w, 0), (j * cell_w, h), (255,255,255), 2)

        # ---------------- SMART DENSITY ----------------
        max_density = np.max(grid_counts)
        avg_density = np.mean(grid_counts)

        if max_density < 3 and avg_density < 1.5:
            density = "LOW"
            color = (0,255,0)
        elif max_density < 6 and avg_density < 3:
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

        cv2.putText(frame, f"Max Zone: {int(max_density)}",
                    (10, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,0), 2)

        cv2.putText(frame, f"Avg Density: {avg_density:.2f}",
                    (10, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,0), 2)

        # ---------------- DISPLAY ----------------
        stframe.image(frame, channels="BGR")

    cap.release()
