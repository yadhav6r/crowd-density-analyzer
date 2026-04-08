import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import pandas as pd
from collections import deque

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00FFC6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🔥 AI Crowd Intelligence Dashboard</p>', unsafe_allow_html=True)

st.markdown("""
### 🚀 Real-time Crowd Monitoring System
- 🔍 AI Detection + Heatmap  
- 📊 Live Crowd Trend  
- ⚠ Smart Alerts  
""")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙ Controls")

    mode = st.selectbox("Mode", [
        "Event Monitoring",
        "Traffic Control",
        "Emergency Detection"
    ])

    conf = st.slider("Detection Confidence", 0.1, 0.9, 0.3)
    frame_skip = st.slider("Performance Speed", 1, 5, 3)

    st.markdown("---")
    st.info("Built for Hackathon Demo 🚀")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- LAYOUT ----------------
left, right = st.columns([2, 1])

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
stop_btn = st.button("🛑 Stop Processing")

history = deque(maxlen=50)

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    frame_count = 0

    while cap.isOpened():

        if stop_btn:
            st.warning("Stopped")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        h, w, _ = frame.shape

        # ---------------- GRID ----------------
        rows, cols = 3, 3
        zone_h = h // rows
        zone_w = w // cols

        zone_counts = np.zeros((rows, cols))

        # ---------------- DETECTION ----------------
        results = model(frame, conf=conf)

        people_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Zone mapping
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    row = min(cy // zone_h, rows - 1)
                    col = min(cx // zone_w, cols - 1)

                    zone_counts[row][col] += 1

        # ---------------- HEATMAP ----------------
        overlay = frame.copy()

        for i in range(rows):
            for j in range(cols):
                count = zone_counts[i][j]

                if count == 0:
                    continue
                elif count < 5:
                    color = (0, 200, 0)      # GREEN (LOW)
                elif count < 10:
                    color = (0, 165, 255)    # ORANGE (MEDIUM)
                else:
                    color = (0, 0, 255)      # RED (HIGH)

                x1 = j * zone_w
                y1 = i * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                cv2.putText(frame, str(int(count)),
                            (x1 + 20, y1 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255,255,255), 2)

        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # ---------------- GRID LINES ----------------
        for i in range(1, rows):
            cv2.line(frame, (0, i * zone_h), (w, i * zone_h), (255,255,255), 2)

        for j in range(1, cols):
            cv2.line(frame, (j * zone_w, 0), (j * zone_w, h), (255,255,255), 2)

        # ---------------- DENSITY LOGIC (FIXED) ----------------
        density_ratio = people_count / (rows * cols * 10)

        if density_ratio < 0.3:
            density = "LOW"
            color = (0,255,0)
            risk_score = 30
        elif density_ratio < 0.6:
            density = "MEDIUM"
            color = (0,165,255)
            risk_score = 60
        else:
            density = "HIGH"
            color = (0,0,255)
            risk_score = 90

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

        # ---------------- GRAPH ----------------
        history.append(people_count)

        # ---------------- DISPLAY ----------------
        with left:
            stframe.image(frame, channels="BGR")

        with right:
            st.subheader("📊 Analytics")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("👥 People", people_count)
            c2.metric("🔥 Density", density)
            c3.metric("⚠ Risk", f"{risk_score}%")
            c4.metric("📈 Trend", "Stable")

            if density == "LOW":
                st.success("🟢 SAFE")
            elif density == "MEDIUM":
                st.warning("🟠 MODERATE")
            else:
                st.error("🔴 HIGH RISK")

            st.markdown("---")
            st.subheader("📈 Crowd Trend")

            smooth = pd.Series(history).rolling(window=5).mean()
            st.line_chart(smooth)

    cap.release()
