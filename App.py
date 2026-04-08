import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import pandas as pd
from collections import deque

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🔥 AI Crowd Density Analyzer (Hackathon Ready)")

mode = st.selectbox("Use Case Mode", [
    "Event Monitoring",
    "Traffic Control",
    "Emergency Detection"
])

st.markdown("""
### 📊 Legend:
- 🟩 Green → Low density
- 🟧 Orange → Medium density
- 🟥 Red → High density
- 📈 Graph → People count vs Time
""")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
stop_btn = st.button("🛑 Stop Processing")

# ---------------- DATA ----------------
history = deque(maxlen=50)

# ---------------- MAIN ----------------
if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    chart = st.empty()

    frame_skip = 3
    frame_count = 0

    while cap.isOpened():

        if stop_btn:
            st.warning("Processing stopped")
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
        results = model(frame, conf=0.3)

        people_count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                    cx = (x1 + x2)//2
                    cy = (y1 + y2)//2

                    row = min(cy // zone_h, rows-1)
                    col = min(cx // zone_w, cols-1)

                    zone_counts[row][col] += 1

        # ---------------- DENSITY ----------------
        avg_density = np.mean(zone_counts)
        max_density = np.max(zone_counts)

        score = 0.7 * max_density + 0.3 * avg_density

        # dynamic thresholds based on mode
        if mode == "Emergency Detection":
            high_th = 4
        else:
            high_th = 6

        if score < 3:
            density = "LOW"
            color = (0,255,0)
        elif score < high_th:
            density = "MEDIUM"
            color = (0,165,255)
        else:
            density = "HIGH"
            color = (0,0,255)

        # ---------------- RISK SCORE ----------------
        risk_score = int((people_count / 30) * 100)
        risk_score = min(risk_score, 100)

        # ---------------- HOTSPOT ----------------
        hotspot = np.unravel_index(np.argmax(zone_counts), zone_counts.shape)

        # ---------------- HEATMAP ----------------
        overlay = frame.copy()

        for i in range(rows):
            for j in range(cols):

                count = zone_counts[i][j]
                if count == 0:
                    continue

                intensity = min(count / 10, 1.0)

                color_zone = (
                    0,
                    int(255 * (1 - intensity)),
                    int(255 * intensity)
                )

                x1 = j * zone_w
                y1 = i * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h

                cv2.rectangle(overlay, (x1,y1), (x2,y2), color_zone, -1)

                cv2.putText(frame, str(int(count)),
                            (x1+20, y1+40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,255,255), 2)

        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # ---------------- GRID ----------------
        for i in range(1, rows):
            cv2.line(frame, (0,i*zone_h), (w,i*zone_h), (255,255,255), 2)
        for j in range(1, cols):
            cv2.line(frame, (j*zone_w,0), (j*zone_w,h), (255,255,255), 2)

        # ---------------- TREND ----------------
        history.append(people_count)

        trend = "STABLE"
        if len(history) > 5:
            if history[-1] > history[-5]:
                trend = "INCREASING"
            elif history[-1] < history[-5]:
                trend = "DECREASING"

        # ---------------- ALERT ----------------
        if density == "HIGH":
            cv2.rectangle(frame, (0,0), (w,80), (0,0,255), -1)
            cv2.putText(frame, "⚠ HIGH CROWD ALERT ⚠",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255,255,255), 3)

        # ---------------- TEXT ----------------
        cv2.putText(frame, f"People: {people_count}",
                    (10, h-160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

        cv2.putText(frame, f"Density: {density}",
                    (10, h-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.putText(frame, f"Risk: {risk_score}%",
                    (10, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255), 2)

        cv2.putText(frame, f"Trend: {trend}",
                    (10, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255), 2)

        cv2.putText(frame, f"Hotspot: {hotspot}",
                    (w-300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255), 2)

        # ---------------- DISPLAY ----------------
        stframe.image(frame, channels="BGR")

        # ---------------- DASHBOARD ----------------
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 People", people_count)
        col2.metric("🔥 Density", density)
        col3.metric("⚠ Risk %", risk_score)

        # ---------------- GRAPH ----------------
        smooth = pd.Series(history).rolling(window=5).mean()
        chart.line_chart(smooth)

    cap.release()

    # ---------------- DOWNLOAD ----------------
    if st.button("📥 Download Report"):
        df = pd.DataFrame(history, columns=["People Count"])
        df.to_csv("crowd_report.csv", index=False)
        st.success("Report saved!")

# ---------------- REAL WORLD ----------------
st.markdown("""
### 🚀 Real-world Applications:
- Crowd control in festivals
- Railway station monitoring
- Disaster evacuation tracking
- Smart city surveillance
""")
