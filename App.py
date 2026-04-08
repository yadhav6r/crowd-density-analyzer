import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import pandas as pd
from collections import deque

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("🔥 AI Crowd Density Analyzer (Smart + Explained)")

st.markdown("""
### 📊 Legend:
- 🟩 Green → Low density
- 🟧 Orange → Medium density
- 🟥 Red → High density
- 📈 Graph → People count over time (trend)
""")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
stop_btn = st.button("🛑 Stop Processing")

# ---------------- DATA ----------------
history = deque(maxlen=30)

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

        # ---------------- DENSITY LOGIC ----------------
        avg_density = np.mean(zone_counts)
        max_density = np.max(zone_counts)

        score = 0.7 * max_density + 0.3 * avg_density

        if score < 3:
            density = "LOW"
            color = (0,255,0)
        elif score < 6:
            density = "MEDIUM"
            color = (0,165,255)
        else:
            density = "HIGH"
            color = (0,0,255)

        # ---------------- HEATMAP ----------------
        overlay = frame.copy()

        for i in range(rows):
            for j in range(cols):

                count = zone_counts[i][j]

                if count == 0:
                    continue

                intensity = min(count / 10, 1.0)

                # gradient (green → red)
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
                    (10, h-100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

        cv2.putText(frame, f"Density: {density}",
                    (10, h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.putText(frame, f"Trend: {trend}",
                    (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255), 2)

        # ---------------- DISPLAY ----------------
        stframe.image(frame, channels="BGR")

        # ---------------- GRAPH ----------------
        smooth = pd.Series(history).rolling(window=5).mean()
        chart.line_chart(smooth)

    cap.release()
