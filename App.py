import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import pandas as pd
from collections import deque

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.markdown("<h1 style='color:#00FFC6;'> Crowd Intelligence Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
### 📊 Crowd Monitoring System  
- Heatmap + Detection  
- Live Trend Graph  
- Smart Alerts  
""")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙ Controls")
    conf = st.slider("Confidence", 0.1, 0.9, 0.3)
    frame_skip = st.slider("Speed (Performance)", 2, 10, 5)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt", task="detect")
    # warmup (prevents first-frame lag)
    model(np.zeros((640,640,3), dtype=np.uint8))
    return model

model = load_model()

# ---------------- LAYOUT ----------------
left, right = st.columns([2,1])

# ---------------- VIDEO PANEL ----------------
with left:
    st.subheader("🎥 Video Input")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
    stop_btn = st.button("🛑 Stop")

    if uploaded_file:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("❌ Video too large (Max: 50MB)")
            st.stop()

    stframe = st.empty()

# ---------------- RIGHT PANEL ----------------
with right:
    st.subheader("📊 Analytics")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    people_metric = col1.empty()
    density_metric = col2.empty()
    risk_metric = col3.empty()
    trend_metric = col4.empty()

    status_box = st.empty()

    st.markdown("---")

    st.subheader("📈 Crowd Trend")
    chart_placeholder = st.empty()

# ---------------- DATA ----------------
history = deque(maxlen=50)

# ---------------- MAIN ----------------
if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0

    # SAFETY LOOP LIMIT (prevents freeze)
    for _ in range(1000):

        if stop_btn:
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
                if int(box.cls[0]) == 0:  # person class
                    people_count += 1

                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cx = (x1+x2)//2
                    cy = (y1+y2)//2

                    row = min(cy//zone_h, rows-1)
                    col = min(cx//zone_w, cols-1)
                    zone_counts[row][col] += 1

        # ---------------- HEATMAP ----------------
        overlay = frame.copy()

        for i in range(rows):
            for j in range(cols):
                count = zone_counts[i][j]

                if count == 0:
                    continue
                elif count < 5:
                    color_zone = (0,200,0)
                elif count < 10:
                    color_zone = (0,165,255)
                else:
                    color_zone = (0,0,255)

                x1 = j*zone_w
                y1 = i*zone_h
                x2 = x1+zone_w
                y2 = y1+zone_h

                cv2.rectangle(overlay,(x1,y1),(x2,y2),color_zone,-1)

                cv2.putText(frame,str(int(count)),
                            (x1+20,y1+40),
                            cv2.FONT_HERSHEY_SIMPLEX,1,
                            (255,255,255),2)

        frame = cv2.addWeighted(overlay,0.4,frame,0.6,0)

        # ---------------- DENSITY ----------------
        density_ratio = people_count / 50

        if density_ratio < 0.3:
            density = "LOW"
            color = (0,255,0)
            risk = 30
        elif density_ratio < 0.6:
            density = "MEDIUM"
            color = (0,165,255)
            risk = 60
        else:
            density = "HIGH"
            color = (0,0,255)
            risk = 90

        # ---------------- ALERT ----------------
        if density == "HIGH":
            cv2.rectangle(frame,(0,0),(w,80),(0,0,255),-1)
            cv2.putText(frame,"⚠ HIGH CROWD ALERT ⚠",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,
                        (255,255,255),3)

        # ---------------- TEXT ----------------
        cv2.putText(frame,f"People: {people_count}",
                    (10,h-60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,255,255),2)

        cv2.putText(frame,f"Density: {density}",
                    (10,h-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    color,2)

        # ---------------- DATA ----------------
        history.append(people_count)

        trend = "Stable"
        if len(history) > 5:
            if history[-1] > history[-5]:
                trend = "Increasing"
            elif history[-1] < history[-5]:
                trend = "Decreasing"

        # ---------------- DISPLAY ----------------
        stframe.image(frame, channels="BGR")

        people_metric.metric("👥 People", people_count)
        density_metric.metric("🔥 Density", density)
        risk_metric.metric("⚠ Risk", f"{risk}%")
        trend_metric.metric("📈 Trend", trend)

        if density == "LOW":
            status_box.success("🟢 SAFE")
        elif density == "MEDIUM":
            status_box.warning("🟠 MODERATE")
        else:
            status_box.error("🔴 HIGH RISK")

        smooth = pd.Series(history).rolling(window=5).mean()
        chart_placeholder.line_chart(smooth)

    cap.release()
