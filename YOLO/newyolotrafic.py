import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO # Corrected import
import math
import sqlite3
import pandas as pd
from datetime import datetime
import torch

# --- SETTINGS ---
CONF_THRESHOLD = 0.5
STOP_LINE_Y = 400

# --- DATABASE FUNCTIONS ---

def setup_database():
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            violation_type TEXT,
            tracker_id INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_violation(tracker_id, violation_type):
    try:
        conn = sqlite3.connect('violations.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO violations (timestamp, violation_type, tracker_id) VALUES (?, ?, ?)",
                  (timestamp, violation_type, tracker_id))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

def get_violation_log():
    try:
        conn = sqlite3.connect('violations.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='violations';")
        if c.fetchone() is None:
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching log: {e}")
        return pd.DataFrame()

# --- MODEL LOADING FUNCTIONS ---
@st.cache_resource
def load_helmet_model():
    try:
        model = YOLO('best2.pt')
        return model
    except Exception as e:
        st.error(f"Error loading helmet model ('best.pt'): {e}")
        st.stop()

@st.cache_resource
def load_traffic_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading traffic model ('yolov8n.pt'): {e}")
        st.stop()

# --- HELPER FUNCTIONS ---

def get_traffic_light_color(frame, traffic_light_box):
    try:
        x1, y1, x2, y2 = map(int, traffic_light_box)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        if y1 >= y2 or x1 >= x2: return 'unknown'

        crop = frame[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([85, 255, 255])

        mask_red = cv2.add(cv2.inRange(hsv_crop, lower_red1, upper_red1), cv2.inRange(hsv_crop, lower_red2, upper_red2))
        mask_yellow = cv2.inRange(hsv_crop, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv_crop, lower_green, upper_green)

        value_mask = hsv_crop[:, :, 2] > 150
        red_pixels = cv2.countNonZero(mask_red & value_mask)
        yellow_pixels = cv2.countNonZero(mask_yellow & value_mask)
        green_pixels = cv2.countNonZero(mask_green & value_mask)

        colors = {'red': red_pixels, 'yellow': yellow_pixels, 'green': green_pixels}
        max_pixels = max(colors.values())

        if max_pixels > 20:
            return max(colors, key=colors.get)
        return 'unknown'
    except Exception:
        return 'unknown'

# --- MAIN PROCESSING FUNCTIONS ---

def process_video(video_file, helmet_model, traffic_model):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    st.subheader("Live Violation Stats")
    col1, col2 = st.columns(2)
    with col1:
        helmet_count_text = st.empty()
    with col2:
        signal_count_text = st.empty()
    st_frame = st.empty()

    helmet_violator_ids, signal_violator_ids = set(), set()
    total_helmet_violations, total_signal_violations = 0, 0
    vehicle_positions = {}

    helmet_names, traffic_names = helmet_model.names, traffic_model.names

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)

        try:
            helmet_results = helmet_model.track(frame, persist=True, verbose=False)
            traffic_results = traffic_model.track(frame, persist=True, verbose=False)
        except Exception as e:
            st.error(f"Model inference error: {e}")
            continue

        annotated_frame = frame.copy()

        if helmet_results and helmet_results[0].boxes.id is not None:
            for tracker_id, cls, conf in zip(helmet_results[0].boxes.id.int().cpu().tolist(),
                                             helmet_results[0].boxes.cls.int().cpu().tolist(),
                                             helmet_results[0].boxes.conf.cpu().tolist()):
                if conf > CONF_THRESHOLD and helmet_names[cls] == 'Without Helmet' and tracker_id not in helmet_violator_ids:
                    total_helmet_violations += 1
                    helmet_violator_ids.add(tracker_id)
                    log_violation(tracker_id, 'Without Helmet')
            annotated_frame = helmet_results[0].plot(img=annotated_frame)

        is_light_red = False
        traffic_light_box_to_draw = None
        if traffic_results and traffic_results[0].boxes.id is not None:
            all_boxes = traffic_results[0].boxes

            traffic_lights = []
            for i in range(len(all_boxes.cls)):
                 cls = all_boxes.cls[i].int().item()
                 conf = all_boxes.conf[i].item()
                 box_coords = all_boxes.xyxy[i].cpu().tolist()
                 if conf > CONF_THRESHOLD and traffic_names[cls] == 'traffic light':
                    traffic_lights.append((conf, box_coords))

            if traffic_lights:
                best_light = max(traffic_lights, key=lambda item: item[0])
                color = get_traffic_light_color(frame, best_light[1])
                traffic_light_box_to_draw = best_light[1]
                if color == 'red':
                    is_light_red = True

            for tracker_id, cls, conf, box in zip(all_boxes.id.int().cpu().tolist(),
                                                  all_boxes.cls.int().cpu().tolist(),
                                                  all_boxes.conf.cpu().tolist(),
                                                  all_boxes.xyxy.cpu().tolist()):
                if conf > CONF_THRESHOLD and traffic_names[cls] in ['car', 'motorcycle', 'bus', 'truck']:
                    y_pos = int(box[3])
                    prev_pos = vehicle_positions.get(tracker_id, 0)
                    if is_light_red and (prev_pos < STOP_LINE_Y) and (y_pos >= STOP_LINE_Y) and (tracker_id not in signal_violator_ids):
                        total_signal_violations += 1
                        signal_violator_ids.add(tracker_id)
                        log_violation(tracker_id, f'Signal Jump ({traffic_names[cls]})')
                    vehicle_positions[tracker_id] = y_pos

        if traffic_results:
             annotated_frame = traffic_results[0].plot(img=annotated_frame)

        if is_light_red and traffic_light_box_to_draw:
             cv2.putText(annotated_frame, "RED LIGHT", (int(traffic_light_box_to_draw[0]), int(traffic_light_box_to_draw[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.line(annotated_frame, (0, STOP_LINE_Y), (annotated_frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)

        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        helmet_count_text.metric("'Without Helmet' Violators", total_helmet_violations)
        signal_count_text.metric("'Signal Jump' Violators", total_signal_violations)

    cap.release()
    tfile.close()
    st.success("Video processing finished.")
    st.toast("Log updated!")


def process_image(image_file, helmet_model, traffic_model):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Detection Results")
    col1, col2 = st.columns(2)

    helmet_results = helmet_model(image, conf=CONF_THRESHOLD)
    total_helmet_violations = 0
    helmet_names = helmet_model.names
    for r in helmet_results:
        for c in r.boxes.cls:
            if helmet_names[int(c)] == 'Without Helmet':
                total_helmet_violations += 1
    annotated_image = helmet_results[0].plot()

    traffic_results = traffic_model(image, conf=CONF_THRESHOLD)
    traffic_names = traffic_model.names
    is_light_red = False
    traffic_light_box_to_draw = None

    if traffic_results and traffic_results[0].boxes:
        all_boxes = traffic_results[0].boxes
        traffic_lights = []
        for i in range(len(all_boxes.cls)):
             cls = all_boxes.cls[i].int().item()
             conf = all_boxes.conf[i].item()
             box_coords = all_boxes.xyxy[i].cpu().tolist()
             if conf > CONF_THRESHOLD and traffic_names[cls] == 'traffic light':
                traffic_lights.append((conf, box_coords))

        if traffic_lights:
            best_light = max(traffic_lights, key=lambda item: item[0])
            color = get_traffic_light_color(image, best_light[1])
            traffic_light_box_to_draw = best_light[1]
            if color == 'red':
                is_light_red = True

    final_annotated_image = traffic_results[0].plot(img=annotated_image)

    if is_light_red and traffic_light_box_to_draw:
        cv2.putText(final_annotated_image, "RED LIGHT", (int(traffic_light_box_to_draw[0]), int(traffic_light_box_to_draw[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    with col1:
        st.metric(f"'Without Helmet' Detections", total_helmet_violations)
        st.markdown(f"**Red Light Detected:** {'Yes' if is_light_red else 'No'}")
    with col2:
         st.image(cv2.cvtColor(final_annotated_image, cv2.COLOR_BGR2RGB),
                 caption='Processed Image with Detections',
                 use_container_width=True)
    st.success("Image processing finished.")


st.set_page_config(page_title="UrbanPulse", layout="wide")

st.markdown("<h1 style='text-align: center;'>UrbanPulse: Smart Traffic Violation Detection</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>Uses AI to detect helmet & signal violations. (Confidence > {int(CONF_THRESHOLD*100)}%)</p>", unsafe_allow_html=True)
st.markdown("---")

load_col1, load_col2 = st.columns([1, 4])
with load_col1:
    st.write("Loading AI Models...")
    helmet_model = load_helmet_model()
    traffic_model = load_traffic_model()

with load_col2:
    if helmet_model and traffic_model:
        st.success("AI Models Loaded Successfully!")

        helmet_ok = 'Without Helmet' in helmet_model.names.values()
        traffic_ok = 'traffic light' in traffic_model.names.values() and ('car' in traffic_model.names.values() or 'motorcycle' in traffic_model.names.values())

        if not helmet_ok: st.error(f"Helmet model ('best.pt') does not have the required 'Without Helmet' class! Found: {helmet_model.names}")
        if not traffic_ok: st.error(f"Traffic model ('yolov8n.pt') does not have 'traffic light' or vehicle classes! Found: {traffic_model.names}")

        if helmet_ok and traffic_ok:
            tab1, tab2 = st.tabs(["Process Image", "Process Video"])

            with tab1:
                st.header("Upload Image")
                uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_uploader", label_visibility="collapsed")
                if uploaded_image:
                    if st.button("Detect Objects in Image", key="image_button", use_container_width=True):
                        with st.spinner('Analyzing image...'):
                            process_image(uploaded_image, helmet_model, traffic_model)

            with tab2:
                st.header("Upload Video")
                uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"], key="video_uploader", label_visibility="collapsed")
                if uploaded_video:
                    if st.button("Detect Violations in Video", key="video_button", use_container_width=True):
                        with st.spinner('Processing video... Please wait.'):
                            process_video(uploaded_video, helmet_model, traffic_model)

            st.markdown("---")
            st.header("Violation Log")
            col_log1, col_log2 = st.columns([4, 1])
            with col_log2:
                if st.button("Refresh Log", use_container_width=True):
                    st.rerun()

            try:
                setup_database()
                violation_df = get_violation_log()
                if not violation_df.empty:
                     st.dataframe(violation_df, use_container_width=True, height=300)
                else:
                     st.info("No violations logged yet.")
            except Exception as e:
                st.error(f"Could not display violation log: {e}")

        else:
            st.error("Cannot proceed due to incorrect model classes.")

    else:
        st.error("Critical error: One or more AI models failed to load.")

st.markdown("---")
st.caption("UrbanPulse v1.0 - AI Detection System")