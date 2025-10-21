# Smart-Traffic-Violation-Detection

**1. Project overview**

    ✅UrbanPulse is a Streamlit-based UI that uses YOLO (Ultralytics) object detection/tracking models to detect:
    ✅Helmet violations (people Without Helmet) using a custom best2.pt model.
    ✅Traffic objects and traffic lights (e.g., traffic light, car, motorcycle) using yolov8n.pt.
    ✅It marks red-light violations when a vehicle crosses a stop line during a detected red light, logs violations
       to a local SQLite DB, and shows annotated frames in the Streamlit UI.

**2. Features**

    ✅Image and video upload (Streamlit file uploader).
    ✅Real-time annotated preview while processing video frames.
    ✅Helmet violation detection (counts & logs).
    ✅Red-light (signal jump) detection (counts & logs).
    ✅SQLite database for persistent logging (violations.db).
    ✅Simple, single-page Streamlit UI with tabs for image/video and a violation log display.

**3. Requirements & recommended versions**

    Pick the correct torch version for your hardware (GPU/CUDA). If you have CUDA 11.8, pick a matching torch wheel.
    Python 3.8 — 3.11
    streamlit >= 1.23
    ultralytics >= 8.x 
    torch 
    opencv-python
    numpy
    pandas
    sqlite3 (builtin)

**Example requirements.txt:**

  streamlit>=1.23
  ultralytics>=8.3
  torch>=2.0   # choose appropriate CUDA build (or cpu only)
  opencv-python
  numpy
  pandas
  tqdm


**Quick pip (CPU fallback):**

    python -m venv venv
    source venv/bin/activate            
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install streamlit ultralytics opencv-python numpy pandas tqdm

**4. File / folder structure (recommended)**

    urbanpulse/
    ├─ app.py                  # your Streamlit app (the code you posted)
    ├─ models/
    │  ├─ best2.pt             # helmet detection model (custom)
    │  └─ yolov8n.pt           # traffic detection model
    ├─ violations.db           # SQLite DB (auto-created)
    ├─ README.md
    ├─ requirements.txt
    └─ logs/                   # optional: for storing processed frames or debug output


Place best2.pt and yolov8n.pt in models/ or update code paths accordingly.

5. Quick install & final execution steps (exact commands)

Clone repo and cd:

git clone <your-repo-url>
cd urbanpulse


Create virtual env & activate:

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate


Install dependencies (example CPU-only; replace torch install for GPU):

pip install -r requirements.txt
# or (if requirements doesn't include torch)
pip install streamlit ultralytics opencv-python numpy pandas
# then install torch as appropriate for your CUDA version


Put model files in models/ (or same folder as app.py). Update the model loading path in code if moved:

models/best2.pt
models/yolov8n.pt


Run the app:

streamlit run app.py


In the Streamlit UI:

Wait for "AI Models Loaded Successfully!"

Use the Process Image or Process Video tab.

Upload image or video and click the respective button to run detection.

Check Violation Log at the bottom.

6. Step-by-step working (code walkthrough & logic)
Start: Settings & imports

CONF_THRESHOLD = 0.5 — detection confidence threshold. Objects with confidence below this are ignored.

STOP_LINE_Y = 400 — y-coordinate of the virtual stop line used for signal-jump detection (pixel coordinate).

Database helper functions

setup_database() — creates violations table if missing.

log_violation(tracker_id, violation_type) — inserts a record with timestamp, violation type and tracker id.

get_violation_log() — reads DB into a pandas DataFrame.

Model loading

@st.cache_resource decorated load_helmet_model() and load_traffic_model() to cache the loaded models, reduces reloads when UI reruns.

They create YOLO instances: YOLO('best2.pt') and YOLO('yolov8n.pt'). If loading fails, st.error() + st.stop() halts further execution.

Note: Model path strings in your posted code are 'best2.pt' and 'yolov8n.pt'. If your files are in models/, change to models/best2.pt.

Helper: get_traffic_light_color(frame, box)

Extracts the traffic light crop from frame, converts to HSV and counts red/yellow/green pixels above a brightness threshold.

Returns 'red', 'yellow', 'green' or 'unknown'.

Uses HSV thresholds for color ranges — may need tuning per camera and lighting.

process_video(video_file, helmet_model, traffic_model)

Flow:

Save uploaded file to a temporary file (tempfile.NamedTemporaryFile).

Open with OpenCV VideoCapture.

Prepare UI elements (two metrics and one image area).

Maintain sets: helmet_violator_ids, signal_violator_ids and counters.

Loop frame-by-frame:

Draw stop line on frame.

Run two model inferences:

helmet_model.track(frame, persist=True, verbose=False)

traffic_model.track(frame, persist=True, verbose=False)
(Note: running two .track() calls per frame is computation heavy — consider combining or running only one if needed.)

For helmet results: iterate boxes.id, boxes.cls, conf. If class is 'Without Helmet' and not previously logged (tracker id uniqueness), increment and log_violation.

For traffic results:

Build traffic_lights list from boxes where class is 'traffic light'

Choose the best traffic light (highest conf) and call get_traffic_light_color() to determine color. If 'red', set is_light_red flag.

For vehicle boxes (car, motorcycle, bus, truck) track bottom y-coordinate y_pos and compare to STOP_LINE_Y. If a vehicle crossed the line while light is red and not already logged, log violation.

Annotate frame with YOLO plot() results and overlays, update Streamlit image and metrics.

Release capture and close temp file.

process_image(image_file, helmet_model, traffic_model)

Reads uploaded image, runs helmet_model(image, conf=CONF_THRESHOLD) and counts 'Without Helmet' boxes.

Runs traffic_model(image, conf=CONF_THRESHOLD) to find traffic light and compute color.

Annotates image, shows results and metrics in columns.

Streamlit UI (bottom of file)

Loads models and checks that model classes (helmet_model.names, traffic_model.names) include required labels.

Tabs: Process Image and Process Video with uploaders and action buttons.

Violation log shown after processing with a refresh button (st.rerun()).

7. Architecture (diagram + responsibilities)

Simple ASCII architecture:

+----------------------+      +------------------+      +----------------+
|     Streamlit UI     | ---> |  Inference Loop  | ---> |  YOLO Models   |
| (file upload, metrics)|      | (frame-by-frame) |      | (helmet,traffic)|
+----------+-----------+      +---------+--------+      +--------+-------+
           |                                |                       |
           |                                v                       v
           |                      +----------------+       +------------------+
           |                      | Violation Logic|       |Traffic Light CV  |
           |                      | (stop-line,    |       |(HSV thresholds)  |
           +--------------------->|  tracker state)|       +------------------+
                                  +--------+-------+
                                           |
                                           v
                                  +------------------+
                                  | SQLite DB (logs) |
                                  +------------------+


Component responsibilities:

Streamlit UI: user interaction, model load, show annotated frames and metrics, let user upload file.

Inference Loop: runs tracking/detection, aggregates trackers, calls color classification.

Violation Logic: decides when to log helmet violations or signal jumps (based on tracker id and position).

Database: stores timestamp, violation_type, and tracker_id.

8. Configurable parameters & tuning

CONF_THRESHOLD — increase to reduce false positives (e.g., 0.6–0.7) or decrease to catch more detections (more false positives).

STOP_LINE_Y — set appropriate stop line depending on camera resolution/angle. Could be derived from calibration steps or UI slider.

HSV thresholds in get_traffic_light_color() — tune for local camera conditions.

value_mask threshold >150 — brightness threshold; reduce for darker scenes.

Frame skip: to improve performance, process every Nth frame (e.g., skip = 2 or 3).

9. Database schema & logs

Table violations:

CREATE TABLE violations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT,
  violation_type TEXT,
  tracker_id INTEGER
);


Query examples:

SELECT * FROM violations ORDER BY timestamp DESC; — get all logs.

The app uses pandas.read_sql_query to display the logs.

10. Troubleshooting & common fixes
1) Models fail to load / YOLO() errors

Ensure best2.pt and yolov8n.pt exist and are compatible with your ultralytics version.

Use absolute path or place model in same directory as app.py.

If ultralytics API changed, check model.names type — sometimes .names is dict or list; adapt code accordingly, e.g.:

helmet_names = helmet_model.names
if isinstance(helmet_names, dict):
    helmet_names = [helmet_names[i] for i in range(len(helmet_names))]

2) Very slow video processing

Use GPU build of PyTorch + appropriate CUDA.

Lower model size (yolov8n is small; yolov8s or yolov8n are good). Use yolov8n.pt for speed.

Skip frames: process every 2–3 frames.

Reduce input resolution before passing to the model: cv2.resize(frame, (640, 360)).

3) False traffic light classification

Adjust HSV thresholds and brightness threshold value_mask.

Sometimes traffic light boxes include background — crop area carefully.

Consider a specialized traffic-light classifier that takes the detection crop and returns color using small CNN — more robust.

4) AttributeError or None boxes.id

Some YOLO outputs may not include tracking metadata if track is not used or track fails. Use safe checks:

boxes = helmet_results[0].boxes
ids = getattr(boxes, 'id', None)
if ids is not None:
    # proceed

5) Duplicate st.stop() or stray code

In your pasted file there was an extra st.stop() out-of-place. Make sure each exception path has st.stop() only when needed. Remove stray st.stop() lines.

11. Improvements & next steps (recommended)

Add an interactive slider to set STOP_LINE_Y in the UI (makes calibration easier).

Add an FPS/processing-time display for profiling.

Store richer log data — vehicle class, confidence, frame number, snapshot image path.

Add model warm-up and frame resizing for consistent performance.

Replace HSV heuristics for traffic light color with a small classifier trained on traffic light crops (more robust to lighting).

Add authentication / user roles for production.

Add option to export logs (CSV) from the UI.

Use Redis or Postgres (instead of SQLite) when scaling multi-user or cloud deployment.

12. Minimal README.md (drop-in)
# UrbanPulse

UrbanPulse is a Streamlit app that detects helmet and red-light violations using YOLO models.  
Place `best2.pt` (helmet) and `yolov8n.pt` (traffic) in the project root or `models/`.

## Quick start

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt  # or pip install streamlit ultralytics opencv-python numpy pandas
streamlit run app.py


Open the UI, upload an image or video and click the appropriate button.

Config

CONF_THRESHOLD in app.py: detection threshold

STOP_LINE_Y: y-coord for stop line

Database

violations.db automatically created in project root.
