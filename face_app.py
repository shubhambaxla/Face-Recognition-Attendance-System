import os, cv2, json, time, csv, threading
import numpy as np
import gradio as gr
from datetime import datetime, time as dtime
from pathlib import Path
try:
    from fer import FER  # Facial emotion recognition
    _FER_AVAILABLE = True
except Exception:
    _FER_AVAILABLE = False

# Reuse your detector implementation (adjust import if you used different name)
from face_recognition_project.utils import get_face_detector
from face_recognition_project.capture_dataset import capture_for_person

# Set the correct path for dataset and model directories
ROOT = Path("C:/Users/shubh/Desktop/machine learning try 2")  # Root directory where 'dataset' and 'models' are stored
DATASET_DIR = ROOT / "dataset"  # Path to the dataset folder
# Use the working project's models directory to stay consistent with backend scripts
MODEL_DIR = ROOT / "face_recognition_project" / "models"
MODEL_PATH = MODEL_DIR / "lbph_model.xml"  # Path to the trained model file
LABELS_PATH = MODEL_DIR / "lbph_labels.json"  # Path to the labels file
ATTENDANCE_FILE = ROOT / "attendance.csv"  # Path to the attendance log file

# Default attendance time slots (24h format HH:MM)
DEFAULT_SLOTS = {
    "Morning (09:00-10:00)": ("09:00", "10:00"),
    "Afternoon (13:00-14:00)": ("13:00", "14:00"),
    "Evening (17:00-18:00)": ("17:00", "18:00"),
}

# Ensure dataset and models directory exists
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Global recognizer + detector + label_map (will be loaded/reloaded)
recognizer_lock = threading.Lock()
detector = get_face_detector()

# Load model if it exists
def load_model_if_exists():
    """Load model if it exists; otherwise create an empty recognizer."""
    rec = cv2.face.LBPHFaceRecognizer_create()
    if MODEL_PATH.exists():
        rec.read(str(MODEL_PATH))
    else:
        # empty recognizer (won't predict until trained)
        pass
    # load labels if exists
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r") as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
    else:
        label_map = {}
    return rec, label_map

rec, label_map = load_model_if_exists()

# Default number of samples to save per click
SAMPLES_PER_CLICK = 500

# --- Simple data augmentation for grayscale images ---
def augment_gray_image(gray: np.ndarray) -> np.ndarray:
    """Create a light random augmentation of a grayscale image for diversity."""
    h, w = gray.shape[:2]
    # random rotation within [-15, 15]
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # random horizontal flip
    if np.random.rand() < 0.5:
        aug = cv2.flip(aug, 1)
    # random brightness/contrast
    alpha = np.random.uniform(0.9, 1.1)  # contrast
    beta = np.random.uniform(-15, 15)    # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    # random gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 5, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug

# Helper: train model from dataset
def train_lbph_and_save():
    X = []
    y = []
    label_map_local = {}
    cur_label = 0
    # iterate persons
    for person in sorted(os.listdir(DATASET_DIR)):
        person_dir = DATASET_DIR / person
        if not person_dir.is_dir(): 
            continue
        label_map_local[cur_label] = person
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            img = cv2.imread(str(person_dir / fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(img)
            y.append(cur_label)
        cur_label += 1
    if len(X) == 0:
        return False, "No images found in dataset. Capture some samples first."
    # train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X, np.array(y))
    recognizer.write(str(MODEL_PATH))
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_map_local, f)
    # reload global rec & labels
    global rec, label_map
    with recognizer_lock:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read(str(MODEL_PATH))
        label_map = label_map_local
    return True, f"Trained on {len(set(y))} people, {len(X)} images."

# Helper: get next filename index
def next_index_for_person(person):
    pdir = DATASET_DIR / person
    pdir.mkdir(parents=True, exist_ok=True)
    existing = [f for f in os.listdir(pdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    return len(existing)

# Save a single sample image (gradio returns RGB numpy array)
def save_sample(name, image, num_samples: int = SAMPLES_PER_CLICK):
    if name is None or len(name.strip()) == 0:
        return False, "Enter a valid name (no spaces recommended)."
    if image is None:
        return False, "No image received. Use webcam or upload."
    safe_name = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-")).strip()
    person_dir = DATASET_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index_for_person(safe_name)
    # Gradio gives RGB; convert to grayscale once
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    base_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    saved = 0
    for i in range(num_samples):
        if i == 0:
            gray_to_save = base_gray
        else:
            gray_to_save = augment_gray_image(base_gray)
        filename = person_dir / f"{start_idx + i:04d}.png"
        ok = cv2.imwrite(str(filename), gray_to_save)
        if ok:
            saved += 1
    return True, f"Saved {saved} samples for {safe_name} (total now {start_idx + saved})"

# Recognize function: annotate image & optionally log attendance
def _parse_hhmm(hhmm: str) -> dtime:
    try:
        hh, mm = hhmm.split(":")
        return dtime(int(hh), int(mm))
    except Exception:
        return None


def _now_in_slot(slot_start: str, slot_end: str) -> bool:
    ts = datetime.now().time()
    t1 = _parse_hhmm(slot_start)
    t2 = _parse_hhmm(slot_end)
    if t1 is None or t2 is None:
        return False
    if t1 <= t2:
        return t1 <= ts <= t2
    # Overnight slot (e.g., 22:00-02:00)
    return ts >= t1 or ts <= t2


def recognize_image(image, do_log=True, conf_threshold=200, slot_label: str = None, slot_start: str = None, slot_end: str = None):
    if image is None:
        return None, "No image"
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    results = []
    for (x,y,w,h) in rects:
        face = gray[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face, (160,160))
        except Exception:
            face_resized = cv2.resize(face, (max(20, face.shape[1]), max(20, face.shape[0])))

        with recognizer_lock:
            if LABELS_PATH.exists() and MODEL_PATH.exists():
                label, conf = rec.predict(face_resized)
            else:
                label, conf = -1, 9999
        if label != -1 and label in label_map and conf < conf_threshold:
            name = label_map.get(label, "Unknown")
            results.append((name, float(conf), (x,y,w,h)))
            if do_log:
                if slot_start and slot_end:
                    if _now_in_slot(slot_start, slot_end):
                        log_attendance(name, slot_label or f"{slot_start}-{slot_end}")
                else:
                    # If no slot provided, behave like previous always-log
                    log_attendance(name, None)
        else:
            results.append(("Unknown", float(conf), (x,y,w,h)))

    # draw boxes and text
    for name, conf, (x,y,w,h) in results:
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(bgr, (x,y), (x+w, y+h), color, 2)
        text = f"{name} ({conf:.0f})"
        cv2.putText(bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, f"Detected {len(results)} faces."

# Attendance logging (append to CSV)
def _already_logged_today(name: str, slot_label: str | None) -> bool:
    if not ATTENDANCE_FILE.exists():
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    n, ts = row[0], row[1]
                    slot = row[2] if len(row) > 2 else None
                    if n == name and ts.startswith(today):
                        if slot_label is None:
                            if slot is None:
                                return True
                        else:
                            if slot == slot_label:
                                return True
    except Exception:
        return False
    return False


def log_attendance(name, slot_label: str | None):
    if _already_logged_today(name, slot_label):
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = not ATTENDANCE_FILE.exists()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["name","timestamp","slot"])  # new header with slot
        if slot_label is None:
            writer.writerow([name, ts])
        else:
            writer.writerow([name, ts, slot_label])

# Gradio UI callbacks
def on_save_sample(name, img):
    ok, msg = save_sample(name, img)
    return msg, show_dataset_counts()

def on_retrain():
    ok, msg = train_lbph_and_save()
    return msg, show_dataset_counts()

def on_recognize(img, slot_choice, start_text, end_text):
    slot_start, slot_end = None, None
    slot_label = None
    if slot_choice and slot_choice in DEFAULT_SLOTS:
        slot_start, slot_end = DEFAULT_SLOTS[slot_choice]
        slot_label = slot_choice
    # Allow overrides via text boxes
    if start_text and end_text:
        slot_start, slot_end = start_text.strip(), end_text.strip()
        slot_label = slot_label or f"{slot_start}-{slot_end}"
    out_img, msg = recognize_image(img, do_log=True, slot_label=slot_label, slot_start=slot_start, slot_end=slot_end)
    return out_img, msg

def on_recognize_stream(img, slot_choice, start_text, end_text):
    """Streaming variant for live webcam recognition with slot-gated logging."""
    slot_start, slot_end = None, None
    slot_label = None
    if slot_choice and slot_choice in DEFAULT_SLOTS:
        slot_start, slot_end = DEFAULT_SLOTS[slot_choice]
        slot_label = slot_choice
    if start_text and end_text:
        slot_start, slot_end = start_text.strip(), end_text.strip()
        slot_label = slot_label or f"{slot_start}-{slot_end}"
    out_img, msg = recognize_image(img, do_log=True, slot_label=slot_label, slot_start=slot_start, slot_end=slot_end)
    return out_img, msg

def on_capture_500(name):
    """Run native webcam capture to collect 500 real frames using the backend script."""
    if name is None or len(name.strip()) == 0:
        return "Enter a valid name first.", show_dataset_counts()
    safe_name = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-")).strip()
    # Run capture in a background thread to avoid blocking Gradio UI
    def _worker():
        try:
            # capture_for_person expects out_dir as str
            capture_for_person(safe_name, out_dir=str(DATASET_DIR), samples=500)
        except Exception as e:
            pass
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return "Started native capture window for 500 samples. Close it or press 'q' to stop.", show_dataset_counts()

def show_dataset_counts():
    lines = []
    for person in sorted(os.listdir(DATASET_DIR)):
        p = DATASET_DIR / person
        if not p.is_dir(): continue
        count = len([f for f in os.listdir(p) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        lines.append(f"{person} : {count}")
    return "\n".join(lines) if lines else "No persons in dataset yet."

# --- Facial Emotion Recognition (FER) ---
_fer_detector = None

def analyze_emotions(image_rgb: np.ndarray):
    if image_rgb is None:
        return None, {"error": "No image"}
    if not _FER_AVAILABLE:
        return image_rgb, {"error": "FER not installed. Run: pip install fer mtcnn"}
    global _fer_detector
    if _fer_detector is None:
        try:
            _fer_detector = FER(mtcnn=True)
        except Exception:
            _fer_detector = FER()
    # FER expects RGB
    results = _fer_detector.detect_emotions(image_rgb)
    annotated = image_rgb.copy()
    payload = []
    for r in results:
        (x, y, w, h) = r.get('box', (0,0,0,0))
        emotions = r.get('emotions', {})
        if emotions:
            top_label = max(emotions, key=emotions.get)
            score = emotions[top_label]
        else:
            top_label, score = 'neutral', 0.0
        # draw
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 215, 0), 2)
        cv2.putText(annotated, f"{top_label} ({score:.2f})", (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        payload.append({"box": [int(x), int(y), int(w), int(h)], "top_emotion": top_label, "score": float(score), "emotions": emotions})
    return annotated, payload

CUSTOM_CSS = """
/* App container */
.app-container { max-width: 1200px; margin: 0 auto; }

/* Navbar */
.app-navbar { position: sticky; top: 0; z-index: 5; background: #0d1117; border-bottom: 1px solid #222; }
.app-navbar-inner { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; }
.brand { color: #e6edf3; font-weight: 700; letter-spacing: .2px; }
.nav a { color: #9bbcf5; text-decoration: none; padding: 8px 10px; border-radius: 8px; }
.nav a:hover { background: #1f6feb22; }

/* Hero */
.hero { padding: 24px 16px 6px; }
.hero h1 { margin: 6px 0 4px; color: #e6edf3; }
.muted { color: #8b949e; }

/* Cards */
.card { border: 1px solid #30363d; background: #0b0f14; border-radius: 12px; padding: 16px; }
.card h3 { margin: 0 0 8px; color: #e6edf3; }
.section-title { margin: 16px 0 8px; color: #c9d1d9; }

/* Footer */
.footer { margin: 24px 0 12px; color: #8b949e; text-align: center; }
"""

# Gradio layout
with gr.Blocks(title="Face Enrollment + LBPH Trainer + Recognizer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    # Navbar
    gr.HTML("""
    <div class='app-navbar'>
      <div class='app-navbar-inner app-container'>
        <div class='brand'>Face Enrollment & Attendance</div>
        <nav class='nav'>
          <a href="#enroll">Enroll</a>
          <a href="#train">Train</a>
          <a href="#recognize">Recognize</a>
          <a href="#live">Live</a>
        </nav>
      </div>
    </div>
    """)

    # Hero / Intro
    gr.HTML("""
    <div class='app-container hero'>
      <h1>LBPH Face Recognition Suite</h1>
      <p class='muted'>Capture samples → Retrain → Recognize & log attendance. Optimized for clarity and speed.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Enroll", id="enroll"):
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("<div class='section-title'>Enrollment</div>")
                    with gr.Group(elem_classes=["card"]):
                        name_input = gr.Textbox(label="Person name", placeholder="e.g. shubh")
                        webcam = gr.Image(sources=["webcam", "upload"], type="numpy", label="Webcam / Upload")
                        with gr.Row():
                            save_btn = gr.Button("Save Sample", variant="primary")
                            capture_500_btn = gr.Button("Capture 500 (native)")
                with gr.Column(scale=6):
                    gr.HTML("<div class='section-title'>Dataset & Training</div>")
                    with gr.Group(elem_classes=["card"], elem_id="train"):
                        dataset_info = gr.Textbox(label="Dataset status", value=show_dataset_counts(), interactive=False)
                        retrain_btn = gr.Button("Retrain Model (LBPH)", variant="primary")
                        retrain_out = gr.Textbox(label="Retrain output")
                    with gr.Group(elem_classes=["card"]):
                        save_out = gr.Textbox(label="Save output")

        with gr.TabItem("Recognize", id="recognize"):
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("<div class='section-title'>Recognize (Test)</div>")
                    with gr.Group(elem_classes=["card"]):
                        slot_choice = gr.Dropdown(label="Attendance Slot (preset)", choices=list(DEFAULT_SLOTS.keys()), value=list(DEFAULT_SLOTS.keys())[0])
                        with gr.Row():
                            slot_start_text = gr.Textbox(label="Start (HH:MM)", value=DEFAULT_SLOTS[list(DEFAULT_SLOTS.keys())[0]][0])
                            slot_end_text = gr.Textbox(label="End (HH:MM)", value=DEFAULT_SLOTS[list(DEFAULT_SLOTS.keys())[0]][1])
                        test_img = gr.Image(type="numpy", label="Recognition Input")
                        recognize_btn = gr.Button("Recognize & Log Attendance", variant="primary")
                with gr.Column(scale=6):
                    gr.HTML("<div class='section-title'>Output</div>")
                    with gr.Group(elem_classes=["card"]):
                        recognized = gr.Image(label="Recognized (output)")
                        rec_status = gr.Textbox(label="Recognition status")

        with gr.TabItem("Live", id="live"):
            with gr.Column():
                gr.HTML("<div class='section-title'>Live Recognition</div>")
                with gr.Group(elem_classes=["card"]):
                    live_slot_choice = gr.Dropdown(label="Attendance Slot (preset)", choices=list(DEFAULT_SLOTS.keys()), value=list(DEFAULT_SLOTS.keys())[0])
                    with gr.Row():
                        live_slot_start_text = gr.Textbox(label="Start (HH:MM)", value=DEFAULT_SLOTS[list(DEFAULT_SLOTS.keys())[0]][0])
                        live_slot_end_text = gr.Textbox(label="End (HH:MM)", value=DEFAULT_SLOTS[list(DEFAULT_SLOTS.keys())[0]][1])
                    live_cam = gr.Image(sources=["webcam"], type="numpy", label="Live Recognition (webcam)", streaming=True)
        
        with gr.TabItem("Emotion", id="emotion"):
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("<div class='section-title'>Facial Emotion Recognition</div>")
                    with gr.Group(elem_classes=["card"]):
                        emo_img = gr.Image(type="numpy", label="Image (RGB)")
                        emo_btn = gr.Button("Analyze Emotions", variant="primary")
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        emo_out = gr.Image(label="Annotated")
                        emo_json = gr.JSON(label="Emotions JSON")

    gr.HTML("""
    <div class='app-container footer'>
      <small>LBPH + Haar Cascade • Built with Gradio Blocks • Attendance CSV logging</small>
    </div>
    """)

    # events
    save_btn.click(on_save_sample, inputs=[name_input, webcam], outputs=[save_out, dataset_info])
    capture_500_btn.click(on_capture_500, inputs=[name_input], outputs=[save_out, dataset_info])
    retrain_btn.click(on_retrain, outputs=[retrain_out, dataset_info])
    recognize_btn.click(on_recognize, inputs=[test_img, slot_choice, slot_start_text, slot_end_text], outputs=[recognized, rec_status])
    live_cam.stream(on_recognize_stream, inputs=[live_cam, live_slot_choice, live_slot_start_text, live_slot_end_text], outputs=[recognized, rec_status])
    emo_btn.click(lambda img: analyze_emotions(img), inputs=[emo_img], outputs=[emo_out, emo_json])

    # refresh dataset_info on load
    demo.load(fn=show_dataset_counts, outputs=[dataset_info])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
