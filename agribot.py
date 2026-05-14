#!/home/agribot/Desktop/agri-workspace/.venv/bin/python
import io
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import RPi.GPIO as GPIO
from flask import Flask, Response, render_template, jsonify, send_file, request
from picamera2 import Picamera2
from ultralytics import YOLO
from agriMove import AgriMove

# ── Pin assignments ───────────────────────────────────────────────────────────
# GPIO 18     = ESC (sysfs PWM, handled by agriMove.py)
# PCA9685 CH0 = Camera tilt servo MG90S (I2C)
# PCA9685 CH2 = Camera pan  servo MG90S (I2C)
# PCA9685 CH4 = Steering servo JX PDI-6621 (I2C, handled by agriMove.py)
# GPIO 22     = Laser
LASER_PIN = 22

DATASET_DIR = Path("/home/agribot/Desktop/agri-workspace/dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ── GPIO setup (laser only — servos use PCA9685 I2C, not RPi.GPIO) ──────────
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LASER_PIN, GPIO.OUT)
GPIO.output(LASER_PIN, GPIO.LOW)

# ── Camera servos — PCA9685 I2C ──────────────────────────────────────────────
# CH2 = pan  (left / right)
# CH0 = tilt (up   / down)
from agriMove import _PCA9685Channel

camera_pan  = _PCA9685Channel(2)
camera_tilt = _PCA9685Channel(0)

camera_pan.set_pulsewidth_us(1500)    # centre on startup
camera_tilt.set_pulsewidth_us(1500)   # centre on startup

# ── Drivetrain ────────────────────────────────────────────────────────────────
gpio_lock = threading.Lock()
agri_move = AgriMove(gpio_lock)

state = {
    'servo1': 0,   # camera pan  (PCA9685 CH2, left/right)
    'servo2': 0,   # camera tilt (PCA9685 CH0, up/down)
    'laser':  False,
}


def _angle_to_us(angle: int) -> int:
    """Map −90..90 ° to 500..2500 µs (MG90S full range)."""
    return int(1500 + (max(-90, min(90, angle)) / 90.0) * 1000)


_servo_devices = {}   # populated after servos are constructed


def _servo_device(servo):
    return _servo_devices.get(servo)


def set_servo(servo, angle):
    angle = max(-90, min(90, int(angle)))
    _servo_device(servo).set_pulsewidth_us(_angle_to_us(angle))
    state[servo] = angle


# Generation counter: incrementing before a new smooth_move causes the running
# thread to notice it has been preempted and exit early.
_smooth_gen = {'servo1': 0, 'servo2': 0}


def smooth_move(servo, target):
    dev = _servo_device(servo)
    _smooth_gen[servo] += 1
    gen = _smooth_gen[servo]

    def _run():
        current = state[servo]
        target_ = max(-90, min(90, int(target)))
        if current == target_:
            return
        step = 1 if target_ > current else -1
        for a in range(current, target_ + step, step):
            if _smooth_gen[servo] != gen:   # preempted by newer call
                return
            dev.set_pulsewidth_us(_angle_to_us(a))
            state[servo] = a
            time.sleep(0.015)

    threading.Thread(target=_run, daemon=True).start()


_servo_devices['servo1'] = camera_pan
_servo_devices['servo2'] = camera_tilt

# ── Auto-tracking ─────────────────────────────────────────────────────────────
auto_track  = False
track_lock  = threading.Lock()

# Approximate Pi Camera v2 FOV; adjust if you have a different lens.
_TRACK_FOV_H      = 62.0   # horizontal degrees
_TRACK_FOV_V      = 48.0   # vertical degrees
_TRACK_W          = 640
_TRACK_H          = 480
_TRACK_DEADZONE   = 18     # pixels — ignore error smaller than this
_TRACK_GAIN       = 0.21   # proportional gain — 70% reduced from 0.7 for smooth lock-in
_TRACK_LABELS     = {'WeedA'}

# Laser box offset — kept in sync with the frontend sliders via /api/laser/offset
laser_ox   = 0
laser_oy   = 0
laser_offset_lock = threading.Lock()


# ── YOLO model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/home/agribot/Desktop/agri-workspace/best.pt")
model = YOLO(str(MODEL_PATH))

# Colours per class (BGR): WeedA=red, Pest=purple
CLASS_COLORS = {
    'WeedA': (0, 60, 220),
    'Pest':  (180, 0, 180),
}
DEFAULT_COLOR = (0, 200, 80)

detect_enabled = True
detect_lock    = threading.Lock()
latest_dets    = []          # list of dicts: {label, conf, x1,y1,x2,y2}
detect_lock_dets = threading.Lock()

CONF_THRESHOLD = 0.35
DETECT_INTERVAL = 0.4        # seconds between inference runs


# ── Camera ────────────────────────────────────────────────────────────────────
camera = Picamera2()
cam_config = camera.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    controls={"AwbEnable": True, "AeEnable": True, "AfMode": 2, "NoiseReductionMode": 2}
)
camera.configure(cam_config)
camera.start()
time.sleep(2)

stream_lock       = threading.Lock()
latest_frame      = None       # annotated JPEG bytes
latest_raw_frame  = None       # raw numpy BGR array (for inference)
last_capture_path = None
frame_lock_raw    = threading.Lock()


def correct_colors(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw_detections(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        label = d['label']
        conf  = d['conf']
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text   = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


_stop_capture = threading.Event()


def capture_frames():
    global latest_frame, latest_raw_frame
    while not _stop_capture.is_set():
        frame     = camera.capture_array()
        corrected = correct_colors(frame)
        with frame_lock_raw:
            latest_raw_frame = corrected.copy()

        with detect_lock:
            en = detect_enabled
        if en:
            with detect_lock_dets:
                dets = list(latest_dets)
            annotated = draw_detections(corrected.copy(), dets)
        else:
            annotated = corrected

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with stream_lock:
            latest_frame = jpeg.tobytes()
        time.sleep(0.033)


def detection_loop():
    while not _stop_capture.is_set():
        with detect_lock:
            en = detect_enabled
        if not en:
            time.sleep(0.1)
            continue

        with frame_lock_raw:
            raw = latest_raw_frame
        if raw is None:
            time.sleep(0.1)
            continue

        results = model(raw, conf=CONF_THRESHOLD, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append({'label': label, 'conf': conf,
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        with detect_lock_dets:
            latest_dets.clear()
            latest_dets.extend(dets)

        time.sleep(DETECT_INTERVAL)


_TRACK_DEBOUNCE = 2.0   # settling window (seconds) after initial snap move


def tracking_loop():
    """Three-phase weed tracker:
      1. INITIAL SNAP  — weed first detected: immediately calculate full-correction
                         pan/tilt angles and move there in one smooth move.
      2. SETTLING      — wait _TRACK_DEBOUNCE seconds for the camera to physically
                         reach the position and for detections to stabilise.
      3. FINE TRACKING — weed still visible after settling: apply proportional
                         corrections continuously until weed leaves frame."""
    deg_px_h = _TRACK_FOV_H / _TRACK_W   # ~0.097 deg/px
    deg_px_v = _TRACK_FOV_V / _TRACK_H   # ~0.100 deg/px
    weed_first_seen = None
    snapped = False   # True after Phase 1 snap fires; prevents reset on YOLO loss

    while not _stop_capture.is_set():
        with track_lock:
            active = auto_track
        if not active:
            weed_first_seen = None
            snapped = False
            time.sleep(0.15)
            continue

        with detect_lock_dets:
            weeds = [d for d in latest_dets if d['label'] in _TRACK_LABELS]

        if not weeds:
            if not snapped:
                # No detection and no snap yet — reset, keep waiting
                weed_first_seen = None
            # After snap: YOLO may flicker — hold position, don't reset
            time.sleep(0.15)
            continue

        best    = max(weeds, key=lambda d: d['conf'])
        weed_cx = (best['x1'] + best['x2']) / 2.0
        weed_cy = (best['y1'] + best['y2']) / 2.0

        # Target is the laser box center (frame center + calibration offset)
        with laser_offset_lock:
            ox, oy = laser_ox, laser_oy
        target_x = _TRACK_W / 2.0 + ox
        target_y = _TRACK_H / 2.0 + oy

        dx = weed_cx - target_x   # + = weed is right of laser box
        dy = weed_cy - target_y   # + = weed is below laser box

        now = time.time()

        # ── Phase 1: initial snap ─────────────────────────────────────────────
        if weed_first_seen is None:
            weed_first_seen = now
            snapped = True
            pan_snap  = max(-90.0, min(90.0, state['servo1'] - dx * deg_px_h))
            tilt_snap = max(-90.0, min(90.0, state['servo2'] + dy * deg_px_v))
            smooth_move('servo1', int(pan_snap))
            smooth_move('servo2', int(tilt_snap))
            time.sleep(0.15)
            continue

        # ── Phase 2: settling window ──────────────────────────────────────────
        if now - weed_first_seen < _TRACK_DEBOUNCE:
            time.sleep(0.15)
            continue

        # ── Phase 3: fine tracking ────────────────────────────────────────────
        if abs(dx) < _TRACK_DEADZONE and abs(dy) < _TRACK_DEADZONE:
            time.sleep(0.15)
            continue

        # Pan: negate dx because JS negates servo1 angle (hardware inversion fix)
        pan_target  = max(-90.0, min(90.0, state['servo1'] - dx * deg_px_h * _TRACK_GAIN))
        tilt_target = max(-90.0, min(90.0, state['servo2'] + dy * deg_px_v * _TRACK_GAIN))

        smooth_move('servo1', int(pan_target))
        smooth_move('servo2', int(tilt_target))

        time.sleep(DETECT_INTERVAL + 0.1)


_capture_thread   = threading.Thread(target=capture_frames,   daemon=True)
_detection_thread = threading.Thread(target=detection_loop,   daemon=True)
_tracking_thread  = threading.Thread(target=tracking_loop,    daemon=True)
_capture_thread.start()
_detection_thread.start()
_tracking_thread.start()

# ── Flask ─────────────────────────────────────────────────────────────────────
# HTML → templates/index.html
# CSS  → static/style.css
# JS   → static/app.js
app = Flask(__name__)


def generate_stream():
    while True:
        with stream_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/servo", methods=["POST"])
def api_servo():
    try:
        data      = request.get_json(force=True, silent=True) or {}
        servo     = data.get("servo")
        angle     = int(data.get("angle", 0))
        do_smooth = bool(data.get("smooth", False))
        if servo not in state or servo == 'laser':
            return jsonify({"error": f"unknown servo '{servo}'"}), 400
        if do_smooth:
            smooth_move(servo, angle)
        else:
            set_servo(servo, angle)
        return jsonify({"success": True, "servo": servo, "angle": state[servo]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/laser/offset", methods=["POST"])
def api_laser_offset():
    global laser_ox, laser_oy
    try:
        data = request.get_json(force=True, silent=True) or {}
        with laser_offset_lock:
            laser_ox = int(data.get("ox", 0))
            laser_oy = int(data.get("oy", 0))
        return jsonify({"success": True, "ox": laser_ox, "oy": laser_oy})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/laser", methods=["POST"])
def api_laser():
    try:
        data = request.get_json(force=True, silent=True) or {}
        on   = bool(data.get("on", False))
        with gpio_lock:
            GPIO.output(LASER_PIN, GPIO.HIGH if on else GPIO.LOW)
            state["laser"] = on
        return jsonify({"success": True, "laser": state["laser"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/track/toggle", methods=["POST"])
def api_track_toggle():
    global auto_track
    with track_lock:
        auto_track = not auto_track
        t = auto_track
    return jsonify({"tracking": t})


@app.route("/api/status")
def api_status():
    with track_lock:
        t = auto_track
    return jsonify({**state, 'auto_track': t})


@app.route("/api/detections")
def api_detections():
    with detect_lock_dets:
        dets = list(latest_dets)
    return jsonify(dets)


@app.route("/api/move", methods=["POST"])
def api_move():
    try:
        data      = request.get_json(force=True, silent=True) or {}
        direction = data.get("direction", "stop")
        if direction == "forward":
            agri_move.forward()
        elif direction == "backward":
            agri_move.backward()
        else:
            agri_move.stop()
        return jsonify({"success": True, "state": agri_move.get_state()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/move/steer", methods=["POST"])
def api_move_steer():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        angle = int(data.get("angle", 0))
        agri_move.steer(angle)
        return jsonify({"success": True, "state": agri_move.get_state()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/move/speed", methods=["POST"])
def api_move_speed():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        speed = int(data.get("speed", 50))
        agri_move.set_speed(speed)
        return jsonify({"success": True, "state": agri_move.get_state()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/move/status")
def api_move_status():
    return jsonify(agri_move.get_state())


@app.route("/api/detect/toggle", methods=["POST"])
def api_detect_toggle():
    global detect_enabled
    with detect_lock:
        detect_enabled = not detect_enabled
        en = detect_enabled
    if not en:
        with detect_lock_dets:
            latest_dets.clear()
    return jsonify({"enabled": en})


@app.route("/capture", methods=["POST"])
def capture():
    global last_capture_path
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"capture_{timestamp}.jpg"
        filepath  = DATASET_DIR / filename
        frame     = camera.capture_array()
        corrected = correct_colors(frame)
        cv2.imwrite(str(filepath), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        last_capture_path = filepath
        return jsonify({"success": True, "filename": filename, "path": str(filepath)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/preview")
def preview():
    if last_capture_path and Path(last_capture_path).exists():
        return send_file(str(last_capture_path), mimetype="image/jpeg")
    return Response(status=404)


def _shutdown():
    _stop_capture.set()
    _capture_thread.join(timeout=2)
    try:
        camera.stop()
    except Exception:
        pass
    try:
        agri_move.cleanup()
    except Exception:
        pass
    for dev in _servo_devices.values():
        try:
            dev.stop()
        except Exception:
            pass
    GPIO.cleanup()
    print("GPIO cleaned up.")


if __name__ == "__main__":
    print("AgriBot Interface running at http://localhost:5000")
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            _shutdown()
        except KeyboardInterrupt:
            GPIO.cleanup()
            print("Force quit.")
