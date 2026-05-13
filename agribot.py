import io
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import RPi.GPIO as GPIO
from flask import Flask, Response, render_template_string, jsonify, send_file, request
from picamera2 import Picamera2
from ultralytics import YOLO
from agriMove import AgriMove

# ── Pin assignments ───────────────────────────────────────────────────────────
# GPIO 18 = ESC (sysfs PWM, handled by agriMove.py)
# GPIO 19 = Camera pan servo MG90S — pwmchip0/pwm3 (hardware PWM, sysfs)
# GPIO 22 = Laser
LASER_PIN = 22

DATASET_DIR = Path("/home/agribot/Desktop/agri-workspace/dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ── GPIO setup (laser only — servos use sysfs hardware PWM, not RPi.GPIO) ────
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LASER_PIN, GPIO.OUT)
GPIO.output(LASER_PIN, GPIO.LOW)

# ── Camera pan servo — sysfs hardware PWM on GPIO 19 (pwmchip0/pwm3) ─────────
# GPIO 19 = PWM0_CHAN3 (confirmed via `pinctrl get 19`).
# Using sysfs instead of RPi.GPIO software PWM eliminates jitter on the MG90S.
from agriMove import _SysfsPWM, PWM_PERIOD_NS

_CAMERA_PWM_CHIP    = 0
_CAMERA_PWM_CHANNEL = 3   # pwmchip0/pwm3 = GPIO 19

camera_servo = _SysfsPWM(_CAMERA_PWM_CHIP, _CAMERA_PWM_CHANNEL)
camera_servo.set_pulsewidth_us(1500)   # centre on startup

# ── Drivetrain ────────────────────────────────────────────────────────────────
gpio_lock = threading.Lock()
agri_move = AgriMove(gpio_lock)

state = {
    'servo1': 0,   # camera pan angle in degrees
    'laser':  False,
}


def _angle_to_us(angle: int) -> int:
    """Map −90..90 ° to 1000..2000 µs (standard servo pulse width)."""
    return int(1500 + (max(-90, min(90, angle)) / 90.0) * 500)


def set_servo(servo, angle):
    angle = max(-90, min(90, int(angle)))
    camera_servo.set_pulsewidth_us(_angle_to_us(angle))
    state[servo] = angle


def smooth_move(servo, target):
    def _run():
        current = state[servo]
        target_ = max(-90, min(90, int(target)))
        direction = 1 if target_ > current else -1
        for a in range(current, target_ + direction, direction):
            camera_servo.set_pulsewidth_us(_angle_to_us(a))
            state[servo] = a
            time.sleep(0.02)
    threading.Thread(target=_run, daemon=True).start()


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


_capture_thread   = threading.Thread(target=capture_frames,   daemon=True)
_detection_thread = threading.Thread(target=detection_loop,   daemon=True)
_capture_thread.start()
_detection_thread.start()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriBot Interface</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117; color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 24px 16px; gap: 24px;
  }
  h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: .05em; color: #90cdf4; }
  .row {
    display: flex; gap: 24px; flex-wrap: wrap;
    justify-content: center; width: 100%; max-width: 1200px;
  }
  .panel {
    background: #1a1d27; border: 1px solid #2d3748;
    border-radius: 12px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .panel-title {
    padding: 10px 16px; font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .12em;
    color: #718096; border-bottom: 1px solid #2d3748;
    display: flex; justify-content: space-between; align-items: center;
  }
  .gpio-tag {
    font-size: .68rem; background: #2d3748;
    padding: 2px 7px; border-radius: 20px; color: #a0aec0; font-weight: 400;
  }

  /* Camera */
  .live-panel    { flex: 2; min-width: 320px; }
  .preview-panel { flex: 1; min-width: 240px; }
  .live-panel img, .preview-panel img { width: 100%; display: block; object-fit: cover; }
  .live-panel img    { min-height: 240px; background: #000; }
  .preview-panel img { min-height: 180px; background: #111; }
  .placeholder {
    width: 100%; min-height: 180px;
    display: flex; align-items: center; justify-content: center;
    color: #4a5568; font-size: .85rem;
  }
  .cam-controls {
    padding: 14px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    border-top: 1px solid #2d3748;
  }
  .capture-info {
    padding: 8px 16px; font-size: .72rem; color: #718096;
    border-top: 1px solid #2d3748; word-break: break-all;
  }

  /* Detection panel */
  .det-panel { flex: 1; min-width: 240px; max-width: 320px; }
  .det-body  { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; flex: 1; }
  .det-toggle-row { display: flex; align-items: center; gap: 10px; }
  .det-list  { display: flex; flex-direction: column; gap: 6px; min-height: 80px; }
  .det-item  {
    display: flex; justify-content: space-between; align-items: center;
    background: #252836; border-radius: 8px; padding: 7px 12px;
    font-size: .82rem;
  }
  .det-label { font-weight: 600; }
  .det-label.WeedA { color: #fc8181; }
  .det-label.Pest  { color: #b794f4; }
  .det-conf  { color: #68d391; font-size: .78rem; font-weight: 700; }
  .det-empty { color: #4a5568; font-size: .82rem; text-align: center; padding: 20px 0; }
  .det-count {
    font-size: .72rem; background: #2d3748;
    padding: 2px 7px; border-radius: 20px; color: #a0aec0;
  }

  /* Controls */
  .ctrl-panel { min-width: 300px; flex: 1; }
  .ctrl-body  { padding: 18px 20px; display: flex; flex-direction: column; gap: 20px; }

  .servo-block { display: flex; flex-direction: column; gap: 8px; }
  .servo-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .8rem; font-weight: 600; color: #a0aec0;
  }
  .servo-label span { font-size: .95rem; color: #68d391; font-weight: 700; min-width: 44px; text-align: right; }
  input[type=range] {
    -webkit-appearance: none; width: 100%; height: 6px;
    background: #2d3748; border-radius: 3px; outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: #4299e1; cursor: pointer;
    box-shadow: 0 0 0 3px rgba(66,153,225,.25);
  }
  .servo-btns { display: flex; gap: 6px; }
  .btn-sm {
    flex: 1; padding: 6px 0;
    background: #2d3748; color: #e2e8f0;
    border: 1px solid #4a5568; border-radius: 6px;
    font-size: .72rem; font-weight: 600; cursor: pointer; transition: background .12s;
  }
  .btn-sm:hover { background: #3d4e63; }
  .smooth-row {
    display: flex; align-items: center; gap: 8px;
    font-size: .75rem; color: #718096;
  }
  .smooth-row input[type=checkbox] { accent-color: #4299e1; }

  /* Laser */
  .laser-block {
    display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
  }
  .laser-info  { font-size: .8rem; font-weight: 600; color: #a0aec0; }
  .laser-status { font-size: .72rem; margin-top: 2px; }
  .laser-status.on  { color: #f6e05e; }
  .laser-status.off { color: #718096; }
  .btn-laser {
    padding: 10px 22px; border: none; border-radius: 8px;
    font-size: .85rem; font-weight: 700; cursor: pointer; transition: all .15s;
  }
  .btn-laser.on  { background: #f6e05e; color: #1a1d27; }
  .btn-laser.off { background: #2d3748; color: #a0aec0; border: 1px solid #4a5568; }
  .btn-laser:active { transform: scale(.96); }

  /* Shared */
  .btn-primary {
    background: #3182ce; color: white; border: none;
    padding: 10px 24px; border-radius: 8px;
    font-size: .9rem; font-weight: 600; cursor: pointer; transition: background .15s;
  }
  .btn-primary:hover    { background: #2b6cb0; }
  .btn-primary:active   { background: #2c5282; transform: scale(.97); }
  .btn-primary:disabled { background: #4a5568; cursor: not-allowed; }
  .btn-toggle {
    padding: 7px 16px; border-radius: 8px;
    font-size: .8rem; font-weight: 700; cursor: pointer; transition: all .15s; border: none;
  }
  .btn-toggle.on  { background: #276749; color: #c6f6d5; }
  .btn-toggle.off { background: #2d3748; color: #718096; border: 1px solid #4a5568; }
  #camStatus { font-size: .8rem; color: #68d391; min-width: 140px; }
  .divider { border: none; border-top: 1px solid #2d3748; }
  #toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #276749; color: #fff; padding: 10px 22px;
    border-radius: 8px; font-size: .85rem; font-weight: 600;
    opacity: 0; transition: opacity .3s; pointer-events: none; z-index: 999;
  }

  /* Movement panel */
  .move-panel { min-width: 300px; flex: 1; }
  .move-body  { padding: 18px 20px; display: flex; flex-direction: column; gap: 18px; }

  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 64px);
    grid-template-rows:    repeat(3, 64px);
    gap: 6px;
    justify-content: center;
  }
  .dpad-btn {
    width: 64px; height: 64px;
    background: #2d3748; color: #e2e8f0;
    border: 1px solid #4a5568; border-radius: 10px;
    font-size: 1.4rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .1s, transform .1s;
    user-select: none;
  }
  .dpad-btn:hover  { background: #3d4e63; }
  .dpad-btn:active { background: #4a5568; transform: scale(.93); }
  .dpad-btn.stop-btn {
    background: #742a2a; color: #feb2b2; border-color: #9b2c2c;
  }
  .dpad-btn.stop-btn:hover  { background: #9b2c2c; }
  .dpad-btn.active-dir {
    background: #2b6cb0; border-color: #4299e1; color: #bee3f8;
  }

  .speed-block { display: flex; flex-direction: column; gap: 8px; }
  .speed-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .8rem; font-weight: 600; color: #a0aec0;
  }
  .speed-label span { font-size: .95rem; color: #68d391; font-weight: 700; }

  .steer-block { display: flex; flex-direction: column; gap: 8px; }
  .steer-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .8rem; font-weight: 600; color: #a0aec0;
  }
  .steer-label span { font-size: .95rem; color: #f6ad55; font-weight: 700; }

  .move-status {
    font-size: .78rem; color: #718096; text-align: center;
  }
  .move-status b { color: #90cdf4; }
</style>
</head>
<body>
<div id="toast"></div>
<h1>AgriBot Interface</h1>

<!-- Camera row -->
<div class="row">
  <div class="panel live-panel">
    <div class="panel-title">Live Feed</div>
    <img src="/stream" alt="Live feed">
    <div class="cam-controls">
      <button class="btn-primary" id="captureBtn" onclick="capture()">Capture</button>
      <span id="camStatus"></span>
    </div>
  </div>
  <div class="panel preview-panel">
    <div class="panel-title">Last Capture</div>
    <div id="previewContainer"><div class="placeholder">No capture yet</div></div>
    <div class="capture-info" id="captureInfo"></div>
  </div>

  <!-- Detection panel -->
  <div class="panel det-panel">
    <div class="panel-title">
      Weed Detection
      <span class="det-count" id="detCount">0</span>
    </div>
    <div class="det-body">
      <div class="det-toggle-row">
        <button class="btn-toggle on" id="detToggle" onclick="toggleDetect()">Overlay ON</button>
        <span style="font-size:.72rem;color:#718096;">~2.5 fps inference</span>
      </div>
      <div class="det-list" id="detList">
        <div class="det-empty">No detections</div>
      </div>
    </div>
  </div>
</div>

<!-- Movement row -->
<div class="row">
  <div class="panel move-panel">
    <div class="panel-title">
      Movement Control
      <span class="gpio-tag">ESC GPIO 18</span>
    </div>
    <div class="move-body">

      <!-- D-Pad -->
      <div class="dpad">
        <!-- row 1 -->
        <div></div>
        <button class="dpad-btn" id="btn-forward"  onmousedown="driveStart('forward')"  onmouseup="driveStop()" onmouseleave="driveStop()"
                                                    ontouchstart="driveStart('forward')" ontouchend="driveStop()">&#9650;</button>
        <div></div>
        <!-- row 2 -->
        <button class="dpad-btn" id="btn-left"     onmousedown="steerDir('left')"    onmouseup="steerCenter()" onmouseleave="steerCenter()"
                                                    ontouchstart="steerDir('left')"  ontouchend="steerCenter()">&#9664;</button>
        <button class="dpad-btn stop-btn"           onclick="driveStop()">&#9632;</button>
        <button class="dpad-btn" id="btn-right"    onmousedown="steerDir('right')"   onmouseup="steerCenter()" onmouseleave="steerCenter()"
                                                    ontouchstart="steerDir('right')" ontouchend="steerCenter()">&#9654;</button>
        <!-- row 3 -->
        <div></div>
        <button class="dpad-btn" id="btn-backward" onmousedown="driveStart('backward')"  onmouseup="driveStop()" onmouseleave="driveStop()"
                                                    ontouchstart="driveStart('backward')" ontouchend="driveStop()">&#9660;</button>
        <div></div>
      </div>

      <div class="move-status" id="moveStatus">Direction: <b>stop</b> &nbsp;|&nbsp; Steering: <b>0°</b></div>

      <!-- Speed -->
      <div class="speed-block">
        <div class="speed-label">
          Speed
          <span id="speedVal">50%</span>
        </div>
        <input type="range" id="speedSlider" min="0" max="100" value="50"
               oninput="onSpeedInput(this.value)"
               onchange="setSpeed(this.value)">
      </div>

      <!-- Steering fine control -->
      <div class="steer-block">
        <div class="steer-label">
          Steering angle
          <span id="steerVal">0°</span>
        </div>
        <input type="range" id="steerSlider" min="-90" max="90" value="0"
               oninput="document.getElementById('steerVal').textContent=this.value+'°'"
               onchange="steerAngle(this.value)">
        <div class="servo-btns">
          <button class="btn-sm" onclick="steerPreset(-90)">Full L</button>
          <button class="btn-sm" onclick="steerPreset(-45)">Left</button>
          <button class="btn-sm" onclick="steerPreset(0)">Center</button>
          <button class="btn-sm" onclick="steerPreset(45)">Right</button>
          <button class="btn-sm" onclick="steerPreset(90)">Full R</button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Controls row -->
<div class="row">
  <div class="panel ctrl-panel">
    <div class="panel-title">
      Camera &amp; Laser Controls
    </div>
    <div class="ctrl-body">

      <!-- Camera Pan -->
      <div class="servo-block">
        <div class="servo-label">
          Camera Pan <span class="gpio-tag">GPIO 19</span>
          <span id="s1val">0°</span>
        </div>
        <input type="range" id="s1" min="-90" max="90" value="0"
               oninput="liveLabel('s1','s1val')"
               onchange="moveServo('servo1', this.value, false)">
        <div class="servo-btns">
          <button class="btn-sm" onclick="setServo('servo1','s1',-90)">Full L</button>
          <button class="btn-sm" onclick="setServo('servo1','s1',-45)">Left</button>
          <button class="btn-sm" onclick="setServo('servo1','s1',0)">Center</button>
          <button class="btn-sm" onclick="setServo('servo1','s1',45)">Right</button>
          <button class="btn-sm" onclick="setServo('servo1','s1',90)">Full R</button>
        </div>
        <div class="smooth-row">
          <input type="checkbox" id="smooth1">
          <label for="smooth1">Smooth move for presets</label>
        </div>
      </div>

      <hr class="divider">

      <!-- Laser -->
      <div class="laser-block">
        <div>
          <div class="laser-info">Laser <span class="gpio-tag">GPIO 22</span></div>
          <div class="laser-status off" id="laserStatus">OFF</div>
        </div>
        <button class="btn-laser off" id="laserBtn" onclick="toggleLaser()">Turn ON</button>
      </div>

    </div>
  </div>
</div>

<script>
let laserOn   = false;
let detectOn  = true;

function toast(msg, ok = true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? '#276749' : '#9b2c2c';
  t.style.opacity = '1';
  clearTimeout(t._tid);
  t._tid = setTimeout(() => { t.style.opacity = '0'; }, 2500);
}

function liveLabel(sliderId, labelId) {
  document.getElementById(labelId).textContent = document.getElementById(sliderId).value + '°';
}

async function moveServo(servo, angle, smooth) {
  try {
    const res  = await fetch('/api/servo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({servo, angle: parseInt(angle), smooth})
    });
    const data = await res.json();
    if (data.success) toast(`${servo} → ${data.angle}°`);
    else toast(`Servo error: ${data.error}`, false);
  } catch(e) { toast('Request failed: ' + e, false); }
}

function setServo(servo, sliderId, angle) {
  const smooth = document.getElementById('smooth' + sliderId.slice(-1)).checked;
  document.getElementById(sliderId).value = angle;
  liveLabel(sliderId, sliderId + 'val');
  moveServo(servo, angle, smooth);
}

async function toggleLaser() {
  try {
    const next = !laserOn;
    const res  = await fetch('/api/laser', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({on: next})
    });
    const data = await res.json();
    if (!data.success) { toast('Laser error: ' + data.error, false); return; }
    laserOn = data.laser;
    const btn  = document.getElementById('laserBtn');
    const stat = document.getElementById('laserStatus');
    if (laserOn) {
      btn.textContent = 'Turn OFF'; btn.className = 'btn-laser on';
      stat.textContent = 'ON';  stat.className = 'laser-status on';
      toast('Laser ON');
    } else {
      btn.textContent = 'Turn ON'; btn.className = 'btn-laser off';
      stat.textContent = 'OFF'; stat.className = 'laser-status off';
      toast('Laser OFF');
    }
  } catch(e) { toast('Request failed: ' + e, false); }
}

async function toggleDetect() {
  try {
    const res  = await fetch('/api/detect/toggle', {method: 'POST'});
    const data = await res.json();
    detectOn = data.enabled;
    const btn = document.getElementById('detToggle');
    if (detectOn) {
      btn.textContent = 'Overlay ON'; btn.className = 'btn-toggle on';
    } else {
      btn.textContent = 'Overlay OFF'; btn.className = 'btn-toggle off';
      document.getElementById('detList').innerHTML = '<div class="det-empty">Detection paused</div>';
      document.getElementById('detCount').textContent = '0';
    }
  } catch(e) { toast('Request failed: ' + e, false); }
}

async function capture() {
  const btn  = document.getElementById('captureBtn');
  const stat = document.getElementById('camStatus');
  btn.disabled = true; stat.textContent = 'Capturing...';
  try {
    const res  = await fetch('/capture', {method: 'POST'});
    const data = await res.json();
    if (data.success) {
      stat.textContent = 'Saved!';
      document.getElementById('previewContainer').innerHTML =
        '<img src="/preview?t=' + Date.now() + '" alt="Last capture">';
      document.getElementById('captureInfo').textContent = data.filename;
      setTimeout(() => { stat.textContent = ''; }, 2000);
    } else {
      stat.textContent = 'Error: ' + data.error;
    }
  } catch(e) { stat.textContent = 'Request failed'; }
  btn.disabled = false;
}

// ── Movement controls ─────────────────────────────────────────────────────────
let _driveTimer = null;

async function moveCmd(endpoint, body = {}) {
  try {
    const res  = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.state) updateMoveStatus(data.state);
    return data;
  } catch(e) { toast('Move error: ' + e, false); }
}

function updateMoveStatus(s) {
  document.getElementById('moveStatus').innerHTML =
    `Direction: <b>${s.direction}</b> &nbsp;|&nbsp; Steering: <b>${s.steering}°</b>`;

  ['forward','backward','left','right'].forEach(d => {
    const el = document.getElementById('btn-' + d);
    if (el) el.classList.remove('active-dir');
  });
  const active = s.direction !== 'stop' ? 'btn-' + s.direction : null;
  if (active) {
    const el = document.getElementById(active);
    if (el) el.classList.add('active-dir');
  }

  const steerSlider = document.getElementById('steerSlider');
  steerSlider.value = s.steering;
  document.getElementById('steerVal').textContent = s.steering + '°';
}

function driveStart(direction) {
  moveCmd('/api/move', {direction});
}

function driveStop() {
  moveCmd('/api/move', {direction: 'stop'});
}

function steerDir(dir) {
  const angle = dir === 'left' ? -45 : 45;
  moveCmd('/api/move/steer', {angle});
}

function steerCenter() {
  moveCmd('/api/move/steer', {angle: 0});
}

function steerAngle(angle) {
  moveCmd('/api/move/steer', {angle: parseInt(angle)});
}

function steerPreset(angle) {
  document.getElementById('steerSlider').value = angle;
  document.getElementById('steerVal').textContent = angle + '°';
  steerAngle(angle);
}

function onSpeedInput(val) {
  document.getElementById('speedVal').textContent = val + '%';
}

async function setSpeed(val) {
  const data = await moveCmd('/api/move/speed', {speed: parseInt(val)});
  if (data && data.state) {
    document.getElementById('speedVal').textContent = data.state.speed + '%';
  }
}

// ── Detection polling ─────────────────────────────────────────────────────────
// Poll detection results every 500 ms
async function pollDetections() {
  if (!detectOn) return;
  try {
    const res  = await fetch('/api/detections');
    const data = await res.json();
    const list = document.getElementById('detList');
    const cnt  = document.getElementById('detCount');
    cnt.textContent = data.length;
    if (data.length === 0) {
      list.innerHTML = '<div class="det-empty">No detections</div>';
    } else {
      list.innerHTML = data.map(d =>
        `<div class="det-item">
          <span class="det-label ${d.label}">${d.label}</span>
          <span class="det-conf">${Math.round(d.conf * 100)}%</span>
        </div>`
      ).join('');
    }
  } catch(e) {}
}
setInterval(pollDetections, 500);
</script>
</body>
</html>
"""


def generate_stream():
    while True:
        with stream_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template_string(HTML)


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


@app.route("/api/status")
def api_status():
    return jsonify(state)


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
    try:
        camera_servo.stop()
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
