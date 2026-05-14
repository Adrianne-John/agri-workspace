let laserOn  = false;
let detectOn = true;

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, ok = true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? '#276749' : '#9b2c2c';
  t.style.opacity = '1';
  clearTimeout(t._tid);
  t._tid = setTimeout(() => { t.style.opacity = '0'; }, 2500);
}

// ── Camera pan servo ──────────────────────────────────────────────────────────
function liveLabel(sliderId, labelId) {
  document.getElementById(labelId).textContent =
    document.getElementById(sliderId).value + '°';
}

async function moveServo(servo, angle, smooth) {
  try {
    const sent = servo === 'servo1' ? -parseInt(angle) : parseInt(angle);
    const res  = await fetch('/api/servo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ servo, angle: sent, smooth })
    });
    const data = await res.json();
    if (data.success) toast(`${servo} → ${data.angle}°`);
    else toast(`Servo error: ${data.error}`, false);
  } catch (e) { toast('Request failed: ' + e, false); }
}

function setServo(servo, sliderId, angle) {
  const smooth = document.getElementById('smooth' + sliderId.slice(-1)).checked;
  document.getElementById(sliderId).value = angle;
  liveLabel(sliderId, sliderId + 'val');
  moveServo(servo, angle, smooth);
}

// ── Laser ─────────────────────────────────────────────────────────────────────
async function toggleLaser() {
  try {
    const res  = await fetch('/api/laser', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on: !laserOn })
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
  } catch (e) { toast('Request failed: ' + e, false); }
}

// ── Detection toggle ──────────────────────────────────────────────────────────
async function toggleDetect() {
  try {
    const res  = await fetch('/api/detect/toggle', { method: 'POST' });
    const data = await res.json();
    detectOn = data.enabled;
    const btn = document.getElementById('detToggle');
    if (detectOn) {
      btn.textContent = 'Overlay ON'; btn.className = 'btn-toggle on';
    } else {
      btn.textContent = 'Overlay OFF'; btn.className = 'btn-toggle off';
      document.getElementById('detList').innerHTML =
        '<div class="det-empty">Detection paused</div>';
      document.getElementById('detCount').textContent = '0';
    }
  } catch (e) { toast('Request failed: ' + e, false); }
}

// ── Capture ───────────────────────────────────────────────────────────────────
async function capture() {
  const btn  = document.getElementById('captureBtn');
  const stat = document.getElementById('camStatus');
  btn.disabled = true; stat.textContent = 'Capturing...';
  try {
    const res  = await fetch('/capture', { method: 'POST' });
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
  } catch (e) { stat.textContent = 'Request failed'; }
  btn.disabled = false;
}

// ── Movement controls ─────────────────────────────────────────────────────────
async function moveCmd(endpoint, body = {}) {
  try {
    const res  = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.state) updateMoveStatus(data.state);
    return data;
  } catch (e) { toast('Move error: ' + e, false); }
}

function updateMoveStatus(s) {
  document.getElementById('moveStatus').innerHTML =
    `Direction: <b>${s.direction}</b> &nbsp;|&nbsp; Steering: <b>${s.steering}°</b>`;

  ['forward', 'backward', 'left', 'right'].forEach(d => {
    const el = document.getElementById('btn-' + d);
    if (el) el.classList.remove('active-dir');
  });
  const active = s.direction !== 'stop' ? 'btn-' + s.direction : null;
  if (active) {
    const el = document.getElementById(active);
    if (el) el.classList.add('active-dir');
  }

  document.getElementById('steerSlider').value = s.steering;
  document.getElementById('steerVal').textContent = s.steering + '°';
}

function driveStart(direction) { moveCmd('/api/move', { direction }); }
function driveStop()           { moveCmd('/api/move', { direction: 'stop' }); }

function steerDir(dir) {
  moveCmd('/api/move/steer', { angle: dir === 'left' ? -45 : 45 });
}
function steerCenter()      { moveCmd('/api/move/steer', { angle: 0 }); }
function steerAngle(angle)  { moveCmd('/api/move/steer', { angle: parseInt(angle) }); }

function steerPreset(angle) {
  document.getElementById('steerSlider').value = angle;
  document.getElementById('steerVal').textContent = angle + '°';
  steerAngle(angle);
}

function onSpeedInput(val) {
  document.getElementById('speedVal').textContent = val + '%';
}

async function setSpeed(val) {
  const data = await moveCmd('/api/move/speed', { speed: parseInt(val) });
  if (data && data.state)
    document.getElementById('speedVal').textContent = data.state.speed + '%';
}

// ── Laser Pointer Overlay ─────────────────────────────────────────────────────
let lp = { x: 320, y: 240, w: 80, h: 80, visible: false };

function updateLP(prop, val) {
  val = parseInt(val);
  if (prop === 'width')  { lp.w = val; document.getElementById('lpWidthVal').textContent = val + ' px'; }
  if (prop === 'length') { lp.h = val; document.getElementById('lpLenVal').textContent   = val + ' px'; }
  if (prop === 'x')      { lp.x = val; document.getElementById('lpXVal').textContent     = val + ' px'; }
  if (prop === 'y')      { lp.y = val; document.getElementById('lpYVal').textContent     = val + ' px'; }
  drawLP();
}

function toggleLP() {
  lp.visible = !lp.visible;
  const btn = document.getElementById('lpToggle');
  btn.textContent  = lp.visible ? 'Overlay ON' : 'Overlay OFF';
  btn.className    = lp.visible ? 'btn-toggle on' : 'btn-toggle off';
  drawLP();
}

function drawLP() {
  const canvas = document.getElementById('laserCanvas');
  const ctx    = canvas.getContext('2d');
  canvas.width  = 640;
  canvas.height = 480;
  ctx.clearRect(0, 0, 640, 480);
  if (!lp.visible) return;

  const x     = lp.x - lp.w / 2;
  const y     = lp.y - lp.h / 2;
  const color = laserOn ? '#f6e05e' : '#68d391';
  const tick  = Math.max(6, Math.min(lp.w, lp.h) * 0.12);

  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;

  // Targeting rectangle
  ctx.strokeRect(x, y, lp.w, lp.h);

  // Corner ticks
  const cx = lp.x, cy = lp.y;
  ctx.beginPath();
  ctx.moveTo(cx - tick, cy); ctx.lineTo(cx + tick, cy);
  ctx.moveTo(cx, cy - tick); ctx.lineTo(cx, cy + tick);
  ctx.stroke();
}

// ── Detection polling ─────────────────────────────────────────────────────────
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
  } catch (e) {}
}
setInterval(pollDetections, 500);
