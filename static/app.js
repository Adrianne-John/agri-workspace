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

async function autoRun() {
  const btn = document.getElementById('autoRunBtn');
  btn.disabled = true;
  btn.textContent = 'Running…';
  try {
    const data = await moveCmd('/api/auto/run');
    if (!data || !data.success) {
      btn.disabled = false;
      btn.textContent = 'Run';
      toast(data && data.error ? data.error : 'Auto run failed', false);
      return;
    }
    const poll = setInterval(async () => {
      try {
        const res  = await fetch('/api/auto/status');
        const stat = await res.json();
        if (stat.state) updateMoveStatus(stat.state);
        if (!stat.running) {
          clearInterval(poll);
          btn.disabled = false;
          btn.textContent = 'Run';
          toast('Autonomous run complete');
        }
      } catch (e) {
        clearInterval(poll);
        btn.disabled = false;
        btn.textContent = 'Run';
      }
    }, 500);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Run';
  }
}

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
// lockedBox: last known WeedA bounding box while auto-track is on.
// Updated each poll when YOLO detects a weed; sticks at last position when YOLO
// loses detection so the duplicate box stays visible during the lock-in phase.
let lp        = { w: 80, h: 80, ox: 0, oy: 0, visible: false };
let autoTrack = false;
let lockedBox = null;   // { x1, y1, x2, y2, cx, cy }

function updateLP(prop, val) {
  val = parseInt(val);
  if (prop === 'width')  { lp.w  = val; document.getElementById('lpWidthVal').textContent = val + ' px'; }
  if (prop === 'length') { lp.h  = val; document.getElementById('lpLenVal').textContent   = val + ' px'; }
  if (prop === 'ox')     { lp.ox = val; document.getElementById('lpOxVal').textContent    = val + ' px'; }
  if (prop === 'oy')     { lp.oy = val; document.getElementById('lpOyVal').textContent    = val + ' px'; }
  drawLP();
}

function toggleLP() {
  lp.visible = !lp.visible;
  const btn = document.getElementById('lpToggle');
  btn.textContent = lp.visible ? 'Overlay ON' : 'Overlay OFF';
  btn.className   = lp.visible ? 'btn-toggle on' : 'btn-toggle off';
  drawLP();
}

async function syncOffset() {
  try {
    await fetch('/api/laser/offset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ox: lp.ox, oy: lp.oy, w: lp.w, h: lp.h })
    });
  } catch (e) { toast('Offset sync failed: ' + e, false); }
}

// Restore saved calibration on page load
(async function loadCalibration() {
  try {
    const res  = await fetch('/api/laser/calibration');
    const data = await res.json();
    if (data && data.ox !== undefined) {
      lp.ox = data.ox; lp.oy = data.oy;
      lp.w  = data.w;  lp.h  = data.h;
      document.getElementById('lpOx').value    = lp.ox;
      document.getElementById('lpOy').value    = lp.oy;
      document.getElementById('lpWidth').value = lp.w;
      document.getElementById('lpLen').value   = lp.h;
      document.getElementById('lpOxVal').textContent    = lp.ox + ' px';
      document.getElementById('lpOyVal').textContent    = lp.oy + ' px';
      document.getElementById('lpWidthVal').textContent = lp.w  + ' px';
      document.getElementById('lpLenVal').textContent   = lp.h  + ' px';
      drawLP();
    }
  } catch (e) {}
})();

async function toggleTrack() {
  try {
    const res  = await fetch('/api/track/toggle', { method: 'POST' });
    const data = await res.json();
    autoTrack = data.tracking;
    if (!autoTrack) { lockedBox = null; drawLP(); }
    const btn = document.getElementById('trackToggle');
    btn.textContent = autoTrack ? 'Auto Track ON' : 'Auto Track OFF';
    btn.className   = autoTrack ? 'btn-toggle on' : 'btn-toggle off';
    toast(autoTrack ? 'Auto Track enabled' : 'Auto Track disabled');
  } catch (e) { toast('Track toggle failed: ' + e, false); }
}

function drawLP() {
  const canvas = document.getElementById('laserCanvas');
  const ctx    = canvas.getContext('2d');
  canvas.width  = 640;
  canvas.height = 480;
  ctx.clearRect(0, 0, 640, 480);

  // Shared crosshair tick length — identical for both boxes so they overlap cleanly
  const TICK   = 14;
  const MARGIN = 10;   // tolerance ring radius — "close enough" zone

  // ── Frozen duplicate detection box ──────────────────────────────────────────
  if (lockedBox) {
    const bx = lockedBox.cx, by = lockedBox.cy;
    const bw = lockedBox.x2 - lockedBox.x1;
    const bh = lockedBox.y2 - lockedBox.y1;

    ctx.strokeStyle = '#f6ad55';
    ctx.lineWidth   = 2;
    ctx.shadowColor = '#f6ad55';
    ctx.shadowBlur  = 6;

    ctx.setLineDash([8, 4]);
    ctx.strokeRect(lockedBox.x1, lockedBox.y1, bw, bh);
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(bx - TICK, by); ctx.lineTo(bx + TICK, by);
    ctx.moveTo(bx, by - TICK); ctx.lineTo(bx, by + TICK);
    ctx.stroke();

    ctx.shadowBlur = 0;
  }

  if (!lp.visible) return;

  // ── Laser box — fixed calibrated position ───────────────────────────────────
  const cx    = 320 + lp.ox;
  const cy    = 240 + lp.oy;
  const x     = cx - lp.w / 2;
  const y     = cy - lp.h / 2;
  const color = laserOn ? '#f6e05e' : (autoTrack ? '#fc8181' : '#68d391');

  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;

  ctx.strokeRect(x, y, lp.w, lp.h);

  // Crosshair — same TICK size as the detection crosshair
  ctx.beginPath();
  ctx.moveTo(cx - TICK, cy); ctx.lineTo(cx + TICK, cy);
  ctx.moveTo(cx, cy - TICK); ctx.lineTo(cx, cy + TICK);
  ctx.stroke();

  // Tolerance margin ring — dashed circle showing the acceptable alignment zone
  ctx.beginPath();
  ctx.arc(cx, cy, MARGIN, 0, Math.PI * 2);
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.shadowBlur = 0;
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

    // While auto-track is on: keep lockedBox updated to the latest highest-conf
    // WeedA detection. When YOLO loses the weed lockedBox sticks at last position
    // so the duplicate box stays visible and the laser stays locked during settling.
    if (autoTrack) {
      const weeds = data.filter(d => d.label === 'WeedA' || d.label === 'Pest');
      if (weeds.length > 0) {
        const best = weeds.reduce((a, b) => a.conf > b.conf ? a : b);
        lockedBox = {
          x1: best.x1, y1: best.y1, x2: best.x2, y2: best.y2,
          cx: Math.round((best.x1 + best.x2) / 2),
          cy: Math.round((best.y1 + best.y2) / 2)
        };
        drawLP();
      }
      // No else: don't clear lockedBox on detection loss — it sticks
    }

  } catch (e) {}
}
setInterval(pollDetections, 500);
