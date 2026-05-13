import os
import time
import threading

# ── Hardware PWM via sysfs ────────────────────────────────────────────────────
# GPIO 18 → QuicRun 1060 ESC signal wire  (pwmchip0 / pwm2 = PWM0_CHAN2)
# GPIO 19 → freed for camera pan servo (RPi.GPIO software PWM in agribot.py)
#
# config.txt overlay: dtoverlay=pwm,pin=18,func=2
#   (single channel — GPIO 19 is NOT claimed by the PWM overlay)
#
# Wiring:
#   GPIO 18 → QuicRun 1060 ESC signal wire
#   Pi GND  → ESC BEC ground (shared)
#   ESC BEC → powers JX PDI-6621 steering servo (servo not driven for now)

PWM_CHIP    = 0
ESC_CHANNEL = 2   # GPIO 18 = PWM0_CHAN2 → pwmchip0/pwm2

PWM_PERIOD_NS  = 20_000_000   # 20 ms = 50 Hz

# ESC pulse widths in microseconds (QuicRun 1060 hobby servo protocol):
#   1000 µs → full reverse
#   1500 µs → neutral / stopped  ← ESC arm point on startup
#   2000 µs → full forward
ESC_NEUTRAL_US = 1500
ESC_FWD_MAX_US = 2000
ESC_REV_MAX_US = 1000

DEFAULT_SPEED = 50  # 0–100 %


class _SysfsPWM:
    """Thin wrapper around the Pi 5 sysfs hardware PWM interface."""

    def __init__(self, chip: int, channel: int):
        self._chip    = chip
        self._channel = channel
        self._base    = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}"

        if not os.path.exists(self._base):
            with open(f"/sys/class/pwm/pwmchip{chip}/export", 'w') as f:
                f.write(str(channel))
            time.sleep(0.15)

        self._write('period',     PWM_PERIOD_NS)
        self._write('duty_cycle', 0)
        self._write('enable',     1)

    def _write(self, attr: str, value):
        with open(f"{self._base}/{attr}", 'w') as f:
            f.write(str(int(value)))

    def set_pulsewidth_us(self, us: int):
        duty_ns = max(0, min(PWM_PERIOD_NS, int(us) * 1000))
        self._write('duty_cycle', duty_ns)

    def stop(self):
        self._write('duty_cycle', 0)
        self._write('enable',     0)
        try:
            with open(f"/sys/class/pwm/pwmchip{self._chip}/unexport", 'w') as f:
                f.write(str(self._channel))
        except OSError:
            pass


def _speed_to_us(speed: int, forward: bool) -> int:
    ratio = max(0, min(100, speed)) / 100.0
    if forward:
        return int(ESC_NEUTRAL_US + ratio * (ESC_FWD_MAX_US - ESC_NEUTRAL_US))
    else:
        return int(ESC_NEUTRAL_US - ratio * (ESC_NEUTRAL_US - ESC_REV_MAX_US))


class AgriMove:
    """
    Drivetrain controller for AgriBot — ESC only (GPIO 18, sysfs hardware PWM).

    Steering servo (GPIO 19) is currently unconnected; steer() calls are
    accepted but do nothing so the UI buttons remain harmless.
    """

    def __init__(self, gpio_lock: threading.Lock):
        self._lock  = threading.Lock()
        self._speed = DEFAULT_SPEED
        self._state = {
            'direction': 'stop',
            'steering':  0,
            'speed':     DEFAULT_SPEED,
        }

        self._esc = _SysfsPWM(PWM_CHIP, ESC_CHANNEL)

        # Arm the QuicRun 1060: hold neutral for 2 s (ESC beeps when ready)
        self._esc.set_pulsewidth_us(ESC_NEUTRAL_US)
        time.sleep(2.0)

    # ── Drive ─────────────────────────────────────────────────────────────────

    def forward(self):
        us = _speed_to_us(self._speed, forward=True)
        with self._lock:
            self._esc.set_pulsewidth_us(us)
            self._state['direction'] = 'forward'

    def backward(self):
        us = _speed_to_us(self._speed, forward=False)
        with self._lock:
            self._esc.set_pulsewidth_us(us)
            self._state['direction'] = 'backward'

    def stop(self):
        with self._lock:
            self._esc.set_pulsewidth_us(ESC_NEUTRAL_US)
            self._state['direction'] = 'stop'

    def set_speed(self, speed: int):
        speed = max(0, min(100, int(speed)))
        self._speed = speed
        self._state['speed'] = speed
        direction = self._state['direction']
        if direction == 'forward':
            with self._lock:
                self._esc.set_pulsewidth_us(_speed_to_us(speed, forward=True))
        elif direction == 'backward':
            with self._lock:
                self._esc.set_pulsewidth_us(_speed_to_us(speed, forward=False))

    # ── Steering (no-op until servo is wired) ────────────────────────────────

    def steer(self, angle: int):
        self._state['steering'] = max(-90, min(90, int(angle)))

    def steer_left(self):
        self.steer(-45)

    def steer_right(self):
        self.steer(45)

    def steer_center(self):
        self.steer(0)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return dict(self._state)

    def cleanup(self):
        self.stop()
        time.sleep(0.2)
        self._esc.stop()
