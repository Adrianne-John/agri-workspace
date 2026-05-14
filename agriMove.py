import os
import time
import threading

import board
from adafruit_pca9685 import PCA9685 as _PCA9685Driver

# ── Hardware PWM via sysfs ────────────────────────────────────────────────────
# GPIO 18  →  PWM0_CHAN2  →  pwmchip0/pwm2   QuicRun 1060 ESC  (stays on sysfs)
#
# config.txt overlay:
#   dtoverlay=pwm-2chan,pin=18,func=2
#
# ── PCA9685 I2C PWM (I2C1, default address 0x40) ─────────────────────────────
# CH0  →  tilt servo   (camera up / down)      — agribot.py
# CH2  →  pan servo    (camera left / right)   — agribot.py
# CH4  →  JX PDI-6621  (bot steering)

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

STEER_CENTER_US = 1500   # 0 °  (trim this if the servo doesn't centre perfectly)
STEER_LEFT_US   = 1000   # −90 °
STEER_RIGHT_US  = 2000   # +90 °

DEFAULT_SPEED = 13  # 0–100 %


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
        self._write('enable',     0)
        self._write('period',     PWM_PERIOD_NS)
        self._write('duty_cycle', 0)
        self._write('enable',     1)

    def _write(self, attr: str, value):
        try:
            with open(f"{self._base}/{attr}", 'w') as f:
                f.write(str(int(value)))
        except OSError as e:
            # Catching the error ensures your app doesn't silently fail 
            print(f"PWM Write Error on {self._base}/{attr}: {e}")

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


# ── PCA9685 channel wrapper ───────────────────────────────────────────────────
_PCA9685_PERIOD_US = 20_000.0   # 50 Hz → 20 ms period

def _init_pca9685() -> _PCA9685Driver:
    i2c = board.I2C()
    pca = _PCA9685Driver(i2c)
    pca.frequency = 50
    return pca

_pca9685     = _init_pca9685()
_pca9685_lock = threading.Lock()   # I2C bus is shared; serialise all channel writes


class _PCA9685Channel:
    """Wraps one channel of the shared PCA9685; same interface as _SysfsPWM."""

    def __init__(self, channel: int):
        self._ch = _pca9685.channels[channel]

    def set_pulsewidth_us(self, us: int):
        duty = int(max(0.0, min(_PCA9685_PERIOD_US, float(us))) / _PCA9685_PERIOD_US * 0xFFFF)
        with _pca9685_lock:
            self._ch.duty_cycle = duty

    def stop(self):
        with _pca9685_lock:
            self._ch.duty_cycle = 0


def _speed_to_us(speed: int, forward: bool) -> int:
    ratio = max(0, min(100, speed)) / 100.0
    if forward:
        return int(ESC_NEUTRAL_US + ratio * (ESC_FWD_MAX_US - ESC_NEUTRAL_US))
    else:
        return int(ESC_NEUTRAL_US - ratio * (ESC_NEUTRAL_US - ESC_REV_MAX_US))


def _angle_to_steer_us(angle: int) -> int:
    """Map −90..90 ° to pulse width for the JX PDI-6621 steering servo."""
    angle = max(-90, min(90, angle))
    return int(1500 + (angle / 90.0) * 500)


class AgriMove:
    """
    Drivetrain controller for AgriBot.

      PCA9685 CH4  JX PDI-6621 steering servo
      GPIO 18      pwmchip0/pwm2  QuicRun 1060 ESC  (sysfs)
    """

    def __init__(self, gpio_lock: threading.Lock):
        self._lock  = threading.Lock()
        self._speed = DEFAULT_SPEED
        self._state = {
            'direction': 'stop',
            'steering':  0,
            'speed':     DEFAULT_SPEED,
        }

        self._steer = _PCA9685Channel(4)
        self._esc   = _SysfsPWM(PWM_CHIP, ESC_CHANNEL)

        self._steer.set_pulsewidth_us(STEER_CENTER_US)
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

    # ── Steering ──────────────────────────────────────────────────────────────

    def steer(self, angle: int):
        angle = max(-90, min(90, int(angle)))
        with self._lock:
            self._steer.set_pulsewidth_us(_angle_to_steer_us(angle))
            self._state['steering'] = angle

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
        self.steer_center()
        time.sleep(0.2)
        self._steer.stop()
        self._esc.stop()
