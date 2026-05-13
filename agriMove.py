import os
import time
import threading

# ── Hardware PWM via sysfs ────────────────────────────────────────────────────
# All three devices use pwmchip0 (RP1 PWM0 block), confirmed via pinctrl:
#
#   GPIO 12  →  PWM0_CHAN0  →  pwmchip0/pwm0   JX PDI-6621 steering servo
#   GPIO 18  →  PWM0_CHAN2  →  pwmchip0/pwm2   QuicRun 1060 ESC
#   GPIO 19  →  PWM0_CHAN3  →  pwmchip0/pwm3   MG90S camera pan (agribot.py)
#
# config.txt overlays:
#   dtoverlay=pwm,pin=12,func=4
#   dtoverlay=pwm-2chan,pin=18,func=2,pin2=19,func2=2
#
# Wiring:
#   GPIO 12 → JX PDI-6621 steering servo signal wire
#   GPIO 18 → QuicRun 1060 ESC signal wire
#   Pi GND  → ESC BEC ground (shared)
#   ESC BEC → powers JX PDI-6621 and MG90S

PWM_CHIP      = 0
STEER_CHANNEL = 0   # GPIO 12 = PWM0_CHAN0 → pwmchip0/pwm0
ESC_CHANNEL   = 2   # GPIO 18 = PWM0_CHAN2 → pwmchip0/pwm2

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


def _angle_to_steer_us(angle: int) -> int:
    """Map −90..90 ° to pulse width for the JX PDI-6621 steering servo."""
    angle = max(-90, min(90, angle))
    return int(1500 + (angle / 90.0) * 500)


class AgriMove:
    """
    Drivetrain controller for AgriBot (sysfs hardware PWM throughout).

      GPIO 12  pwmchip0/pwm0  JX PDI-6621 steering servo
      GPIO 18  pwmchip0/pwm2  QuicRun 1060 ESC
    """

    def __init__(self, gpio_lock: threading.Lock):
        self._lock  = threading.Lock()
        self._speed = DEFAULT_SPEED
        self._state = {
            'direction': 'stop',
            'steering':  0,
            'speed':     DEFAULT_SPEED,
        }

        self._steer = _SysfsPWM(PWM_CHIP, STEER_CHANNEL)
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
