"""
Sound Manager — Procedural Audio for CARLA Driving
====================================================
Koi external audio file nahi chahiye.
Sare sounds numpy se real-time generate hote hain.

Sounds:
  - Engine idle / driving  (pitch speed ke saath badhta hai)
  - Brake squeal           (hard braking pe)
  - Collision impact       (crash pe)
  - Horn                   (H key se)
  - Turn signal tick       (indicator sound)

Usage:
    sm = SoundManager()
    sm.update(speed_kmh=60, throttle=0.6, brake=0.0, collision=False)
    sm.horn()        # H key
    sm.cleanup()     # exit pe
"""

import numpy as np
import pygame
import threading
import math

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz
CHUNK_SIZE    = 512     # samples per chunk
MAX_VOL       = 0.6     # master volume (0.0 – 1.0)


def _make_sound(array: np.ndarray) -> pygame.mixer.Sound:
    """Convert float32 numpy array (range -1..1) to pygame Sound."""
    pcm = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
    # pygame needs shape (N, 2) for stereo
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)


def _sine(freq: float, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(sr * duration)
    return (np.random.uniform(-1, 1, n)).astype(np.float32)


def _envelope(signal: np.ndarray, attack: float, release: float,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply linear attack / release envelope."""
    a = int(attack * sr)
    r = int(release * sr)
    env = np.ones(len(signal), dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0 and r <= len(signal):
        env[-r:] = np.linspace(1, 0, r)
    return signal * env


# ══════════════════════════════════════════════════════════
#  SoundManager
# ══════════════════════════════════════════════════════════
class SoundManager:
    """Manages all vehicle sounds during simulation."""

    def __init__(self):
        # Sound disabled per user request
        self._available = False
        print("[INFO] SoundManager: Audio completely disabled.")
        return

    # ── Engine bank: pre-generate one loop per speed range ─────
    def _gen_engine_bank(self) -> dict:
        """
        Generate engine tones for speed buckets 0..7.
        Engine = fundamental + harmonics (rich sound).
        """
        bank = {}
        # Speed buckets: 0=idle, 1=<20, 2=<35, 3=<50, 4=<65, 5=<80, 6=<100, 7=100+
        # Base RPM → frequency mapping (rough)
        frequencies = [35, 50, 65, 85, 105, 130, 160, 200]

        for bucket, base_f in enumerate(frequencies):
            dur = 0.5   # 0.5 sec loop
            t   = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)

            # Fundamental + 2nd + 3rd harmonic for rich engine timbre
            wave = (
                0.50 * np.sin(2 * np.pi * base_f * t) +
                0.30 * np.sin(2 * np.pi * base_f * 2 * t) +
                0.15 * np.sin(2 * np.pi * base_f * 3 * t) +
                0.05 * np.random.uniform(-1, 1, len(t)).astype(np.float32)  # grit
            ).astype(np.float32)

            # Slight volume envelope to avoid clicks on loop
            wave = _envelope(wave, attack=0.02, release=0.02)
            wave *= 0.8

            bank[bucket] = _make_sound(wave)
        return bank

    def _speed_to_bucket(self, speed: float) -> int:
        thresholds = [5, 20, 35, 50, 65, 80, 100]
        for i, t in enumerate(thresholds):
            if speed < t:
                return i
        return 7

    def _play_engine(self, speed: float):
        if not self._available:
            return
        bucket = self._speed_to_bucket(speed)
        if bucket != self._current_bucket:
            snd = self._engine_sounds[bucket]
            self._engine_channel.play(snd, loops=-1)   # loop forever
            self._current_bucket = bucket

        # Volume: idle=0.25, full speed=0.65
        vol = 0.25 + min(speed / 120.0, 1.0) * 0.40
        self._engine_channel.set_volume(vol * MAX_VOL)

    # ── Static sound generators ─────────────────────────────────
    def _gen_brake(self) -> pygame.mixer.Sound:
        """High-pitched squeal with noise."""
        dur  = 1.2
        t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        freq = np.linspace(900, 600, len(t))   # pitch falling
        wave = (
            0.6 * np.sin(2 * np.pi * freq * t / SAMPLE_RATE * SAMPLE_RATE) +
            0.4 * np.random.uniform(-1, 1, len(t)).astype(np.float32)
        ).astype(np.float32)
        wave = _envelope(wave, attack=0.05, release=0.3)
        wave *= 0.55
        return _make_sound(wave)

    def _gen_collision(self) -> pygame.mixer.Sound:
        """Sharp loud impact thud."""
        dur  = 0.7
        t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        freq = np.linspace(120, 40, len(t))
        wave = (
            0.5 * np.sin(2 * np.pi * freq * t / SAMPLE_RATE * SAMPLE_RATE) +
            0.5 * np.random.uniform(-1, 1, len(t)).astype(np.float32)
        ).astype(np.float32)
        wave = _envelope(wave, attack=0.01, release=0.35)
        wave *= 0.9
        return _make_sound(wave)

    def _gen_horn(self) -> pygame.mixer.Sound:
        """Car horn — two-tone (440 Hz + 550 Hz)."""
        dur = 0.6
        t   = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        wave = (
            0.55 * np.sin(2 * np.pi * 440 * t) +
            0.45 * np.sin(2 * np.pi * 550 * t)
        ).astype(np.float32)
        wave = _envelope(wave, attack=0.03, release=0.08)
        wave *= 0.7
        return _make_sound(wave)

    def _gen_tick(self) -> pygame.mixer.Sound:
        """Short click for turn signal / mode change."""
        dur  = 0.04
        t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        wave = np.sin(2 * np.pi * 800 * t).astype(np.float32)
        wave = _envelope(wave, attack=0.002, release=0.01)
        wave *= 0.4
        return _make_sound(wave)

    # ── Public API ──────────────────────────────────────────────
    def update(self, speed_kmh: float, throttle: float,
               brake: float, collision: bool):
        """
        Call every frame with current vehicle state.
        speed_kmh  — current speed in km/h
        throttle   — 0.0 .. 1.0
        brake      — 0.0 .. 1.0
        collision  — True if collision occurred this frame
        """
        if not self._available:
            return

        # Engine
        self._play_engine(speed_kmh)

        # Brake squeal
        if brake > 0.5 and speed_kmh > 8:
            if not self._brake_playing:
                ch = pygame.mixer.find_channel()
                if ch:
                    ch.play(self._snd_brake)
                    self._brake_playing = True
        else:
            self._brake_playing = False

        # Collision
        if collision and not self._collision_played:
            ch = pygame.mixer.find_channel()
            if ch:
                ch.play(self._snd_collision)
            self._collision_played = True
        elif not collision:
            self._collision_played = False

    def horn(self):
        """Play car horn (call on H key press)."""
        if not self._available:
            return
        ch = pygame.mixer.find_channel()
        if ch:
            ch.play(self._snd_horn)

    def tick(self):
        """Play a short click (mode switch, parking, RL toggle)."""
        if not self._available:
            return
        ch = pygame.mixer.find_channel()
        if ch:
            ch.play(self._snd_tick)

    def cleanup(self):
        """Stop all sounds and quit mixer."""
        if not self._available:
            return
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
            print("[OK] SoundManager: shutdown complete")
        except Exception:
            pass


# ── Quick self-test ───────────────────────────────────────
if __name__ == "__main__":
    import time
    pygame.init()
    sm = SoundManager()

    print("Engine idle (3 sec)...")
    sm.update(speed_kmh=0, throttle=0.0, brake=0.0, collision=False)
    time.sleep(3)

    print("Accelerating 0 -> 100 km/h...")
    for spd in range(0, 105, 5):
        sm.update(speed_kmh=spd, throttle=0.8, brake=0.0, collision=False)
        time.sleep(0.3)

    print("Braking hard!")
    sm.update(speed_kmh=80, throttle=0.0, brake=1.0, collision=False)
    time.sleep(2)

    print("Horn!")
    sm.horn()
    time.sleep(1)

    print("Collision!")
    sm.update(speed_kmh=20, throttle=0.0, brake=0.0, collision=True)
    time.sleep(2)

    sm.cleanup()
    print("Done!")
