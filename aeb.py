"""
Automatic Emergency Braking (AEB) System
==========================================
Detects persons and vehicles ahead and applies graduated braking.
Works as a SAFETY OVERRIDE — active in BOTH manual and auto modes.

Braking Zones (distance to obstacle):
  PERSON (pedestrian — more sensitive):
    WARNING  : < 18 m  — alert only, throttle cut
    SOFT     : < 10 m  — brake 0.4
    HARD     : <  6 m  — brake 0.8
    EMERGENCY: <  3 m  — brake 1.0 (full stop)

  VEHICLE (car, truck, bus):
    WARNING  : < 25 m  — alert only, throttle cut
    SOFT     : < 15 m  — brake 0.35
    HARD     : <  8 m  — brake 0.75
    EMERGENCY: <  4 m  — brake 1.0 (full stop)

Output:
  .update() returns AEBResult with:
    - brake        : override brake value (0.0 if no action)
    - throttle_cut : True = force throttle to 0
    - level        : 'CLEAR' / 'WARNING' / 'SOFT' / 'HARD' / 'EMERGENCY'
    - target_name  : 'person' / 'car' etc.
    - target_dist  : metres to nearest danger object
    - color_bgr    : HUD color for this level
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Braking profiles per class ───────────────────────────────────────
PROFILES = {
    # class_name : (warn_m, soft_m, hard_m, emrg_m,
    #               soft_brake, hard_brake)
    'person': (20.0, 15.0, 10.0, 5.0, 0.5, 0.9),
    'car':    (15.0, 8.0, 4.5, 2.5, 0.35, 0.75),
    'truck':  (18.0, 10.0, 5.5, 3.0, 0.35, 0.80),
    'bus':    (18.0, 10.0, 5.5, 3.0, 0.35, 0.80),
}

# HUD colors (BGR) per level
LEVEL_COLORS = {
    'CLEAR':     (0, 230, 76),      # green
    'WARNING':   (0, 165, 255),     # orange
    'SOFT':      (0, 100, 255),     # orange-red
    'HARD':      (0, 30, 255),      # red
    'EMERGENCY': (0, 0, 255),       # full red
}

LEVEL_PRIORITY = ['CLEAR', 'WARNING', 'SOFT', 'HARD', 'EMERGENCY']


@dataclass
class AEBResult:
    brake:        float = 0.0
    throttle_cut: bool  = False
    level:        str   = 'CLEAR'
    target_name:  str   = ''
    target_dist:  float = 999.0
    color_bgr:    tuple = (0, 230, 76)
    active:       bool  = False   # True = AEB is intervening


class AEB:
    """Automatic Emergency Braking System."""

    def __init__(self, min_speed_kmh: float = 1.0):
        """
        min_speed_kmh: AEB only activates above this speed (avoid false
                       triggers when parked / reversing slowly).
        """
        self._min_speed    = min_speed_kmh
        self._last_result  = AEBResult()
        self._warn_start   = None   # timestamp when WARNING began (for flash)
        print("[OK] AEB: Automatic Emergency Braking initialized")

    # ── Main update — call every frame ──────────────────────────────
    def update(self,
               tracked_objects: List[Dict],
               current_speed_kmh: float) -> AEBResult:
        """
        Args:
            tracked_objects: list of dicts from tracker/fusion.
                Each must have: class_name, distance (metres)
            current_speed_kmh: vehicle speed

        Returns:
            AEBResult with brake/throttle override values.
        """
        result = AEBResult()   # default CLEAR

        # AEB inactive when nearly stopped (avoid reverse-park false triggers)
        if current_speed_kmh < self._min_speed:
            self._last_result = result
            return result

        worst_level_idx = 0   # index into LEVEL_PRIORITY

        for obj in tracked_objects:
            cls  = obj.get('class_name', '').lower()
            dist = obj.get('distance',  999.0)

            # ── Distance validity check ────────────────────────────────
            # dist <= 1.0  : LiDAR found no valid points in bbox
            #                (fusion returns 0 when no match)
            # dist >= 80.0 : too far, not a real threat
            if cls not in PROFILES or dist <= 1.0 or dist >= 80.0:
                continue

            # ── Forward-facing filter using bbox position ──────────────
            # Only trigger AEB for objects in upper 75% of frame
            # (objects below centre = beside ego / already passed)
            y2 = obj.get('y2', 0)
            frame_h = obj.get('frame_h', 600)   # fallback if not provided
            if frame_h > 0 and (y2 / frame_h) > 0.82:
                continue   # object is at very bottom — skip


            warn_m, soft_m, hard_m, emrg_m, soft_b, hard_b = PROFILES[cls]

            # Classify this object's threat level
            if dist < emrg_m:
                lvl   = 'EMERGENCY'
                brake = 1.0
            elif dist < hard_m:
                # Interpolate brake between soft_b and hard_b
                t     = 1.0 - (dist - emrg_m) / max(hard_m - emrg_m, 0.1)
                brake = soft_b + t * (hard_b - soft_b)
                lvl   = 'HARD'
            elif dist < soft_m:
                # Interpolate between 0 and soft_b
                t     = 1.0 - (dist - hard_m) / max(soft_m - hard_m, 0.1)
                brake = t * soft_b
                lvl   = 'SOFT'
            elif dist < warn_m:
                lvl   = 'WARNING'
                brake = 0.0
            else:
                continue   # outside all zones

            # Keep worst threat
            idx = LEVEL_PRIORITY.index(lvl)
            if idx > worst_level_idx or (idx == worst_level_idx and dist < result.target_dist):
                worst_level_idx  = idx
                result.level     = lvl
                result.brake     = float(np.clip(brake, 0.0, 1.0))
                result.target_name = cls
                result.target_dist = dist
                result.color_bgr   = LEVEL_COLORS[lvl]
                result.active      = (lvl != 'WARNING')   # warning = alert only
                result.throttle_cut = True   # always cut throttle in any AEB zone

        self._last_result = result
        return result

    # ── Draw AEB overlay on camera frame ────────────────────────────
    def draw_overlay(self, frame: np.ndarray, result: AEBResult) -> np.ndarray:
        """Draw AEB warning on camera frame."""
        if result.level == 'CLEAR':
            return frame

        h, w = frame.shape[:2]
        color = result.color_bgr

        # ── AEB Warning banner (Subtle) ──────────────────────────────
        banner_h = 40
        banner_y = h - 100

        if result.level in ('HARD', 'EMERGENCY'):
            cv2.rectangle(frame, (20, banner_y), (w - 20, banner_y + banner_h),
                          color, 2)
            cv2.rectangle(frame, (20, banner_y), (w - 20, banner_y + banner_h),
                          (0, 0, 0), -1)
            cv2.putText(frame,
                        f"AEB {result.level} BRAKE: {result.target_name.upper()} {result.target_dist:.1f}m",
                        (30, banner_y + 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        elif result.level == 'SOFT':
            cv2.rectangle(frame, (20, banner_y), (w - 20, banner_y + banner_h),
                          (0, 0, 0), -1)
            cv2.putText(frame,
                        f"AEB Braking: {result.target_name.upper()} {result.target_dist:.1f}m",
                        (30, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elif result.level == 'WARNING':
            cv2.putText(frame,
                        f"AEB Alert: {result.target_name.upper()} {result.target_dist:.1f}m",
                        (30, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── Distance gauge (bottom strip) ───────────────────────────
        if result.target_dist < 30:
            gauge_w = w - 40
            gauge_y = h - 45
            cv2.rectangle(frame, (20, gauge_y), (20 + gauge_w, gauge_y + 6),
                          (30, 30, 30), -1)
            fill_pct = max(0.0, 1.0 - result.target_dist / 30.0)
            cv2.rectangle(frame, (20, gauge_y),
                          (20 + int(gauge_w * fill_pct), gauge_y + 6),
                          color, -1)
            cv2.putText(frame, f"AEB DIST: {result.target_dist:.1f} m",
                        (20, gauge_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame
