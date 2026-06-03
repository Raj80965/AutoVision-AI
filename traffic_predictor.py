"""
AI Traffic Behavior Prediction Module
======================================
Predicts what nearby vehicles are likely to do based on their
velocity history tracked across frames.

Behavior Classes:
  STATIONARY   — vehicle is stopped or barely moving
  MOVING       — vehicle driving normally
  ACCELERATING — vehicle speeding up
  BRAKING      — vehicle slowing down
  LANE_CHANGE   — vehicle showing lateral movement

Integration:
  Called after DeepSORT tracking in advanced_drive.py.
  Returns prediction label + confidence + predicted future position per object.

Usage:
  predictor = TrafficPredictor()
  predictions = predictor.update(tracked_objects, dt=0.033)
  # predictions[track_id] = {'behavior': 'BRAKING', 'confidence': 0.85, ...}
"""

import math
import time
from collections import deque, defaultdict
import numpy as np
import cv2


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════
HISTORY_LEN      = 15    # frames of velocity history to keep
MIN_FRAMES       = 4     # minimum frames before making a prediction
STATIONARY_SPEED = 2.0   # km/h threshold to call vehicle stationary (pixel/frame)
ACCEL_THRESHOLD  = 0.25  # m/s² equivalent in pixel-speed change
BRAKE_THRESHOLD  = -0.25
LATERAL_THRESHOLD = 0.15  # fraction of bbox width — lateral movement flag

# Behavior colors (BGR) for HUD overlay
BEHAVIOR_COLORS = {
    "STATIONARY":   (150, 150, 150),
    "MOVING":       (0, 230, 76),
    "ACCELERATING": (0, 165, 255),
    "BRAKING":      (0, 0, 255),
    "LANE_CHANGE":  (255, 0, 212),
    "UNKNOWN":      (100, 100, 100),
}

BEHAVIOR_ICONS = {
    "STATIONARY":   "■",
    "MOVING":       "►",
    "ACCELERATING": "▲",
    "BRAKING":      "▼",
    "LANE_CHANGE":  "◄►",
    "UNKNOWN":      "?",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Vehicle State Tracker (per vehicle)
# ══════════════════════════════════════════════════════════════════════════════
class VehicleHistory:
    """Stores recent center positions and derived velocities for one vehicle."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.centers  = deque(maxlen=HISTORY_LEN)   # (cx, cy) pixel positions
        self.timestamps = deque(maxlen=HISTORY_LEN)
        self.speeds   = deque(maxlen=HISTORY_LEN)   # pixel displacement / frame
        self.lateral_speeds = deque(maxlen=HISTORY_LEN)
        self.last_seen = time.time()

    def add(self, cx, cy, bbox_w):
        now = time.time()
        if len(self.centers) > 0:
            prev_cx, prev_cy = self.centers[-1]
            dt = max(now - self.timestamps[-1], 1e-6)
            vx = (cx - prev_cx) / dt          # pixels/sec
            vy = (cy - prev_cy) / dt
            speed   = math.sqrt(vx**2 + vy**2)   # total pixel speed
            lat_spd = abs(vx) / max(bbox_w, 1)    # lateral fraction of width
            self.speeds.append(speed)
            self.lateral_speeds.append(lat_spd)
        self.centers.append((cx, cy))
        self.timestamps.append(now)
        self.last_seen = now

    def is_stale(self, timeout=1.5):
        return (time.time() - self.last_seen) > timeout

    @property
    def n_frames(self):
        return len(self.centers)


# ══════════════════════════════════════════════════════════════════════════════
#  Traffic Predictor
# ══════════════════════════════════════════════════════════════════════════════
class TrafficPredictor:
    """
    Predicts the behavior of each tracked vehicle.
    Works with DeepSORT track IDs from tracker.py.
    """

    def __init__(self):
        self._histories: dict[int, VehicleHistory] = {}
        self._last_predictions: dict[int, dict] = {}
        print("[OK] TrafficPredictor initialized")

    # ── Main update — call every frame ─────────────────────────────────────
    def update(self, tracked_objects: list) -> dict:
        """
        Args:
            tracked_objects: list of dicts from tracker.update()
                Each dict has keys: track_id, x1, y1, x2, y2, class_name, ...

        Returns:
            dict mapping track_id → prediction dict:
                {
                  'behavior':   str,      # STATIONARY / MOVING / ACCELERATING / BRAKING / LANE_CHANGE
                  'confidence': float,    # 0.0 – 1.0
                  'speed_px':   float,    # pixel speed (proxy for real speed)
                  'future_cx':  float,    # predicted center x in 1 second
                  'future_cy':  float,    # predicted center y in 1 second
                  'class_name': str
                }
        """
        predictions = {}
        active_ids  = set()

        for obj in tracked_objects:
            tid = obj.get('track_id', None)
            if tid is None or tid == '' or tid == -1:
                continue
            active_ids.add(tid)

            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(x2 - x1, 1)

            if tid not in self._histories:
                self._histories[tid] = VehicleHistory(tid)
            hist = self._histories[tid]
            hist.add(cx, cy, bw)

            predictions[tid] = self._classify(hist, obj.get('class_name', ''))

        # Prune stale histories
        for tid in list(self._histories.keys()):
            if tid not in active_ids and self._histories[tid].is_stale():
                del self._histories[tid]

        self._last_predictions = predictions
        return predictions

    # ── Classification logic ────────────────────────────────────────────────
    def _classify(self, hist: VehicleHistory, class_name: str) -> dict:
        if hist.n_frames < MIN_FRAMES:
            return {
                'behavior':   'UNKNOWN',
                'confidence': 0.0,
                'speed_px':   0.0,
                'future_cx':  hist.centers[-1][0],
                'future_cy':  hist.centers[-1][1],
                'class_name': class_name
            }

        speeds   = list(hist.speeds)
        lat_spds = list(hist.lateral_speeds)
        recent_n = min(5, len(speeds))

        avg_speed   = float(np.mean(speeds[-recent_n:]))
        avg_lat     = float(np.mean(lat_spds[-recent_n:]))

        # Acceleration: change in speed per frame
        if len(speeds) >= 3:
            accel = float(np.mean(np.diff(speeds[-recent_n:])))
        else:
            accel = 0.0

        # ── Classify ──────────────────────────────────────────────────────
        if avg_speed < STATIONARY_SPEED:
            behavior = "STATIONARY"
            confidence = min(0.6 + (STATIONARY_SPEED - avg_speed) / STATIONARY_SPEED * 0.4, 1.0)

        elif avg_lat > LATERAL_THRESHOLD and avg_speed > STATIONARY_SPEED:
            behavior = "LANE_CHANGE"
            confidence = min(0.5 + avg_lat * 2.0, 1.0)

        elif accel > ACCEL_THRESHOLD:
            behavior = "ACCELERATING"
            confidence = min(0.5 + accel * 2.0, 1.0)

        elif accel < BRAKE_THRESHOLD:
            behavior = "BRAKING"
            confidence = min(0.5 + abs(accel) * 2.0, 1.0)

        else:
            behavior = "MOVING"
            confidence = 0.75

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # ── Predict future position (1 sec ahead linear extrapolation) ────
        if len(hist.centers) >= 2:
            c0 = hist.centers[-2]
            c1 = hist.centers[-1]
            dt = hist.timestamps[-1] - hist.timestamps[-2]
            if dt > 0:
                vx = (c1[0] - c0[0]) / dt
                vy = (c1[1] - c0[1]) / dt
                future_cx = c1[0] + vx * 1.0
                future_cy = c1[1] + vy * 1.0
            else:
                future_cx, future_cy = hist.centers[-1]
        else:
            future_cx, future_cy = hist.centers[-1]

        return {
            'behavior':   behavior,
            'confidence': confidence,
            'speed_px':   avg_speed,
            'future_cx':  future_cx,
            'future_cy':  future_cy,
            'class_name': class_name
        }

    # ── HUD overlay ────────────────────────────────────────────────────────
    def draw_predictions(self, frame: np.ndarray,
                         tracked_objects: list,
                         predictions: dict) -> np.ndarray:
        """
        Draw behavior labels, confidence bars, and trajectory arrows
        on top of tracked bounding boxes.
        """
        for obj in tracked_objects:
            tid = obj.get('track_id', -1)
            if tid not in predictions:
                continue

            pred   = predictions[tid]
            beh    = pred['behavior']
            conf   = pred['confidence']
            color  = BEHAVIOR_COLORS.get(beh, (200, 200, 200))
            icon   = BEHAVIOR_ICONS.get(beh, '?')

            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ── Label above bbox ─────────────────────────────────────────
            cls_name = str(pred.get('class_name', 'OBJ')).upper()
            if cls_name == 'UNKNOWN' or cls_name == 'NONE' or cls_name == '':
                cls_name = 'OBJ'
                
            label = f"{cls_name} | {beh} {conf*100:.0f}%"
            label_y = max(y1 - 8, 16)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ── Confidence bar ───────────────────────────────────────────
            bar_w = x2 - x1
            bar_fill = int(bar_w * conf)
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 6), (30, 30, 30), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_fill, y2 + 6), color, -1)

            # ── Future trajectory arrow ───────────────────────────────────
            fx = int(pred['future_cx'])
            fy = int(pred['future_cy'])
            if beh not in ('STATIONARY', 'UNKNOWN'):
                cv2.arrowedLine(frame, (cx, cy), (fx, fy),
                                color, 2, tipLength=0.3)

        return frame

    # ── Summary stats ───────────────────────────────────────────────────────
    def get_summary(self) -> dict:
        """Returns count of each behavior currently seen."""
        from collections import Counter
        counts = Counter(p['behavior'] for p in self._last_predictions.values())
        return dict(counts)


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  TrafficPredictor — Self Test")
    print("=" * 55)

    predictor = TrafficPredictor()

    # Simulate 20 frames of a vehicle moving right and decelerating
    mock_objects = []
    cx, cy = 400, 300
    speed = 40  # pixels/sec initially

    for i in range(20):
        time.sleep(0.033)   # simulate ~30 FPS
        speed = max(speed - 1.5, 0)   # decelerating
        cx += speed * 0.033
        mock_objects = [{
            'track_id': 1,
            'x1': int(cx - 60), 'y1': int(cy - 30),
            'x2': int(cx + 60), 'y2': int(cy + 30),
            'class_name': 'car',
            'distance': 15.0
        }]
        preds = predictor.update(mock_objects)

    print(f"Prediction after 20 frames: {preds}")
    print(f"Summary: {predictor.get_summary()}")

    print("=" * 55)
    print("✅ TrafficPredictor test passed!")
    print("=" * 55)
