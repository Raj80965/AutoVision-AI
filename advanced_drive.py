import sys
import glob
import time
import os
import json
import numpy as np
import cv2
import random
from queue import Queue
from enum import Enum
import math

# ===== CARLA PATH =====
try:
    sys.path.append(glob.glob('D:/carla/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass
sys.path.append('D:/carla/WindowsNoEditor/PythonAPI/carla')

import carla
from ultralytics import YOLO
from agents.navigation.behavior_agent import BehaviorAgent

from config import *
from sensors import SensorManager
from fusion import SensorFusion
from tracker import ObjectTracker
from rl_agent import RLAgent
from parking import ParkingManager, ParkingState
from lane_detection import LaneDetector  # Fixed: lane detection added
from traffic_predictor import TrafficPredictor  # NEW: AI traffic behavior prediction
from sound_manager import SoundManager           # NEW: Vehicle sound effects
from aeb import AEB                              # NEW: Automatic Emergency Braking


class DrivingMode(Enum):
    NORMAL = 1
    BRAKING = 2
    STOPPED_AT_SIGNAL = 3
    AVOIDING = 4
    EMERGENCY = 5
    PARKING = 6
    RL_CONTROL = 7
    MANUAL = 8      # Manual override from dashboard
    YIELDING = 9    # Yielding to pedestrians


class AdvancedDrivingSystem:
    def __init__(self):
        print("[..] Connecting to CARLA...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.blueprint_library = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()

        # Sync mode setup
        settings = self.world.get_settings()
        settings.synchronous_mode = SYNC_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(SYNC_MODE)

        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter(VEHICLE_MODEL)[0]
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', 'hero')

        spawn_points = self.map.get_spawn_points()
        self.vehicle = None
        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if self.vehicle:
                break

        if not self.vehicle:
            raise RuntimeError("Failed to spawn vehicle!")

        print("[OK] Vehicle spawned:", VEHICLE_MODEL)

        # Fixed: tick once to let vehicle settle before creating agent
        self.world.tick()

        # BehaviorAgent — handles road following + waypoints automatically
        self.agent = BehaviorAgent(self.vehicle, behavior='normal')
        self.set_new_destination()
        print("[OK] BehaviorAgent initialized and destination set")

        # Sensors
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        self.camera = self.sensor_manager.setup_camera()
        self.lidar = self.sensor_manager.setup_lidar()
        self.collision = self.sensor_manager.setup_collision_sensor()

        self.image_queue = Queue(maxsize=2)
        self.camera.listen(
            lambda image: self.image_queue.put(image) if self.image_queue.qsize() < 2 else None
        )

        # Spectator follows vehicle
        self.spectator = self.world.get_spectator()

        # AI Models
        print("[..] Loading AI Models...")
        self.yolo = YOLO(YOLO_MODEL)
        self.tracker = ObjectTracker()
        self.fusion = SensorFusion(CAMERA_IMAGE_SIZE_X, CAMERA_IMAGE_SIZE_Y, CAMERA_FOV)
        self.lane_detector = LaneDetector()   # Fixed: lane detector initialized
        self.rl_agent = RLAgent()
        self.rl_agent.load()
        self.parking = ParkingManager()
        self.traffic_predictor = TrafficPredictor()  # NEW: traffic behavior predictor
        self.sound_manager = SoundManager()          # NEW: vehicle audio
        self.aeb = AEB(min_speed_kmh=2.0)            # NEW: Automatic Emergency Braking
        print("[OK] All AI models loaded")
        
        self.walkers_list = []
        self.controllers_list = []
        self.spawn_random_pedestrians(10)

        # State
        self.target_speed = TARGET_SPEED
        self.current_speed_limit = TARGET_SPEED   # NEW: updated by speed limit signs
        self.driving_mode = DrivingMode.NORMAL
        self.use_rl = False
        self.parking_requested = False
        
        # Stop Sign Logic State
        self.stop_sign_active = False
        self.stop_timer_start = 0.0
        self.stop_cooldown_end = 0.0
        
        # Weather Control
        self.weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.WetCloudyNoon,
            carla.WeatherParameters.ClearSunset
        ]
        self.weather_names = ["Clear Noon", "Cloudy Sunset", "Hard Rain", "Wet Cloudy", "Clear Sunset"]
        self.weather_index = 0
        self.world.set_weather(self.weather_presets[self.weather_index])

        # RL tracking across frames
        self._prev_rl_state = None
        self._prev_rl_action = None

        self.telemetry = {
            "mode": "NORMAL",
            "speed": 0,
            "lane_center": 0.5,
            "detected_signs": [],
            "closest_obstacle": "None",
            "ttc": 999.0,
            "speed_limit": TARGET_SPEED,
            "traffic_behaviors": {},
            "aeb_level": "CLEAR",
            "aeb_dist": 999.0,
            "aeb_target": ""
        }
        # Manual control — absolute path so both scripts find same file
        self._manual_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manual_control.json')
        # Write default state at startup (manual OFF)
        with open(self._manual_file, 'w') as f:
            json.dump({'manual_mode': False, 'throttle': 0.0, 'steer': 0.0, 'brake': 0.0, 'reverse': False}, f)
        print(f"[OK] Manual control file: {self._manual_file}")

    # ──────────────────────────────────────────────────────────────────
    def set_new_destination(self):
        """Pick a random far-away spawn point as destination"""
        spawn_points = self.map.get_spawn_points()
        # Pick a destination that is reasonably far (not same as current position)
        current_loc = self.vehicle.get_location()
        far_points = [sp for sp in spawn_points
                      if sp.location.distance(current_loc) > 50]
        destination = random.choice(far_points if far_points else spawn_points)
        self.agent.set_destination(destination.location)
        print("[OK] New destination set")

    def update_spectator(self):
        """Keep spectator camera behind the vehicle"""
        transform = self.vehicle.get_transform()
        yaw_rad = math.radians(transform.rotation.yaw)
        spec_loc = carla.Location(
            x=transform.location.x - 8 * math.cos(yaw_rad),
            y=transform.location.y - 8 * math.sin(yaw_rad),
            z=transform.location.z + 4
        )
        self.spectator.set_transform(
            carla.Transform(spec_loc, carla.Rotation(pitch=-20, yaw=transform.rotation.yaw))
        )

    def get_next_direction(self) -> str:
        """
        Reads next waypoint from BehaviorAgent's local planner queue.
        Returns: 'LEFT' / 'RIGHT' / 'STRAIGHT' / 'UTURN' / 'FOLLOW'
        """
        try:
            from agents.navigation.local_planner import RoadOption
            # Get upcoming waypoints queue
            queue = list(self.agent._local_planner._waypoints_queue)
            if not queue:
                return 'FOLLOW'

            # Look ahead a few waypoints to find a turn
            for wp, cmd in queue[:12]:
                if cmd == RoadOption.LEFT:
                    return 'LEFT'
                elif cmd == RoadOption.RIGHT:
                    return 'RIGHT'
                elif cmd == RoadOption.STRAIGHT:
                    return 'STRAIGHT'
                elif cmd == RoadOption.LANEFOLLOW:
                    continue
            return 'FOLLOW'
        except Exception:
            return 'FOLLOW'

    def draw_direction_arrow(self, img, direction: str):
        """
        Draw a big glowing navigation arrow on the top-center of the frame.
        LEFT=blue arrow, RIGHT=blue arrow, STRAIGHT=green arrow
        """
        h, w = img.shape[:2]
        cx = w // 2
        cy = 70   # vertical position

        # Colors
        ARROW_COLORS = {
            'LEFT':     (255, 180, 0),    # blue
            'RIGHT':    (255, 180, 0),    # blue
            'STRAIGHT': (0, 230, 76),     # green
            'UTURN':    (0, 100, 255),    # red
            'FOLLOW':   (100, 100, 100),  # gray (no turn soon)
        }
        color = ARROW_COLORS.get(direction, (100, 100, 100))

        # Background pill
        cv2.rectangle(img, (cx - 70, cy - 45), (cx + 70, cy + 15),
                      (0, 0, 0), -1)
        cv2.rectangle(img, (cx - 70, cy - 45), (cx + 70, cy + 15),
                      color, 2)

        # Arrow shapes using polylines
        if direction == 'LEFT':
            # Left arrow: shaft + head
            pts_shaft = [(cx + 30, cy - 15), (cx - 10, cy - 15),
                         (cx - 10, cy - 25), (cx - 10, cy - 5),
                         (cx - 10, cy - 15)]
            # arrowedLine approach
            cv2.arrowedLine(img, (cx + 25, cy - 15), (cx - 25, cy - 15),
                            color, 4, tipLength=0.4)
            cv2.putText(img, 'TURN LEFT',
                        (cx - 58, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif direction == 'RIGHT':
            cv2.arrowedLine(img, (cx - 25, cy - 15), (cx + 25, cy - 15),
                            color, 4, tipLength=0.4)
            cv2.putText(img, 'TURN RIGHT',
                        (cx - 58, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif direction == 'STRAIGHT':
            cv2.arrowedLine(img, (cx, cy + 5), (cx, cy - 35),
                            color, 4, tipLength=0.35)
            cv2.putText(img, 'STRAIGHT',
                        (cx - 45, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif direction == 'UTURN':
            cv2.putText(img, 'U-TURN',
                        (cx - 35, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Glow effect (repeat with alpha)
        overlay = img.copy()
        if direction == 'LEFT':
            cv2.arrowedLine(overlay, (cx + 25, cy - 15), (cx - 25, cy - 15),
                            color, 8, tipLength=0.4)
        elif direction == 'RIGHT':
            cv2.arrowedLine(overlay, (cx - 25, cy - 15), (cx + 25, cy - 15),
                            color, 8, tipLength=0.4)
        elif direction == 'STRAIGHT':
            cv2.arrowedLine(overlay, (cx, cy + 5), (cx, cy - 35),
                            color, 8, tipLength=0.35)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


    def write_telemetry(self):
        try:
            with open('telemetry.json', 'w') as f:
                json.dump(self.telemetry, f, indent=2)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    def process_frame(self, image_data):
        img = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
        img = img.reshape((image_data.height, image_data.width, 4))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]

        vel = self.vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)
        current_yaw = self.vehicle.get_transform().rotation.yaw

        # Reset stale collision flag
        self.sensor_manager.reset_collision_if_old()

        # ── Check Manual Control from Dashboard ────────────────────
        manual_override = False
        try:
            with open(self._manual_file, 'r') as f:
                mc = json.load(f)
            if mc.get('manual_mode', False):
                manual_override = True
                m_throttle = float(mc.get('throttle', 0.0))
                m_steer    = float(mc.get('steer',    0.0))
                m_brake    = float(mc.get('brake',    0.0))
                m_reverse  = bool(mc.get('reverse',   False))
        except Exception as e:
            pass  # file missing or corrupt — stay in auto

        # ── Lane Detection ─────────────────────────────────────────
        # Fixed: detect lanes and draw them on the frame
        lane_center, lane_detected, left_lane, right_lane = self.lane_detector.detect(img)
        img = self.lane_detector.draw_lanes(img, left_lane, right_lane, lane_center, lane_detected)

        # Lane deviation — how far from center (0=center, negative=left, positive=right)
        lane_deviation = lane_center - 0.5  # -0.5 to +0.5

        # ── YOLO Detection ─────────────────────────────────────────
        results = self.yolo(img, verbose=False, conf=YOLO_CONFIDENCE)[0]
        raw_obstacles = []
        detected_signs = []

        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.yolo.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if name in ['car', 'truck', 'bus', 'person']:
                # ── Ego-vehicle filter (third-person camera fix) ──────
                # In 3rd-person view our own car appears in lower-center.
                # Skip any detection whose center-x is near frame center
                # AND bottom edge is in lower 40% of frame.
                cx_norm = (x1 + x2) / 2.0 / max(w, 1)   # 0..1
                cy_bot  = y2 / max(h, 1)                  # 0..1
                is_ego  = (0.25 < cx_norm < 0.75) and (cy_bot > 0.55)
                if is_ego:
                    continue   # skip — that's our own car

                raw_obstacles.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'name': name, 'conf': conf
                })
            elif name == 'stop sign':
                detected_signs.append('STOP SIGN')
                cv2.putText(img, 'STOP SIGN', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            elif 'speed limit' in name.lower():
                # ── NEW: Parse speed limit value from label ────────────
                # YOLOv8 COCO label is "speed limit 30" / "speed limit 60" etc.
                parts = name.split()
                limit_val = None
                for part in parts:
                    if part.isdigit():
                        limit_val = int(part)
                        break

                # Fallback: try OCR-lite — read digits from cropped bbox
                if limit_val is None:
                    crop = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                    if crop.size > 0:
                        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                        # Look for common speed limits using pixel density heuristic
                        # (simple, no tesseract dependency)
                        brightness = np.mean(th)
                        if brightness > 180:
                            limit_val = 30
                        elif brightness > 120:
                            limit_val = 60
                        else:
                            limit_val = 90

                if limit_val is None:
                    limit_val = 30   # safe default

                sign_label = f'SPEED LIMIT {limit_val}'
                detected_signs.append(sign_label)

                # Dynamically adjust target speed
                self.current_speed_limit = float(limit_val)
                # OVERRIDE: Disabled auto-slowing to let the car drive at high speed (70+ km/h)
                # self.target_speed = float(limit_val)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(img, sign_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # ── DeepSORT Tracking + Traffic Behavior Prediction ─────────
        tracked_objects = self.tracker.update(raw_obstacles, frame=img)

        # ── NEW: AI Traffic Behavior Prediction ───────────────────────
        traffic_predictions = self.traffic_predictor.update(tracked_objects)
        behavior_summary = self.traffic_predictor.get_summary()

        # ── Sound Update ──────────────────────────────────────────────
        collision_now = self.sensor_manager.collision_occurred

        # ── Sensor Fusion: LiDAR + Camera ─────────────────────────
        lidar_points = self.sensor_manager.get_lidar_points()
        projected_lidar = self.fusion.project_lidar_to_camera(lidar_points)

        closest_dist = 999
        closest_ttc = 999
        closest_name = "None"

        for obj in tracked_objects:
            dist = self.fusion.get_distance_for_bbox(
                projected_lidar, [obj['x1'], obj['y1'], obj['x2'], obj['y2']]
            )
            obj['distance'] = dist
            obj['frame_h']  = h   # pass frame height for AEB forward filter
            
            obj_x1, obj_y1, obj_x2, obj_y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            cx_norm = (obj_x1 + obj_x2) / 2.0 / max(w, 1)
            cy_bot  = obj_y2 / max(h, 1)

            # Wider detection for pedestrians (30% to 70% of screen) to catch people in slightly curved lanes
            if obj['class_name'] == 'person':
                is_in_front = (0.28 < cx_norm < 0.72) and (cy_bot < 0.85)
            else:
                is_in_front = (0.32 < cx_norm < 0.68) and (cy_bot < 0.85)
            
            if is_in_front and 3.0 < dist < closest_dist:
                closest_dist = dist
                closest_ttc = dist / max(current_speed / 3.6, 0.1)
                closest_name = obj['class_name']


        # ── MANUAL MODE — AI completely OFF ───────────────────────
        # Jab manual on ho: BehaviorAgent, RL, Parking — sab skip
        if manual_override:
            self.driving_mode = DrivingMode.MANUAL   # HUD fix
            control = carla.VehicleControl(
                throttle=float(np.clip(m_throttle, 0.0, 1.0)),
                steer=float(np.clip(m_steer,    -1.0, 1.0)),
                brake=float(np.clip(m_brake,     0.0, 1.0)),
                reverse=m_reverse
            )
            
            # Explicitly set gear for reverse to work reliably in CARLA
            if m_reverse:
                control.manual_gear_shift = True
                control.gear = -1
            else:
                control.manual_gear_shift = False

            self.vehicle.apply_control(control)
            mode_label = "MANUAL"
            throttle = m_throttle
            steer    = m_steer
            brake    = m_brake

            # Sound for manual mode
            self.sound_manager.update(
                speed_kmh=current_speed,
                throttle=m_throttle,
                brake=m_brake,
                collision=collision_now
            )

        else:
            # ── AUTO MODE — BehaviorAgent + Supervisor ─────────────
            # BehaviorAgent handles road + lane following
            if self.agent.done():
                self.set_new_destination()

            base_control = self.agent.run_step()
            base_control.manual_gear_shift = False

            throttle = base_control.throttle
            steer    = base_control.steer
            brake    = base_control.brake
            reverse  = False
            
            # Detect if we are at a RED or YELLOW light
            is_at_signal = self.vehicle.is_at_traffic_light()
            signal_state = self.vehicle.get_traffic_light_state() if is_at_signal else None
            is_red_light = is_at_signal and (signal_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow))
            
            # Check for upcoming turn
            next_dir = self.get_next_direction()
            is_turning = next_dir in ('LEFT', 'RIGHT', 'UTURN')
            
            # Trust agent braking ONLY if it's for a Red light or a Turn.
            # Otherwise (at green lights), ignore its paranoid "caution" braking for distant pedestrians.
            agent_is_braking = (base_control.brake > 0.05)
            should_trust_agent = is_red_light or is_turning or (agent_is_braking and closest_ttc < 5.0)

            if not should_trust_agent and current_speed < self.target_speed and closest_ttc > 5.0:
                brake = 0.0  # Override agent's paranoid/cruise braking
                # Add extra throttle based on how far we are from target speed
                throttle_boost = (self.target_speed - current_speed) / 25.0
                throttle = min(1.0, throttle + throttle_boost)

            # Gentle visual lane correction (only in clear road)
            if lane_detected and closest_ttc > 5.0 and not self.parking_requested:
                steer = float(np.clip(steer + lane_deviation * 0.15, -1.0, 1.0))

            # ── STOP SIGN LOGIC ────────────────────────────────────
            import time
            current_time = time.time()
            if 'STOP SIGN' in detected_signs and current_time > self.stop_cooldown_end and not self.stop_sign_active:
                self.stop_sign_active = True
                self.stop_timer_start = current_time
                print("[INFO] STOP SIGN Detected! Yielding for 3 seconds...")
                self.sound_manager.tick()

            # ── SUPERVISOR LOGIC ───────────────────────────────────
            if self.sensor_manager.collision_occurred:
                self.driving_mode = DrivingMode.EMERGENCY
                mode_label = "COLLISION_STOP"
                brake, throttle = 1.0, 0.0

            elif self.parking_requested:
                if self.driving_mode != DrivingMode.PARKING:
                    self.parking.start_parking(current_yaw)
                    self.driving_mode = DrivingMode.PARKING

                throttle, steer, brake, reverse, active = self.parking.update(
                    current_speed, current_yaw
                )
                if not active:
                    self.parking_requested = False
                    self.driving_mode = DrivingMode.NORMAL

            elif self.stop_sign_active:
                if current_time - self.stop_timer_start < 3.0:
                    self.driving_mode = DrivingMode.STOPPED_AT_SIGNAL
                    brake = 1.0
                    throttle = 0.0
                else:
                    self.stop_sign_active = False
                    self.stop_cooldown_end = current_time + 15.0 # 15 sec cooldown
                    self.driving_mode = DrivingMode.NORMAL

            elif closest_name == 'person' and closest_dist < 8.0:
                self.driving_mode = DrivingMode.YIELDING
                brake = 1.0
                throttle = 0.0

            elif closest_ttc < 1.5:
                self.driving_mode = DrivingMode.EMERGENCY
                brake = 1.0
                throttle = 0.0

            elif closest_ttc < 3.0:
                self.driving_mode = DrivingMode.BRAKING
                brake = max(brake, 0.5)
                throttle = 0.0

            elif self.use_rl:
                self.driving_mode = DrivingMode.RL_CONTROL
                has_obstacle = closest_ttc < 5.0
                # NEW: pass lane_deviation to DQN for richer state
                state_key = self.rl_agent.get_state_key(
                    current_speed, lane_center, has_obstacle, closest_ttc,
                    lane_deviation
                )
                action = self.rl_agent.get_action(state_key)

                if action == 0:
                    throttle = 0.4; brake = 0.0; steer = 0.0
                elif action == 1:
                    throttle = 0.0; brake = 0.5; steer = 0.0
                elif action == 2:
                    steer = -0.3; throttle = 0.3
                elif action == 3:
                    steer = 0.3; throttle = 0.3

                if self._prev_rl_state is not None:
                    reward = self.rl_agent.calculate_reward(
                        current_speed, self.target_speed,   # use dynamic speed limit
                        self.sensor_manager.collision_occurred,
                        closest_ttc, lane_center,
                        obstacle_avoided=(closest_ttc > 3.0 and has_obstacle)
                    )
                    self.rl_agent.update(
                        self._prev_rl_state, self._prev_rl_action, reward, state_key
                    )

                self._prev_rl_state = state_key
                self._prev_rl_action = action

            else:
                self.driving_mode = DrivingMode.NORMAL
                self._prev_rl_state = None
                self._prev_rl_action = None

            # Apply AI control
            control = carla.VehicleControl(
                throttle=float(np.clip(throttle, 0.0, 1.0)),
                steer=float(np.clip(steer,    -1.0, 1.0)),
                brake=float(np.clip(brake,     0.0, 1.0)),
                reverse=reverse
            )
            self.vehicle.apply_control(control)
            mode_label = self.driving_mode.name

            # Sound for auto mode
            self.sound_manager.update(
                speed_kmh=current_speed,
                throttle=throttle,
                brake=brake,
                collision=collision_now
            )

        # ══════════════════════════════════════════════════════
        # AEB SAFETY OVERRIDE — runs AFTER both manual & auto
        # Overrides brake/throttle regardless of driving mode
        # ══════════════════════════════════════════════════════
        is_reversing = m_reverse if manual_override else reverse
        
        # Disable AEB if we are reversing OR if we are in Manual Mode
        # (gives 100% full control to user in manual mode without AI brake overrides)
        if is_reversing or manual_override:
            aeb_result = self.aeb._last_result
            aeb_result.active = False
            aeb_result.level = 'CLEAR'
        else:
            aeb_result = self.aeb.update(tracked_objects, current_speed)

        if aeb_result.active:
            # Build AEB control — keep existing steer, override brake/throttle
            existing_ctrl = self.vehicle.get_control()
            aeb_ctrl = carla.VehicleControl(
                throttle = 0.0,                                          # always cut throttle
                steer    = float(np.clip(existing_ctrl.steer, -1.0, 1.0)),
                brake    = float(np.clip(aeb_result.brake, 0.0, 1.0)),
                reverse  = existing_ctrl.reverse
            )
            self.vehicle.apply_control(aeb_ctrl)

            # Sound: brake squeal for AEB
            self.sound_manager.update(
                speed_kmh=current_speed,
                throttle=0.0,
                brake=aeb_result.brake,
                collision=collision_now
            )

            if aeb_result.level in ('HARD', 'EMERGENCY'):
                mode_label = f"AEB_{aeb_result.level}"
                if aeb_result.level == 'EMERGENCY':
                    self.driving_mode = DrivingMode.EMERGENCY


        # ── Telemetry ──────────────────────────────────────────────
        self.telemetry = {
            "mode": mode_label,
            "speed": round(current_speed, 1),
            "lane_center": round(lane_center, 3),
            "lane_detected": lane_detected,
            "detected_signs": detected_signs,
            "closest_obstacle": closest_name.upper(),
            "ttc": round(closest_ttc, 2) if closest_ttc < 999 else 999.0,
            "speed_limit": self.current_speed_limit,
            "traffic_behaviors": behavior_summary,
            "aeb_level":  aeb_result.level,
            "aeb_dist":   round(aeb_result.target_dist, 1) if aeb_result.target_dist < 999 else 999.0,
            "aeb_target": aeb_result.target_name,
            "next_turn":  self.get_next_direction()   # Navigation direction
        }
        self.write_telemetry()

        # ── Draw HUD ───────────────────────────────────────────────────
        next_dir = self.get_next_direction()
        img = self.tracker.draw_tracks(img, tracked_objects)
        img = self.traffic_predictor.draw_predictions(img, tracked_objects, traffic_predictions)
        img = self.aeb.draw_overlay(img, aeb_result)    # AEB warning overlay
        self.draw_direction_arrow(img, next_dir)         # Navigation arrow
        self._draw_hud(img, h, w, current_speed, lane_center, lane_detected,
                       lane_deviation, closest_dist, closest_ttc, detected_signs)

        cv2.imshow("Advanced AI Drive", img)
        self.update_spectator()

    # ──────────────────────────────────────────────────────────────────
    def _draw_hud(self, img, h, w, speed, lane_center, lane_detected,
                  lane_deviation, dist, ttc, signs):
        """Draw all HUD elements"""
        # Mode badge
        mode_color = {
            DrivingMode.NORMAL:    (0, 200, 0),
            DrivingMode.BRAKING:   (0, 165, 255),
            DrivingMode.STOPPED_AT_SIGNAL: (0, 100, 255),
            DrivingMode.EMERGENCY: (0, 0, 255),
            DrivingMode.PARKING:   (255, 200, 0),
            DrivingMode.RL_CONTROL:(200, 0, 255),
            DrivingMode.MANUAL:    (0, 140, 255),   # orange ─ clearly visible
            DrivingMode.YIELDING:  (255, 100, 255), # pinkish/purple
        }.get(self.driving_mode, (0, 200, 0))

        cv2.rectangle(img, (0, 0), (340, 130), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 0), (340, 130), mode_color, 2)

        cv2.putText(img, f"MODE: {self.driving_mode.name}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2)
        cv2.putText(img, f"SPEED: {speed:.1f} km/h  TARGET: {TARGET_SPEED} km/h",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Lane info
        lane_status = "DETECTED" if lane_detected else "NOT FOUND"
        lane_col = (0, 255, 0) if lane_detected else (0, 100, 255)
        dev_str = f"{lane_deviation:+.2f}" if lane_detected else "N/A"
        cv2.putText(img, f"LANE: {lane_status}  DEV: {dev_str}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lane_col, 1)

        # Obstacle / LiDAR
        if dist < 999:
            obs_col = (0, 0, 255) if ttc < 2.0 else (0, 255, 255)
            cv2.putText(img, f"OBSTACLE: {dist:.1f}m  TTC: {ttc:.1f}s",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, obs_col, 1)
        else:
            cv2.putText(img, "OBSTACLE: clear",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Traffic signs
        if signs:
            cv2.putText(img, f"SIGN: {', '.join(signs)}",
                        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # NEW: Speed limit indicator (top-right corner)
        sl_color = (0, 165, 255) if self.current_speed_limit < TARGET_SPEED else (0, 230, 118)
        cv2.rectangle(img, (w - 120, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(img, "LIMIT",
                    (w - 115, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(img, f"{int(self.current_speed_limit)} km/h",
                    (w - 115, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sl_color, 2)

        # Lane center bar (bottom)
        bar_x = 50
        bar_y = h - 30
        bar_w = w - 100
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        center_px = int(bar_x + lane_center * bar_w)
        mid_px = bar_x + bar_w // 2
        bar_col = (0, 255, 0) if abs(lane_deviation) < 0.1 else (0, 165, 255)
        cv2.rectangle(img, (min(center_px, mid_px), bar_y),
                      (max(center_px, mid_px), bar_y + 12), bar_col, -1)
        cv2.circle(img, (center_px, bar_y + 6), 7, (255, 255, 0), -1)
        cv2.line(img, (mid_px, bar_y - 4), (mid_px, bar_y + 16), (255, 255, 255), 1)
        cv2.putText(img, "LANE CENTER", (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # RL epsilon
        if self.use_rl:
            cv2.putText(img, f"RL eps={self.rl_agent.epsilon:.3f}",
                        (w - 160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # BIG YIELDING BANNER removed as requested
        if self.driving_mode == DrivingMode.YIELDING:
            pass
                        
        elif self.driving_mode == DrivingMode.STOPPED_AT_SIGNAL:
            banner_y = h // 2 - 40
            cv2.rectangle(img, (0, banner_y), (w, banner_y + 60), (0, 0, 255), -1)
            cv2.rectangle(img, (0, banner_y), (w, banner_y + 60), (0, 0, 0), 2)
            cv2.putText(img, "STOP SIGN - WAITING...",
                        (w // 2 - 200, banner_y + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Controls hint
        cv2.putText(img, "[R] RL  [P] Park  [C] Weather  [T] Test Pedestrian  [ESC] Exit",
                    (w - 480, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ──────────────────────────────────────────────────────────────────
    def spawn_random_pedestrians(self, num_pedestrians=30):
        """Spawns pedestrians randomly around the map so they walk on sidewalks."""
        print(f"[..] Spawning {num_pedestrians} random pedestrians on sidewalks...")
        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        spawn_points = []
        for i in range(num_pedestrians * 2):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(loc))
            if len(spawn_points) >= num_pedestrians:
                break

        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            walker = self.world.try_spawn_actor(walker_bp, spawn_point)
            if walker:
                self.walkers_list.append(walker)
                
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for walker in self.walkers_list:
            controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if controller:
                self.controllers_list.append(controller)
                
        self.world.tick()
        
        for controller in self.controllers_list:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.0 + random.random())
            
        print(f"[OK] Successfully spawned {len(self.walkers_list)} pedestrians wandering around.")

    # ──────────────────────────────────────────────────────────────────
    def spawn_pedestrian_in_front(self):
        """Spawns a pedestrian ~12m ahead to walk across the road for demonstration."""
        try:
            ego_trans = self.vehicle.get_transform()
            fwd = ego_trans.get_forward_vector()
            rgt = ego_trans.get_right_vector()
            
            # Spawn 12m ahead, 4m to the right (so they cross the lane)
            spawn_loc = ego_trans.location + carla.Location(x=fwd.x*12 + rgt.x*4, y=fwd.y*12 + rgt.y*4, z=0.5)
            
            walker_bps = self.world.get_blueprint_library().filter('walker.*')
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            # Face left across the road
            spawn_trans = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_trans.rotation.yaw - 90))
            
            walker = self.world.try_spawn_actor(walker_bp, spawn_trans)
            if walker:
                controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker)
                if controller:
                    self.world.tick()
                    controller.start()
                    # Walk 10m across the road
                    target_loc = spawn_loc - carla.Location(x=rgt.x*10, y=rgt.y*10, z=0)
                    controller.go_to_location(target_loc)
                    controller.set_max_speed(1.8)
                    print("[DEMO] Spawning pedestrian 12m ahead!")
                else:
                    walker.destroy()
        except Exception as e:
            print(f"[ERR] Pedestrian spawn failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "=" * 55)
        print("  ADVANCED AUTONOMOUS DRIVING SYSTEM - STARTED")
        print("  BehaviorAgent + YOLO + DeepSORT + LiDAR + RL")
        print("=" * 55 + "\n")

        try:
            while True:
                # Fixed: TICK FIRST in sync mode — then read sensors
                # This ensures BehaviorAgent gets fresh waypoint data
                self.world.tick()

                # Check for weather command from dashboard
                if os.path.exists('weather_cmd.txt'):
                    try:
                        os.remove('weather_cmd.txt')
                    except Exception:
                        pass
                    self.weather_index = (self.weather_index + 1) % len(self.weather_presets)
                    self.world.set_weather(self.weather_presets[self.weather_index])
                    self.sound_manager.tick()
                    print(f"[DASHBOARD] Weather changed to: {self.weather_names[self.weather_index]}")

                if not self.image_queue.empty():
                    self.process_frame(self.image_queue.get())

                # Single waitKey location (Fixed: was in two places before)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:           # ESC
                    break
                elif key == ord('p'):
                    self.parking_requested = not self.parking_requested
                    self.sound_manager.tick()
                    print(f"[P] Parking: {'ON' if self.parking_requested else 'OFF'}")
                elif key == ord('r'):
                    self.use_rl = not self.use_rl
                    self.sound_manager.tick()
                    print(f"[R] RL Control: {'ON' if self.use_rl else 'OFF'}")
                elif key == ord('h'):
                    self.sound_manager.horn()  # Honk!
                    print("[H] Horn!")
                elif key == 82:  # UP arrow
                    self.target_speed = min(self.target_speed + 5, MAX_SPEED)
                    print(f"[SPEED] Target speed increased to: {self.target_speed} km/h")
                elif key == 84:  # DOWN arrow
                    self.target_speed = max(self.target_speed - 5, MIN_SPEED)
                    print(f"[SPEED] Target speed decreased to: {self.target_speed} km/h")
                elif key == ord('t'):
                    self.spawn_pedestrian_in_front()
                elif key == ord('c'):
                    # Change Weather
                    self.weather_index = (self.weather_index + 1) % len(self.weather_presets)
                    self.world.set_weather(self.weather_presets[self.weather_index])
                    self.sound_manager.tick()
                    print(f"[C] Weather changed to: {self.weather_names[self.weather_index]}")
                elif key == ord('d'):
                    # Debug: print current state
                    loc = self.vehicle.get_location()
                    wp = self.map.get_waypoint(loc)
                    print(f"[D] Location: ({loc.x:.1f}, {loc.y:.1f}) | "
                          f"Road ID: {wp.road_id} | Lane: {wp.lane_id}")

        finally:
            print("\n[..] Cleaning up...")
            
            for controller in self.controllers_list:
                controller.stop()
                controller.destroy()
            for walker in self.walkers_list:
                walker.destroy()
                
            self.sound_manager.cleanup()   # Stop all sounds first
            self.sensor_manager.cleanup()
            if self.vehicle:
                self.vehicle.destroy()
            cv2.destroyAllWindows()
            self.world.apply_settings(self.original_settings)
            self.traffic_manager.set_synchronous_mode(False)
            self.rl_agent.save()
            print("[OK] System shutdown complete!")


if __name__ == '__main__':
    system = AdvancedDrivingSystem()
    system.run()
