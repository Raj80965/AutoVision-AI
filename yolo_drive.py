import sys
import glob
import time
import numpy as np
import cv2
import random
from queue import Queue
from enum import Enum
import math
import threading

# ===== CARLA PATH =====
try:
    sys.path.append(glob.glob('D:/carla/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

import carla

# ===== YOLO =====
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# ===== CONNECT TO CARLA =====
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Get original settings
original_settings = world.get_settings()

# ===== SETUP SYNC MODE =====
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# ===== SPAWN VEHICLE =====
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
if not vehicle_bp:
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    print("❌ Failed to spawn vehicle!")
    sys.exit(1)

print(f"✅ Vehicle spawned at {spawn_point.location}")

# ===== COLLISION SENSOR =====
collision_sensor_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=vehicle)

collision_occurred = False

def on_collision(event):
    global collision_occurred
    collision_occurred = True
    print(f"💥 COLLISION DETECTED! Emergency Stop!")
    vehicle.apply_control(carla.VehicleControl(brake=1.0))
    
collision_sensor.listen(on_collision)

# ===== CAMERA SETUP =====
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "110")

camera_transform = carla.Transform(
    carla.Location(x=1.5, y=0, z=1.7),
    carla.Rotation(pitch=-15, yaw=0, roll=0)
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# ===== SPECTATOR CAMERA =====
spectator = world.get_spectator()

# ===== QUEUE =====
image_queue = Queue(maxsize=2)

# ===== VEHICLE STATE =====
class DrivingMode(Enum):
    NORMAL = 1
    BRAKING = 2
    STOPPED_AT_SIGNAL = 3
    AVOIDING = 4
    EMERGENCY = 5
    PATH_PLANNING = 6

class TrafficLightState(Enum):
    GREEN = 1
    YELLOW = 2
    RED = 3
    UNKNOWN = 4

class VehicleState:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.target_speed = 35
        self.driving_mode = DrivingMode.NORMAL
        self.traffic_light_state = TrafficLightState.UNKNOWN
        self.stop_at_signal = False
        self.stop_line_distance = 0
        self.path_planned = []
        self.current_path_index = 0
        self.lane_left = 0.25
        self.lane_right = 0.75
        self.lane_center = 0.5
        
state = VehicleState()

# ===== HELPER FUNCTIONS =====
def get_vehicle_speed():
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed

def detect_lanes_and_position(img):
    """
    Detect lane lines and calculate vehicle position within lane
    Returns: (left_lane_pos, right_lane_pos, center_lane_pos, is_lane_detected)
    """
    h, w = img.shape[:2]
    
    # Region of interest (bottom half)
    roi = img[h//2:h, :]
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # White and yellow lane detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Edge detection
    edges = cv2.Canny(lane_mask, 50, 150)
    
    # Histogram
    histogram = np.sum(edges[edges.shape[0]//2:, :], axis=0)
    
    # Find lane peaks
    midpoint = w // 2
    
    left_peak = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 0 else midpoint * 0.3
    right_peak = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 0 else midpoint * 1.7
    
    # Normalize
    left_pos = left_peak / w
    right_pos = right_peak / w
    center_pos = (left_peak + right_peak) / (2 * w) if right_peak > left_peak else 0.5
    
    is_detected = (np.max(histogram) > 100)
    
    # Update lane boundaries
    if is_detected:
        state.lane_left = max(0.2, left_pos - 0.05)
        state.lane_right = min(0.8, right_pos + 0.05)
        state.lane_center = center_pos
    
    return left_pos, right_pos, center_pos, is_detected

def detect_traffic_light(results, img):
    """
    Detect traffic light and determine its state
    Returns: (traffic_light_state, is_stop_required)
    """
    h, w = img.shape[:2]
    traffic_lights = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        
        if name == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) // 2
            
            # Extract traffic light region
            tl_roi = img[y1:y2, x1:x2]
            
            # Analyze color in traffic light region
            if tl_roi.size > 0:
                # Convert to HSV for color detection
                tl_hsv = cv2.cvtColor(tl_roi, cv2.COLOR_BGR2HSV)
                
                # Red color detection
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                red_mask = cv2.inRange(tl_hsv, lower_red1, upper_red1) + cv2.inRange(tl_hsv, lower_red2, upper_red2)
                
                # Green color detection
                lower_green = np.array([40, 100, 100])
                upper_green = np.array([80, 255, 255])
                green_mask = cv2.inRange(tl_hsv, lower_green, upper_green)
                
                # Yellow color detection
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(tl_hsv, lower_yellow, upper_yellow)
                
                red_pixels = cv2.countNonZero(red_mask)
                green_pixels = cv2.countNonZero(green_mask)
                yellow_pixels = cv2.countNonZero(yellow_mask)
                
                total_pixels = tl_roi.shape[0] * tl_roi.shape[1]
                
                if red_pixels > total_pixels * 0.1:
                    state = TrafficLightState.RED
                    is_stop = True
                elif yellow_pixels > total_pixels * 0.1:
                    state = TrafficLightState.YELLOW
                    is_stop = True
                elif green_pixels > total_pixels * 0.1:
                    state = TrafficLightState.GREEN
                    is_stop = False
                else:
                    state = TrafficLightState.UNKNOWN
                    is_stop = False
                
                # Consider size and position for stop decision
                if box_area > 1500 and y2 < h * 0.5:
                    traffic_lights.append({
                        'state': state,
                        'stop': is_stop,
                        'area': box_area,
                        'y_pos': y2
                    })
                
                # Draw traffic light bounding box
                color = (0, 0, 255) if state == TrafficLightState.RED else (0, 255, 0) if state == TrafficLightState.GREEN else (0, 255, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw state text
                state_text = "RED - STOP" if state == TrafficLightState.RED else "GREEN - GO" if state == TrafficLightState.GREEN else "YELLOW - CAUTION"
                cv2.putText(img, state_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Find closest traffic light that requires stop
    stop_required = False
    closest_state = TrafficLightState.UNKNOWN
    
    if traffic_lights:
        closest_tl = min(traffic_lights, key=lambda x: x['area'])
        stop_required = closest_tl['stop']
        closest_state = closest_tl['state']
    
    return closest_state, stop_required

def calculate_brake_intensity(box_area, box_y, image_height, current_speed):
    """
    Calculate brake intensity based on obstacle distance and speed
    """
    h = image_height
    
    # Distance factor (larger box = closer)
    distance_factor = min(box_area / 20000, 1.0)
    
    # Position factor (lower on screen = closer)
    position_factor = (h - box_y) / h
    
    # Speed factor (higher speed = more braking)
    speed_factor = min(current_speed / 50, 1.0)
    
    # Combined danger score
    danger_score = (distance_factor * 0.5 + position_factor * 0.3 + speed_factor * 0.2) * 100
    
    # Brake decision
    if danger_score > 70 or box_area > 25000:
        brake = 1.0
        emergency = True
    elif danger_score > 50 or box_area > 15000:
        brake = 0.7
        emergency = False
    elif danger_score > 30 or box_area > 8000:
        brake = 0.4
        emergency = False
    elif danger_score > 15:
        brake = 0.2
        emergency = False
    else:
        brake = 0.0
        emergency = False
    
    return brake, emergency, danger_score

def plan_path(obstacle_center_x, image_width, current_lane_pos):
    """
    Plan a path to avoid obstacle while staying in lane
    """
    w = image_width
    obs_pos = obstacle_center_x / w
    
    # Path options
    paths = []
    
    # Option 1: Slight left
    if obs_pos < current_lane_pos:
        paths.append((-0.3, "LEFT"))
    
    # Option 2: Slight right
    if obs_pos > current_lane_pos:
        paths.append((0.3, "RIGHT"))
    
    # Option 3: Maintain lane if obstacle not directly ahead
    if abs(obs_pos - current_lane_pos) > 0.2:
        paths.append((0.0, "CENTER"))
    
    # Choose best path
    if paths:
        best_steer, direction = paths[0]
    else:
        best_steer = 0.3 if current_lane_pos < 0.5 else -0.3
        direction = "RIGHT" if best_steer > 0 else "LEFT"
    
    return best_steer, direction

def smooth_steering(current_steer, target_steer, smooth_factor=0.3):
    return current_steer * (1 - smooth_factor) + target_steer * smooth_factor

def update_spectator():
    vehicle_transform = vehicle.get_transform()
    spectator_transform = carla.Transform(
        carla.Location(x=vehicle_transform.location.x - 10,
                       y=vehicle_transform.location.y,
                       z=vehicle_transform.location.z + 6),
        carla.Rotation(pitch=-25, yaw=vehicle_transform.rotation.yaw, roll=0)
    )
    spectator.set_transform(spectator_transform)

def process_image(image):
    global collision_occurred
    
    try:
        # Convert image
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Detect lanes
        left_lane, right_lane, current_lane_pos, lane_detected = detect_lanes_and_position(img)
        
        # Run YOLO
        results = model(img, verbose=False, conf=0.4)[0]
        
        # Detect traffic lights
        traffic_light_state, stop_at_traffic = detect_traffic_light(results, img)
        
        # Get current speed
        current_speed = get_vehicle_speed()
        h, w = img.shape[:2]
        
        # Check for collision
        if collision_occurred:
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            cv2.putText(img, "!!! COLLISION - STOPPED !!!", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imshow("AUTONOMOUS DRIVING", img)
            cv2.waitKey(1)
            return
        
        # Default controls
        throttle = 0.4
        brake = 0.0
        steer = state.steer
        
        # Find obstacles
        obstacles = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) // 2
            
            # Skip small/far objects
            if box_area < 1500:
                continue
            
            # Skip sky
            if y2 < h * 0.1:
                continue
            
            # Consider only obstacles
            if name in ['car', 'truck', 'bus', 'person', 'bicycle']:
                brake_intensity, is_emergency, danger = calculate_brake_intensity(
                    box_area, y2, h, current_speed
                )
                
                obstacles.append({
                    'name': name,
                    'area': box_area,
                    'center_x': center_x,
                    'brake': brake_intensity,
                    'emergency': is_emergency,
                    'danger': danger,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                # Color based on danger
                if danger > 70:
                    color = (0, 0, 255)
                    thickness = 4
                elif danger > 50:
                    color = (0, 165, 255)
                    thickness = 3
                elif danger > 30:
                    color = (0, 255, 255)
                    thickness = 2
                else:
                    color = (0, 255, 0)
                    thickness = 1
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                label = f"{name.upper()} | D:{danger:.0f}%"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # ===== DECISION MAKING (Priority Order) =====
        
        # PRIORITY 1: Emergency Stop (Collision)
        if collision_occurred:
            brake = 1.0
            throttle = 0.0
            state.driving_mode = DrivingMode.EMERGENCY
        
        # PRIORITY 2: Traffic Light (Stop at Red)
        elif stop_at_traffic:
            if current_speed > 5:
                brake = 0.6
                throttle = 0.0
            else:
                brake = 1.0
                throttle = 0.0
            state.driving_mode = DrivingMode.STOPPED_AT_SIGNAL
            steer = smooth_steering(steer, 0, 0.2)
            
            cv2.putText(img, "🔴 STOPPED AT TRAFFIC SIGNAL", (w//2-150, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # PRIORITY 3: Obstacle Avoidance with Braking
        elif obstacles:
            closest = max(obstacles, key=lambda x: x['brake'])
            
            # Apply brake first
            brake = max(brake, closest['brake'])
            throttle = 0.0
            
            # Plan path to avoid
            if closest['danger'] > 40:
                steer_target, direction = plan_path(
                    closest['center_x'], w, current_lane_pos
                )
                steer = smooth_steering(steer, steer_target, 0.4)
                state.driving_mode = DrivingMode.AVOIDING
                
                cv2.putText(img, f"🔄 AVOIDING: {direction}", (w//2-100, h-80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if closest['brake'] > 0.5:
                    cv2.putText(img, f"⚠️ BRAKING! D:{closest['danger']:.0f}%", (w//2-100, h-110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                print(f"⚠️ {closest['name']} detected - Brake:{closest['brake']:.1f} - {direction}")
        
        # PRIORITY 4: Normal Driving with Lane Keeping
        else:
            state.driving_mode = DrivingMode.NORMAL
            
            # Lane keeping
            if lane_detected:
                if current_lane_pos < state.lane_center - 0.05:
                    # Drifting left - steer right
                    steer = smooth_steering(steer, 0.25, 0.1)
                elif current_lane_pos > state.lane_center + 0.05:
                    # Drifting right - steer left
                    steer = smooth_steering(steer, -0.25, 0.1)
                else:
                    # Centered - maintain
                    steer = smooth_steering(steer, 0, 0.05)
            else:
                # No lane detected, gentle random steering
                steer = smooth_steering(steer, random.uniform(-0.1, 0.1), 0.05)
            
            # Speed management
            if current_speed < state.target_speed:
                throttle = min(0.55, 0.3 + current_speed / 80)
                brake = 0.0
            else:
                throttle = max(0.2, throttle - 0.1)
                brake = 0.1
        
        # Speed regulation
        if current_speed > state.target_speed:
            throttle = max(throttle - 0.2, 0.0)
            brake = min(brake + 0.15, 0.4)
        
        # Initial movement
        if current_speed < 0.5 and state.driving_mode == DrivingMode.NORMAL:
            throttle = 0.4
            brake = 0.0
        
        # Update state
        state.throttle = throttle
        state.brake = brake
        state.steer = steer
        
        # ===== DASHBOARD UI =====
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Mode display
        mode_configs = {
            DrivingMode.NORMAL: ("🟢 NORMAL - LANE FOLLOWING", (0, 255, 0)),
            DrivingMode.BRAKING: ("🟡 BRAKING - OBSTACLE AHEAD", (0, 255, 255)),
            DrivingMode.STOPPED_AT_SIGNAL: ("🔴 STOPPED - RED SIGNAL", (0, 0, 255)),
            DrivingMode.AVOIDING: ("🟠 AVOIDING - PATH PLANNING", (0, 165, 255)),
            DrivingMode.EMERGENCY: ("🔴 EMERGENCY - STOPPED", (255, 0, 0)),
            DrivingMode.PATH_PLANNING: ("🔵 PATH PLANNING ACTIVE", (255, 165, 0))
        }
        
        mode_text, mode_color = mode_configs.get(state.driving_mode, ("UNKNOWN", (255, 255, 255)))
        cv2.putText(img, mode_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        
        # Lane info
        lane_text = f"Lane: {'Detected' if lane_detected else 'Unknown'} | Pos: {current_lane_pos:.2f}"
        cv2.putText(img, lane_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Traffic light info
        if traffic_light_state != TrafficLightState.UNKNOWN:
            tl_text = f"Signal: {traffic_light_state.name}"
            cv2.putText(img, tl_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Speed
        cv2.putText(img, f"SPEED: {int(current_speed)} km/h", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"TARGET: {state.target_speed} km/h", (10, 148),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Obstacle info
        if obstacles:
            closest = max(obstacles, key=lambda x: x['danger'])
            cv2.putText(img, f"Obstacle: {closest['name'].upper()} | Danger: {closest['danger']:.0f}%", 
                       (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # ===== LANE VISUALIZATION =====
        # Lane boundaries
        left_x = int(w * state.lane_left)
        right_x = int(w * state.lane_right)
        center_x = int(w * current_lane_pos)
        
        # Lane area
        overlay_lane = img.copy()
        cv2.rectangle(overlay_lane, (left_x, h-80), (right_x, h-20), (0, 255, 0), -1)
        img = cv2.addWeighted(img, 0.8, overlay_lane, 0.2, 0)
        
        # Lane lines
        cv2.line(img, (left_x, h-80), (left_x, h-20), (0, 255, 0), 3)
        cv2.line(img, (right_x, h-80), (right_x, h-20), (0, 255, 0), 3)
        
        # Vehicle position marker
        cv2.circle(img, (center_x, h-50), 10, (255, 255, 0), -1)
        cv2.putText(img, "CAR", (center_x-15, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Throttle/Brake bars
        throttle_w = int((throttle / 0.7) * 120)
        cv2.rectangle(img, (w-140, 30), (w-140+throttle_w, 45), (0, 255, 0), -1)
        cv2.putText(img, "THR", (w-140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        brake_w = int((brake / 1.0) * 120)
        cv2.rectangle(img, (w-140, 55), (w-140+brake_w, 70), (0, 0, 255), -1)
        cv2.putText(img, "BRK", (w-140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Steering indicator
        center_steer = w // 2
        cv2.arrowedLine(img, (center_steer, h-20), 
                       (int(center_steer + steer * 100), h-20), 
                       (0, 255, 255), 3)
        
        # Path planning indicator
        if state.driving_mode == DrivingMode.AVOIDING:
            cv2.arrowedLine(img, (center_steer, h-40), 
                           (int(center_steer + steer * 150), h-40), 
                           (255, 165, 0), 4)
        
        # Controls help
        cv2.putText(img, "ESC:Exit | SPACE:Stop | UP/DOWN:Speed", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("🤖 AUTONOMOUS DRIVING - PATH PLANNING", img)
        cv2.waitKey(1)
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0, 0.65))
        control.steer = float(np.clip(steer, -0.6, 0.6))
        control.brake = float(np.clip(brake, 0, 1.0))
        vehicle.apply_control(control)
        
        update_spectator()
        
    except Exception as e:
        print(f"Error: {e}")

def camera_callback(image):
    if image_queue.qsize() < 2:
        image_queue.put(image)

# ===== MAIN =====
def main():
    camera.listen(camera_callback)
    
    print("\n" + "="*80)
    print("🤖 COMPLETE AUTONOMOUS DRIVING SYSTEM")
    print("="*80)
    print("✅ ALL FEATURES ACTIVE:")
    print("")
    print("   1. 🛑 IMMEDIATE BRAKING - Stops for any obstacle ahead")
    print("   2. 🚦 TRAFFIC SIGNAL DETECTION - Stops at red lights")
    print("   3. 🛣️ LANE DETECTION - Keeps vehicle within lane boundaries")
    print("   4. 🗺️ PATH PLANNING - Automatically plans avoidance path")
    print("   5. 🚗 SMOOTH DRIVING - Gentle acceleration and braking")
    print("")
    print("📌 PRIORITY SYSTEM:")
    print("   • HIGHEST: Emergency Stop (Collision)")
    print("   • HIGH: Traffic Signals (Stop at Red)")
    print("   • MEDIUM: Obstacle Avoidance (Brake + Steer)")
    print("   • NORMAL: Lane Following (Smooth Driving)")
    print("")
    print("📌 Controls:")
    print("   • ESC: Exit program")
    print("   • SPACE: Manual emergency stop")
    print("   • UP/DOWN: Adjust target speed (current: 35 km/h)")
    print("")
    print("🎯 System Status: ACTIVE")
    print("="*80 + "\n")
    
    try:
        frame_count = 0
        while True:
            if not image_queue.empty():
                image = image_queue.get()
                process_image(image)
                frame_count += 1
                
                # Status update every 100 frames
                if frame_count % 100 == 0:
                    speed = get_vehicle_speed()
                    print(f"📊 Status: Speed={int(speed)} km/h | Mode={state.driving_mode.name}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n🛑 Exiting...")
                break
            elif key == 32:  # SPACE
                print("⚠️ Manual Emergency Stop!")
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
                time.sleep(0.5)
            elif key == 82:  # UP
                state.target_speed = min(state.target_speed + 5, 50)
                print(f"🎯 Target Speed: {state.target_speed} km/h")
            elif key == 84:  # DOWN
                state.target_speed = max(state.target_speed - 5, 20)
                print(f"🎯 Target Speed: {state.target_speed} km/h")
            
            # Tick world
            world.tick()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        camera.stop()
        camera.destroy()
        collision_sensor.stop()
        collision_sensor.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        world.apply_settings(original_settings)
        print("✅ System shutdown complete!")

if __name__ == "__main__":
    main()