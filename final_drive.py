import sys
import glob
import time
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

import carla

# ===== YOLO =====
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# ===== CONNECT TO CARLA =====
print("🔌 Connecting to CARLA...")
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Get original settings
original_settings = world.get_settings()

# ===== SETUP SYNC MODE =====
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.03
world.apply_settings(settings)

# ===== SPAWN VEHICLE =====
print("🚗 Spawning vehicle...")
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
if not vehicle_bp:
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    print("❌ Failed to spawn vehicle!")
    sys.exit(1)

# REMOVED: vehicle.set_color() - this was causing the error
print(f"✅ Vehicle spawned successfully!")

# ===== COLLISION SENSOR =====
collision_sensor_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=vehicle)
collision_occurred = False
collision_time = 0

def on_collision(event):
    global collision_occurred, collision_time
    collision_occurred = True
    collision_time = time.time()
    print(f"💥 COLLISION DETECTED!")
    
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
print("📷 Camera attached")

# ===== SPECTATOR CAMERA =====
spectator = world.get_spectator()

def update_spectator():
    """Update spectator to follow vehicle from behind"""
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    
    # Calculate position behind vehicle
    yaw_rad = math.radians(vehicle_rotation.yaw)
    
    # Camera parameters
    distance_behind = 10.0  # meters
    height_above = 5.0      # meters
    
    cam_x = vehicle_location.x - (math.cos(yaw_rad) * distance_behind)
    cam_y = vehicle_location.y - (math.sin(yaw_rad) * distance_behind)
    cam_z = vehicle_location.z + height_above
    
    spectator_transform = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=-15, yaw=vehicle_rotation.yaw, roll=0)
    )
    spectator.set_transform(spectator_transform)

# ===== QUEUE =====
image_queue = Queue(maxsize=1)

# ===== VEHICLE STATE =====
class DrivingMode(Enum):
    NORMAL = 1
    BRAKING = 2
    AVOIDING = 3
    EMERGENCY = 4

class VehicleState:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.target_speed = 40
        self.driving_mode = DrivingMode.NORMAL
        self.lane_center = 0.5
        
state = VehicleState()

# ===== HELPER FUNCTIONS =====
def get_vehicle_speed():
    velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed

def calculate_time_to_collision(box_area, box_y, image_height, speed):
    """Calculate Time to Collision in seconds"""
    if speed < 0.5:
        return 999
    
    area_ratio = min(box_area / 25000, 1.0)
    vertical_ratio = (image_height - box_y) / image_height
    ttc = (1 - area_ratio) * 4 / max(speed / 30, 0.5)
    return max(ttc, 0.3)

def simple_lane_detection(img):
    """Simple lane detection for position tracking"""
    h, w = img.shape[:2]
    roi = img[h//2:h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    histogram = np.sum(lane_mask[lane_mask.shape[0]//2:, :], axis=0)
    
    if np.max(histogram) > 50:
        midpoint = len(histogram) // 2
        left = np.argmax(histogram[:midpoint])
        right = np.argmax(histogram[midpoint:]) + midpoint
        center = (left + right) / (2 * len(histogram))
        return center, True
    else:
        return 0.5, False

def calculate_evasive_steering(obstacle_x, image_width, current_lane, ttc):
    """Calculate steering direction and amount to avoid collision"""
    w = image_width
    obs_pos = obstacle_x / w
    
    if ttc < 1.2:
        # Emergency - aggressive steering
        if obs_pos < 0.4:
            return 0.7, "RIGHT"
        elif obs_pos > 0.6:
            return -0.7, "LEFT"
        else:
            return 0.6, "RIGHT"
    else:
        # Normal avoidance
        if obs_pos < 0.4:
            return 0.5, "RIGHT"
        elif obs_pos > 0.6:
            return -0.5, "LEFT"
        else:
            return 0.4, "RIGHT"

def smooth_value(current, target, factor=0.35):
    return current * (1 - factor) + target * factor

# Frame counter
frame_counter = 0

def process_image(image):
    global collision_occurred, frame_counter
    
    try:
        # Convert image
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Lane detection
        lane_pos, lane_detected = simple_lane_detection(img)
        if lane_detected:
            state.lane_center = smooth_value(state.lane_center, lane_pos, 0.2)
        
        # Run YOLO
        results = model(img, verbose=False, conf=0.4)[0]
        
        current_speed = get_vehicle_speed()
        h, w = img.shape[:2]
        
        # Check collision recovery
        if collision_occurred:
            if time.time() - collision_time < 2:
                vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                cv2.putText(img, "!!! COLLISION - STOPPED !!!", (w//2-150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imshow("AUTONOMOUS DRIVING", img)
                cv2.waitKey(1)
                return
            else:
                collision_occurred = False
        
        # Default controls
        throttle = 0.4
        brake = 0.0
        steer = state.steer
        
        # Find closest obstacle
        closest_obstacle = None
        min_ttc = float('inf')
        
        for box in results.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            
            if name in ['car', 'truck', 'bus', 'person', 'bicycle']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) // 2
                
                if box_area > 1500:  # Only consider close obstacles
                    ttc = calculate_time_to_collision(box_area, y2, h, current_speed)
                    
                    if ttc < min_ttc:
                        min_ttc = ttc
                        closest_obstacle = {
                            'name': name,
                            'area': box_area,
                            'center_x': center_x,
                            'ttc': ttc,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                        }
                    
                    # Color coding based on TTC
                    if ttc < 1.0:
                        color = (0, 0, 255)  # RED
                        thickness = 4
                    elif ttc < 2.0:
                        color = (0, 165, 255)  # ORANGE
                        thickness = 3
                    elif ttc < 3.5:
                        color = (0, 255, 255)  # YELLOW
                        thickness = 2
                    else:
                        color = (0, 255, 0)  # GREEN
                        thickness = 1
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(img, f"{name.upper()} | TTC:{ttc:.1f}s", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # ===== DECISION MAKING =====
        if closest_obstacle and closest_obstacle['ttc'] < 3.5:
            # Obstacle detected - brake and avoid
            ttc = closest_obstacle['ttc']
            
            # Calculate brake intensity based on TTC
            if ttc < 1.0:
                brake = 1.0
                throttle = 0.0
                state.driving_mode = DrivingMode.EMERGENCY
            elif ttc < 2.0:
                brake = 0.7
                throttle = 0.0
                state.driving_mode = DrivingMode.BRAKING
            else:
                brake = 0.4
                throttle = 0.1
                state.driving_mode = DrivingMode.AVOIDING
            
            # Calculate evasive steering
            steer_target, direction = calculate_evasive_steering(
                closest_obstacle['center_x'], w, state.lane_center, ttc
            )
            steer = smooth_value(steer, steer_target, 0.5)
            
            # Draw avoidance indicator
            cv2.arrowedLine(img, (w//2, h-80), 
                           (int(w//2 + steer_target * 150), h-80), 
                           (0, 255, 255), 5)
            cv2.putText(img, f"AVOIDING: {direction}", (w//2-80, h-100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # TTC warning
            if ttc < 1.5:
                cv2.putText(img, f"⚠️ COLLISION IN {ttc:.1f}s!", (w//2-120, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Console log
            if frame_counter % 30 == 0:
                print(f"⚠️ {closest_obstacle['name']} | TTC:{ttc:.1f}s | Brake:{brake:.0%} | {direction}")
        
        else:
            # Normal driving
            state.driving_mode = DrivingMode.NORMAL
            
            # Lane keeping
            if lane_detected:
                error = state.lane_center - 0.5
                steer_target = -error * 0.6
                steer = smooth_value(steer, steer_target, 0.15)
            else:
                steer = smooth_value(steer, 0, 0.08)
            
            # Speed management
            if current_speed < state.target_speed:
                throttle = min(0.5, 0.25 + current_speed/80)
                brake = 0.0
            else:
                throttle = max(0.15, throttle - 0.1)
                brake = 0.05
        
        # Speed limit control
        if current_speed > state.target_speed + 5:
            throttle = max(throttle - 0.2, 0.0)
            brake = min(brake + 0.15, 0.5)
        
        # Initial movement
        if current_speed < 1.0 and state.driving_mode == DrivingMode.NORMAL:
            throttle = 0.35
            brake = 0.0
        
        # Update state
        state.throttle = throttle
        state.brake = brake
        state.steer = steer
        
        # ===== DASHBOARD UI =====
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
        
        # Mode display
        mode_configs = {
            DrivingMode.NORMAL: ("🟢 NORMAL DRIVING", (0, 255, 0)),
            DrivingMode.BRAKING: ("🟡 BRAKING - OBSTACLE AHEAD", (0, 255, 255)),
            DrivingMode.AVOIDING: ("🟠 AVOIDING OBSTACLE", (0, 165, 255)),
            DrivingMode.EMERGENCY: ("🔴 EMERGENCY STOP", (0, 0, 255))
        }
        
        mode_text, mode_color = mode_configs.get(state.driving_mode, ("UNKNOWN", (255, 255, 255)))
        cv2.putText(img, mode_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Speed display
        cv2.putText(img, f"SPEED: {int(current_speed)} km/h", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"TARGET: {state.target_speed} km/h", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Obstacle info
        if closest_obstacle:
            cv2.putText(img, f"⚠️ {closest_obstacle['name'].upper()} | TTC: {closest_obstacle['ttc']:.1f}s", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Lane info
        if lane_detected:
            cv2.putText(img, f"LANE POS: {state.lane_center:.2f}", (10, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # Throttle/Brake bars
        throttle_w = int((throttle / 0.6) * 100)
        cv2.rectangle(img, (w-120, 30), (w-120+throttle_w, 45), (0, 255, 0), -1)
        cv2.putText(img, "THR", (w-120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        brake_w = int((brake / 1.0) * 100)
        cv2.rectangle(img, (w-120, 55), (w-120+brake_w, 70), (0, 0, 255), -1)
        cv2.putText(img, "BRK", (w-120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Steering indicator
        cv2.arrowedLine(img, (w//2, h-30), (int(w//2 + steer * 120), h-30), (0, 255, 255), 3)
        cv2.putText(img, f"STEER: {steer:.2f}", (w//2-40, h-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Controls help
        cv2.putText(img, "ESC:Exit | SPACE:Stop | UP/DOWN:Speed", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("AUTONOMOUS DRIVING SYSTEM", img)
        cv2.waitKey(1)
        
        # Apply control
        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0, 0.6))
        control.steer = float(np.clip(steer, -0.7, 0.7))
        control.brake = float(np.clip(brake, 0, 1.0))
        vehicle.apply_control(control)
        
        # Update spectator view
        update_spectator()
        
        frame_counter += 1
        
    except Exception as e:
        print(f"Error: {e}")

def camera_callback(image):
    if image_queue.empty():
        image_queue.put(image)

# ===== MAIN =====
def main():
    camera.listen(camera_callback)
    
    print("\n" + "="*70)
    print("🚗 AUTONOMOUS DRIVING SYSTEM - FINAL WORKING")
    print("="*70)
    print("")
    print("✅ WORKING FEATURES:")
    print("   • Predictive Braking (TTC based)")
    print("   • Evasive Steering")
    print("   • Lane Detection")
    print("   • Obstacle Detection (YOLO)")
    print("")
    print("📌 WHAT TO DO:")
    print("   1. Look at CARLA Simulator window - Your car is there!")
    print("   2. Dashboard window shows camera view")
    print("   3. Car will automatically drive")
    print("")
    print("📌 CONTROLS:")
    print("   • UP ARROW  → Increase speed")
    print("   • DOWN ARROW → Decrease speed")
    print("   • SPACE     → Emergency stop")
    print("   • ESC       → Exit")
    print("")
    print("="*70)
    print("🚀 SYSTEM STARTING... Car will move now!\n")
    
    try:
        while True:
            if not image_queue.empty():
                image = image_queue.get()
                process_image(image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n🛑 Exiting...")
                break
            elif key == 32:  # SPACE
                print("⚠️ Emergency Stop!")
                vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
                time.sleep(0.5)
            elif key == 82:  # UP
                state.target_speed = min(state.target_speed + 5, 60)
                print(f"🎯 Target Speed: {state.target_speed} km/h")
            elif key == 84:  # DOWN
                state.target_speed = max(state.target_speed - 5, 25)
                print(f"🎯 Target Speed: {state.target_speed} km/h")
            
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