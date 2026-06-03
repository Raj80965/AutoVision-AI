"""
Main Entry Point - Complete Autonomous Driving System
"""

import sys
import time
import cv2
import numpy as np  # Fixed: was missing

from carla_manager import CarlaManager
from sensors import SensorManager
from detection import DetectionManager
from lane_detection import LaneDetector
from path_planning import PathPlanner
from rl_agent import RLAgent
from vehicle_control import VehicleController
from dashboard import Dashboard
from config import *

class AutonomousDrivingSystem:
    def __init__(self):
        self.carla = CarlaManager()
        self.sensors = None
        self.detector = DetectionManager(YOLO_MODEL)
        self.lane_detector = LaneDetector()
        self.path_planner = PathPlanner()
        self.rl_agent = RLAgent()
        self.controller = VehicleController()
        self.dashboard = Dashboard()
        
        self.frame_count = 0
        
    def setup(self):
        """Setup all components"""
        print("\n" + "="*80)
        print("🚗 COMPLETE AUTONOMOUS DRIVING SYSTEM")
        print("="*80)
        
        # Connect to CARLA
        if not self.carla.connect():
            return False
        self.carla.setup_sync_mode()
        
        # Spawn vehicle
        if not self.carla.spawn_vehicle():
            return False
        
        # Setup sensors
        self.sensors = SensorManager(self.carla.world, self.carla.vehicle)
        self.sensors.setup_camera()
        self.sensors.setup_lidar()
        self.sensors.setup_collision_sensor()
        
        return True
    
    def process_camera_image(self, image):
        """Process camera image and run detection"""
        # Convert CARLA image to numpy array
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def run(self):
        """Main loop"""
        if not self.setup():
            print("❌ Setup failed!")
            return
        
        print("\n🚀 SYSTEM STARTING...\n")
        print("📌 Controls: UP/DOWN=Speed, SPACE=Stop, ESC=Exit\n")
        
        try:
            while True:
                # Get camera image from queue
                if not self.carla.camera_queue.empty():
                    image = self.carla.camera_queue.get()
                    img = self.process_camera_image(image)
                    
                    # Lane detection
                    lane_center, lane_detected, left_lane, right_lane = self.lane_detector.detect(img)
                    
                    # Object detection
                    results = self.detector.detect(img, YOLO_CONFIDENCE)
                    obstacles = self.detector.get_obstacles(results, img.shape[0])
                    traffic_signs = self.detector.detect_traffic_signs(results, img)
                    
                    # Get current speed
                    current_speed = self.carla.get_vehicle_speed()
                    
                    # Process obstacles and calculate TTC
                    closest_obstacle = None
                    min_ttc = float('inf')
                    
                    for obs in obstacles:
                        ttc = self.path_planner.calculate_ttc(
                            obs['area'], obs['y2'], img.shape[0], current_speed
                        )
                        if ttc < min_ttc:
                            min_ttc = ttc
                            closest_obstacle = obs
                            obs['ttc'] = ttc
                    
                    # RL decision
                    state_key = self.rl_agent.get_state_key(
                        current_speed, lane_center, closest_obstacle is not None, min_ttc
                    )
                    rl_action = self.rl_agent.get_action(state_key)
                    
                    # Vehicle control decision
                    throttle, brake, steer, driving_mode = self.controller.decide_controls(
                        closest_obstacle, min_ttc, lane_center, lane_detected,
                        TARGET_SPEED, current_speed, rl_action
                    )
                    
                    # Update controls
                    throttle, brake, steer = self.controller.update_controls(throttle, steer, brake)
                    
                    # Calculate reward for RL
                    reward = self.rl_agent.calculate_reward(
                        current_speed, TARGET_SPEED, self.sensors.collision_occurred,
                        min_ttc, lane_center, closest_obstacle is not None and min_ttc < 2.0
                    )
                    
                    # Update RL agent
                    next_state_key = self.rl_agent.get_state_key(
                        current_speed, lane_center, closest_obstacle is not None, min_ttc
                    )
                    self.rl_agent.update(state_key, rl_action, reward, next_state_key)
                    
                    # Apply control to vehicle
                    self.carla.apply_control(throttle, steer, brake)
                    
                    # Draw detections on image
                    img = self.detector.draw_detections(img, obstacles, [obs.get('ttc', 5) for obs in obstacles])
                    img = self.lane_detector.draw_lanes(img, left_lane, right_lane, lane_center, lane_detected)
                    
                    # Draw dashboard
                    img = self.dashboard.draw(
                        img, (throttle, brake, steer), current_speed, TARGET_SPEED,
                        closest_obstacle, min_ttc, lane_center, lane_detected,
                        driving_mode, reward, len(self.sensors.lidar_data) > 0
                    )
                    
                    # Show image
                    self.dashboard.show(img)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == 32:  # SPACE
                        self.carla.apply_control(0, 0, 1.0)
                        time.sleep(0.5)
                    elif key == 82:  # UP arrow
                        global TARGET_SPEED  # Fixed: single global declaration
                        TARGET_SPEED = min(TARGET_SPEED + 5, MAX_SPEED)
                        print(f"Speed: {TARGET_SPEED} km/h")
                    elif key == 84:  # DOWN arrow
                        TARGET_SPEED = max(TARGET_SPEED - 5, MIN_SPEED)
                        print(f"Speed: {TARGET_SPEED} km/h")
                    
                    self.frame_count += 1
                
                # Tick world
                self.carla.world.tick()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\n🧹 Cleaning up...")
        if self.sensors:
            self.sensors.cleanup()
        self.carla.cleanup()
        self.dashboard.destroy()
        print("✅ System shutdown complete!")

if __name__ == "__main__":
    system = AutonomousDrivingSystem()
    system.run()