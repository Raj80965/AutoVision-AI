"""
Path Planning and Obstacle Avoidance
"""

import math

class PathPlanner:
    def __init__(self):
        self.avoid_direction = 0
        self.avoid_timer = 0
        
    def calculate_ttc(self, box_area, box_y, image_height, speed):
        """Calculate Time to Collision"""
        if speed < 0.5:
            return 999
        
        area_ratio = min(box_area / 25000, 1.0)
        vertical_ratio = (image_height - box_y) / image_height
        ttc = (1 - area_ratio) * 4 / max(speed / 30, 0.5)
        return max(ttc, 0.3)
    
    def calculate_brake_intensity(self, ttc, box_area):
        """Calculate brake intensity based on TTC"""
        if ttc < 1.0 or box_area > 25000:
            return 1.0, True
        elif ttc < 2.0 or box_area > 15000:
            return 0.7, False
        elif ttc < 3.5 or box_area > 8000:
            return 0.4, False
        elif ttc < 5.0:
            return 0.2, False
        else:
            return 0.0, False
    
    def calculate_evasive_steering(self, obstacle_x, image_width, lane_center, ttc):
        """Calculate steering direction for obstacle avoidance"""
        w = image_width
        obs_pos = obstacle_x / w
        
        if ttc < 1.2:
            if obs_pos < 0.4:
                return 0.7, "RIGHT"
            elif obs_pos > 0.6:
                return -0.7, "LEFT"
            else:
                return 0.6, "RIGHT"
        else:
            if obs_pos < 0.35:
                return 0.5, "RIGHT"
            elif obs_pos > 0.65:
                return -0.5, "LEFT"
            else:
                if lane_center < 0.5:
                    return 0.45, "RIGHT"
                else:
                    return -0.45, "LEFT"

if __name__ == "__main__":
    print("="*50)
    print("Testing path_planning.py")
    print("="*50)
    
    planner = PathPlanner()
    
    ttc = planner.calculate_ttc(10000, 500, 720, 30)
    print(f"TTC: {ttc:.1f} seconds")
    
    brake, emergency = planner.calculate_brake_intensity(1.5, 10000)
    print(f"Brake: {brake:.1f}, Emergency: {emergency}")
    
    steer, direction = planner.calculate_evasive_steering(400, 1280, 0.5, 1.5)
    print(f"Steer: {steer:.2f}, Direction: {direction}")
    
    print("✅ path_planning.py working!")