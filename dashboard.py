"""
Dashboard UI Display
"""

import cv2
import numpy as np
from config import COLORS

class Dashboard:
    def __init__(self, window_name="AUTONOMOUS DRIVING SYSTEM"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
    def draw(self, image, state, current_speed, target_speed, obstacle, 
             ttc, lane_center, lane_detected, driving_mode, reward, lidar_active):
        """Draw futuristic cyberpunk dashboard on image"""
        h, w = image.shape[:2]
        
        # Cyberpunk Theme Colors (BGR)
        CYAN = (255, 255, 0)
        MAGENTA = (255, 0, 255)
        WHITE = (255, 255, 255)
        DARK_BG = (10, 10, 15)
        
        # Top Dashboard Overlay (Semi-transparent)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), DARK_BG, -1)
        # Bottom Dashboard Overlay
        cv2.rectangle(overlay, (0, h-80), (w, h), DARK_BG, -1)
        image = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
        
        # Draw techy borders
        cv2.line(image, (0, 110), (w, 110), CYAN, 2)
        cv2.line(image, (0, h-80), (w, h-80), CYAN, 2)
        
        # Mode and color
        mode_config = {
            "NORMAL": ("SYS: ONLINE | CRUISE", CYAN),
            "BRAKING": ("SYS: ALERT | BRAKING", (0, 200, 255)),
            "AVOIDING": ("SYS: WARNING | AVOID", MAGENTA),
            "EMERGENCY": ("SYS: CRITICAL | STOP", (0, 0, 255))
        }
        mode_text, mode_color = mode_config.get(driving_mode, ("SYS: UNKNOWN", WHITE))
        
        cv2.putText(image, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(image, f"VELOCITY: {int(current_speed):03d} KM/H", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
        cv2.putText(image, f"TGT: {target_speed:03d} KM/H", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        
        # Obstacle info in center top
        if obstacle:
            cv2.rectangle(image, (w//2 - 160, 15), (w//2 + 160, 65), (0, 0, 150), -1)
            cv2.rectangle(image, (w//2 - 160, 15), (w//2 + 160, 65), (0, 0, 255), 2)
            cv2.putText(image, f"!! COLLISION WARNING !!", (w//2 - 115, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(image, f"TGT: {obstacle['name'].upper()} | TTC: {ttc:.2f}S", (w//2 - 110, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        else:
            cv2.putText(image, "[ RADAR: CLEAR ]", (w//2 - 70, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        
        # Lane info
        if lane_detected:
            cv2.putText(image, f"LANE DEV: {lane_center:+.2f}", (w//2 - 60, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1)
        
        # Sensors & RL Status (Top Right)
        cv2.putText(image, f"AI REWARD: {reward:+.2f}", (w-220, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1)
        lidar_color = CYAN if lidar_active else (100, 100, 100)
        cv2.putText(image, "LDR: ACTIVE" if lidar_active else "LDR: OFFLINE", (w-220, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, lidar_color, 1)
        
        # Throttle/Brake/Steer Visualization (Bottom)
        throttle, brake, steer = state
        
        # Draw a central steering wheel indicator
        center_x, center_y = w//2, h - 40
        cv2.circle(image, (center_x, center_y), 25, CYAN, 1)
        cv2.line(image, (center_x, center_y), 
                 (int(center_x + steer * 35), int(center_y - 25)), MAGENTA, 3)
        cv2.putText(image, f"STR: {steer:+.2f}", (center_x - 35, center_y - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
        # Throttle bar
        t_width = int((throttle / 1.0) * 150)
        cv2.rectangle(image, (40, h - 50), (190, h - 35), (40, 40, 40), -1)
        cv2.rectangle(image, (40, h - 50), (40 + t_width, h - 35), CYAN, -1)
        cv2.putText(image, f"THR: {throttle*100:.0f}%", (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        
        # Brake bar
        b_width = int((brake / 1.0) * 150)
        cv2.rectangle(image, (220, h - 50), (370, h - 35), (40, 40, 40), -1)
        cv2.rectangle(image, (220, h - 50), (220 + b_width, h - 35), (0, 0, 255), -1)
        cv2.putText(image, f"BRK: {brake*100:.0f}%", (220, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Controls help
        cv2.putText(image, "[ESC] EXIT | [^/v] SPEED | [SPACE] E-BRAKE", (w - 420, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(image, "MODULES: LKA | ACC | AEB | DRL", (w - 420, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1)
        
        return image
    
    def show(self, image):
        """Display image"""
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
    
    def destroy(self):
        """Destroy window"""
        cv2.destroyAllWindows()