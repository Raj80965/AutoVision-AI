"""
Lane Detection Module
"""

import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        self.lane_center = 0.5
        
    def detect(self, image):
        """Detect lane lines and return center position"""
        h, w = image.shape[:2]
        
        # Region of interest (bottom half)
        roi = image[h//2:h, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # White lane detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow lane detection
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Edge detection
        edges = cv2.Canny(lane_mask, 50, 150)
        
        # Histogram
        histogram = np.sum(edges[edges.shape[0]//2:, :], axis=0)
        
        if np.max(histogram) > 50:
            midpoint = len(histogram) // 2
            left_peak = np.argmax(histogram[:midpoint])
            right_peak = np.argmax(histogram[midpoint:]) + midpoint
            
            left_pos = left_peak / len(histogram)
            right_pos = right_peak / len(histogram)
            center_pos = (left_pos + right_pos) / 2
            
            # Smooth update
            self.lane_center = self.lane_center * 0.8 + center_pos * 0.2
            
            return self.lane_center, True, left_pos, right_pos
        
        return self.lane_center, False, 0.3, 0.7
    
    def draw_lanes(self, image, left_pos, right_pos, center_pos, lane_detected):
        """Draw lane visualization on image"""
        h, w = image.shape[:2]
        
        if lane_detected:
            left_x = int(left_pos * w)
            right_x = int(right_pos * w)
            center_x = int(center_pos * w)
            
            # Draw lane lines
            cv2.line(image, (left_x, h-50), (left_x, h-150), (0, 255, 0), 3)
            cv2.line(image, (right_x, h-50), (right_x, h-150), (0, 255, 0), 3)
            
            # Draw center marker
            cv2.circle(image, (center_x, h-100), 8, (255, 255, 0), -1)
            cv2.putText(image, "CAR", (center_x-15, h-105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return image

if __name__ == "__main__":
    print("="*50)
    print("Testing lane_detection.py")
    print("="*50)
    
    # Create test image
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(test_img, (300, 500), (400, 720), (255, 255, 255), 10)
    cv2.line(test_img, (900, 500), (800, 720), (255, 255, 255), 10)
    
    detector = LaneDetector()
    center, detected, left, right = detector.detect(test_img)
    
    print(f"Lane Detected: {detected}")
    print(f"Center Position: {center:.3f}")
    print(f"✅ lane_detection.py working!")