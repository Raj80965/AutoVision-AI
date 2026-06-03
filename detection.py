"""
YOLO Detection and Traffic Sign Recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import COLORS

class DetectionManager:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.obstacles = []
        self.traffic_signs = []
        
    def detect(self, image, conf_threshold=0.4):
        """Run YOLO detection on image"""
        results = self.model(image, verbose=False, conf=conf_threshold)[0]
        return results
    
    def get_obstacles(self, results, image_height, min_area=1500):
        """Extract obstacle information from results"""
        obstacles = []
        h = image_height
        
        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.model.names[cls]
            
            if name in ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) // 2
                
                if box_area > min_area and y2 > h * 0.1:
                    obstacles.append({
                        'name': name,
                        'area': box_area,
                        'center_x': center_x,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
        return obstacles
    
    def detect_traffic_signs(self, results, image):
        """Detect and recognize traffic signs"""
        signs = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.model.names[cls]
            
            if name in ['stop sign', 'traffic light']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                
                signs.append({
                    'name': name,
                    'area': box_area,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                # Color based on sign type
                color = COLORS['emergency'] if name == 'stop sign' else COLORS['traffic_light']
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image, f"SIGN: {name.upper()}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        return signs
    
    def draw_detections(self, image, obstacles, ttc_values=None):
        """Draw bounding boxes on image"""
        for i, obs in enumerate(obstacles):
            # Determine color based on TTC
            if ttc_values and i < len(ttc_values):
                ttc = ttc_values[i]
                if ttc < 1.0:
                    color = COLORS['emergency']
                    thickness = 4
                elif ttc < 2.0:
                    color = COLORS['danger']
                    thickness = 3
                elif ttc < 3.5:
                    color = COLORS['warning']
                    thickness = 2
                else:
                    color = COLORS['safe']
                    thickness = 1
            else:
                color = COLORS['car']
                thickness = 2
            
            cv2.rectangle(image, (obs['x1'], obs['y1']), (obs['x2'], obs['y2']), color, thickness)
            
            text = f"{obs['name'].upper()}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (obs['x1'], obs['y1'] - h - 10), (obs['x1'] + w, obs['y1']), color, -1)
            cv2.putText(image, text, (obs['x1'], obs['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image