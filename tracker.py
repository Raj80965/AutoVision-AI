"""
DeepSORT Object Tracking Module
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        # Initialize DeepSORT tracker
        # max_age: Maximum number of missed misses before a track is deleted
        # n_init: Number of consecutive detections before the track is confirmed
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
        
        # Store history to calculate velocity/prediction
        self.history = {}
        
    def update(self, obstacles, frame):
        """
        Update tracker with YOLO detections
        obstacles: list of dicts with 'x1','y1','x2','y2', 'conf', 'name'
        """
        # DeepSORT expects detections in format: [ [left,top,w,h], confidence, detection_class ]
        formatted_detections = []
        for obs in obstacles:
            x1, y1, x2, y2 = obs['x1'], obs['y1'], obs['x2'], obs['y2']
            w = x2 - x1
            h = y2 - y1
            conf = obs.get('conf', 0.5) # Default conf if not provided
            cls_name = obs['name']
            
            formatted_detections.append(([x1, y1, w, h], conf, cls_name))
            
        # Update tracks
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        
        # Format output and calculate predictions
        tracked_objects = []
        current_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            current_ids.append(track_id)
            ltrb = track.to_ltrb() # left, top, right, bottom
            
            x1, y1, x2, y2 = map(int, ltrb)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Update history
            if track_id not in self.history:
                self.history[track_id] = []
            
            self.history[track_id].append((center_x, center_y))
            if len(self.history[track_id]) > 10: # Keep last 10 frames
                self.history[track_id].pop(0)
                
            # Predict next position (simple linear velocity)
            pred_x, pred_y = center_x, center_y
            if len(self.history[track_id]) >= 3:
                # Calculate avg movement over last 3 frames
                dx = (self.history[track_id][-1][0] - self.history[track_id][-3][0]) / 2.0
                dy = (self.history[track_id][-1][1] - self.history[track_id][-3][1]) / 2.0
                # Predict position 10 frames ahead
                pred_x = int(center_x + dx * 10)
                pred_y = int(center_y + dy * 10)
            
            tracked_objects.append({
                'track_id': track_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center_x': center_x,
                'center_y': center_y,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'class_name': track.det_class if track.det_class else 'unknown'
            })
            
        # Cleanup history for lost tracks
        lost_ids = set(self.history.keys()) - set(current_ids)
        for lost_id in lost_ids:
            del self.history[lost_id]
            
        return tracked_objects
        
    def draw_tracks(self, img, tracked_objects):
        """Visualize tracking and predictions"""
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            track_id = obj['track_id']
            pred_x, pred_y = obj['pred_x'], obj['pred_y']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"ID: {track_id}", (x1, y1 - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                       
            # Draw prediction vector
            center_x, center_y = obj['center_x'], obj['center_y']
            if pred_x != center_x or pred_y != center_y:
                cv2.arrowedLine(img, (center_x, center_y), (pred_x, pred_y), 
                              (0, 255, 255), 2, tipLength=0.3)
                              
        return img
