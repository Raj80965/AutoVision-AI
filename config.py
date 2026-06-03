"""
Configuration file for Autonomous Driving System
"""

# CARLA Settings
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0

# Vehicle Settings
VEHICLE_MODEL = 'vehicle.tesla.model3'
TARGET_SPEED = 40  # Further reduced as requested
MAX_SPEED = 70
MIN_SPEED = 15

# Camera Settings - Third Person View (full car visible)
CAMERA_IMAGE_SIZE_X = 1280
CAMERA_IMAGE_SIZE_Y = 720
CAMERA_FOV = 100
CAMERA_POSITION = {'x': -8.0, 'y': 0, 'z': 4.0}   # Behind + above car
CAMERA_ROTATION = {'pitch': -20, 'yaw': 0, 'roll': 0}  # Looking slightly down

# LiDAR Settings
LIDAR_CHANNELS = 32
LIDAR_POINTS_PER_SECOND = 50000
LIDAR_ROTATION_FREQUENCY = 20
LIDAR_RANGE = 80  # Fixed: explicit range in metres (default was only 50m)

# YOLO Settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.4

# Driving Settings
SYNC_MODE = True
FIXED_DELTA_SECONDS = 0.03

# Safety Settings
SAFE_DISTANCE_AREA = 1500
EMERGENCY_BRAKE_AREA = 25000
TTC_EMERGENCY = 1.0
TTC_HARD_BRAKE = 2.0
TTC_SOFT_BRAKE = 3.5

# Colors (BGR format) - Cyberpunk Theme
COLORS = {
    'safe': (255, 255, 0),        # Cyan
    'warning': (0, 165, 255),     # Orange
    'danger': (0, 0, 255),        # Red
    'emergency': (255, 0, 255),   # Magenta
    'info': (255, 255, 255),      # White
    'car': (255, 255, 0),         # Cyan
    'person': (255, 0, 255),      # Magenta
    'traffic_light': (0, 255, 255)# Yellow
}

print("[OK] config.py loaded")