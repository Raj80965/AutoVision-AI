"""
Sensor Management - Camera, LiDAR, Collision
Fixed:
  - Collision flag now auto-resets after 2 seconds (was never resetting)
  - LiDAR data is now thread-safe using threading.Lock
"""

import numpy as np
import threading
import time
import carla
from config import *

class SensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.camera = None
        self.lidar = None
        self.collision_sensor = None
        
        # Thread-safe LiDAR data
        self._lidar_lock = threading.Lock()
        self._lidar_data = []
        
        # Collision — grace period ignores spawn collisions
        self.collision_occurred = False
        self.collision_time = 0
        self._spawn_time = time.time()       # ignore collisions in first 3s
        self._COLLISION_GRACE  = 3.0         # seconds after spawn
        self._COLLISION_RESET_DELAY = 2.0    # Reduced for faster recovery
        self._last_collision_print = 0       # throttle prints

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def setup_camera(self):
        """Setup RGB Camera"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(CAMERA_IMAGE_SIZE_X))
        camera_bp.set_attribute("image_size_y", str(CAMERA_IMAGE_SIZE_Y))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))

        camera_transform = carla.Transform(
            carla.Location(x=CAMERA_POSITION['x'],
                           y=CAMERA_POSITION['y'],
                           z=CAMERA_POSITION['z']),
            carla.Rotation(pitch=CAMERA_ROTATION['pitch'],
                           yaw=CAMERA_ROTATION['yaw'],
                           roll=CAMERA_ROTATION['roll'])
        )

        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print("[OK] Camera attached")
        return self.camera

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    def setup_lidar(self):
        """Setup LiDAR Sensor (thread-safe data storage)"""
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))  # Fixed: explicit range from config

        lidar_transform = carla.Transform(carla.Location(x=0, z=1.8))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

        def on_lidar_data(lidar_measurement):
            points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            new_data = points[:, :3]  # Only keep x, y, z
            # Fixed: use lock to prevent race condition with main thread
            with self._lidar_lock:
                self._lidar_data = new_data

        self.lidar.listen(on_lidar_data)
        print("[OK] LiDAR attached")
        return self.lidar

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def setup_collision_sensor(self):
        """Setup Collision Sensor"""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle)

        def on_collision(event):
            now = time.time()
            # Ignore collisions right after spawn (ground/physics settle)
            if now - self._spawn_time < self._COLLISION_GRACE:
                return
            self.collision_occurred = True
            self.collision_time = now
            # Throttle print — once every 3 seconds max
            if now - self._last_collision_print > 3.0:
                print("[!!] COLLISION DETECTED!")
                self._last_collision_print = now

        self.collision_sensor.listen(on_collision)
        print("[OK] Collision sensor attached")
        return self.collision_sensor

    def reset_collision_if_old(self):
        """Auto-reset collision flag after delay. Call every frame."""
        if self.collision_occurred and (time.time() - self.collision_time > self._COLLISION_RESET_DELAY):
            self.collision_occurred = False

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------
    def get_lidar_points(self):
        """Get LiDAR point cloud data (thread-safe copy)"""
        with self._lidar_lock:
            return np.copy(self._lidar_data) if len(self._lidar_data) > 0 else np.array([])

    @property
    def lidar_data(self):
        """Backward-compatible property for legacy code"""
        return self.get_lidar_points()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Cleanup sensors"""
        for sensor in [self.camera, self.lidar, self.collision_sensor]:
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        print("[OK] Sensors cleanup complete")