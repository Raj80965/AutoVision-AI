"""
CARLA Connection and Vehicle Management
"""

import random
import math
import carla
from config import *

class CarlaManager:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.original_settings = None
        
    def connect(self):
        """Connect to CARLA server"""
        print("🔌 Connecting to CARLA...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        return True
    
    def setup_sync_mode(self):
        """Setup synchronous mode"""
        settings = self.world.get_settings()
        settings.synchronous_mode = SYNC_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        
    def spawn_vehicle(self):
        """Spawn vehicle at random spawn point"""
        print("🚗 Spawning vehicle...")
        blueprint_library = self.world.get_blueprint_library()
        
        # Try to get specified vehicle
        vehicle_bp = blueprint_library.filter(VEHICLE_MODEL)
        if not vehicle_bp:
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp[0], spawn_point)
        
        if self.vehicle is None:
            print("❌ Failed to spawn vehicle!")
            return False
        
        print(f"✅ Vehicle spawned at: {spawn_point.location}")
        return True
    
    def get_vehicle_speed(self):
        """Get current vehicle speed in km/h"""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return 0
    
    def apply_control(self, throttle, steer, brake):
        """Apply control to vehicle"""
        if self.vehicle:
            control = carla.VehicleControl()
            control.throttle = float(max(0, min(throttle, 0.6)))
            control.steer = float(max(-0.7, min(steer, 0.7)))
            control.brake = float(max(0, min(brake, 1.0)))
            self.vehicle.apply_control(control)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.vehicle:
            self.vehicle.destroy()
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        print("✅ CARLA cleanup complete")