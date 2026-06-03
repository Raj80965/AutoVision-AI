"""
Vehicle Control Logic
"""

class VehicleController:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        
    def smooth_value(self, current, target, factor=0.35):
        """Smooth transition between values"""
        return current * (1 - factor) + target * factor
    
    def update_controls(self, throttle, steer, brake):
        """Update control values with smoothing"""
        self.throttle = self.smooth_value(self.throttle, throttle, 0.3)
        self.steer = self.smooth_value(self.steer, steer, 0.35)
        self.brake = self.smooth_value(self.brake, brake, 0.4)
        return self.throttle, self.steer, self.brake
    
    def decide_controls(self, obstacle, ttc, lane_center, lane_detected, 
                        target_speed, current_speed, rl_action=None):
        """Main decision logic"""
        
        throttle = 0.4
        brake = 0.0
        steer = self.steer
        driving_mode = "NORMAL"
        
        if obstacle and ttc < 1.0:
            throttle = 0.0
            brake = 1.0
            driving_mode = "EMERGENCY"
            if obstacle['center_x'] < 400:
                steer = 0.7
            else:
                steer = -0.7
                
        elif obstacle and ttc < 2.0:
            throttle = 0.0
            brake = 0.7
            driving_mode = "BRAKING"
            if obstacle['center_x'] < 450:
                steer = 0.5
            elif obstacle['center_x'] > 550:
                steer = -0.5
                
        elif obstacle and ttc < 3.5:
            throttle = 0.15
            brake = 0.4
            driving_mode = "AVOIDING"
            if obstacle['center_x'] < 400:
                steer = 0.45
            elif obstacle['center_x'] > 600:
                steer = -0.45
            else:
                steer = 0.4 if lane_center < 0.5 else -0.4
                
        else:
            if lane_detected:
                error = lane_center - 0.5
                steer = -error * 0.6
            else:
                steer = self.smooth_value(steer, 0, 0.08)
            
            if current_speed < target_speed:
                throttle = min(0.55, 0.25 + current_speed / 80)
            else:
                throttle = 0.2
                brake = 0.05
        
        return throttle, brake, steer, driving_mode

if __name__ == "__main__":
    print("="*50)
    print("Testing vehicle_control.py")
    print("="*50)
    
    controller = VehicleController()
    
    # Test cases
    test_cases = [
        ("No obstacle", None, 5.0, 0.5, True, 40, 30),
        ("Far obstacle", {'center_x': 500}, 4.0, 0.5, True, 40, 30),
        ("Medium obstacle", {'center_x': 500}, 2.5, 0.5, True, 40, 30),
        ("Close obstacle", {'center_x': 500}, 1.5, 0.5, True, 40, 30),
    ]
    
    for name, obs, ttc, lane, detected, target, current in test_cases:
        throttle, brake, steer, mode = controller.decide_controls(
            obs, ttc, lane, detected, target, current
        )
        print(f"{name}: Mode={mode}, T={throttle:.2f}, B={brake:.2f}, S={steer:.2f}")
    
    print("✅ vehicle_control.py working!")