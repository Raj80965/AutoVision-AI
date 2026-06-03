"""
Autonomous Parking Module
Fixed:
  - Negative throttle replaced with reverse=True in VehicleControl
    (CARLA throttle must be 0.0–1.0; negative values are clamped to 0)
"""

import time
import carla
from enum import Enum


class ParkingState(Enum):
    IDLE = 0
    SEARCHING = 1
    POSITIONING = 2
    REVERSING_IN = 3
    REVERSING_STRAIGHT = 4
    STRAIGHTENING_OUT = 5
    ADJUSTING_FORWARD = 6
    PARKED = 7


class ParkingManager:
    def __init__(self):
        self.state = ParkingState.IDLE
        self.start_time = 0
        self.start_yaw = 0

    def start_parking(self, current_yaw):
        """Initiate the parking sequence"""
        self.state = ParkingState.POSITIONING
        self.start_time = time.time()
        self.start_yaw = current_yaw
        print("[P] Initiating Autonomous Parking Sequence")

    def update(self, current_speed, current_yaw, lidar_distance_right=None):
        """
        Updates the parking state machine.
        Returns: (throttle, steer, brake, reverse, is_parking_active)
        Fixed: added 'reverse' return value — use carla.VehicleControl(reverse=reverse)
        """
        if self.state in (ParkingState.IDLE, ParkingState.PARKED):
            return 0.0, 0.0, 1.0, False, False

        throttle = 0.0
        steer = 0.0
        brake = 0.0
        reverse = False

        elapsed = time.time() - self.start_time

        # ── POSITIONING ────────────────────────────────────────────────
        if self.state == ParkingState.POSITIONING:
            # Move forward slightly past the gap
            if elapsed < 2.0:
                throttle = 0.2
                steer = 0.0
            else:
                brake = 1.0
                if current_speed < 0.1:
                    self.state = ParkingState.REVERSING_IN
                    self.start_time = time.time()
                    self.start_yaw = current_yaw

        # ── REVERSING_IN ───────────────────────────────────────────────
        elif self.state == ParkingState.REVERSING_IN:
            yaw_diff = abs((current_yaw - self.start_yaw + 180) % 360 - 180)

            if yaw_diff < 40 and elapsed < 4.0:
                throttle = 0.15   # Fixed: positive value + reverse=True
                steer = 0.8
                reverse = True
            else:
                brake = 1.0
                if current_speed < 0.1:
                    self.state = ParkingState.REVERSING_STRAIGHT
                    self.start_time = time.time()

        # ── REVERSING_STRAIGHT ─────────────────────────────────────────
        elif self.state == ParkingState.REVERSING_STRAIGHT:
            if elapsed < 1.5:
                throttle = 0.15   # Fixed: positive + reverse=True
                steer = 0.0
                reverse = True
            else:
                brake = 1.0
                if current_speed < 0.1:
                    self.state = ParkingState.STRAIGHTENING_OUT
                    self.start_time = time.time()
                    self.start_yaw = current_yaw

        # ── STRAIGHTENING_OUT ──────────────────────────────────────────
        elif self.state == ParkingState.STRAIGHTENING_OUT:
            yaw_diff = abs((current_yaw - self.start_yaw + 180) % 360 - 180)

            if yaw_diff < 38 and elapsed < 4.0:
                throttle = 0.15   # Fixed: positive + reverse=True
                steer = -0.8
                reverse = True
            else:
                brake = 1.0
                if current_speed < 0.1:
                    self.state = ParkingState.ADJUSTING_FORWARD
                    self.start_time = time.time()

        # ── ADJUSTING_FORWARD ──────────────────────────────────────────
        elif self.state == ParkingState.ADJUSTING_FORWARD:
            if elapsed < 1.0:
                throttle = 0.15
                steer = 0.0
            else:
                self.state = ParkingState.PARKED
                brake = 1.0
                print("[OK] Parking Complete!")

        return throttle, steer, brake, reverse, True

    def get_state_name(self):
        return self.state.name
