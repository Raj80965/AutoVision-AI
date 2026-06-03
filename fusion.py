"""
Sensor Fusion Module - Combines Camera and LiDAR
Fixed:
  - Camera-LiDAR extrinsic offsets now read from config.py instead of
    being hardcoded, so changing CAMERA_POSITION in config automatically
    updates the projection.
"""

import numpy as np
import math
from config import CAMERA_IMAGE_SIZE_X, CAMERA_IMAGE_SIZE_Y, CAMERA_FOV
from config import CAMERA_POSITION, CAMERA_ROTATION


class SensorFusion:
    def __init__(self, w=CAMERA_IMAGE_SIZE_X, h=CAMERA_IMAGE_SIZE_Y, fov=CAMERA_FOV):
        self.w = w
        self.h = h
        self.fov = fov

        # Intrinsic matrix
        self.focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w / 2.0
        self.K[1, 2] = h / 2.0

        # Fixed: read camera extrinsic offsets from config (was hardcoded)
        # LiDAR is mounted at z=1.8; camera at CAMERA_POSITION z
        self._cam_x_offset = CAMERA_POSITION['x']          # e.g. 1.5
        self._cam_z_offset = CAMERA_POSITION['z'] - 1.8    # e.g. 1.7 - 1.8 = -0.1
        self._cam_pitch_rad = math.radians(CAMERA_ROTATION['pitch'])  # e.g. -15 deg

    # ──────────────────────────────────────────────────────────────────
    def project_lidar_to_camera(self, lidar_points):
        """
        Projects LiDAR points (N, 3) to Camera image plane.
        Returns: array of [u, v, depth] for points within image.
        """
        if len(lidar_points) == 0:
            return np.array([])

        points = np.copy(lidar_points).astype(np.float64)

        # 1. Translate LiDAR origin to Camera origin (Fixed: uses config values)
        points[:, 0] -= self._cam_x_offset   # x relative to camera
        points[:, 2] -= (-self._cam_z_offset) # z relative to camera

        # 2. Convert CARLA coords → Standard camera coords
        #    CARLA: x forward, y right, z up
        #    Std camera: z forward, x right, y down
        cam_z = points[:, 0]   # forward
        cam_x = points[:, 1]   # right
        cam_y = -points[:, 2]  # down (flip z-up to y-down)

        # 3. Apply camera pitch rotation (Fixed: uses config value)
        pitch = self._cam_pitch_rad
        cam_y_rot = cam_y * math.cos(pitch) - cam_z * math.sin(pitch)
        cam_z_rot = cam_y * math.sin(pitch) + cam_z * math.cos(pitch)
        cam_y = cam_y_rot
        cam_z = cam_z_rot

        # 4. Filter points behind camera
        valid = cam_z > 0
        cam_x, cam_y, cam_z = cam_x[valid], cam_y[valid], cam_z[valid]

        if len(cam_z) == 0:
            return np.array([])

        # 5. Project using intrinsic matrix
        u = (cam_x * self.focal) / cam_z + self.w / 2.0
        v = (cam_y * self.focal) / cam_z + self.h / 2.0

        # 6. Keep only points inside image bounds
        in_img = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        result = np.vstack((u[in_img], v[in_img], cam_z[in_img])).T
        return result

    # ──────────────────────────────────────────────────────────────────
    def get_distance_for_bbox(self, projected_points, bbox):
        """
        Returns the median depth of LiDAR points inside the bounding box.
        Returns -1 if no points found.
        """
        if len(projected_points) == 0:
            return -1

        x1, y1, x2, y2 = bbox
        u = projected_points[:, 0]
        v = projected_points[:, 1]

        inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        points_in_box = projected_points[inside]

        if len(points_in_box) > 0:
            return float(np.median(points_in_box[:, 2]))

        return -1
