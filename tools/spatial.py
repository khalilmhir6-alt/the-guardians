import numpy as np
from typing import Dict, Any

class LunarSpatialProjector:
    def __init__(self, camera_height: float = 0.5, camera_pitch_deg: float = 15.0, hfov_deg: float = 60.0):
        self.h = camera_height
        self.pitch = np.radians(camera_pitch_deg)
        
        self.cx, self.cy = 640 / 2.0, 480 / 2.0 
        self.fx = self.cx / np.tan(np.radians(hfov_deg) / 2.0)
        self.fy = self.fx 
        
        self.R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ])

    def pixel_to_spatial(self, u: float, v: float) -> Dict[str, float]:
        """Projects 2D pixels to 3D lunar coordinates."""
        ray_camera = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        ray_rover = self.R_pitch @ ray_camera
        ray_x, ray_y, ray_z = ray_rover
        
        if ray_z >= 0: 
            return {"x": 0.0, "y": 0.0, "z": 0.0, "valid": False} 
            
        scale = -self.h / ray_z
        return {"x": round(ray_x * scale, 3), "y": round(ray_y * scale, 3), "z": 0.0, "valid": True}

    def assess_kinematic_threat(self, distance_y: float, lateral_x: float, rover_speed_ms: float = 1.2) -> Dict[str, Any]:
        """
        Calculates Time-To-Impact (TTI) and Autonomous Evasive Actions.
        rover_speed_ms: Assuming the rover moves at 1.2 meters per second.
        """
        # If the hazard is behind us or we are stopped
        if distance_y <= 0 or rover_speed_ms == 0:
            return {"tti": 0.0, "threat": "SAFE", "action": "NONE"}

        tti = distance_y / rover_speed_ms
        
        # Assume the rover chassis is 1.0 meter wide. 
        # If lateral_x is > 0.5 or < -0.5, the hazard will miss the wheels.
        in_direct_path = abs(lateral_x) <= 0.5 

        if not in_direct_path:
            return {"tti": round(tti, 1), "threat": "SAFE", "action": "MAINTAIN_HEADING"}
        
        if tti <= 2.5:
            return {"tti": round(tti, 1), "threat": "CRITICAL", "action": "EMERGENCY_BRAKE"}
        elif tti <= 5.0:
            return {"tti": round(tti, 1), "threat": "WARNING", "action": "INITIATE_EVASION"}
        else:
            return {"tti": round(tti, 1), "threat": "NOMINAL", "action": "MONITOR_APPROACH"}