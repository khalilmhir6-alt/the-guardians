"""Spatial math: pixel -> ground-plane position (the defended geometry).

Pinhole camera model, flat-ground assumption (z = 0).

World frame: x/y horizontal, z up. Origin (0,0) = landing site.
Camera pose: x, y, z (meters), yaw_deg (0 = looking along +y),
pitch_deg (positive = tilted down toward the ground).

Camera frame (OpenCV convention): +x right in the image, +y down in the
image, +z along the optical axis.

Ray from pixel (u, v):
    d_cam = ((u - cx) / fx, (v - cy) / fy, 1)
Rotate into the world, intersect with z = 0. Rays at or above the horizon
(direction z-component >= 0) are rejected.
"""
import math

# Tolerance for "at or above horizon" rejection (radians).
HORIZON_EPS = 1e-9


def intrinsics(cam: dict):
    """Pinhole intrinsics from camera.json-style parameters.

    Returns (fx, fy, cx, cy). Square pixels: fy = fx.
    """
    fx = cam["focal_mm"] / cam["sensor_width_mm"] * cam["width_px"]
    fy = fx
    cx = cam["width_px"] / 2.0
    cy = cam["height_px"] / 2.0
    return fx, fy, cx, cy


def _rotation_columns(yaw_deg: float, pitch_deg: float):
    """Unit vectors of camera x (image right), y (image down), z (optical
    axis) expressed in the world frame."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    sy, cy_ = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    x_axis = (cy_, -sy, 0.0)
    z_axis = (sy * cp, cy_ * cp, -sp)
    y_axis = (-sp * sy, -sp * cy_, -cp)
    return x_axis, y_axis, z_axis


def project_point(point, pose: dict, cam: dict):
    """World ground point -> pixel (u, v), or None if behind the camera.

    pose: {"x","y","z","yaw_deg","pitch_deg"}; point: (x, y, z).
    """
    fx, fy, cx, cy = intrinsics(cam)
    xa, ya, za = _rotation_columns(pose["yaw_deg"], pose["pitch_deg"])
    dx = point[0] - pose["x"]
    dy = point[1] - pose["y"]
    dz = point[2] - pose["z"]
    # Camera coordinates = R^T * (point - cam_pos); R columns are xa, ya, za.
    xc = dx * xa[0] + dy * xa[1] + dz * xa[2]
    yc = dx * ya[0] + dy * ya[1] + dz * ya[2]
    zc = dx * za[0] + dy * za[1] + dz * za[2]
    if zc <= 1e-6:
        return None
    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    return u, v


def pixel_to_ground(u: float, v: float, pose: dict, cam: dict):
    """Pixel of a ground-contact point -> world (x, y) on z = 0, or None
    if the ray points at or above the horizon.
    """
    fx, fy, cx, cy = intrinsics(cam)
    dxc = (u - cx) / fx
    dyc = (v - cy) / fy
    dzc = 1.0
    xa, ya, za = _rotation_columns(pose["yaw_deg"], pose["pitch_deg"])
    dxw = dxc * xa[0] + dyc * ya[0] + dzc * za[0]
    dyw = dxc * xa[1] + dyc * ya[1] + dzc * za[1]
    dzw = dxc * xa[2] + dyc * ya[2] + dzc * za[2]
    if dzw >= -HORIZON_EPS:
        return None  # at or above the horizon: flat-ground model undefined
    t = -pose["z"] / dzw
    if t <= 0:
        return None
    return pose["x"] + t * dxw, pose["y"] + t * dyw


def localize_detection(box, pose: dict, cam: dict):
    """Bottom-center of a pixel box -> world (x, y). None if rejected."""
    u = (box[0] + box[2]) / 2.0
    v = box[3]
    return pixel_to_ground(u, v, pose, cam)


def heading_to_steps(heading_deg: float) -> int:
    """Heading in degrees -> 4-bit field (22.5-degree steps, 0..15)."""
    return int(round(heading_deg / 22.5)) % 16


def steps_to_heading(steps: int) -> float:
    """4-bit field -> heading in degrees (0..337.5)."""
    return (steps % 16) * 22.5
