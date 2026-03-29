# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write transforms.json for LichtFeld Studio with COLMAP-derived poses.

Handles coordinate conversion from COLMAP (OpenCV, world-to-camera) to
LichtFeld's transforms.json format (OpenGL, camera-to-world with 180 deg Y
pre-compensation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def colmap_pose_to_c2w_opengl(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Convert COLMAP world-to-camera (R, t) to 4x4 camera-to-world in OpenGL convention.

    COLMAP stores poses as world-to-camera: p_cam = R @ p_world + t,
    in OpenCV convention (Y down, Z forward).

    LichtFeld's transforms.json expects camera-to-world matrices in OpenGL
    convention (Y up, Z back), with a 180 deg Y pre-compensation to cancel
    the rotation that the loader applies internally.

    Conversion steps:
        1. Invert w2c to c2w: R_c2w = R^T, t_c2w = -R^T @ t
        2. Build 4x4 matrix
        3. OpenCV -> OpenGL: flip Y and Z columns (c2w[:3, 1:3] *= -1)
        4. Pre-compensate for loader's 180 deg Y rotation: left-multiply by diag(-1, 1, -1, 1)

    Args:
        R_w2c: 3x3 rotation matrix (world-to-camera).
        t_w2c: 3-element translation vector (world-to-camera).

    Returns:
        4x4 camera-to-world matrix in OpenGL convention with Y pre-compensation.
    """
    # Step 1: Invert w2c to c2w
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    # Step 2: Build 4x4
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w

    # Step 3: OpenCV -> OpenGL: flip Y and Z columns
    c2w[:3, 1:3] *= -1

    # Step 4: 180 deg Y pre-compensation (cancels loader's internal rotation)
    y180 = np.diag([-1.0, 1.0, -1.0, 1.0])
    c2w = y180 @ c2w

    return c2w


def write_transforms_json(
    output_path: str | Path,
    camera_model: str,
    w: int,
    h: int,
    fl_x: float,
    fl_y: float,
    frames: list[dict],
    ply_file_path: str | None = None,
) -> None:
    """Write a transforms.json file in LichtFeld Studio format.

    Args:
        output_path: Destination file path.
        camera_model: Camera model string (e.g. "EQUIRECTANGULAR", "PINHOLE").
        w: Image width in pixels.
        h: Image height in pixels.
        fl_x: Focal length X. For ERP: w / 2.0.
        fl_y: Focal length Y. For ERP: h (full height).
        frames: List of frame dicts, each with "file_path" and "transform_matrix".
        ply_file_path: Optional relative path to a point cloud PLY file.
    """
    data = {
        "camera_model": camera_model,
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "frames": frames,
    }
    if ply_file_path is not None:
        data["ply_file_path"] = ply_file_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
