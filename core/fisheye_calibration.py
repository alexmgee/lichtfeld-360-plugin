# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Fisheye camera calibration dataclasses and default Osmo 360 calibration.

Ported from reconstruction-zone (prep360/core/fisheye_calibration.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class FisheyeCalibration:
    """Calibration for a single fisheye lens."""

    camera_matrix: np.ndarray       # 3x3 K matrix
    dist_coeffs: np.ndarray         # 4x1 (k1, k2, k3, k4)
    image_size: Tuple[int, int]     # (width, height)
    rms_error: float                # reprojection error (-1 = prior, not measured)
    num_images_used: int
    fov_degrees: float              # estimated diagonal FOV

    def save(self, path: str):
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "image_size": list(self.image_size),
            "rms_error": self.rms_error,
            "num_images_used": self.num_images_used,
            "fov_degrees": self.fov_degrees,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> FisheyeCalibration:
        data = json.loads(Path(path).read_text())
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(4, 1),
            image_size=tuple(data["image_size"]),
            rms_error=data["rms_error"],
            num_images_used=data["num_images_used"],
            fov_degrees=data["fov_degrees"],
        )


@dataclass
class DualFisheyeCalibration:
    """Calibration for a dual-fisheye camera (e.g., DJI Osmo 360).

    front/back are independent optical systems. back_rotation_deg=180
    means the back lens points opposite to the front.

    Rig geometry:
        baseline_m:    distance between optical centers in metres.
        baseline_axis: unit vector from front to back center in front
                       camera's coordinate frame.
    """

    front: FisheyeCalibration
    back: FisheyeCalibration
    front_rotation_deg: float = 0.0
    back_rotation_deg: float = 180.0
    camera_model: str = "DJI Osmo 360"
    baseline_m: float = 0.0
    baseline_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def save(self, path: str):
        data = {
            "camera_model": self.camera_model,
            "front_rotation_deg": self.front_rotation_deg,
            "back_rotation_deg": self.back_rotation_deg,
            "baseline_m": self.baseline_m,
            "baseline_axis": list(self.baseline_axis),
            "front": {
                "camera_matrix": self.front.camera_matrix.tolist(),
                "dist_coeffs": self.front.dist_coeffs.flatten().tolist(),
                "image_size": list(self.front.image_size),
                "rms_error": self.front.rms_error,
                "num_images_used": self.front.num_images_used,
                "fov_degrees": self.front.fov_degrees,
            },
            "back": {
                "camera_matrix": self.back.camera_matrix.tolist(),
                "dist_coeffs": self.back.dist_coeffs.flatten().tolist(),
                "image_size": list(self.back.image_size),
                "rms_error": self.back.rms_error,
                "num_images_used": self.back.num_images_used,
                "fov_degrees": self.back.fov_degrees,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> DualFisheyeCalibration:
        data = json.loads(Path(path).read_text())
        return cls(
            front=FisheyeCalibration(
                camera_matrix=np.array(data["front"]["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["front"]["dist_coeffs"], dtype=np.float64).reshape(4, 1),
                image_size=tuple(data["front"]["image_size"]),
                rms_error=data["front"]["rms_error"],
                num_images_used=data["front"]["num_images_used"],
                fov_degrees=data["front"]["fov_degrees"],
            ),
            back=FisheyeCalibration(
                camera_matrix=np.array(data["back"]["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["back"]["dist_coeffs"], dtype=np.float64).reshape(4, 1),
                image_size=tuple(data["back"]["image_size"]),
                rms_error=data["back"]["rms_error"],
                num_images_used=data["back"]["num_images_used"],
                fov_degrees=data["back"]["fov_degrees"],
            ),
            front_rotation_deg=data.get("front_rotation_deg", 0.0),
            back_rotation_deg=data.get("back_rotation_deg", 180.0),
            camera_model=data.get("camera_model", "Unknown"),
            baseline_m=data.get("baseline_m", 0.0),
            baseline_axis=tuple(data.get("baseline_axis", (0.0, 0.0, 1.0))),
        )


def default_osmo360_calibration() -> DualFisheyeCalibration:
    """Empirical calibration for DJI Osmo 360 (3840x3840 per lens).

    Per-lens intrinsics from Metashape 2.3 alignment (248 pairs, equisolid
    fisheye, two independent sensors, scale bars verified to 0.0%).
    Rig geometry from the same session + COLMAP cross-validation.

    The rms_error=-1.0 indicates these are priors, not from a ChArUco
    calibration of this specific unit.
    """
    # Front lens: f=1047.9, cx offset=-2.4, cy offset=-0.1
    K_front = np.array([
        [1047.898, 0, 1920.0 - 2.403],
        [0, 1047.898, 1920.0 - 0.124],
        [0, 0, 1],
    ], dtype=np.float64)

    # Back lens: f=1044.9, cx offset=-8.3, cy offset=-2.1
    K_back = np.array([
        [1044.882, 0, 1920.0 - 8.334],
        [0, 1044.882, 1920.0 - 2.097],
        [0, 0, 1],
    ], dtype=np.float64)

    # Distortion: from Metashape equisolid model (reasonable priors for
    # cv2.fisheye equidistant; BA refines in COLMAP)
    D_front = np.array([[0.0559], [0.0114], [-0.0095], [0.0005]], dtype=np.float64)
    D_back = np.array([[0.0572], [0.0076], [-0.0072], [0.0001]], dtype=np.float64)

    front = FisheyeCalibration(
        camera_matrix=K_front, dist_coeffs=D_front,
        image_size=(3840, 3840), rms_error=-1.0,
        num_images_used=0, fov_degrees=190.0,
    )
    back = FisheyeCalibration(
        camera_matrix=K_back, dist_coeffs=D_back,
        image_size=(3840, 3840), rms_error=-1.0,
        num_images_used=0, fov_degrees=190.0,
    )

    return DualFisheyeCalibration(
        front=front, back=back,
        front_rotation_deg=0.0, back_rotation_deg=180.0,
        camera_model="DJI Osmo 360 (empirical 2026-04-21)",
        baseline_m=0.026,
        baseline_axis=(0.0, 0.0, 1.0),
    )
