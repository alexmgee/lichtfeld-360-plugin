# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Per-camera-family OPENCV_FISHEYE intrinsic priors for COLMAP.

In CameraMode.PER_FOLDER, COLMAP creates one camera per subfolder
(`images/front/` and `images/back/`) and initializes both from the
same `camera_params` string. Bundle adjustment then refines each camera
independently. So the values below are **family averages** — what BA
should converge to is documented in the round-2 spec §5 per-lens table.

Sources:
  - DJI Osmo 360: empirical Metashape calibration on unit 95SXN7S02213TB
    (see osmo360_rig_calibration_report.md). Family-average values
    rounded for prior use.
  - Insta360: not yet calibrated — returns None and the pipeline falls
    back to COLMAP's default_focal_length_factor.

Override path: a user-supplied JSON in cv2.fisheye format (K + D) gets
converted via colmap_params_from_cv2_fisheye().
"""

from __future__ import annotations

from typing import Optional, Sequence

# Family → (fx, fy, cx, cy, k1, k2, k3, k4) for OPENCV_FISHEYE.
FISHEYE_PRIORS: dict[str, Optional[tuple[float, float, float, float, float, float, float, float]]] = {
    "dji_osmo360": (1046.0, 1046.0, 1915.0, 1919.0, 0.0, 0.0, 0.0, 0.0),
    "insta360":    None,  # TBD — calibrate a representative unit
}


def infer_fisheye_camera_params(family: Optional[str]) -> Optional[str]:
    """Return the COLMAP `camera_params` string for a known family.

    Returns None if the family is unknown or has no calibrated prior.
    Caller falls back to default_focal_length_factor or a user-provided
    override JSON in that case.
    """
    if not family:
        return None
    prior = FISHEYE_PRIORS.get(family)
    if prior is None:
        return None
    return ",".join(f"{p}" for p in prior)


def colmap_params_from_cv2_fisheye(
    K: Sequence[Sequence[float]],
    D: Sequence[float],
) -> str:
    """Convert a cv2.fisheye-format calibration to COLMAP OPENCV_FISHEYE.

    Args:
        K: 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        D: 4-element distortion vector (k1, k2, k3, k4).

    Returns:
        Comma-separated string `"fx,fy,cx,cy,k1,k2,k3,k4"` ready for
        `ColmapConfig.camera_params`.
    """
    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    if len(D) != 4:
        raise ValueError(f"Expected 4-element distortion vector, got {len(D)}")
    k1, k2, k3, k4 = (float(d) for d in D)
    return f"{fx},{fy},{cx},{cy},{k1},{k2},{k3},{k4}"
