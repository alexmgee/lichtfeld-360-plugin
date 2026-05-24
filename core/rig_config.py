# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
COLMAP rig configuration generator.

Generates rig constraint JSON from the plugin's ring-based view geometry.
COLMAP rig constraints tell the mapper that multiple cameras share the same
optical center with fixed relative rotations — exactly the relationship
between the pinhole views extracted from a single equirectangular frame.

See https://colmap.github.io/rigs.html for the format specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .presets import ViewConfig
from .reframer import create_rotation_matrix

# Lazy import to avoid circular dependency — fisheye types only needed
# by generate_fisheye_pinhole_rig_config.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .fisheye_reframer import FisheyeViewConfig


def rotation_matrix_to_quaternion(R: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion.

    Uses Shepperd's method for numerical stability across all rotations.

    Args:
        R: A 3x3 orthogonal rotation matrix.

    Returns:
        Quaternion as [w, x, y, z] with w >= 0 (canonical form).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = [float(w), float(x), float(y), float(z)]

    # Canonical form: ensure w >= 0
    if q[0] < 0:
        q = [-c for c in q]

    # Normalize to unit quaternion
    norm = np.sqrt(sum(c * c for c in q))
    return [c / norm for c in q]


def generate_rig_config(view_config: ViewConfig) -> list[dict]:
    """Generate COLMAP rig config JSON from a ViewConfig.

    Image layout (camera-first, produced by the reframer):
        images/{view_name}/{station}.jpg

    COLMAP stores these as ``{view_name}/{station}.jpg`` in its database.
    Each camera is identified by a literal folder prefix such as
    ``"00_00/"`` and the remaining filename ``"{station}.jpg"`` becomes
    the frame key shared across all virtual cameras for the same panorama.

    Rotation convention:
        ``create_rotation_matrix()`` returns ``world_from_cam`` for each
        virtual view because the reframer rotates camera-space rays into
        world-space before sampling the ERP image. COLMAP's
        ``cam_from_rig_rotation`` expects the inverse relationship:
        a transform from rig coordinates into each camera's coordinates.

        With the reference sensor defining the rig coordinate system, the
        correct relative rotation for camera ``i`` is therefore::

            cam_i_from_rig = cam_i_from_world * world_from_rig
                           = R_i.T * R_ref

        Writing the forward rotation ``R_i * R_ref.T`` mirrors the rig and
        causes COLMAP to assemble geometrically inconsistent reconstructions.

    Args:
        view_config: The view configuration defining all perspective views.

    Returns:
        Rig config structure ready for JSON serialization.
    """
    views = view_config.get_all_views()

    if not views:
        return [{"cameras": []}]

    ref_yaw, ref_pitch, _ref_fov, _ref_name, _ref_flip = views[0]
    R_ref = create_rotation_matrix(ref_yaw, ref_pitch)

    cameras: List[dict] = []
    for i, (yaw, pitch, fov, name, _flip) in enumerate(views):
        R = create_rotation_matrix(yaw, pitch)

        cam_entry: dict = {
            "image_prefix": f"{name}/",
        }

        if i == 0:
            cam_entry["ref_sensor"] = True
        else:
            R_relative = R @ R_ref.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_relative)
            cam_entry["cam_from_rig_rotation"] = [qw, qx, qy, qz]
            cam_entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]

        cameras.append(cam_entry)

    return [{"cameras": cameras}]


def write_rig_config(view_config: ViewConfig, output_path: str) -> str:
    """Generate and write rig config to a JSON file.

    Args:
        view_config: The view configuration defining all perspective views.
        output_path: Path to write the JSON file.

    Returns:
        The absolute path to the written file.
    """
    rig = generate_rig_config(view_config)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig, indent=2))
    return str(out.resolve())


def write_dual_fisheye_rig_config(
    output_path: str,
    baseline_m: float = 0.025,
    rotation_quat: list[float] | None = None,
) -> str:
    """Write a two-sensor rig config for dual fisheye cameras.

    Front lens is the reference sensor (identity). Back lens is rotated
    180 deg around Y with a translational offset along -Z in the back
    camera's frame.

    Args:
        output_path: JSON file path to write.
        baseline_m: Inter-lens baseline in metres (default 25mm).
        rotation_quat: [qw, qx, qy, qz] for back sensor's
            cam_from_rig rotation. Default: 180 deg Y = [0, 0, 1, 0].

    Returns:
        Absolute path to the written file.
    """
    if rotation_quat is None:
        rotation_quat = [0, 0, 1, 0]

    # cam_from_rig_translation: rig origin (front sensor) expressed in
    # back camera's coordinate system. Back's +Z points opposite to
    # front's +Z, so rig origin is at -baseline along back's Z.
    translation = [0.0, 0.0, -baseline_m]

    # Spec §4.5 sign assertion
    assert translation[2] < 0, (
        f"cam_from_rig_translation Z must be negative, got {translation[2]}"
    )

    rig = [{
        "cameras": [
            {"image_prefix": "front/", "ref_sensor": True},
            {
                "image_prefix": "back/",
                "cam_from_rig_rotation": rotation_quat,
                "cam_from_rig_translation": translation,
            },
        ]
    }]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig, indent=2))
    return str(out.resolve())


def write_fisheye_pinhole_rig_config(
    view_config: "FisheyeViewConfig",
    output_path: str,
    baseline_m: float = 0.026,
) -> str:
    """Generate and write rig config for fisheye → pinhole reframed views.

    Each virtual pinhole camera's world-space rotation combines:
      1. View-in-lens rotation: _rotation_matrix(yaw, pitch)
      2. Lens-in-world rotation: front = identity, back = 180° Y

    Reference sensor: first view in the config (front center).
    Back-lens views get a translational offset for the baseline.

    Args:
        view_config: FisheyeViewConfig with all virtual views.
        output_path: JSON file path to write.
        baseline_m: Inter-lens baseline in metres.

    Returns:
        Absolute path to the written file.
    """
    from .fisheye_reframer import _rotation_matrix

    # 180° Y rotation for back lens
    R_back = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ], dtype=np.float64)

    views = view_config.views
    if not views:
        rig = [{"cameras": []}]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rig, indent=2))
        return str(out.resolve())

    # Compute world-space rotation for each view
    def _world_rotation(view) -> np.ndarray:
        R_view = _rotation_matrix(view.yaw_deg, view.pitch_deg)
        if view.source_lens == "back":
            return R_view @ R_back
        return R_view

    R_ref = _world_rotation(views[0])

    cameras: List[dict] = []
    for i, view in enumerate(views):
        cam_entry: dict = {"image_prefix": f"{view.name}/"}

        if i == 0:
            cam_entry["ref_sensor"] = True
        else:
            R = _world_rotation(view)
            R_relative = R @ R_ref.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_relative)
            cam_entry["cam_from_rig_rotation"] = [qw, qx, qy, qz]

            if view.source_lens == "back":
                cam_entry["cam_from_rig_translation"] = [0.0, 0.0, -baseline_m]
            else:
                cam_entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]

        cameras.append(cam_entry)

    rig = [{"cameras": cameras}]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rig, indent=2))
    return str(out.resolve())
