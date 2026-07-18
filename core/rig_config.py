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
        ``create_rotation_matrix()`` returns ``cam_from_world`` for each
        virtual view — rows ``[right, up, -forward]``. (Ground truth:
        ``reframe_view`` samples along ``R.T @ ray``, rotating a
        camera-space ray into world space, so ``R`` is world-to-camera.)
        COLMAP's ``cam_from_rig_rotation`` is the transform from rig
        coordinates into each camera's coordinates.

        With the reference sensor (view ``[0]``) defining the rig frame,
        ``world_from_rig = R_ref.T``, so the relative rotation for camera
        ``i`` is::

            cam_i_from_rig = cam_i_from_world @ world_from_rig
                           = R_i @ R_ref.T

        implemented below as ``R @ R_ref.T`` — a proper rotation (det +1).
        The cancellation assumes every view shares the same output-axis
        convention: the reframer's universal ``fliplr`` cancels in the
        relative rotation. Per-view ``flip_vertical`` is deliberately not
        honored here (the ``_flip`` field is ignored) — such a view is an
        improper reflection with no quaternion representation and must not
        be used with rig alignment. No built-in preset sets ``flip_vertical``.

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
