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

    Returns a list with one rig dict containing all cameras.
    The first view is the reference sensor (identity rotation).
    All other views have quaternion rotations relative to the reference
    and zero translation (views share the same optical center).

    Args:
        view_config: The view configuration defining all perspective views.

    Returns:
        Rig config structure ready for JSON serialization.
    """
    views = view_config.get_all_views()

    if not views:
        return [{"cameras": []}]

    # Reference view rotation
    ref_yaw, ref_pitch, _ref_fov, _ref_name = views[0]
    R_ref = create_rotation_matrix(ref_yaw, ref_pitch)

    cameras: List[dict] = []
    for i, (yaw, pitch, fov, name) in enumerate(views):
        if i == 0:
            cameras.append({
                "image_prefix": f"frame_*_view_{name}",
                "ref_sensor": True,
            })
        else:
            R = create_rotation_matrix(yaw, pitch)
            R_relative = R @ R_ref.T
            q = rotation_matrix_to_quaternion(R_relative)

            cameras.append({
                "image_prefix": f"frame_*_view_{name}",
                "cam_from_rig_rotation": q,
                "cam_from_rig_translation": [0, 0, 0],
            })

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
