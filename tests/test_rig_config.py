# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the COLMAP rig configuration generator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.presets import VIEW_PRESETS, ViewConfig, Ring
from core.reframer import create_rotation_matrix
from core.rig_config import (
    generate_rig_config,
    rotation_matrix_to_quaternion,
    write_rig_config,
)


# ---------------------------------------------------------------------------
# Quaternion conversion tests
# ---------------------------------------------------------------------------


def test_quaternion_identity():
    """Identity rotation matrix should produce identity quaternion."""
    R = np.eye(3)
    q = rotation_matrix_to_quaternion(R)
    assert len(q) == 4
    assert abs(q[0] - 1.0) < 1e-6  # w
    assert abs(q[1]) < 1e-6  # x
    assert abs(q[2]) < 1e-6  # y
    assert abs(q[3]) < 1e-6  # z


def test_quaternion_90_degree_yaw():
    """90-degree yaw rotation should produce a valid unit quaternion."""
    R = create_rotation_matrix(90, 0)
    q = rotation_matrix_to_quaternion(R)
    norm = sum(x**2 for x in q) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_quaternion_180_degree_yaw():
    """180-degree yaw — a degenerate case for some algorithms."""
    R = create_rotation_matrix(180, 0)
    q = rotation_matrix_to_quaternion(R)
    norm = sum(x**2 for x in q) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_quaternion_pitch_up():
    """Pure pitch rotation should produce a valid unit quaternion."""
    R = create_rotation_matrix(0, 45)
    q = rotation_matrix_to_quaternion(R)
    norm = sum(x**2 for x in q) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_quaternion_roundtrip():
    """Converting R -> q -> R should recover the original matrix."""
    from scipy.spatial.transform import Rotation as ScipyR

    for yaw, pitch in [(30, 10), (90, -45), (0, 90), (270, 0), (180, -30)]:
        R_orig = create_rotation_matrix(yaw, pitch)
        q = rotation_matrix_to_quaternion(R_orig)
        # scipy uses [x, y, z, w] order
        R_back = ScipyR.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        np.testing.assert_allclose(R_orig, R_back, atol=1e-10)


def test_quaternion_canonical_form():
    """Quaternion w component should always be non-negative."""
    for yaw in range(0, 360, 30):
        for pitch in [-60, -30, 0, 30, 60]:
            R = create_rotation_matrix(yaw, pitch)
            q = rotation_matrix_to_quaternion(R)
            assert q[0] >= 0, f"w < 0 for yaw={yaw}, pitch={pitch}: {q}"


# ---------------------------------------------------------------------------
# Rig config structure tests
# ---------------------------------------------------------------------------


def test_rig_config_structure():
    """Rig config should be a list with one rig containing all cameras."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    assert isinstance(rig, list)
    assert len(rig) == 1
    cameras = rig[0]["cameras"]
    assert len(cameras) == config.total_views()


def test_reference_sensor():
    """First camera should be the reference sensor with no rotation."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    first_cam = rig[0]["cameras"][0]
    assert first_cam.get("ref_sensor") is True
    assert "cam_from_rig_rotation" not in first_cam
    assert "cam_from_rig_translation" not in first_cam


def test_non_ref_has_rotation():
    """Non-reference cameras should have rotation and translation."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    second_cam = rig[0]["cameras"][1]
    assert "cam_from_rig_rotation" in second_cam
    assert len(second_cam["cam_from_rig_rotation"]) == 4
    assert "cam_from_rig_translation" in second_cam


def test_all_translations_zero():
    """All translations should be [0, 0, 0] (shared optical center)."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    for cam in rig[0]["cameras"]:
        if "cam_from_rig_translation" in cam:
            assert cam["cam_from_rig_translation"] == [0, 0, 0]


def test_all_quaternions_unit():
    """All quaternions should be unit length across all presets."""
    for name, config in VIEW_PRESETS.items():
        rig = generate_rig_config(config)
        for cam in rig[0]["cameras"]:
            if "cam_from_rig_rotation" in cam:
                q = cam["cam_from_rig_rotation"]
                norm = sum(x**2 for x in q) ** 0.5
                assert abs(norm - 1.0) < 1e-6, (
                    f"Non-unit quaternion in preset '{name}': {q}"
                )


def test_image_prefix_format():
    """Image prefixes should match the expected naming convention."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    for cam in rig[0]["cameras"]:
        prefix = cam["image_prefix"]
        assert prefix.startswith("frame_*_view_"), f"Bad prefix: {prefix}"


def test_full_sphere_preset():
    """Full sphere preset should include zenith and nadir cameras."""
    config = VIEW_PRESETS["full"]
    rig = generate_rig_config(config)
    prefixes = [cam["image_prefix"] for cam in rig[0]["cameras"]]
    zenith_found = any("ZN_00" in p for p in prefixes)
    nadir_found = any("ND_00" in p for p in prefixes)
    assert zenith_found, "Zenith camera missing"
    assert nadir_found, "Nadir camera missing"


def test_empty_config():
    """Empty config should produce a rig with no cameras."""
    config = ViewConfig(rings=[], include_zenith=False, include_nadir=False)
    rig = generate_rig_config(config)
    assert rig == [{"cameras": []}]


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


def test_write_rig_config():
    """write_rig_config should produce valid JSON on disk."""
    config = VIEW_PRESETS["cubemap"]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "rig.json")
        result = write_rig_config(config, out_path)
        assert Path(result).exists()

        data = json.loads(Path(result).read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "cameras" in data[0]
        assert len(data[0]["cameras"]) == config.total_views()
