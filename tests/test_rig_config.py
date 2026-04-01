# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the COLMAP rig configuration generator."""

from __future__ import annotations

import json
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
from core.colmap_runner import (
    ColmapConfig,
    MATCH_BUDGETS,
    _summarize_registration,
    infer_shared_pinhole_camera_params,
    resolve_match_budget,
)


def quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
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
    for yaw, pitch in [(30, 10), (90, -45), (0, 90), (270, 0), (180, -30)]:
        R_orig = create_rotation_matrix(yaw, pitch)
        q = rotation_matrix_to_quaternion(R_orig)
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R_orig, R_back, atol=1e-10)


def test_quaternion_canonical_form():
    """Quaternion w component should always be non-negative."""
    for yaw in range(0, 360, 30):
        for pitch in [-60, -30, 0, 30, 60]:
            R = create_rotation_matrix(yaw, pitch)
            q = rotation_matrix_to_quaternion(R)
            assert q[0] >= 0, f"w < 0 for yaw={yaw}, pitch={pitch}: {q}"


def test_infer_shared_pinhole_camera_params_for_cubemap():
    """Shared cubemap FOV should produce known PINHOLE camera params."""
    view_fovs = [fov for _yaw, _pitch, fov, _name, _flip in VIEW_PRESETS["cubemap"].get_all_views()]
    params, factor, fov_deg = infer_shared_pinhole_camera_params(view_fovs, 1920)

    assert fov_deg == pytest.approx(90.0)
    assert factor == pytest.approx(0.5)
    fx, fy, cx, cy = [float(x) for x in params.split(",")]
    assert fx == pytest.approx(960.0)
    assert fy == pytest.approx(960.0)
    assert cx == pytest.approx(960.0)
    assert cy == pytest.approx(960.0)


def test_infer_shared_pinhole_camera_params_returns_none_for_mixed_fov():
    """Mixed-FOV presets cannot use one shared PINHOLE parameter string."""
    params, factor, fov_deg = infer_shared_pinhole_camera_params([75.0, 90.0], 1920)
    assert params is None
    assert factor is None
    assert fov_deg is None


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


def test_cubemap_rotations_are_rig_to_camera_transforms():
    """Rig rotations should map reference-sensor coordinates into each camera."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    cams = {
        cam["image_prefix"].rstrip("/"): cam
        for cam in rig[0]["cameras"]
    }

    views = config.get_all_views()
    ref_yaw, ref_pitch, _ref_fov, _ref_name, _ref_flip = views[0]
    R_ref = create_rotation_matrix(ref_yaw, ref_pitch)

    for yaw, pitch, _fov, name, _flip in views[1:]:
        rig_pitch = -pitch if abs(pitch) == 90 else pitch
        R_expected = create_rotation_matrix(yaw, rig_pitch).T @ R_ref
        R_actual = quaternion_to_rotation_matrix(cams[name]["cam_from_rig_rotation"])
        np.testing.assert_allclose(R_actual, R_expected, atol=1e-10)


@pytest.mark.parametrize("preset_name", ["balanced", "standard", "dense", "full"])
def test_mixed_angle_preset_rotations_are_rig_to_camera_transforms(preset_name: str):
    """Non-cubemap presets should also produce rig-consistent mixed-angle cameras."""
    config = VIEW_PRESETS[preset_name]
    rig = generate_rig_config(config)
    cams = {
        cam["image_prefix"].rstrip("/"): cam
        for cam in rig[0]["cameras"]
    }

    views = config.get_all_views()
    ref_yaw, ref_pitch, _ref_fov, _ref_name, _ref_flip = views[0]
    R_ref = create_rotation_matrix(ref_yaw, ref_pitch)

    for yaw, pitch, _fov, name, _flip in views[1:]:
        rig_pitch = -pitch if abs(pitch) == 90 else pitch
        R_expected = create_rotation_matrix(yaw, rig_pitch).T @ R_ref
        R_actual = quaternion_to_rotation_matrix(cams[name]["cam_from_rig_rotation"])
        np.testing.assert_allclose(R_actual, R_expected, atol=1e-10)


def test_right_face_rotation_points_reference_forward_to_camera_left():
    """The +90 yaw cubemap face should see the reference forward direction on its left."""
    config = VIEW_PRESETS["cubemap"]
    rig = generate_rig_config(config)
    right_cam = next(
        cam for cam in rig[0]["cameras"]
        if cam["image_prefix"].rstrip("/") == "00_01"
    )

    R_right = quaternion_to_rotation_matrix(right_cam["cam_from_rig_rotation"])
    ref_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_view_of_ref_forward = R_right @ ref_forward

    np.testing.assert_allclose(
        right_view_of_ref_forward,
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        atol=1e-10,
    )


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
        assert prefix.endswith("/"), f"Bad prefix: {prefix}"
        assert "*" not in prefix, f"Prefix must be literal: {prefix}"


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


def test_registration_summary_reports_complete_partial_and_dropped_frames():
    """Registration summary should distinguish whole, partial, and missing rig frames."""
    expected = [
        "00_00/frame_001.jpg",
        "00_01/frame_001.jpg",
        "00_02/frame_001.jpg",
        "00_00/frame_002.jpg",
        "00_01/frame_002.jpg",
        "00_02/frame_002.jpg",
        "00_00/frame_003.jpg",
        "00_01/frame_003.jpg",
        "00_02/frame_003.jpg",
    ]
    registered = [
        "00_00/frame_001.jpg",
        "00_01/frame_001.jpg",
        "00_02/frame_001.jpg",
        "00_00/frame_002.jpg",
        "00_01/frame_002.jpg",
    ]

    summary = _summarize_registration(expected, registered)

    assert summary.expected_frames == 3
    assert summary.registered_frames == 2
    assert summary.complete_frames == 1
    assert summary.partial_frames == 1
    assert summary.views_per_frame == 3
    assert summary.expected_images_by_view == {
        "00_00": 3,
        "00_01": 3,
        "00_02": 3,
    }
    assert summary.registered_images_by_view == {
        "00_00": 2,
        "00_01": 2,
        "00_02": 1,
    }
    assert summary.partial_frame_examples == ["frame_002.jpg (2/3)"]
    assert summary.dropped_frame_examples == ["frame_003.jpg"]


def test_resolve_match_budget_uses_tiers_and_override():
    assert resolve_match_budget("fast") == MATCH_BUDGETS["fast"]
    assert resolve_match_budget("balanced") == MATCH_BUDGETS["balanced"]
    assert resolve_match_budget("default") == MATCH_BUDGETS["default"]
    assert resolve_match_budget("high") == MATCH_BUDGETS["high"]
    assert resolve_match_budget("custom", 24576) == 24576


def test_colmap_config_uses_match_budget_override():
    cfg = ColmapConfig(match_budget_tier="balanced", max_num_matches_override=24576)
    assert cfg.sift_max_num_matches == 24576


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


def test_write_rig_config(tmp_path: Path):
    """write_rig_config should produce valid JSON on disk."""
    config = VIEW_PRESETS["cubemap"]
    out_path = tmp_path / "rig.json"

    try:
        result = write_rig_config(config, str(out_path))
        assert Path(result).exists()

        data = json.loads(Path(result).read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "cameras" in data[0]
        assert len(data[0]["cameras"]) == config.total_views()
    finally:
        out_path.unlink(missing_ok=True)
