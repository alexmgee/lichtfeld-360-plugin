# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the reframe engine (rotation matrix + reprojection)."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from core.presets import VIEW_PRESETS
from core.reframer import Reframer, _collect_image_files, create_rotation_matrix, reframe_view


def test_identity_rotation():
    R = create_rotation_matrix(0, 0)
    assert R.shape == (3, 3)
    # w2c matrix: forward is [0,0,1], so row 2 = -fwd = [0,0,-1]
    forward = -R[2]
    np.testing.assert_allclose(forward, [0, 0, 1], atol=1e-6)


def test_90_degree_yaw():
    R = create_rotation_matrix(90, 0)
    assert R.shape == (3, 3)
    # yaw=90: forward = [sin(90),0,cos(90)] = [1,0,0]
    forward = -R[2]
    assert abs(forward[0] - 1.0) < 0.01


def test_pitch_90_up():
    R = create_rotation_matrix(0, 90)
    forward = -R[2]
    # Looking up: forward should be [0,1,0]
    assert abs(forward[1] - 1.0) < 0.01


def test_mixed_yaw_pitch_matches_expected_forward_direction():
    """Combined yaw+pitch should keep the same upward tilt at every yaw."""
    for yaw, pitch in [(0, 25), (45, 25), (90, 25), (180, 25), (270, 25)]:
        R = create_rotation_matrix(yaw, pitch)
        forward = -R[2]
        expected = np.array(
            [
                np.sin(np.radians(yaw)) * np.cos(np.radians(pitch)),
                np.sin(np.radians(pitch)),
                np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
            ]
        )
        np.testing.assert_allclose(forward, expected, atol=1e-6)


def test_high_preset_mixed_angle_views_point_where_their_angles_say():
    """High preset views should honour both yaw and pitch simultaneously."""
    for yaw, pitch, _fov, _name, _flip in VIEW_PRESETS["high"].get_all_views():
        R = create_rotation_matrix(yaw, pitch)
        forward = -R[2]
        expected = np.array(
            [
                np.sin(np.radians(yaw)) * np.cos(np.radians(pitch)),
                np.sin(np.radians(pitch)),
                np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
            ]
        )
        np.testing.assert_allclose(forward, expected, atol=1e-6)


def test_reframe_view_output_shape():
    equirect = np.zeros((200, 400, 3), dtype=np.uint8)
    equirect[90:110, 190:210, :] = 255
    result = reframe_view(equirect, fov_deg=90, yaw_deg=0, pitch_deg=0, out_size=128)
    assert result.shape == (128, 128, 3)


def test_reframe_center_captures_content():
    equirect = np.zeros((200, 400, 3), dtype=np.uint8)
    equirect[95:105, 195:205, :] = 255
    result = reframe_view(equirect, fov_deg=90, yaw_deg=0, pitch_deg=0, out_size=128)
    center_brightness = result[60:68, 60:68, :].mean()
    assert center_brightness > 50, "Center should capture the white dot"


def test_reframe_nearest_mode():
    equirect = np.zeros((200, 400, 3), dtype=np.uint8)
    equirect[95:105, 195:205, :] = 255
    result = reframe_view(
        equirect, fov_deg=90, yaw_deg=0, pitch_deg=0, out_size=128, mode="nearest"
    )
    assert result.shape == (128, 128, 3)


def test_reframe_grayscale():
    equirect = np.zeros((200, 400), dtype=np.uint8)
    equirect[95:105, 195:205] = 255
    result = reframe_view(
        equirect, fov_deg=90, yaw_deg=0, pitch_deg=0, out_size=64, mode="nearest"
    )
    assert result.shape == (64, 64)


def test_reframe_single_writes_camera_first_layout(tmp_path):
    image_path = tmp_path / "station_001.jpg"
    equirect = np.zeros((200, 400, 3), dtype=np.uint8)
    equirect[95:105, 195:205, :] = 255
    cv2.imwrite(str(image_path), equirect)

    outputs, error = Reframer(VIEW_PRESETS["cubemap"]).reframe_single(
        str(image_path),
        str(tmp_path / "images"),
    )

    assert error is None
    assert len(outputs) == VIEW_PRESETS["cubemap"].total_views()

    expected_paths = [
        tmp_path / "images" / view_name / "station_001.jpg"
        for _, _, _, view_name, _ in VIEW_PRESETS["cubemap"].get_all_views()
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing output image: {path}"


def test_reframe_single_writes_masks_in_matching_view_folders(tmp_path):
    image_path = tmp_path / "station_001.jpg"
    mask_path = tmp_path / "station_001.png"

    equirect = np.zeros((200, 400, 3), dtype=np.uint8)
    equirect[95:105, 195:205, :] = 255
    cv2.imwrite(str(image_path), equirect)

    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[80:120, 180:220] = 255
    cv2.imwrite(str(mask_path), mask)

    outputs, error = Reframer(VIEW_PRESETS["cubemap"]).reframe_single(
        str(image_path),
        str(tmp_path / "images"),
        mask_path=str(mask_path),
    )

    assert error is None
    assert len(outputs) == VIEW_PRESETS["cubemap"].total_views()

    expected_mask_paths = [
        tmp_path / "masks" / view_name / "station_001.png"
        for _, _, _, view_name, _ in VIEW_PRESETS["cubemap"].get_all_views()
    ]
    for path in expected_mask_paths:
        assert path.exists(), f"Missing output mask: {path}"


def test_collect_image_files_deduplicates_case_insensitive_glob_results(tmp_path, monkeypatch):
    image_path = tmp_path / "frame_001.jpg"
    image_path.write_bytes(b"test")

    original_glob = Path.glob

    def fake_glob(self: Path, pattern: str) -> Iterator[Path]:
        if self == tmp_path and pattern in ("*.jpg", "*.JPG"):
            return iter([image_path])
        return iter([])

    monkeypatch.setattr(Path, "glob", fake_glob)

    files = _collect_image_files(tmp_path)

    assert files == [image_path]
