# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the reframe engine (rotation matrix + reprojection)."""

import numpy as np

from core.reframer import create_rotation_matrix, reframe_view


def test_identity_rotation():
    R = create_rotation_matrix(0, 0)
    assert R.shape == (3, 3)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_90_degree_yaw():
    R = create_rotation_matrix(90, 0)
    assert R.shape == (3, 3)
    # Ry(90) rotates z-forward into +x: [sin(90), 0, cos(90)] = [1, 0, 0]
    forward = R @ np.array([0, 0, 1])
    assert abs(forward[0] - 1.0) < 0.01


def test_pitch_90_up():
    R = create_rotation_matrix(0, 90)
    forward = R @ np.array([0, 0, 1])
    # Looking up: z-forward should become y-up
    assert abs(forward[1] - (-1.0)) < 0.01


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
