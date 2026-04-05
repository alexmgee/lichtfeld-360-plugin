# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for transforms_writer module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.transforms_writer import colmap_pose_to_c2w_opengl, write_transforms_json


class TestColmapPoseToCw:
    """Tests for the COLMAP-to-OpenGL coordinate conversion."""

    def test_identity_pose(self):
        """Identity COLMAP pose should produce valid 4x4 matrix."""
        R = np.eye(3)
        t = np.zeros(3)
        c2w = colmap_pose_to_c2w_opengl(R, t)
        assert c2w.shape == (4, 4)
        assert c2w[3, 3] == 1.0
        assert c2w[3, 0] == 0.0

    def test_pose_is_orthogonal(self):
        """Output rotation part should be orthogonal."""
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        c2w = colmap_pose_to_c2w_opengl(R, t)
        R_out = c2w[:3, :3]
        product = R_out @ R_out.T
        assert np.allclose(product, np.eye(3), atol=1e-6)

    def test_random_rotation_is_orthogonal(self):
        """Random valid rotation should produce orthogonal output."""
        # Build a random rotation via QR decomposition
        rng = np.random.default_rng(42)
        M = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(M)
        # Ensure proper rotation (det = +1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.standard_normal(3)

        c2w = colmap_pose_to_c2w_opengl(Q, t)
        R_out = c2w[:3, :3]
        product = R_out @ R_out.T
        assert np.allclose(product, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R_out), 1.0, atol=1e-6)

    def test_bottom_row_is_0001(self):
        """Bottom row of output should always be [0, 0, 0, 1]."""
        R = np.eye(3)
        t = np.array([5.0, -3.0, 1.0])
        c2w = colmap_pose_to_c2w_opengl(R, t)
        np.testing.assert_array_equal(c2w[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_identity_pose_values(self):
        """Identity R, zero t should produce the y180 pre-compensation matrix.

        Steps: c2w = I, flip Y/Z cols -> diag(1,-1,-1), then y180 @ that =
        diag(-1,-1,1) for rotation, zero translation.
        """
        R = np.eye(3)
        t = np.zeros(3)
        c2w = colmap_pose_to_c2w_opengl(R, t)
        expected = np.diag([-1.0, -1.0, 1.0, 1.0])
        np.testing.assert_allclose(c2w, expected, atol=1e-12)

    def test_translation_transformed(self):
        """Translation should be inverted, then sign-flipped by y180."""
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        c2w = colmap_pose_to_c2w_opengl(R, t)
        # c2w_t before y180: -R^T @ t = -t = [-1, -2, -3]
        # OpenCV->OpenGL flip doesn't change column 3 (translation)
        # y180 = diag(-1,1,-1,1) flips x and z: [1, -2, 3]
        np.testing.assert_allclose(c2w[:3, 3], [1.0, -2.0, 3.0], atol=1e-12)


class TestWriteTransformsJson:
    """Tests for the JSON writer."""

    def test_structure(self, tmp_path):
        """Written JSON should have all required fields."""
        frames = [
            {
                "file_path": "images/frame_0001.jpg",
                "transform_matrix": np.eye(4).tolist(),
            }
        ]
        out = str(tmp_path / "transforms.json")
        write_transforms_json(
            out, "EQUIRECTANGULAR", 7680, 3840, 3840.0, 3840.0, frames
        )
        with open(out) as f:
            data = json.load(f)
        assert data["camera_model"] == "EQUIRECTANGULAR"
        assert data["w"] == 7680
        assert data["h"] == 3840
        assert data["fl_x"] == 3840.0
        assert data["fl_y"] == 3840.0
        assert len(data["frames"]) == 1

    def test_ply_file_path_absent_by_default(self, tmp_path):
        """ply_file_path should not appear when not provided."""
        out = str(tmp_path / "t.json")
        write_transforms_json(out, "EQUIRECTANGULAR", 100, 50, 50.0, 50.0, [])
        with open(out) as f:
            data = json.load(f)
        assert "ply_file_path" not in data

    def test_ply_file_path_present_when_given(self, tmp_path):
        """ply_file_path should appear when provided."""
        out = str(tmp_path / "t.json")
        write_transforms_json(
            out, "EQUIRECTANGULAR", 100, 50, 50.0, 50.0, [], ply_file_path="pc.ply"
        )
        with open(out) as f:
            data = json.load(f)
        assert data["ply_file_path"] == "pc.ply"

    def test_creates_parent_directories(self, tmp_path):
        """Should create intermediate directories if they don't exist."""
        out = tmp_path / "sub" / "dir" / "transforms.json"
        write_transforms_json(
            str(out), "EQUIRECTANGULAR", 100, 50, 50.0, 50.0, []
        )
        assert out.exists()

    def test_multiple_frames(self, tmp_path):
        """Should handle multiple frames correctly."""
        frames = [
            {"file_path": f"images/frame_{i:04d}.jpg", "transform_matrix": np.eye(4).tolist()}
            for i in range(5)
        ]
        out = str(tmp_path / "transforms.json")
        write_transforms_json(out, "EQUIRECTANGULAR", 7680, 3840, 3840.0, 3840.0, frames)
        with open(out) as f:
            data = json.load(f)
        assert len(data["frames"]) == 5
        assert data["frames"][2]["file_path"] == "images/frame_0002.jpg"
