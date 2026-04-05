# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for CubemapProjection class."""

import numpy as np
import pytest
from core.cubemap_projection import CubemapProjection


def test_equirect2cubemap_returns_six_faces():
    """Decomposing an ERP image produces exactly 6 named faces."""
    proj = CubemapProjection(face_size=64)
    erp = np.zeros((100, 200, 3), dtype=np.uint8)
    faces = proj.equirect2cubemap(erp)
    assert set(faces.keys()) == {"front", "back", "left", "right", "up", "down"}
    for name, face in faces.items():
        assert face.shape == (64, 64, 3), f"{name} has wrong shape: {face.shape}"


def test_equirect2cubemap_face_size_from_width():
    """Default face size is min(1024, w//4)."""
    proj = CubemapProjection(face_size=None)
    erp = np.zeros((200, 400, 3), dtype=np.uint8)
    faces = proj.equirect2cubemap(erp)
    assert faces["front"].shape[0] == 100  # min(1024, 400//4) = 100


def test_cubemap_round_trip_preserves_mask():
    """A mask painted on one face survives decompose → merge round-trip."""
    proj = CubemapProjection(face_size=64)
    # Create an ERP mask with a white rectangle in the front-center region
    erp_mask = np.zeros((100, 200), dtype=np.uint8)
    erp_mask[40:60, 90:110] = 1  # center of ERP = front face

    # Decompose to cubemap, then merge back
    # Stack to 3-channel so cv2.remap returns 3D
    erp_3ch = np.stack([erp_mask] * 3, axis=-1)
    faces = proj.equirect2cubemap(erp_3ch)
    face_masks = {}
    for name, face in faces.items():
        face_masks[name] = (face[:, :, 0] > 0).astype(np.uint8)
    merged = proj.cubemap2equirect(face_masks, (200, 100))

    # The center region should be preserved (allow some interpolation loss)
    center_recall = merged[45:55, 95:105].mean()
    assert center_recall > 0.5, f"Center region lost: recall={center_recall:.2f}"
