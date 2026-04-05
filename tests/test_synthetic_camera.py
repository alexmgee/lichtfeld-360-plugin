# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for synthetic fisheye camera projection functions.

Covers Task A1 from the masking v1 final plan:
- Camera creation and pycolmap round-trip
- ERP → fisheye rendering
- Fisheye mask → ERP back-projection
- Render → mask → backproject round-trip
"""
from __future__ import annotations

import numpy as np
import pycolmap
import pytest


# ── A1.1: Camera creation and cam_from_img / img_from_cam round-trip ──


class TestCreateSyntheticCamera:
    """Verify the ideal equidistant fisheye camera model."""

    def test_camera_model_is_opencv_fisheye(self):
        from core.masker import _create_synthetic_camera

        cam = _create_synthetic_camera(256)
        assert cam.model == pycolmap.CameraModelId.OPENCV_FISHEYE

    def test_camera_dimensions_match_requested_size(self):
        from core.masker import _create_synthetic_camera

        cam = _create_synthetic_camera(512)
        assert cam.width == 512
        assert cam.height == 512

    def test_center_pixel_maps_to_forward(self):
        """Center pixel should map to [0, 0] (camera +Z = forward)."""
        from core.masker import _create_synthetic_camera

        cam = _create_synthetic_camera(256)
        center = cam.width / 2
        ray_2d = cam.cam_from_img(np.array([[center, center]], dtype=np.float64))
        assert ray_2d.shape == (1, 2)
        np.testing.assert_allclose(ray_2d[0], [0.0, 0.0], atol=1e-10)

    def test_45_degree_ray(self):
        """A pixel at 45° from center should map to [1, 0] in normalized coords."""
        from core.masker import _create_synthetic_camera

        size = 256
        cam = _create_synthetic_camera(size)
        # For equidistant: r = f*θ. At θ=45°=π/4, r = f*��/4 = (size/2)/2 = size/4
        # So pixel is at (center + size/4, center)
        center = size / 2
        offset = size / 4
        px = np.array([[center + offset, center]], dtype=np.float64)
        ray_2d = cam.cam_from_img(px)
        np.testing.assert_allclose(ray_2d[0], [1.0, 0.0], atol=1e-6)

    def test_img_from_cam_round_trip(self):
        """cam_from_img → append z=1 → img_from_cam should round-trip exactly."""
        from core.masker import _create_synthetic_camera

        cam = _create_synthetic_camera(512)
        # Sample several pixels across the image
        test_pixels = np.array([
            [100.0, 80.0],
            [256.0, 256.0],
            [200.0, 300.0],
            [400.0, 150.0],
        ], dtype=np.float64)

        rays_2d = cam.cam_from_img(test_pixels)
        rays_3d = np.hstack([rays_2d, np.ones((len(rays_2d), 1))])
        recovered = cam.img_from_cam(rays_3d)
        np.testing.assert_allclose(recovered, test_pixels, atol=1e-6)

    def test_hemisphere_edge_is_asymptotic(self):
        """Near 90° from center, the normalized coordinate should be very large,
        not a finite value like 1.0."""
        from core.masker import _create_synthetic_camera

        size = 256
        cam = _create_synthetic_camera(size)
        center = size / 2
        # Pixel near the edge of the inscribed circle (just inside radius)
        near_edge = np.array([[size - 1.0, center]], dtype=np.float64)
        ray_2d = cam.cam_from_img(near_edge)
        # Should be much larger than 1.0 (the 45° value)
        assert ray_2d[0, 0] > 5.0, f"Expected large value near horizon, got {ray_2d[0, 0]}"


# ── A1.2: ERP → fisheye rendering ──────────────────────────────────


class TestRenderSyntheticFisheye:
    """Verify ERP → fisheye rendering produces correct output."""

    def test_output_is_square_and_correct_size(self):
        from core.masker import _create_synthetic_camera, _render_synthetic_fisheye

        size = 128
        cam = _create_synthetic_camera(size)
        erp = np.zeros((64, 128, 3), dtype=np.uint8)
        R = np.eye(3)  # identity = look forward (+Z)
        fisheye = _render_synthetic_fisheye(erp, cam, R)
        assert fisheye.shape == (size, size, 3)

    def test_center_pixel_samples_forward_direction(self):
        """With identity rotation, the fisheye center should sample the ERP
        forward direction (center column of the ERP)."""
        from core.masker import _create_synthetic_camera, _render_synthetic_fisheye

        size = 128
        cam = _create_synthetic_camera(size)
        # Create ERP with a bright vertical stripe at the center (forward)
        erp = np.zeros((64, 128, 3), dtype=np.uint8)
        erp[:, 62:66, :] = 255  # bright stripe at center column
        R = np.eye(3)
        fisheye = _render_synthetic_fisheye(erp, cam, R)
        # Center pixel of fisheye should be bright
        cx, cy = size // 2, size // 2
        assert fisheye[cy, cx].mean() > 200, "Center pixel should sample the bright stripe"

    def test_output_has_circular_coverage(self):
        """Fisheye output should have content in a circle, black corners."""
        from core.masker import _create_synthetic_camera, _render_synthetic_fisheye

        size = 256
        cam = _create_synthetic_camera(size)
        # White ERP
        erp = np.full((128, 256, 3), 255, dtype=np.uint8)
        R = np.eye(3)
        fisheye = _render_synthetic_fisheye(erp, cam, R)
        # Corner pixels should be black (outside fisheye circle)
        assert fisheye[0, 0].sum() == 0, "Top-left corner should be black"
        assert fisheye[0, -1].sum() == 0, "Top-right corner should be black"
        # Center should be white
        cx, cy = size // 2, size // 2
        assert fisheye[cy, cx].mean() > 200, "Center should be bright"


# ── A1.3: Fisheye mask → ERP back-projection ──────────────────────


class TestBackprojectFisheyeMaskToErp:
    """Verify fisheye mask → ERP back-projection."""

    def test_output_dimensions_match_erp_size(self):
        from core.masker import (
            _create_synthetic_camera,
            _backproject_fisheye_mask_to_erp,
        )

        size = 128
        cam = _create_synthetic_camera(size)
        mask = np.zeros((size, size), dtype=np.uint8)
        R = np.eye(3)
        erp_mask = _backproject_fisheye_mask_to_erp(mask, (256, 128), cam, R)
        assert erp_mask.shape == (128, 256)

    def test_center_blob_backprojects_to_forward(self):
        """A blob at the fisheye center should back-project to the ERP forward
        direction (center of ERP) when rotation is identity."""
        from core.masker import (
            _create_synthetic_camera,
            _backproject_fisheye_mask_to_erp,
        )

        size = 256
        cam = _create_synthetic_camera(size)
        # Create a blob at the center of the fisheye
        mask = np.zeros((size, size), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        mask[cy - 10 : cy + 10, cx - 10 : cx + 10] = 1
        R = np.eye(3)
        erp_w, erp_h = 512, 256
        erp_mask = _backproject_fisheye_mask_to_erp(mask, (erp_w, erp_h), cam, R)
        # The blob should appear near the center of the ERP
        # (forward = center column, equator = center row)
        ys, xs = np.where(erp_mask > 0)
        if len(xs) > 0:
            mean_x = xs.mean()
            mean_y = ys.mean()
            assert abs(mean_x - erp_w / 2) < erp_w * 0.1, (
                f"Blob center x={mean_x}, expected near {erp_w/2}"
            )
            assert abs(mean_y - erp_h / 2) < erp_h * 0.1, (
                f"Blob center y={mean_y}, expected near {erp_h/2}"
            )
        else:
            pytest.fail("Back-projected mask has no positive pixels")


# ── A1.4: Render → mask → backproject round-trip ──────────────────


class TestFisheyeRoundTrip:
    """Verify render → create mask on fisheye → backproject preserves coverage."""

    def test_erp_blob_survives_round_trip(self):
        """Place a blob in the ERP, render to fisheye, threshold the fisheye
        to create a mask, backproject, and verify the blob location is preserved."""
        from core.masker import (
            _create_synthetic_camera,
            _render_synthetic_fisheye,
            _backproject_fisheye_mask_to_erp,
        )

        size = 256
        cam = _create_synthetic_camera(size)
        erp_w, erp_h = 512, 256

        # ERP with a bright blob at the forward direction (center)
        erp = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        blob_cx, blob_cy = erp_w // 2, erp_h // 2
        erp[blob_cy - 15 : blob_cy + 15, blob_cx - 15 : blob_cx + 15] = 255

        R = np.eye(3)

        # Render to fisheye
        fisheye = _render_synthetic_fisheye(erp, cam, R)

        # Create mask from fisheye (threshold)
        gray = fisheye.mean(axis=2) if fisheye.ndim == 3 else fisheye.astype(float)
        fish_mask = (gray > 128).astype(np.uint8)

        # Backproject
        erp_mask = _backproject_fisheye_mask_to_erp(fish_mask, (erp_w, erp_h), cam, R)

        # Verify: the backprojected mask should overlap with the original blob
        original_blob = np.zeros((erp_h, erp_w), dtype=np.uint8)
        original_blob[blob_cy - 15 : blob_cy + 15, blob_cx - 15 : blob_cx + 15] = 1
        overlap = (erp_mask > 0) & (original_blob > 0)
        overlap_ratio = overlap.sum() / max(original_blob.sum(), 1)
        assert overlap_ratio > 0.5, (
            f"Round-trip overlap with original blob is only {overlap_ratio:.2%}"
        )

    def test_round_trip_with_rotated_direction(self):
        """Same round-trip but with the synthetic camera aimed 90° right."""
        from core.masker import (
            _create_synthetic_camera,
            _render_synthetic_fisheye,
            _backproject_fisheye_mask_to_erp,
            _look_at_rotation,
        )

        size = 256
        cam = _create_synthetic_camera(size)
        erp_w, erp_h = 512, 256

        # ERP with a blob at yaw=90° (right side, 3/4 across the image)
        erp = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        blob_cx = int(erp_w * 0.75)  # yaw=90° is at u=0.75
        blob_cy = erp_h // 2
        erp[blob_cy - 15 : blob_cy + 15, blob_cx - 15 : blob_cx + 15] = 255

        # Aim camera at [1, 0, 0] (yaw=90°)
        R = _look_at_rotation(np.array([1.0, 0.0, 0.0]))

        fisheye = _render_synthetic_fisheye(erp, cam, R)
        gray = fisheye.mean(axis=2) if fisheye.ndim == 3 else fisheye.astype(float)
        fish_mask = (gray > 128).astype(np.uint8)
        erp_mask = _backproject_fisheye_mask_to_erp(fish_mask, (erp_w, erp_h), cam, R)

        original_blob = np.zeros((erp_h, erp_w), dtype=np.uint8)
        original_blob[blob_cy - 15 : blob_cy + 15, blob_cx - 15 : blob_cx + 15] = 1
        overlap = (erp_mask > 0) & (original_blob > 0)
        overlap_ratio = overlap.sum() / max(original_blob.sum(), 1)
        assert overlap_ratio > 0.5, (
            f"Rotated round-trip overlap is only {overlap_ratio:.2%}"
        )


# ── A2.1: Center-of-mass ─────────────────────────────────────────


class TestComputeDetectionCom:
    def test_known_blob_centroid(self):
        from core.masker import _compute_detection_com

        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 80:120] = 1  # blob centered at (100, 50)
        result = _compute_detection_com(mask)
        assert result is not None
        cx, cy = result
        assert abs(cx - 99.5) < 1.0
        assert abs(cy - 49.5) < 1.0

    def test_empty_mask_returns_none(self):
        from core.masker import _compute_detection_com

        mask = np.zeros((100, 200), dtype=np.uint8)
        assert _compute_detection_com(mask) is None


# ── A2.2: Pixel CoM to 3D direction ─────────────────────────────


class TestPixelComTo3dDirection:
    """Verify _pixel_com_to_3d_direction inverts the reframer's fliplr + rotation.

    Preflight confirmed: for an even-sized flipped detection image, the
    optical center in x is at (size/2 - 1) due to the fliplr, and the
    optical center in y is at (size/2).
    """

    def test_forward_direction_at_yaw0_pitch0(self):
        """Center pixel of a yaw=0, pitch=0 view should map to [0, 0, 1]."""
        from core.masker import _pixel_com_to_3d_direction

        size = 1024
        # Optical center of flipped image: x = size/2 - 1, y = size/2
        cx = size / 2 - 1
        cy = size / 2
        d = _pixel_com_to_3d_direction(cx, cy, 90.0, 0.0, 0.0, size, False)
        np.testing.assert_allclose(d, [0, 0, 1], atol=0.01)

    def test_right_direction_at_yaw90(self):
        """Center pixel of a yaw=90 view should map to [1, 0, 0]."""
        from core.masker import _pixel_com_to_3d_direction

        size = 1024
        cx = size / 2 - 1
        cy = size / 2
        d = _pixel_com_to_3d_direction(cx, cy, 90.0, 90.0, 0.0, size, False)
        np.testing.assert_allclose(d, [1, 0, 0], atol=0.01)

    def test_pitch45_direction(self):
        """Center pixel of a pitch=45 view should map to [0, sin45, cos45]."""
        from core.masker import _pixel_com_to_3d_direction

        size = 1024
        cx = size / 2 - 1
        cy = size / 2
        d = _pixel_com_to_3d_direction(cx, cy, 90.0, 0.0, 45.0, size, False)
        expected = np.array([0, np.sin(np.radians(45)), np.cos(np.radians(45))])
        np.testing.assert_allclose(d, expected, atol=0.01)

    def test_flip_v_inverts_vertical(self):
        """With flip_v=True, a pixel in the top half of the flipped image
        should recover a direction that points downward compared to flip_v=False."""
        from core.masker import _pixel_com_to_3d_direction

        size = 1024
        cx = size / 2 - 1
        top_cy = size / 4  # top quarter

        d_no_flip = _pixel_com_to_3d_direction(cx, top_cy, 90.0, 0.0, 0.0, size, False)
        d_flip = _pixel_com_to_3d_direction(cx, top_cy, 90.0, 0.0, 0.0, size, True)

        # flip_v should invert the y component
        assert d_no_flip[1] * d_flip[1] < 0, (
            f"Expected opposite y: no_flip={d_no_flip[1]:.3f}, flip={d_flip[1]:.3f}"
        )


# ── A2.3: Weighted person direction ─────────────────────────────


class TestComputeWeightedPersonDirection:
    def test_equal_weight_average(self):
        from core.masker import _compute_weighted_person_direction

        d1 = np.array([1.0, 0.0, 0.0])
        d2 = np.array([0.0, 0.0, 1.0])
        result = _compute_weighted_person_direction([(d1, 1.0), (d2, 1.0)])
        assert result is not None
        expected = np.array([1, 0, 1], dtype=float)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_empty_list_returns_none(self):
        from core.masker import _compute_weighted_person_direction

        assert _compute_weighted_person_direction([]) is None

    def test_single_direction(self):
        from core.masker import _compute_weighted_person_direction

        d = np.array([0.0, 1.0, 0.0])
        result = _compute_weighted_person_direction([(d, 5.0)])
        assert result is not None
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-6)


# ── A2.4: Temporal fallback direction ────────────────────────────


class TestTemporalFallbackDirection:
    def test_borrows_from_nearest(self):
        from core.masker import _temporal_fallback_direction

        fwd = np.array([0, 0, 1.0])
        dirs = [None, None, fwd, None, None]
        # Frame 0 should borrow from frame 2 (nearest with a direction)
        result = _temporal_fallback_direction(0, dirs)
        assert result is not None
        np.testing.assert_allclose(result, fwd)

    def test_prefers_closer_frame(self):
        from core.masker import _temporal_fallback_direction

        d_near = np.array([1, 0, 0.0])
        d_far = np.array([0, 1, 0.0])
        dirs = [d_far, None, None, None, d_near]
        # Frame 3 should borrow from frame 4 (distance 1) not frame 0 (distance 3)
        result = _temporal_fallback_direction(3, dirs)
        assert result is not None
        np.testing.assert_allclose(result, d_near)

    def test_all_none_returns_none(self):
        from core.masker import _temporal_fallback_direction

        dirs = [None, None, None]
        assert _temporal_fallback_direction(1, dirs) is None

    def test_returns_own_if_valid(self):
        from core.masker import _temporal_fallback_direction

        d = np.array([0, 0, 1.0])
        dirs = [None, d, None]
        result = _temporal_fallback_direction(1, dirs)
        np.testing.assert_allclose(result, d)


# ── A2.5: Look-at rotation ──────────────────────────────────────


class TestLookAtRotation:
    def test_forward_produces_near_identity(self):
        from core.masker import _look_at_rotation

        R = _look_at_rotation(np.array([0, 0, 1.0]))
        result = R @ np.array([0, 0, 1.0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_right_rotates_z_to_x(self):
        from core.masker import _look_at_rotation

        R = _look_at_rotation(np.array([1, 0, 0.0]))
        result = R @ np.array([0, 0, 1.0])
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_up_direction(self):
        from core.masker import _look_at_rotation

        R = _look_at_rotation(np.array([0, 1, 0.0]))
        result = R @ np.array([0, 0, 1.0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_is_orthonormal(self):
        from core.masker import _look_at_rotation

        R = _look_at_rotation(np.array([1, 1, 1.0]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ── A2.6: Direction to yaw/pitch ─────────────────────────────────


class TestDirectionToYawPitch:
    def test_forward(self):
        from core.masker import _direction_to_yaw_pitch

        yaw, pitch = _direction_to_yaw_pitch(np.array([0, 0, 1.0]))
        assert abs(yaw) < 0.1
        assert abs(pitch) < 0.1

    def test_right(self):
        from core.masker import _direction_to_yaw_pitch

        yaw, pitch = _direction_to_yaw_pitch(np.array([1, 0, 0.0]))
        assert abs(yaw - 90.0) < 0.1
        assert abs(pitch) < 0.1

    def test_up(self):
        from core.masker import _direction_to_yaw_pitch

        yaw, pitch = _direction_to_yaw_pitch(np.array([0, 1, 0.0]))
        assert abs(pitch - 90.0) < 0.1
