# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Closest-camera Voronoi masks for anti-overlap feature extraction.

Ported from FullCircle (omni2perspective.py:148-153).
For each pixel in each pinhole view, determines which camera center
has the most similar viewing direction. Pixels owned by another camera
are masked black to prevent duplicate COLMAP feature extraction.

Purely geometric — computed from camera rotations, no ML.
Precomputed once per preset, reused for every frame.
"""
from __future__ import annotations

import numpy as np

from .reframer import create_rotation_matrix


def compute_overlap_masks(
    views: list[tuple[float, float, float, str, bool]],
    output_size: int,
) -> dict[str, np.ndarray] | None:
    """Compute per-view Voronoi ownership masks.

    Args:
        views: List of (yaw, pitch, fov, name, flip_vertical) from ViewConfig.
        output_size: Square output image size in pixels.

    Returns:
        Dict mapping view_name → uint8 mask (255=own, 0=other camera closer).
        Returns None if no overlap exists (e.g. cubemap preset).
    """
    if len(views) <= 1:
        return None

    # Compute camera forward directions (center of each view)
    cam_centers = []
    view_names = []
    for yaw, pitch, fov, name, flip_v in views:
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        fwd = np.array([
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.cos(yaw_rad),
        ])
        cam_centers.append(fwd)
        view_names.append(name)
    cam_centers = np.array(cam_centers)  # (N, 3)

    # Check if any views overlap by computing the minimum angular separation
    # between any two camera centers and comparing to the FOV.
    # If the smallest angle between any two cameras > max FOV, no views overlap.
    dots = cam_centers @ cam_centers.T
    np.fill_diagonal(dots, -2)  # ignore self
    max_dot = np.max(dots)
    max_fov = max(fov for _, _, fov, _, _ in views)
    # Two cameras overlap when their angular separation < sum of their half-FOVs.
    # Conservative: use max_fov as the threshold (assumes same FOV for both).
    if max_dot < np.cos(np.radians(max_fov)) + 1e-9:
        return None  # No overlap — skip Voronoi computation

    masks = {}
    for idx, (yaw, pitch, fov, name, flip_v) in enumerate(views):
        half_fov = np.radians(fov / 2.0)

        # Build ray directions for every pixel in this view
        size = output_size
        u = np.linspace(-1, 1, size)
        v = np.linspace(-1, 1, size)
        uu, vv = np.meshgrid(u, v)

        # Pixel rays in camera space (pinhole projection)
        focal = 1.0 / np.tan(half_fov)
        rays_cam = np.stack([uu, -vv, -np.full_like(uu, focal)], axis=-1)
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

        # Rotate to world space using the view's rotation matrix
        # create_rotation_matrix takes degrees, not radians
        R = create_rotation_matrix(yaw, pitch)
        # R is w2c (rows = right, up, -forward), so R.T = c2w
        rays_world = rays_cam @ R  # (H, W, 3) @ (3, 3) → (H, W, 3)

        # For each pixel, find closest camera center
        # rays_world: (H, W, 3), cam_centers: (N, 3)
        dots_per_cam = np.einsum("hwc,nc->hwn", rays_world, cam_centers)
        closest = np.argmax(dots_per_cam, axis=-1)  # (H, W)

        mask = ((closest == idx) * 255).astype(np.uint8)

        if flip_v:
            mask = np.flipud(mask)

        masks[name] = mask

    return masks
