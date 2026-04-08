# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator masking via per-view detection with ERP OR-merge.

Pipeline (matching FullCircle's two-step approach):
  1. Reframe ERP → pinhole views at detection resolution
  2. Detect person on each view (YOLO+SAM or SAM 3)
  3. Back-project all per-view detections to ERP space, OR-merge
  4. Postprocess merged ERP mask (morph close + flood fill)
  5. Save ERP mask — reframer reprojects to pinhole masks

The OR-merge means if ANY view detects the person at a given ERP
pixel, that pixel is masked — even in views where YOLO missed them.

Masks use COLMAP polarity: white (255) = keep, black (0) = remove.
"""
from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import pycolmap

from .backends import (
    get_backend,
    get_backend_name,
    MaskingBackend,
    select_primary_person_box,
    VideoTrackingBackend,
)
from .reframer import create_rotation_matrix

logger = logging.getLogger(__name__)

CUBEMAP_DIRECT_CONFIDENCE = 0.35
SYNTHETIC_FALLBACK_CONFIDENCE = 0.25
SYNTHETIC_PRIMARY_BOX_MODE = "center"
SYNTHETIC_PRIMARY_BOX_CONSTRAIN = True
SYNTHETIC_PRIMARY_BOX_PADDING = 0.35
ALT_VIEW_RESCUE_CONFIDENCE = 0.25
ALT_VIEW_RESCUE_MAX_CENTER_DIST = 420.0
ALT_VIEW_RESCUE_MIN_CONFIDENCE = 0.18
ALT_VIEW_RESCUE_MIN_MASK_PIXELS = 1536
ALT_VIEW_RESCUE_PROPAGATED_GAIN = 1.20
ALT_VIEW_RESCUE_PROPAGATED_MIN_GAIN = 512
ALT_VIEW_RESCUE_PROPAGATED_OVERRIDE_CONFIDENCE = 0.28
ALT_VIEW_RESCUE_PROPAGATED_OVERRIDE_PIXELS = 2048
SAM2_PROMPT_BOX_CONFIDENCE = 0.18
DIRECTION_SEARCH_YAW_OFFSETS = (-18.0, -9.0, 9.0, 18.0)
DIRECTION_SEARCH_PITCH_OFFSETS = (-10.0, 0.0, 10.0)
DIRECTION_SEARCH_MAX_CENTER_DIST = 300.0
DIRECTION_SEARCH_EXTENDED_CENTER_DIST = 560.0
DIRECTION_SEARCH_MIN_CONFIDENCE = 0.22
DIRECTION_SEARCH_EXTENDED_MIN_CONFIDENCE = 0.24
DIRECTION_SEARCH_MIN_MASK_PIXELS = 2048
DIRECTION_SEARCH_EXTENDED_MIN_MASK_PIXELS = 4096
DIRECTION_SEARCH_PROPAGATED_GAIN = 1.25
DIRECTION_SEARCH_PROPAGATED_MIN_GAIN = 1024
DIRECTION_SEARCH_PROPAGATED_SCORE_GAIN = 1.12
DIRECTION_SEARCH_PROPAGATED_SCORE_MARGIN = 4.0


# ── Substage timing ─────────────────────────────────────────────


class _SubstageTimer:
    """Lightweight accumulating timer for masking substages."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def time(self, label: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - t0
            self._totals[label] = self._totals.get(label, 0.0) + elapsed
            self._counts[label] = self._counts.get(label, 0) + 1

    def report(self, log: logging.Logger | None = None) -> None:
        if not self._totals:
            return
        _log = log or logger
        total = sum(self._totals.values())
        lines = ["Masking substage timing:"]
        for label in self._totals:
            t = self._totals[label]
            n = self._counts[label]
            lines.append(f"  {label:24s} {t:7.1f}s  ({n} calls)")
        lines.append(f"  {'TOTAL':24s} {total:7.1f}s")
        _log.debug("\n".join(lines))


# ── Synthetic fisheye camera (Task A1) ───────────────────────────


def _create_synthetic_camera(size: int = 2048) -> pycolmap.Camera:
    """Create an ideal equidistant fisheye camera for synthetic views.

    Uses OPENCV_FISHEYE with zero distortion (k1-k4=0), giving pure
    equidistant projection: r = f*θ. At 180° FOV the full hemisphere
    is inscribed in the image circle.

    Convention: camera +Z = forward (FullCircle convention, NOT the
    reframer's -Z convention). These are isolated and never mixed.
    """
    focal = size / 2 / (np.pi / 2)
    center = size / 2.0
    return pycolmap.Camera(
        camera_id=0,
        model=pycolmap.CameraModelId.OPENCV_FISHEYE,
        width=size,
        height=size,
        params=[focal, focal, center, center, 0.0, 0.0, 0.0, 0.0],
    )


def _look_at_rotation(center_dir: np.ndarray) -> np.ndarray:
    """Compute world_from_cam rotation with camera +Z aimed at center_dir.

    Port of FullCircle's look_at_camZ() from lib/cam_utils.py:66-75.
    Returns a 3×3 rotation matrix R where R @ [0,0,1] ≈ normalize(center_dir).
    Columns are [right, up, forward] in world space.
    """
    z = center_dir.astype(np.float64).ravel()
    n = np.linalg.norm(z)
    if n < 1e-12:
        return np.eye(3)
    z = z / n

    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    return np.stack([x, y, z], axis=1)  # 3×3, columns = [x, y, z]


def _render_synthetic_fisheye(
    erp: np.ndarray,
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> np.ndarray:
    """Render an ERP image into a synthetic fisheye view.

    Port of FullCircle's omni2synthetic.py:107-124.

    Args:
        erp: Equirectangular image (H, W, 3) uint8.
        camera: pycolmap OPENCV_FISHEYE camera.
        R_world_from_cam: 3×3 world_from_cam rotation (cam +Z = forward).

    Returns:
        Fisheye image (size, size, 3) uint8. Pixels outside the lens
        circle are black.
    """
    size = camera.width
    h_eq, w_eq = erp.shape[:2]

    # Generate pixel grid for the fisheye image
    xs = np.arange(size, dtype=np.float64) + 0.5
    ys = np.arange(size, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Radial mask: only pixels inside the inscribed circle
    cx, cy = size / 2.0, size / 2.0
    r = np.sqrt((xx.ravel() - cx) ** 2 + (yy.ravel() - cy) ** 2)
    radius = size / 2.0
    valid = r < (radius - 0.5)

    # cam_from_img returns 2D normalized coords [x/z, y/z]
    rays_2d = camera.cam_from_img(pixels[valid])
    # Form 3D camera-space directions: [x/z, y/z, 1] then normalize
    rays_cam = np.hstack([rays_2d, np.ones((len(rays_2d), 1))])
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # Rotate to world space
    rays_world = (R_world_from_cam @ rays_cam.T).T  # (N, 3)

    # World rays to ERP coordinates
    x_w, y_w, z_w = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
    lon = np.arctan2(x_w, z_w)  # [-π, π]
    lat = np.arcsin(np.clip(y_w, -1, 1))  # [-π/2, π/2]

    u_eq = ((lon / np.pi + 1) / 2) * w_eq
    v_eq = (0.5 - lat / np.pi) * h_eq

    # Build remap arrays
    map_x = np.full(size * size, -1.0, dtype=np.float32)
    map_y = np.full(size * size, -1.0, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    map_x[valid_idx] = u_eq.astype(np.float32) % w_eq
    map_y[valid_idx] = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)

    map_x = map_x.reshape(size, size)
    map_y = map_y.reshape(size, size)

    result = cv2.remap(
        erp, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return result


# Scale factor for downsampled backprojection. At 0.5, the ERP grid
# has 4× fewer points (e.g., 3840×1920 instead of 7680×3840), giving
# ~3.8× speedup with IoU > 0.99 vs full-resolution. The binary result
# is upscaled with nearest-neighbor.
BACKPROJECT_SCALE = 0.5


def _backproject_fisheye_mask_to_erp(
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a fisheye detection mask to ERP space.

    When BACKPROJECT_SCALE < 1.0, computes at reduced resolution and
    upscales with nearest-neighbor for speed.

    Port of FullCircle's synthetic2omni.py:44-90.

    Args:
        mask: Fisheye mask (size, size) uint8, 1=detected.
        erp_size: (width, height) of ERP.
        camera: pycolmap OPENCV_FISHEYE camera.
        R_world_from_cam: 3×3 world_from_cam rotation.

    Returns:
        ERP mask (erp_h, erp_w) uint8, 1=detected.
    """
    erp_w, erp_h = erp_size

    # Downsample: compute at reduced resolution, upscale result
    if BACKPROJECT_SCALE < 1.0:
        reduced_w = max(1, int(erp_w * BACKPROJECT_SCALE))
        reduced_h = max(1, int(erp_h * BACKPROJECT_SCALE))
        reduced = _backproject_fisheye_mask_to_erp_full(
            mask, (reduced_w, reduced_h), camera, R_world_from_cam,
        )
        return cv2.resize(reduced, (erp_w, erp_h), interpolation=cv2.INTER_NEAREST)

    return _backproject_fisheye_mask_to_erp_full(
        mask, erp_size, camera, R_world_from_cam,
    )


def _backproject_fisheye_mask_to_erp_full(
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> np.ndarray:
    """Full-resolution backprojection (no downsampling)."""
    erp_w, erp_h = erp_size
    fish_size = camera.width

    # For each ERP pixel, compute world ray direction
    u = np.arange(erp_w, dtype=np.float64) + 0.5
    v = np.arange(erp_h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    lon = ((uu / erp_w) * 2 - 1) * np.pi
    lat = (0.5 - vv / erp_h) * np.pi

    # World ray directions
    x_w = np.cos(lat) * np.sin(lon)
    y_w = np.sin(lat)
    z_w = np.cos(lat) * np.cos(lon)
    dirs_world = np.stack([x_w.ravel(), y_w.ravel(), z_w.ravel()], axis=1)

    # Rotate to camera space: cam_from_world = R_world_from_cam.T
    R_cam_from_world = R_world_from_cam.T
    dirs_cam = (R_cam_from_world @ dirs_world.T).T

    # Only forward-pointing rays (z > 0 in camera space)
    forward = dirs_cam[:, 2] > 1e-8

    erp_mask = np.zeros(erp_h * erp_w, dtype=np.uint8)

    if not np.any(forward):
        return erp_mask.reshape(erp_h, erp_w)

    # Project forward rays to fisheye pixel coords
    pts_cam = dirs_cam[forward]
    px_py = camera.img_from_cam(pts_cam)

    # Radial validity check
    cx, cy = fish_size / 2.0, fish_size / 2.0
    radius = fish_size / 2.0
    r = np.sqrt((px_py[:, 0] - cx) ** 2 + (px_py[:, 1] - cy) ** 2)
    in_lens = r < (radius - 0.5)

    # Bounds check
    in_bounds = (
        in_lens
        & (px_py[:, 0] >= 0) & (px_py[:, 0] < fish_size)
        & (px_py[:, 1] >= 0) & (px_py[:, 1] < fish_size)
    )

    if np.any(in_bounds):
        forward_idx = np.where(forward)[0]
        valid_idx = forward_idx[in_bounds]
        px_int = np.clip(np.round(px_py[in_bounds, 0]).astype(int), 0, fish_size - 1)
        py_int = np.clip(np.round(px_py[in_bounds, 1]).astype(int), 0, fish_size - 1)
        erp_mask[valid_idx] = mask[py_int, px_int]

    return erp_mask.reshape(erp_h, erp_w)


# ── Shared backprojection map ────────────────────────────────────


@dataclass
class _BackprojectMap:
    """Precomputed fisheye→ERP backprojection lookup.

    For each valid ERP pixel (one inside the fisheye's forward hemisphere
    and lens circle), stores the corresponding fisheye pixel coordinate.
    Applying a mask is then a single numpy index operation.

    When BACKPROJECT_SCALE < 1.0, the map is built at reduced resolution
    and apply() upscales the result to full_erp_size.
    """
    grid_h: int              # reduced grid height
    grid_w: int              # reduced grid width
    full_erp_h: int          # original full ERP height
    full_erp_w: int          # original full ERP width
    valid_idx: np.ndarray    # flat indices into (grid_h * grid_w,)
    fish_px: np.ndarray      # int, fisheye x coords for valid pixels
    fish_py: np.ndarray      # int, fisheye y coords for valid pixels

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Sample a fisheye mask using the precomputed map."""
        erp_mask = np.zeros(self.grid_h * self.grid_w, dtype=np.uint8)
        erp_mask[self.valid_idx] = mask[self.fish_py, self.fish_px]
        result = erp_mask.reshape(self.grid_h, self.grid_w)
        if result.shape != (self.full_erp_h, self.full_erp_w):
            result = cv2.resize(
                result, (self.full_erp_w, self.full_erp_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return result


def _build_backproject_map(
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> _BackprojectMap:
    """Precompute the fisheye→ERP backprojection lookup table.

    When BACKPROJECT_SCALE < 1.0, builds the map at reduced resolution.
    The result can be reused across frames when the person direction
    (and thus R_world_from_cam) is stable.
    """
    erp_w, erp_h = erp_size
    grid_w = max(1, int(erp_w * BACKPROJECT_SCALE))
    grid_h = max(1, int(erp_h * BACKPROJECT_SCALE))
    fish_size = camera.width

    u = np.arange(grid_w, dtype=np.float64) + 0.5
    v = np.arange(grid_h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    lon = ((uu / grid_w) * 2 - 1) * np.pi
    lat = (0.5 - vv / grid_h) * np.pi

    x_w = np.cos(lat) * np.sin(lon)
    y_w = np.sin(lat)
    z_w = np.cos(lat) * np.cos(lon)
    dirs_world = np.stack([x_w.ravel(), y_w.ravel(), z_w.ravel()], axis=1)

    R_cam_from_world = R_world_from_cam.T
    dirs_cam = (R_cam_from_world @ dirs_world.T).T

    forward = dirs_cam[:, 2] > 1e-8
    if not np.any(forward):
        return _BackprojectMap(
            grid_h, grid_w, erp_h, erp_w,
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
        )

    pts_cam = dirs_cam[forward]
    px_py = camera.img_from_cam(pts_cam)

    cx, cy = fish_size / 2.0, fish_size / 2.0
    radius = fish_size / 2.0
    r = np.sqrt((px_py[:, 0] - cx) ** 2 + (px_py[:, 1] - cy) ** 2)
    in_lens = r < (radius - 0.5)
    in_bounds = (
        in_lens
        & (px_py[:, 0] >= 0) & (px_py[:, 0] < fish_size)
        & (px_py[:, 1] >= 0) & (px_py[:, 1] < fish_size)
    )

    forward_idx = np.where(forward)[0]
    valid_idx = forward_idx[in_bounds]
    fish_px = np.clip(np.round(px_py[in_bounds, 0]).astype(int), 0, fish_size - 1)
    fish_py = np.clip(np.round(px_py[in_bounds, 1]).astype(int), 0, fish_size - 1)

    return _BackprojectMap(grid_h, grid_w, erp_h, erp_w, valid_idx, fish_px, fish_py)


def _direction_angular_spread(directions: list[np.ndarray]) -> float:
    """Max pairwise angle (degrees) among a list of unit directions."""
    if len(directions) < 2:
        return 0.0
    max_angle = 0.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dot = float(np.clip(np.dot(directions[i], directions[j]), -1, 1))
            angle = np.degrees(np.arccos(dot))
            if angle > max_angle:
                max_angle = angle
    return max_angle


# ── Direction computation helpers (Task A2) ──────────────────────


def _compute_detection_com(
    mask: np.ndarray,
) -> tuple[float, float] | None:
    """Compute center-of-mass of a detection mask.

    Returns (cx, cy) in pixel coordinates, or None if mask is empty.
    Ref: FullCircle mask_perspectives.py:138-144.
    """
    M = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if M["m00"] < 1.0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _pixel_com_to_3d_direction(
    cx: float,
    cy: float,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    view_size: int,
    flip_v: bool,
) -> np.ndarray:
    """Convert a pixel CoM in a detection view to a world-space 3D unit direction.

    Inverts the transforms applied by _reframe_to_detection():
    1. Undo fliplr: cx = (view_size - 1) - cx
    2. Undo flipud if flip_v: cy = (view_size - 1) - cy
    3. Compute camera-space ray using the same pinhole math
    4. Rotate to world space using the same rotation matrix

    Ref: FullCircle perspective2omni.py:390-410 (adapted for our conventions).
    """
    # Undo fliplr (always applied by _reframe_to_detection)
    cx_unflipped = (view_size - 1) - cx
    cy_unflipped = cy
    # Undo flipud if flip_v
    if flip_v:
        cy_unflipped = (view_size - 1) - cy

    # Same pinhole math as _reframe_to_detection
    fov_rad = np.radians(fov_deg)
    f = (view_size / 2) / np.tan(fov_rad / 2)
    half = view_size / 2

    crx = (cx_unflipped - half) / f
    cry = -(cy_unflipped - half) / f

    # World direction: w = R[row0]*crx + R[row1]*cry - R[row2]
    # (R rows are [right, up, -forward] in w2c convention)
    R = create_rotation_matrix(yaw_deg, pitch_deg)
    wx = R[0, 0] * crx + R[1, 0] * cry - R[2, 0]
    wy = R[0, 1] * crx + R[1, 1] * cry - R[2, 1]
    wz = R[0, 2] * crx + R[1, 2] * cry - R[2, 2]

    direction = np.array([wx, wy, wz], dtype=np.float64)
    return direction / np.linalg.norm(direction)


def _compute_weighted_person_direction(
    directions_and_weights: list[tuple[np.ndarray, float]],
) -> np.ndarray | None:
    """Weighted average of 3D person directions.

    Each entry is (unit_direction, weight). Weight is typically mask area.
    Returns normalized unit vector, or None if list is empty.
    Ref: FullCircle omni2synthetic.py:92-99.
    """
    if not directions_and_weights:
        return None

    directions = np.array([d for d, _ in directions_and_weights])
    weights = np.array([w for _, w in directions_and_weights])

    if np.all(weights <= 0):
        avg = np.mean(directions, axis=0)
    else:
        avg = np.average(directions, axis=0, weights=weights)

    n = np.linalg.norm(avg)
    if n < 1e-12:
        return None
    return avg / n


def _temporal_fallback_direction(
    frame_idx: int,
    all_directions: list[np.ndarray | None],
) -> np.ndarray | None:
    """Search nearest frames for a valid person direction.

    Returns None only if the entire clip has no valid direction.
    Ref: FullCircle omni2synthetic.py:62-89.
    """
    return _temporal_fallback_direction_with_meta(
        frame_idx,
        [""] * len(all_directions),
        all_directions,
    ).direction


@dataclass
class _ResolvedDirectionInfo:
    """Resolved Pass 2 synthetic-view direction and where it came from."""
    direction: np.ndarray | None
    source: str
    source_index: int | None = None
    source_stem: str | None = None
    peer_index: int | None = None
    peer_stem: str | None = None
    interp_angle_deg: float | None = None


def _temporal_fallback_direction_with_meta(
    frame_idx: int,
    frame_order: list[str],
    all_directions: list[np.ndarray | None],
) -> _ResolvedDirectionInfo:
    """Resolve a direction and record whether it was direct or borrowed."""
    if all_directions[frame_idx] is not None:
        stem = frame_order[frame_idx] if frame_idx < len(frame_order) else None
        return _ResolvedDirectionInfo(
            direction=all_directions[frame_idx],
            source="direct",
            source_index=frame_idx,
            source_stem=stem,
        )

    n = len(all_directions)
    prev_idx: int | None = None
    next_idx: int | None = None
    for j in range(frame_idx - 1, -1, -1):
        if all_directions[j] is not None:
            prev_idx = j
            break
    for j in range(frame_idx + 1, n):
        if all_directions[j] is not None:
            next_idx = j
            break

    # If we have good neighboring directions on both sides and they agree
    # reasonably well, interpolate instead of borrowing one side verbatim.
    if prev_idx is not None and next_idx is not None:
        prev_dir = all_directions[prev_idx]
        next_dir = all_directions[next_idx]
        assert prev_dir is not None and next_dir is not None
        prev_gap = frame_idx - prev_idx
        next_gap = next_idx - frame_idx
        dot = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        if angle <= 25.0 and prev_gap <= 2 and next_gap <= 2:
            prev_w = 1.0 / max(float(prev_gap), 1.0)
            next_w = 1.0 / max(float(next_gap), 1.0)
            mixed = (prev_dir * prev_w) + (next_dir * next_w)
            norm = float(np.linalg.norm(mixed))
            if norm > 1e-8:
                mixed = mixed / norm
                prev_stem = frame_order[prev_idx] if prev_idx < len(frame_order) else None
                next_stem = frame_order[next_idx] if next_idx < len(frame_order) else None
                return _ResolvedDirectionInfo(
                    direction=mixed,
                    source="temporal_interp",
                    source_index=prev_idx,
                    source_stem=prev_stem,
                    peer_index=next_idx,
                    peer_stem=next_stem,
                    interp_angle_deg=angle,
                )

    if prev_idx is not None and next_idx is not None:
        prev_gap = frame_idx - prev_idx
        next_gap = next_idx - frame_idx
        if next_gap < prev_gap:
            stem = frame_order[next_idx] if next_idx < len(frame_order) else None
            return _ResolvedDirectionInfo(
                direction=all_directions[next_idx],
                source="temporal_next",
                source_index=next_idx,
                source_stem=stem,
            )

    if prev_idx is not None:
        stem = frame_order[prev_idx] if prev_idx < len(frame_order) else None
        return _ResolvedDirectionInfo(
            direction=all_directions[prev_idx],
            source="temporal_prev",
            source_index=prev_idx,
            source_stem=stem,
        )
    if next_idx is not None:
        stem = frame_order[next_idx] if next_idx < len(frame_order) else None
        return _ResolvedDirectionInfo(
            direction=all_directions[next_idx],
            source="temporal_next",
            source_index=next_idx,
            source_stem=stem,
        )

    return _ResolvedDirectionInfo(
        direction=None,
        source="none",
        source_index=None,
        source_stem=None,
    )


def _direction_to_yaw_pitch(direction: np.ndarray) -> tuple[float, float]:
    """Convert a 3D unit direction to (yaw_deg, pitch_deg)."""
    dx, dy, dz = direction
    yaw = np.degrees(np.arctan2(dx, dz))
    pitch = np.degrees(np.arcsin(np.clip(dy, -1, 1)))
    return float(yaw), float(pitch)


def _yaw_pitch_to_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Convert yaw/pitch in degrees to a world-space unit direction."""
    yaw_r = np.radians(yaw_deg)
    pitch_r = np.radians(pitch_deg)
    direction = np.array([
        np.cos(pitch_r) * np.sin(yaw_r),
        np.sin(pitch_r),
        np.cos(pitch_r) * np.cos(yaw_r),
    ], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / norm


# ── Dedicated detection layout (FullCircle-style) ────────────────
#
# Pass 1 uses this fixed 16-camera layout for person detection,
# independent of the user's reconstruction preset. Matches FullCircle's
# omni2perspective.py: 8 yaw × 2 pitch bands at ±35°, 90° FOV,
# upper row offset by 22.5° for staggered coverage.
#
# Format: (yaw, pitch, fov, name, flip_v)

DETECTION_LAYOUT: list[tuple[float, float, float, str, bool]] = [
    # Lower band: pitch -35°, 8 cameras at 45° spacing
    (0.0, -35.0, 90.0, "det_00", False),
    (45.0, -35.0, 90.0, "det_01", False),
    (90.0, -35.0, 90.0, "det_02", False),
    (135.0, -35.0, 90.0, "det_03", False),
    (180.0, -35.0, 90.0, "det_04", False),
    (-135.0, -35.0, 90.0, "det_05", False),
    (-90.0, -35.0, 90.0, "det_06", False),
    (-45.0, -35.0, 90.0, "det_07", False),
    # Upper band: pitch +35°, 8 cameras offset 22.5°
    (22.5, 35.0, 90.0, "det_08", False),
    (67.5, 35.0, 90.0, "det_09", False),
    (112.5, 35.0, 90.0, "det_10", False),
    (157.5, 35.0, 90.0, "det_11", False),
    (-157.5, 35.0, 90.0, "det_12", False),
    (-112.5, 35.0, 90.0, "det_13", False),
    (-67.5, 35.0, 90.0, "det_14", False),
    (-22.5, 35.0, 90.0, "det_15", False),
]


def is_masking_available() -> bool:
    """Return True only when the full operator masking stack is ready."""
    from .setup_checks import is_operator_masking_ready

    return is_operator_masking_ready()


@dataclass
class MaskConfig:
    """Configuration for operator masking."""
    targets: list[str] = field(default_factory=lambda: ["person"])
    device: str = "cuda"
    output_size: int = 1920
    backend_preference: str | None = None  # "sam3", "yolo_sam1", or None (default)
    dilate_px: int = 2  # expand detected region by N pixels to catch edge artifacts
    # View list: (yaw, pitch, fov, name, flip_vertical) from ViewConfig.get_all_views()
    views: list[tuple[float, float, float, str, bool]] = field(default_factory=list)
    # Synthetic camera (Pass 2)
    enable_synthetic: bool = True
    synthetic_size: int = 2048
    # Diagnostics
    enable_diagnostics: bool = False


@dataclass
class MaskResult:
    """Result of a masking run."""
    success: bool
    total_frames: int = 0
    masked_frames: int = 0
    masks_dir: str = ""
    diagnostics_path: str = ""
    error: str = ""
    backend_name: str = ""
    video_backend_name: str = ""
    used_fallback_video_backend: bool = False
    video_backend_error: str = ""


def _json_float(value: object, digits: int | None = None) -> float | None:
    """Convert numeric values to JSON-safe Python floats."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return round(out, digits) if digits is not None else out


def _json_int(value: object) -> int:
    """Convert numeric values to JSON-safe Python ints."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ── Diagnostics ────────────────────────────────────────────────


@dataclass
class _DiagFrame:
    """Per-frame diagnostics accumulator."""

    stem: str
    # Pass 1
    p1_views_detected: int = 0
    p1_views_total: int = 0
    p1_max_confidence: float = 0.0
    p1_mean_confidence: float = 0.0
    p1_confidences_by_view: dict[str, float] = field(default_factory=dict)
    p1_coverage_by_view: dict[str, float] = field(default_factory=dict)
    p1_direction_yaw: float | None = None
    p1_direction_pitch: float | None = None
    p1_direction_mode: str = ""
    # Pass 2
    p2_tracked: bool | None = None       # None = Pass 2 didn't run
    p2_mask_pixels: int = 0
    p2_rescued: bool = False
    p2_box_center_dist: float | None = None
    p2_box_confidence: float | None = None
    p2_selection_source: str = ""
    p2_clip_padding: float | None = None
    p2_direction_yaw: float | None = None
    p2_direction_pitch: float | None = None
    p2_direction_source: str = ""
    p2_direction_source_stem: str = ""
    p2_direction_source_offset: int | None = None
    p2_direction_source_peer_stem: str = ""
    p2_direction_source_peer_offset: int | None = None
    p2_direction_interp_angle_deg: float | None = None
    p2_prompt_frame_idx: int | None = None
    p2_prompt_syn_idx: int | None = None
    p2_prompt_stem: str = ""
    p2_prompt_frame_offset: int | None = None
    p2_prompt_mode: str = ""
    p2_prompt_box_confidence: float | None = None
    p2_prompt_box_center_dist: float | None = None
    p2_local_redetect_attempted: bool = False
    p2_local_redetect_candidates: int = 0
    p2_local_redetect_window: list[int] | None = None
    p2_local_redetect_selected_iou: float | None = None
    p2_local_redetect_selected_center_shift: float | None = None
    p2_local_redetect_replaced_propagation: bool = False
    p2_center_redetect_attempted: bool = False
    p2_center_redetect_candidates: int = 0
    p2_center_redetect_window: list[int] | None = None
    p2_center_redetect_selected_center_dist: float | None = None
    p2_center_redetect_selected_continuity_shift: float | None = None
    p2_center_redetect_replaced_selection: bool = False
    p2_alt_view_attempted: bool = False
    p2_alt_view_candidates: int = 0
    p2_alt_view_raw_candidates: int = 0
    p2_alt_view_valid_candidates: int = 0
    p2_alt_view_best_raw_confidence: float | None = None
    p2_alt_view_best_raw_center_dist: float | None = None
    p2_alt_view_best_raw_mask_pixels: int | None = None
    p2_alt_view_best_confidence: float | None = None
    p2_alt_view_best_center_dist: float | None = None
    p2_alt_view_best_mask_pixels: int | None = None
    p2_alt_view_selected_source_stem: str = ""
    p2_alt_view_selected_source_offset: int | None = None
    p2_alt_view_replaced_selection: bool = False
    p2_direction_search_attempted: bool = False
    p2_direction_search_raw_candidates: int = 0
    p2_direction_search_valid_candidates: int = 0
    p2_direction_search_best_raw_confidence: float | None = None
    p2_direction_search_best_raw_center_dist: float | None = None
    p2_direction_search_best_raw_mask_pixels: int | None = None
    p2_direction_search_best_raw_yaw_offset: float | None = None
    p2_direction_search_best_raw_pitch_offset: float | None = None
    p2_direction_search_best_confidence: float | None = None
    p2_direction_search_best_center_dist: float | None = None
    p2_direction_search_best_mask_pixels: int | None = None
    p2_direction_search_best_yaw_offset: float | None = None
    p2_direction_search_best_pitch_offset: float | None = None
    p2_direction_search_replaced_selection: bool = False
    p2_dilation_applied: bool = False
    p2_dilation_kernel: int | None = None
    p2_reprompt_applied: bool = False
    p2_reprompt_gain_pixels: int | None = None
    p2_completeness_applied: bool = False
    p2_completeness_kernel: int | None = None
    p2_propagation_gap: int | None = None
    # Final
    final_mask_pixels: int = 0
    final_source: str = ""  # pass2_tracked, pass2_rescued, pass1_fallback, no_direction, empty

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        d: dict = {"stem": self.stem}
        d["pass1"] = {
            "views_detected": _json_int(self.p1_views_detected),
            "views_total": _json_int(self.p1_views_total),
            "max_confidence": _json_float(self.p1_max_confidence, 4),
            "mean_confidence": _json_float(self.p1_mean_confidence, 4),
            "by_view": {
                name: {
                    "confidence": _json_float(self.p1_confidences_by_view.get(name, 0.0), 4),
                    "coverage_pct": _json_float(self.p1_coverage_by_view.get(name, 0.0), 2),
                }
                for name in self.p1_confidences_by_view
            },
            "direction_yaw": _json_float(self.p1_direction_yaw, 1),
            "direction_pitch": _json_float(self.p1_direction_pitch, 1),
            "direction_mode": self.p1_direction_mode or None,
        }
        d["pass2"] = {
            "tracked": None if self.p2_tracked is None else bool(self.p2_tracked),
            "rescued": bool(self.p2_rescued),
            "mask_pixels": _json_int(self.p2_mask_pixels),
            "box_confidence": _json_float(self.p2_box_confidence, 4),
            "box_center_dist": _json_float(self.p2_box_center_dist, 1),
            "selection_source": self.p2_selection_source or None,
            "clip_padding": _json_float(self.p2_clip_padding, 2),
            "direction_yaw": _json_float(self.p2_direction_yaw, 1),
            "direction_pitch": _json_float(self.p2_direction_pitch, 1),
            "direction_source": self.p2_direction_source or None,
            "direction_source_stem": self.p2_direction_source_stem or None,
            "direction_source_offset": _json_int(self.p2_direction_source_offset),
            "direction_source_peer_stem": self.p2_direction_source_peer_stem or None,
            "direction_source_peer_offset": _json_int(self.p2_direction_source_peer_offset),
            "direction_interp_angle_deg": _json_float(self.p2_direction_interp_angle_deg, 1),
            "prompt_frame_idx": _json_int(self.p2_prompt_frame_idx),
            "prompt_syn_idx": _json_int(self.p2_prompt_syn_idx),
            "prompt_stem": self.p2_prompt_stem or None,
            "prompt_frame_offset": _json_int(self.p2_prompt_frame_offset),
            "prompt_mode": self.p2_prompt_mode or None,
            "prompt_box_confidence": _json_float(self.p2_prompt_box_confidence, 4),
            "prompt_box_center_dist": _json_float(self.p2_prompt_box_center_dist, 1),
            "local_redetect_attempted": bool(self.p2_local_redetect_attempted),
            "local_redetect_candidates": _json_int(self.p2_local_redetect_candidates),
            "local_redetect_window": self.p2_local_redetect_window,
            "local_redetect_selected_iou": _json_float(self.p2_local_redetect_selected_iou, 3),
            "local_redetect_selected_center_shift": _json_float(
                self.p2_local_redetect_selected_center_shift, 1,
            ),
            "local_redetect_replaced_propagation": bool(
                self.p2_local_redetect_replaced_propagation,
            ),
            "center_redetect_attempted": bool(self.p2_center_redetect_attempted),
            "center_redetect_candidates": _json_int(self.p2_center_redetect_candidates),
            "center_redetect_window": self.p2_center_redetect_window,
            "center_redetect_selected_center_dist": _json_float(
                self.p2_center_redetect_selected_center_dist, 1,
            ),
            "center_redetect_selected_continuity_shift": _json_float(
                self.p2_center_redetect_selected_continuity_shift, 1,
            ),
            "center_redetect_replaced_selection": bool(
                self.p2_center_redetect_replaced_selection,
            ),
            "alt_view_attempted": bool(self.p2_alt_view_attempted),
            "alt_view_candidates": _json_int(self.p2_alt_view_candidates),
            "alt_view_raw_candidates": _json_int(self.p2_alt_view_raw_candidates),
            "alt_view_valid_candidates": _json_int(self.p2_alt_view_valid_candidates),
            "alt_view_best_raw_confidence": _json_float(self.p2_alt_view_best_raw_confidence, 4),
            "alt_view_best_raw_center_dist": _json_float(self.p2_alt_view_best_raw_center_dist, 1),
            "alt_view_best_raw_mask_pixels": _json_int(self.p2_alt_view_best_raw_mask_pixels),
            "alt_view_best_confidence": _json_float(self.p2_alt_view_best_confidence, 4),
            "alt_view_best_center_dist": _json_float(self.p2_alt_view_best_center_dist, 1),
            "alt_view_best_mask_pixels": _json_int(self.p2_alt_view_best_mask_pixels),
            "alt_view_selected_source_stem": self.p2_alt_view_selected_source_stem or None,
            "alt_view_selected_source_offset": _json_int(self.p2_alt_view_selected_source_offset),
            "alt_view_replaced_selection": bool(self.p2_alt_view_replaced_selection),
            "direction_search_attempted": bool(self.p2_direction_search_attempted),
            "direction_search_raw_candidates": _json_int(self.p2_direction_search_raw_candidates),
            "direction_search_valid_candidates": _json_int(self.p2_direction_search_valid_candidates),
            "direction_search_best_raw_confidence": _json_float(self.p2_direction_search_best_raw_confidence, 4),
            "direction_search_best_raw_center_dist": _json_float(self.p2_direction_search_best_raw_center_dist, 1),
            "direction_search_best_raw_mask_pixels": _json_int(self.p2_direction_search_best_raw_mask_pixels),
            "direction_search_best_raw_yaw_offset": _json_float(self.p2_direction_search_best_raw_yaw_offset, 1),
            "direction_search_best_raw_pitch_offset": _json_float(self.p2_direction_search_best_raw_pitch_offset, 1),
            "direction_search_best_confidence": _json_float(self.p2_direction_search_best_confidence, 4),
            "direction_search_best_center_dist": _json_float(self.p2_direction_search_best_center_dist, 1),
            "direction_search_best_mask_pixels": _json_int(self.p2_direction_search_best_mask_pixels),
            "direction_search_best_yaw_offset": _json_float(self.p2_direction_search_best_yaw_offset, 1),
            "direction_search_best_pitch_offset": _json_float(self.p2_direction_search_best_pitch_offset, 1),
            "direction_search_replaced_selection": bool(self.p2_direction_search_replaced_selection),
            "dilation_applied": bool(self.p2_dilation_applied),
            "dilation_kernel": _json_int(self.p2_dilation_kernel),
            "reprompt_applied": bool(self.p2_reprompt_applied),
            "reprompt_gain_pixels": _json_int(self.p2_reprompt_gain_pixels),
            "completeness_applied": bool(self.p2_completeness_applied),
            "completeness_kernel": _json_int(self.p2_completeness_kernel),
            "propagation_gap": _json_int(self.p2_propagation_gap),
        }
        d["final"] = {
            "mask_pixels": _json_int(self.final_mask_pixels),
            "source": self.final_source,
        }
        return d


def _write_diagnostics_json(
    path: Path,
    frames: list[_DiagFrame],
    backend_name: str,
    video_backend_name: str,
    used_fallback: bool,
) -> None:
    """Write the diagnostics JSON file."""
    n_with_direction = sum(1 for f in frames if f.p1_direction_yaw is not None)
    n_tracked = sum(1 for f in frames if bool(f.p2_tracked))
    n_rescued = sum(1 for f in frames if f.p2_rescued)
    n_empty = sum(1 for f in frames if f.final_source in ("empty", "no_direction"))

    doc = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "video_backend": video_backend_name,
        "used_fallback": bool(used_fallback),
        "detection_layout_views": _json_int(len(DETECTION_LAYOUT)),
        "total_frames": _json_int(len(frames)),
        "frames_with_direction": _json_int(n_with_direction),
        "frames_tracked": _json_int(n_tracked),
        "frames_rescued": _json_int(n_rescued),
        "frames_empty": _json_int(n_empty),
        "frames": [f.to_dict() for f in frames],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.info("Diagnostics written to %s", path)


def _dilate_detection_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Expand detected regions by px pixels to catch segmentation edge artifacts.

    Operates on detection masks (1=detected, 0=background).
    """
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)
    )
    return cv2.dilate(mask, kernel, iterations=1)


def _postprocess_erp_mask(mask: np.ndarray) -> np.ndarray:
    """Morph close + flood-fill at full ERP resolution.

    Ported from Reconstruction Zone reconstruction_pipeline.py:648-667.
    Bridges gaps between per-view detections and fills holes.
    Mask values: 0/1 uint8 throughout.
    """
    binary = ((mask > 0).astype(np.uint8)) * 255
    h, w = binary.shape[:2]

    close_k = max(15, min(51, int(w * 0.004) | 1))
    close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kern)

    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = closed
    inv = cv2.bitwise_not(padded)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(inv, flood_mask, (0, 0), 0)
    holes = inv[1:-1, 1:-1]
    filled = closed | holes

    return (filled > 0).astype(np.uint8)


def _build_detection_remap(
    fov_deg: float, yaw_deg: float, pitch_deg: float,
    out_size: int, erp_w: int, erp_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (map_x, map_y) remap tables for a detection view.

    Pure geometry — depends only on view params and ERP dimensions,
    not on image content. Tables are float32, ready for cv2.remap.
    """
    fov_rad = np.radians(fov_deg)
    f = (out_size / 2) / np.tan(fov_rad / 2)
    cx = out_size / 2
    cy = out_size / 2

    R = create_rotation_matrix(yaw_deg, pitch_deg)
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    px_arr = np.arange(out_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px_arr, px_arr)
    crx = (px_grid - cx) / f
    cry = -(py_grid - cy) / f

    wx = r00 * crx + r10 * cry - r20
    wy = r01 * crx + r11 * cry - r21
    wz = r02 * crx + r12 * cry - r22

    l = np.sqrt(wx * wx + wy * wy + wz * wz)
    wx, wy, wz = wx / l, wy / l, wz / l

    theta = np.arctan2(wx, wz)
    phi = np.arcsin(np.clip(wy, -1, 1))

    u_eq = ((theta / np.pi + 1) / 2) * erp_w
    v_eq = (0.5 - phi / np.pi) * erp_h

    map_x = u_eq.astype(np.float32) % erp_w
    map_y = np.clip(v_eq.astype(np.float32), 0, erp_h - 1)
    return map_x, map_y


def _apply_detection_remap(
    erp: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, flip_v: bool,
) -> np.ndarray:
    """Apply precomputed remap tables to produce a detection view.

    Just cv2.remap + flips — no geometry computation.
    """
    result = cv2.remap(erp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    result = np.fliplr(result)
    if flip_v:
        result = np.flipud(result)
    return result


def _reframe_to_detection(
    erp: np.ndarray, fov_deg: float, yaw_deg: float, pitch_deg: float,
    out_size: int, flip_v: bool,
) -> np.ndarray:
    """Produce a perspective view for detection using reframer conventions.

    Matches reframe_view() output exactly (same rotation matrix, same
    fliplr) so detection masks align with output images.

    Standalone convenience function — computes remap tables on every call.
    For batch use, prefer _build_detection_remap + _apply_detection_remap.
    """
    h_eq, w_eq = erp.shape[:2]
    map_x, map_y = _build_detection_remap(fov_deg, yaw_deg, pitch_deg, out_size, w_eq, h_eq)
    return _apply_detection_remap(erp, map_x, map_y, flip_v)


def _backproject_mask_to_erp(
    mask: np.ndarray, erp_size: tuple[int, int],
    fov_deg: float, yaw_deg: float, pitch_deg: float,
    flip_v: bool,
) -> np.ndarray:
    """Back-project a pinhole detection mask to ERP space.

    For each ERP pixel, computes which pinhole pixel it maps to and
    samples the detection mask. Uses the same math as the reframer
    (including fliplr) to ensure consistent mapping.

    Args:
        mask: Detection mask (H, W) uint8, 1=detected.
        erp_size: (width, height) of ERP.
        fov_deg: View FOV in degrees.
        yaw_deg: View yaw in degrees.
        pitch_deg: View pitch in degrees.
        flip_v: Whether this view uses vertical flip.

    Returns:
        ERP-sized mask (erp_h, erp_w) uint8, 1=detected in this view.
    """
    erp_w, erp_h = erp_size
    det_size = mask.shape[0]  # square

    # Undo flip_v on the mask before back-projecting
    if flip_v:
        mask = np.flipud(mask)
    # Undo fliplr (reframer applies it after sampling)
    mask = np.fliplr(mask)

    fov_rad = np.radians(fov_deg)
    f = (det_size / 2) / np.tan(fov_rad / 2)
    cx = det_size / 2
    cy = det_size / 2

    R = create_rotation_matrix(yaw_deg, pitch_deg)

    # For each ERP pixel, compute world ray direction
    u_eq = np.arange(erp_w, dtype=np.float64)
    v_eq = np.arange(erp_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_eq, v_eq)

    theta = ((uu / erp_w) * 2 - 1) * np.pi  # longitude
    phi = (0.5 - vv / erp_h) * np.pi          # latitude

    # World ray directions
    wx = np.cos(phi) * np.sin(theta)
    wy = np.sin(phi)
    wz = np.cos(phi) * np.cos(theta)

    # Transform to camera space: cam_ray = R @ world_ray
    cx_ray = R[0, 0] * wx + R[0, 1] * wy + R[0, 2] * wz
    cy_ray = R[1, 0] * wx + R[1, 1] * wy + R[1, 2] * wz
    cz_ray = R[2, 0] * wx + R[2, 1] * wy + R[2, 2] * wz

    # Only pixels in front of the camera (cz < 0 because -forward convention)
    # The reframer uses -r20 in its ray computation, meaning the camera
    # looks along -z in camera space
    valid = cz_ray < 0

    # Project to pinhole pixel coordinates
    # crx = (px - cx) / f, cry = -(py - cy) / f
    # So: px = crx * f + cx, py = -cry * f + cy
    # And: crx = cx_ray / (-cz_ray), cry = cy_ray / (-cz_ray)
    erp_mask = np.zeros((erp_h, erp_w), dtype=np.uint8)

    with np.errstate(divide='ignore', invalid='ignore'):
        crx = cx_ray / (-cz_ray)
        cry = cy_ray / (-cz_ray)

    px = crx * f + cx
    py = -cry * f + cy

    # Check which ERP pixels map to valid pinhole pixels
    in_bounds = valid & (px >= 0) & (px < det_size) & (py >= 0) & (py < det_size)

    if np.any(in_bounds):
        px_int = np.clip(np.round(px[in_bounds]).astype(int), 0, det_size - 1)
        py_int = np.clip(np.round(py[in_bounds]).astype(int), 0, det_size - 1)
        erp_mask[in_bounds] = mask[py_int, px_int]

    return erp_mask


class Masker:
    """Two-pass operator masking with ERP OR-merge and synthetic camera.

    Pass 1: Per-view detection on preset views + center-of-mass recording.
    Pass 2: Synthetic fisheye camera aimed at person + video tracking/fallback.
    Pass 3: Postprocess and save merged ERP masks.
    """

    def __init__(self, config: MaskConfig | None = None) -> None:
        self.config = config or MaskConfig()
        self._backend: MaskingBackend | None = None
        self._video_backend: VideoTrackingBackend | None = None
        self._backend_name: str = ""
        self._video_backend_name: str = ""
        self._used_fallback_video_backend: bool = False
        self._video_backend_error: str = ""
        # Cached remap tables for DETECTION_LAYOUT — keyed on
        # (detection_size, erp_w, erp_h). Built once on first frame.
        self._detection_remap_cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._detection_remap_key: tuple[int, int, int] | None = None

    def initialize(self) -> None:
        """Load the user's chosen detection backend and video backend."""
        from .backends import HAS_SAM2, get_video_backend

        self._backend = get_backend(self.config.backend_preference)
        if self._backend is None:
            raise ImportError(
                "No masking backend available. "
                "Install ultralytics + segment-anything for default tier."
            )
        self._backend.initialize()
        self._backend_name = type(self._backend).__name__
        self._used_fallback_video_backend = False
        self._video_backend_error = ""
        self._video_backend_name = ""

        if self.config.enable_synthetic:
            if not HAS_SAM2:
                raise ImportError(
                    "SAM v2 video tracking is required for operator masking. "
                    "Install the full masking stack before enabling masking."
                )
            self._video_backend = get_video_backend(
                preference=self.config.backend_preference,
                fallback_image_backend=None,
                targets=self.config.targets,
            )
            if self._video_backend is not None:
                self._video_backend_name = type(self._video_backend).__name__
                self._used_fallback_video_backend = (
                    self._video_backend_name == "FallbackVideoBackend"
                )
                logger.debug("Pass 2 video backend: %s", self._video_backend_name)
                print(f"[360] Pass 2 backend: {self._video_backend_name}")
                self._video_backend.initialize()

    def cleanup(self) -> None:
        if self._video_backend is not None:
            self._video_backend.cleanup()
            self._video_backend = None
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None

    # ── Cubemap direct masking (public API) ──────────────────────

    def estimate_person_directions(
        self,
        frames_dir: str | Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, np.ndarray | None]:
        """Run Pass 1 direction estimation on ERP frames.

        Returns a dict mapping frame_stem → person direction (unit vector)
        or None if no person was detected in that frame.
        Uses the 16-camera DETECTION_LAYOUT with YOLO-only bounding boxes.
        """
        if self._backend is None:
            raise RuntimeError("Masker not initialized")

        frames_path = Path(frames_dir)
        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            return {}

        n_frames = len(frame_files)
        detection_size: int | None = None
        timer = _SubstageTimer()
        directions: dict[str, np.ndarray | None] = {}

        for fi, frame_file in enumerate(frame_files):
            with timer.time("cubemap_p1_imread"):
                erp = cv2.imread(str(frame_file))
            if erp is None:
                continue

            erp_h, erp_w = erp.shape[:2]
            if detection_size is None:
                detection_size = min(1024, erp_w // 4)

            with timer.time("cubemap_p1_direction"):
                _, direction, n_det = self._primary_detection(erp, detection_size, timer=timer)

            stem = frame_file.stem
            directions[stem] = direction

            if direction is not None:
                yaw_d, pitch_d = _direction_to_yaw_pitch(direction)
                print(f"[360] Pass 1: frame {fi+1}/{n_frames} — "
                      f"{n_det}/{len(DETECTION_LAYOUT)} views, "
                      f"direction yaw={yaw_d:.0f}° pitch={pitch_d:.0f}°")
            else:
                print(f"[360] Pass 1: frame {fi+1}/{n_frames} — "
                      f"{n_det}/{len(DETECTION_LAYOUT)} views, no direction")

            if progress_callback:
                progress_callback(fi + 1, n_frames, f"Pass 1: frame {fi+1}/{n_frames}")

        n_detected = sum(1 for d in directions.values() if d is not None)
        print(f"[360] Pass 1 complete: {n_detected}/{n_frames} frames with person direction")
        timer.report()
        return directions

    @staticmethod
    def _cubemap_face_visible(
        face_yaw: float, face_pitch: float, face_fov: float,
        person_dir: np.ndarray,
        margin_deg: float = 10.0,
    ) -> bool:
        """Check if a person direction falls within a cubemap face's FOV.

        Uses the half-diagonal angle of the square FOV for the gate,
        plus a configurable margin to avoid clipping at corners.
        """
        # Face center direction
        yaw_r = np.radians(face_yaw)
        pitch_r = np.radians(face_pitch)
        face_dir = np.array([
            np.cos(pitch_r) * np.sin(yaw_r),
            np.sin(pitch_r),
            np.cos(pitch_r) * np.cos(yaw_r),
        ])

        cos_angle = np.clip(np.dot(face_dir, person_dir), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Half-diagonal of a square FOV
        half_diag_deg = np.degrees(
            np.arctan(np.sqrt(2) * np.tan(np.radians(face_fov / 2)))
        )

        return angle_deg < half_diag_deg + margin_deg

    def process_reframed_views(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        views: list[tuple[float, float, float, str, bool]],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> MaskResult:
        """Direct per-view masking on already-reframed cubemap images.

        Runs YOLO+SAM v1 detect_and_segment on every cubemap view image
        and writes final COLMAP-polarity masks directly to
        masks_dir/{view_name}/{frame_stem}.png.

        Args:
            images_dir: Directory containing reframed images in
                {view_name}/{frame_stem}.jpg layout.
            masks_dir: Output directory for per-view masks.
            views: List of (yaw, pitch, fov, view_name, flip_v) from the preset.
            progress_callback: Optional (current, total, message) callback.
        """
        if self._backend is None:
            return MaskResult(success=False, error="Not initialized")

        cfg = self.config
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        masks_path.mkdir(parents=True, exist_ok=True)
        diagnostics_path = ""

        timer = _SubstageTimer()
        total_images = 0
        masked_count = 0

        # Collect all view images
        view_images: list[tuple[str, str, Path]] = []  # (view_name, frame_stem, path)
        for yaw, pitch, fov, view_name, flip_v in views:
            view_dir = images_path / view_name
            if not view_dir.is_dir():
                continue
            for img_path in sorted(view_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg"):
                    view_images.append((view_name, img_path.stem, img_path))

        total_images = len(view_images)
        if total_images == 0:
            return MaskResult(success=False, error=f"No images found in {images_dir}")

        # Per-view erosion kernel (matches reframer behavior)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        unmasked_count = 0

        # Cubemap diagnostics: per-frame → per-view records
        cubemap_diag: dict[str, dict[str, dict]] = {} if cfg.enable_diagnostics else {}

        for idx, (view_name, frame_stem, img_path) in enumerate(view_images):
            mask_view_dir = masks_path / view_name
            mask_view_dir.mkdir(parents=True, exist_ok=True)
            mask_out_path = mask_view_dir / f"{frame_stem}.png"

            with timer.time("cubemap_imread"):
                img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            with timer.time("cubemap_segment"):
                detection_mask = self._backend.detect_and_segment(
                    img,
                    cfg.targets,
                    detection_confidence=CUBEMAP_DIRECT_CONFIDENCE,
                    single_primary_box=True,
                )

            detected = detection_mask.sum() > 0
            mask_pixels = int(detection_mask.sum()) if detected else 0

            # Capture per-view box confidence if diagnostics enabled
            view_box_conf: float | None = None
            if cfg.enable_diagnostics and detected:
                diag_boxes = self._backend.batch_detect_boxes(
                    [img], detection_confidence=CUBEMAP_DIRECT_CONFIDENCE,
                )
                if diag_boxes and diag_boxes[0]:
                    view_box_conf = max(c for _, c in diag_boxes[0])

            # Convert: detection (1=detected) → COLMAP polarity (255=keep, 0=remove)
            if detected:
                keep_mask = ((detection_mask == 0).astype(np.uint8)) * 255
                # Per-view erosion (erode keep region = dilate remove region)
                keep_mask = cv2.erode(keep_mask, erode_kernel, iterations=1)
                masked_count += 1
            else:
                keep_mask = np.full((h, w), 255, dtype=np.uint8)
                unmasked_count += 1

            with timer.time("cubemap_write_mask"):
                cv2.imwrite(str(mask_out_path), keep_mask)

            # Record cubemap diagnostics
            if cfg.enable_diagnostics:
                if frame_stem not in cubemap_diag:
                    cubemap_diag[frame_stem] = {}
                cubemap_diag[frame_stem][view_name] = {
                    "detected": bool(detected),
                    "mask_pixels": _json_int(mask_pixels),
                    "confidence": _json_float(view_box_conf, 4),
                }

            if progress_callback:
                progress_callback(
                    idx + 1, total_images,
                    f"Cubemap masking: {idx+1}/{total_images}",
                )

            if (idx + 1) % len(views) == 0:
                frame_num = (idx + 1) // len(views)
                total_frames = total_images // max(len(views), 1)
                print(f"[360] Cubemap masking: frame {frame_num}/{total_frames} "
                      f"({masked_count} views with detections, {unmasked_count} without)")

        print(f"[360] Cubemap masking complete: {masked_count} views with detections, "
              f"{unmasked_count} views without detections")
        timer.report()

        # Write cubemap diagnostics JSON
        if cfg.enable_diagnostics and cubemap_diag:
            try:
                n_views = len(views)
                doc = {
                    "version": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": "cubemap",
                    "backend": self._backend_name,
                    "views_per_frame": _json_int(n_views),
                    "total_frames": _json_int(len(cubemap_diag)),
                    "frames": [
                        {
                            "stem": stem,
                            "views_detected": _json_int(
                                sum(1 for v in per_view.values() if v["detected"])
                            ),
                            "views_total": _json_int(len(per_view)),
                            "by_view": per_view,
                        }
                        for stem, per_view in sorted(cubemap_diag.items())
                    ],
                }
                diag_path = masks_path / "masking_diagnostics.json"
                diag_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
                logger.info("Cubemap diagnostics written to %s", diag_path)
                diagnostics_path = str(diag_path)
                if progress_callback:
                    progress_callback(
                        total_images,
                        total_images,
                        f"Mask diagnostics ready: {diagnostics_path}",
                    )
            except Exception as diag_exc:
                logger.warning("Failed to write cubemap diagnostics: %s", diag_exc)
                print(f"[360] Diagnostics write failed: {diag_exc}")

        total_frames = total_images // max(len(views), 1)
        return MaskResult(
            success=True,
            total_frames=total_frames,
            masked_frames=masked_count,
            masks_dir=str(masks_path),
            diagnostics_path=diagnostics_path,
            backend_name=self._backend_name,
        )

    def _primary_detection(
        self, erp: np.ndarray, detection_size: int,
        timer: _SubstageTimer | None = None,
        diag: _DiagFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Pass 1: YOLO-only person localization on dedicated FullCircle layout.

        Uses DETECTION_LAYOUT (16 cameras, 8 yaw × 2 pitch at ±35°, 90° FOV).
        Runs YOLO bounding box detection only — no SAM v1 segmentation.
        Pass 1's only job is to estimate the person's direction on the sphere.
        Pass 2 (synthetic camera + SAM v2 tracking) produces the actual masks.

        Returns:
            (empty_erp_mask, weighted_person_direction_or_None, n_detections)
            The ERP mask is always empty — Pass 2 is authoritative for mask shape.
        """
        erp_h, erp_w = erp.shape[:2]
        directions_and_weights: list[tuple[np.ndarray, float]] = []
        direction_mode = ""

        detection_views = DETECTION_LAYOUT
        n_views = len(detection_views)
        n_detections = 0

        # Build or reuse cached remap tables for DETECTION_LAYOUT.
        # Tables depend only on view geometry + ERP dimensions — not frame content.
        cache_key = (detection_size, erp_w, erp_h)
        if self._detection_remap_key != cache_key:
            self._detection_remap_cache = [
                _build_detection_remap(fov, yaw, pitch, detection_size, erp_w, erp_h)
                for yaw, pitch, fov, _name, _fv in detection_views
            ]
            self._detection_remap_key = cache_key

        # Build all 16 detection images from cached remap tables (BGR)
        detection_images: list[np.ndarray] = []
        for vi in range(n_views):
            map_x, map_y = self._detection_remap_cache[vi]
            flip_v = detection_views[vi][4]
            face_img = _apply_detection_remap(erp, map_x, map_y, flip_v)
            detection_images.append(face_img)

        # Batched detection via backend protocol
        batch_detections = self._backend.batch_detect_boxes(
            detection_images, detection_confidence=0.35,
        )

        # Parse results per view and accumulate strong direction candidates.
        all_confidences: list[float] = []
        strong_direction_candidates: list[tuple[np.ndarray, float]] = []
        MIN_DIRECTION_COVERAGE_PCT = 5.0
        LARGE_BOX_FILTER_COVERAGE_PCT = 35.0
        LARGE_BOX_FILTER_MIN_CONF = 0.70
        for vi, (yaw, pitch, fov, view_name, flip_v) in enumerate(detection_views):
            raw_detections = batch_detections[vi]
            all_boxes = [box for box, _conf in raw_detections]
            view_max_conf = max((conf for _, conf in raw_detections), default=0.0)

            if diag is not None:
                diag.p1_confidences_by_view[view_name] = view_max_conf

            if not all_boxes:
                if diag is not None:
                    diag.p1_coverage_by_view[view_name] = 0.0
                continue

            total_px = float(detection_size * detection_size)
            filtered_detections: list[tuple[np.ndarray, float]] = []
            for box, conf in raw_detections:
                x1b, y1b, x2b, y2b = box
                area_px = max(0.0, float(x2b - x1b)) * max(0.0, float(y2b - y1b))
                area_pct = 100.0 * area_px / total_px if total_px > 0 else 0.0
                if area_pct >= LARGE_BOX_FILTER_COVERAGE_PCT and conf < LARGE_BOX_FILTER_MIN_CONF:
                    continue
                filtered_detections.append((box, conf))

            # Prevent giant low-confidence false positives from dominating the
            # Pass 1 direction estimate. If everything filters out, fall back to
            # the single highest-confidence detection instead of unioning all boxes.
            usable_detections = filtered_detections
            if not usable_detections and raw_detections:
                usable_detections = [max(raw_detections, key=lambda item: float(item[1]))]

            # Union bounding box of the usable detections for direction estimation.
            boxes = np.array([box for box, _conf in usable_detections])
            x1 = boxes[:, 0].min()
            y1 = boxes[:, 1].min()
            x2 = boxes[:, 2].max()
            y2 = boxes[:, 3].max()
            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0
            box_area = float((x2 - x1) * (y2 - y1))
            coverage_pct = 100.0 * box_area / total_px if total_px > 0 else 0.0

            n_detections += 1
            all_confidences.append(view_max_conf)

            if diag is not None:
                diag.p1_coverage_by_view[view_name] = coverage_pct

            direction = _pixel_com_to_3d_direction(
                box_cx, box_cy, fov, yaw, pitch, detection_size, flip_v,
            )
            if coverage_pct >= MIN_DIRECTION_COVERAGE_PCT:
                strong_direction_candidates.append(
                    (direction, box_area * max(0.25, max((conf for _box, conf in usable_detections), default=0.0)))
                )

        if strong_direction_candidates:
            directions_and_weights = strong_direction_candidates
            direction_mode = "strong"

        person_direction = _compute_weighted_person_direction(
            directions_and_weights
        )

        if diag is not None:
            diag.p1_views_detected = n_detections
            diag.p1_views_total = n_views
            diag.p1_max_confidence = max(all_confidences) if all_confidences else 0.0
            diag.p1_mean_confidence = (
                sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            )
            if person_direction is not None:
                yaw_d, pitch_d = _direction_to_yaw_pitch(person_direction)
                diag.p1_direction_yaw = yaw_d
                diag.p1_direction_pitch = pitch_d
                diag.p1_direction_mode = direction_mode

        # Return empty ERP mask — Pass 2 is authoritative for mask shape
        empty_mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
        return empty_mask, person_direction, n_detections

    def _synthetic_pass(
        self,
        frame_files: list[Path],
        primary_masks: dict[str, np.ndarray],
        person_directions: dict[str, np.ndarray | None],
        detection_counts: dict[str, int],
        frame_order: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
        timer: _SubstageTimer | None = None,
        diag_frames: dict[str, _DiagFrame] | None = None,
    ) -> dict[str, np.ndarray]:
        """Pass 2: Synthetic fisheye camera + video tracking/fallback detection.

        Renders synthetic fisheye views aimed at the detected person, runs
        the currently selected video backend, and backprojects results to ERP.
        Does NOT own backend recovery — process_frames() handles that.

        Returns:
            dict mapping frame_stem → synthetic ERP mask (uint8, 1=detected).
        """
        cfg = self.config
        syn_size = cfg.synthetic_size
        camera = _create_synthetic_camera(syn_size)
        n_total = len(frame_order)

        # Resolve directions with temporal fallback
        all_dirs = [person_directions.get(stem) for stem in frame_order]
        resolved_info: list[_ResolvedDirectionInfo] = []
        resolved_dirs: list[np.ndarray | None] = []
        for i in range(len(frame_order)):
            info = _temporal_fallback_direction_with_meta(i, frame_order, all_dirs)
            resolved_info.append(info)
            resolved_dirs.append(info.direction)

        # Find best prompt frame: the frame with the most Pass 1 detections.
        # More detections = stronger direction = person better centered
        # in the synthetic fisheye = SAM2 center-click more likely to
        # land on the person.
        best_idx = 0
        best_count = 0
        for i, stem in enumerate(frame_order):
            count = detection_counts.get(stem, 0)
            if count > best_count:
                best_count = count
                best_idx = i

        # Render synthetic fisheye views for all frames with valid directions
        synthetic_frames: list[np.ndarray] = []
        valid_frame_indices: list[int] = []
        stem_to_syn_idx: dict[str, int] = {}

        for i, stem in enumerate(frame_order):
            d = resolved_dirs[i]
            if d is None:
                continue

            erp = cv2.imread(str(frame_files[i]))
            if erp is None:
                continue

            R = _look_at_rotation(d)
            with timer.time("p2_render_fisheye") if timer else contextmanager(lambda: (yield))():
                fisheye = _render_synthetic_fisheye(erp, camera, R)
            synthetic_frames.append(fisheye)
            stem_to_syn_idx[stem] = len(synthetic_frames) - 1
            valid_frame_indices.append(i)
            print(f"[360] Pass 2: rendering synthetic view {len(synthetic_frames)}/{n_total}")

            if progress_callback:
                progress_callback(
                    i + 1, n_total,
                    f"Pass 2 — rendering synthetic views: {i+1}/{n_total}",
                )

        if not synthetic_frames:
            logger.debug("Pass 2: no synthetic frames to process (no valid directions)")
            return {}

        def _detect_synthetic_candidate(
            syn_frame: np.ndarray,
        ) -> tuple[np.ndarray | None, float | None, float | None, np.ndarray | None]:
            """Detect and segment a person on a synthetic fisheye frame."""
            if self._backend is None:
                return None, None, None, None

            syn_mask = self._backend.detect_and_segment(
                syn_frame,
                cfg.targets,
                detection_confidence=ALT_VIEW_RESCUE_CONFIDENCE,
                single_primary_box=True,
                primary_box_mode=SYNTHETIC_PRIMARY_BOX_MODE,
                constrain_to_primary_box=SYNTHETIC_PRIMARY_BOX_CONSTRAIN,
                primary_box_padding=SYNTHETIC_PRIMARY_BOX_PADDING,
            )
            if syn_mask is None or syn_mask.sum() <= 0:
                return None, None, None, None

            boxes_per_frame = self._backend.batch_detect_boxes(
                [syn_frame],
                detection_confidence=ALT_VIEW_RESCUE_CONFIDENCE,
            )
            if not boxes_per_frame or not boxes_per_frame[0]:
                return syn_mask, None, None, None

            boxes = [box for box, _ in boxes_per_frame[0]]
            confs = [conf for _, conf in boxes_per_frame[0]]
            selected = select_primary_person_box(
                boxes,
                confs,
                syn_frame.shape,
                mode=SYNTHETIC_PRIMARY_BOX_MODE,
            )
            if selected is None:
                return syn_mask, None, None, None

            best_box, best_conf = selected
            bx = (float(best_box[0]) + float(best_box[2])) / 2.0
            by = (float(best_box[1]) + float(best_box[3])) / 2.0
            img_cx = syn_frame.shape[1] / 2.0
            img_cy = syn_frame.shape[0] / 2.0
            center_dist = float(np.hypot(bx - img_cx, by - img_cy))
            return syn_mask, float(best_conf), center_dist, best_box

        def _apply_post_backend_completeness(
            syn_mask: np.ndarray,
            selected_box: np.ndarray | None,
            selection_source: str,
            selection_conf: float | None,
            selection_center_dist: float | None,
            syn_frame_shape: tuple[int, int, int] | tuple[int, int],
        ) -> tuple[np.ndarray, bool, int | None]:
            """Apply the fallback backend's trusted completeness recovery to post-backend replacements."""
            if (
                selected_box is None
                or not hasattr(self._video_backend, "_should_apply_completeness_recovery")
                or not hasattr(self._video_backend, "_apply_completeness_recovery")
            ):
                return syn_mask, False, None
            try:
                should_apply = bool(self._video_backend._should_apply_completeness_recovery(  # type: ignore[attr-defined]
                    selection_source,
                    selection_conf,
                    selection_center_dist,
                ))
                if not should_apply:
                    return syn_mask, False, None
                improved = self._video_backend._apply_completeness_recovery(  # type: ignore[attr-defined]
                    syn_mask,
                    selected_box,
                    syn_frame_shape,
                    SYNTHETIC_PRIMARY_BOX_PADDING,
                )
                kernel = int(getattr(self._video_backend, "_completeness_kernel", 0) or 0)
                return improved, True, (kernel if kernel > 0 else None)
            except Exception:
                return syn_mask, False, None

        def _apply_post_backend_reprompt(
            syn_frame: np.ndarray,
            syn_mask: np.ndarray,
            selected_box: np.ndarray | None,
            selection_source: str,
            selection_conf: float | None,
            selection_center_dist: float | None,
        ) -> tuple[np.ndarray, bool, int | None]:
            """Apply the fallback backend's trusted re-prompt recovery to post-backend replacements."""
            if (
                selected_box is None
                or not hasattr(self._video_backend, "_should_apply_reprompt_recovery")
                or not hasattr(self._video_backend, "_apply_reprompt_recovery")
            ):
                return syn_mask, False, None
            try:
                should_apply = bool(self._video_backend._should_apply_reprompt_recovery(  # type: ignore[attr-defined]
                    selection_source,
                    selection_conf,
                    selection_center_dist,
                ))
                if not should_apply:
                    return syn_mask, False, None
                improved, gain_pixels = self._video_backend._apply_reprompt_recovery(  # type: ignore[attr-defined]
                    syn_frame,
                    syn_mask,
                    selected_box,
                )
                gain = int(gain_pixels or 0)
                return improved, (gain > 0), (gain if gain > 0 else None)
            except Exception:
                return syn_mask, False, None

        def _alt_view_should_replace(
            current_selection_source: str,
            current_mask: np.ndarray | None,
            alt_conf: float | None,
            alt_center_dist: float | None,
            alt_mask_pixels: int,
        ) -> bool:
            """Decide whether an alternate direct-direction synthetic view should win."""
            if alt_mask_pixels < ALT_VIEW_RESCUE_MIN_MASK_PIXELS:
                return False
            if alt_center_dist is None or alt_center_dist > ALT_VIEW_RESCUE_MAX_CENTER_DIST:
                return False
            if alt_conf is not None and alt_conf < ALT_VIEW_RESCUE_MIN_CONFIDENCE:
                return False

            current_pixels = int(current_mask.sum()) if current_mask is not None else 0
            current_nonempty = current_pixels > 0

            if current_selection_source == "none" or not current_nonempty:
                return True

            if current_selection_source.startswith("propagated"):
                required_pixels = max(
                    ALT_VIEW_RESCUE_MIN_MASK_PIXELS,
                    int(math.ceil(current_pixels * ALT_VIEW_RESCUE_PROPAGATED_GAIN)),
                    current_pixels + ALT_VIEW_RESCUE_PROPAGATED_MIN_GAIN,
                )
                if alt_mask_pixels >= required_pixels:
                    return True
                if (
                    alt_conf is not None
                    and alt_conf >= ALT_VIEW_RESCUE_PROPAGATED_OVERRIDE_CONFIDENCE
                    and alt_mask_pixels >= max(
                        ALT_VIEW_RESCUE_PROPAGATED_OVERRIDE_PIXELS,
                        int(math.ceil(current_pixels * 0.60)),
                    )
                    and alt_center_dist <= (ALT_VIEW_RESCUE_MAX_CENTER_DIST * 0.75)
                ):
                    return True

            return False

        def _direction_search_continuity_factor(yaw_off: float, pitch_off: float) -> float:
            yaw_norm = abs(float(yaw_off)) / max(18.0, max(abs(v) for v in DIRECTION_SEARCH_YAW_OFFSETS))
            pitch_norm = abs(float(pitch_off)) / max(10.0, max(abs(v) for v in DIRECTION_SEARCH_PITCH_OFFSETS))
            return max(0.0, 1.0 - min(1.0, math.hypot(yaw_norm, pitch_norm) / 1.5))

        def _direction_search_candidate_score(
            conf: float | None,
            center_dist: float | None,
            mask_pixels: int,
            yaw_off: float,
            pitch_off: float,
        ) -> float:
            if mask_pixels <= 0 or center_dist is None:
                return 0.0
            continuity = _direction_search_continuity_factor(yaw_off, pitch_off)
            return (
                math.sqrt(float(mask_pixels))
                * (1.0 + 0.35 * max(0.0, float(conf or 0.0)))
                * (1.0 + 0.20 * continuity)
                / (1.0 + max(0.0, float(center_dist)) / 450.0)
            )

        def _direction_search_is_valid_candidate(
            conf: float | None,
            center_dist: float | None,
            mask_pixels: int,
            yaw_off: float,
            pitch_off: float,
        ) -> bool:
            if center_dist is None or mask_pixels < DIRECTION_SEARCH_MIN_MASK_PIXELS:
                return False
            if conf is not None and conf < DIRECTION_SEARCH_MIN_CONFIDENCE:
                return False
            if center_dist <= DIRECTION_SEARCH_MAX_CENTER_DIST:
                return True
            continuity = _direction_search_continuity_factor(yaw_off, pitch_off)
            return (
                center_dist <= DIRECTION_SEARCH_EXTENDED_CENTER_DIST
                and mask_pixels >= DIRECTION_SEARCH_EXTENDED_MIN_MASK_PIXELS
                and float(conf or 0.0) >= DIRECTION_SEARCH_EXTENDED_MIN_CONFIDENCE
                and continuity >= 0.18
            )

        def _direction_search_should_replace(
            current_selection_source: str,
            current_mask: np.ndarray | None,
            current_center_dist: float | None,
            cand_conf: float | None,
            cand_center_dist: float | None,
            cand_mask_pixels: int,
            cand_yaw_off: float,
            cand_pitch_off: float,
        ) -> bool:
            """Decide whether a local direction-search candidate should win."""
            if not _direction_search_is_valid_candidate(
                cand_conf,
                cand_center_dist,
                cand_mask_pixels,
                cand_yaw_off,
                cand_pitch_off,
            ):
                return False

            current_pixels = int(current_mask.sum()) if current_mask is not None else 0
            current_nonempty = current_pixels > 0
            cand_score = _direction_search_candidate_score(
                cand_conf,
                cand_center_dist,
                cand_mask_pixels,
                cand_yaw_off,
                cand_pitch_off,
            )

            if current_selection_source == "none" or not current_nonempty:
                return cand_score >= 32.0

            if current_selection_source.startswith("propagated"):
                current_dist = (
                    DIRECTION_SEARCH_EXTENDED_CENTER_DIST
                    if current_center_dist is None
                    else float(current_center_dist)
                )
                current_score = math.sqrt(float(current_pixels)) / (
                    1.0 + max(0.0, current_dist) / 450.0
                )
                current_score *= 0.80  # propagated results are less trustworthy than fresh current-frame candidates
                required_score = max(
                    current_score * DIRECTION_SEARCH_PROPAGATED_SCORE_GAIN,
                    current_score + DIRECTION_SEARCH_PROPAGATED_SCORE_MARGIN,
                )
                if cand_score >= required_score:
                    return True

                required_pixels = max(
                    DIRECTION_SEARCH_MIN_MASK_PIXELS,
                    int(math.ceil(current_pixels * DIRECTION_SEARCH_PROPAGATED_GAIN)),
                    current_pixels + DIRECTION_SEARCH_PROPAGATED_MIN_GAIN,
                )
                if cand_mask_pixels >= required_pixels and cand_score >= max(current_score, 28.0):
                    return True

                if (
                    cand_conf is not None
                    and cand_conf >= max(DIRECTION_SEARCH_MIN_CONFIDENCE + 0.06, 0.28)
                    and cand_mask_pixels >= max(
                        DIRECTION_SEARCH_MIN_MASK_PIXELS,
                        int(math.ceil(current_pixels * 0.90)),
                    )
                    and cand_score >= max(current_score * 0.95, 30.0)
                ):
                    return True

            return False

        # Default prompt seed from Pass 1 strongest frame. This is only the
        # starting hint now; the actual SAM2 prompt frame is chosen from the
        # rendered synthetic views based on real current-frame detections.
        best_stem = frame_order[best_idx]
        initial_frame_idx = stem_to_syn_idx.get(best_stem, 0)
        initial_prompt_box: np.ndarray | None = None
        prompt_mode = "center_point"
        prompt_box_confidence: float | None = None
        prompt_box_center_dist: float | None = None

        prompt_candidates: list[tuple[tuple[float, float, int, int], int, str, int, float, float, np.ndarray]] = []

        if self._backend is not None:
            try:
                prompt_detections = self._backend.batch_detect_boxes(
                    synthetic_frames,
                    detection_confidence=SAM2_PROMPT_BOX_CONFIDENCE,
                )
                best_prompt_candidate: tuple[float, float, int, int, np.ndarray] | None = None
                best_prompt_meta: tuple[int, str, int, float, float] | None = None
                for syn_idx, detections in enumerate(prompt_detections):
                    if not detections:
                        continue
                    prompt_boxes = [box for box, _ in detections]
                    prompt_confs = [conf for _, conf in detections]
                    selected_prompt = select_primary_person_box(
                        prompt_boxes,
                        prompt_confs,
                        synthetic_frames[syn_idx].shape,
                        mode=SYNTHETIC_PRIMARY_BOX_MODE,
                    )
                    if selected_prompt is None:
                        continue
                    prompt_box, prompt_conf = selected_prompt
                    bx = (float(prompt_box[0]) + float(prompt_box[2])) / 2.0
                    by = (float(prompt_box[1]) + float(prompt_box[3])) / 2.0
                    img_cx = synthetic_frames[syn_idx].shape[1] / 2.0
                    img_cy = synthetic_frames[syn_idx].shape[0] / 2.0
                    center_dist = float(np.hypot(bx - img_cx, by - img_cy))
                    frame_idx = valid_frame_indices[syn_idx]
                    stem = frame_order[frame_idx]
                    det_count = int(detection_counts.get(stem, 0))
                    candidate_key = (center_dist, -float(prompt_conf), -det_count, abs(frame_idx - best_idx))
                    prompt_candidates.append((
                        candidate_key,
                        syn_idx,
                        stem,
                        frame_idx,
                        float(prompt_conf),
                        center_dist,
                        prompt_box.copy(),
                    ))
                    if best_prompt_candidate is None or candidate_key < best_prompt_candidate:
                        best_prompt_candidate = candidate_key
                        best_prompt_meta = (
                            syn_idx,
                            stem,
                            frame_idx,
                            float(prompt_conf),
                            center_dist,
                        )
                        initial_prompt_box = prompt_box

                if best_prompt_meta is not None and initial_prompt_box is not None:
                    initial_frame_idx, best_stem, best_idx, prompt_box_confidence, prompt_box_center_dist = best_prompt_meta
                    prompt_mode = "box"
                    best_count = int(detection_counts.get(best_stem, best_count))
            except Exception as exc:
                logger.debug("SAM2 prompt-box detection failed, using center point: %s", exc)

        if initial_prompt_box is None:
            raise RuntimeError(
                "SAM2 prompt box could not be established on any synthetic view. "
                "Masking now requires an explicit synthetic prompt instead of a blind center click."
            )

        if diag_frames is not None:
            for i, stem in enumerate(frame_order):
                if stem not in diag_frames:
                    continue
                df = diag_frames[stem]
                info = resolved_info[i]
                if info.direction is not None:
                    yaw_r, pitch_r = _direction_to_yaw_pitch(info.direction)
                df.p2_direction_yaw = yaw_r
                df.p2_direction_pitch = pitch_r
                df.p2_direction_source = info.source
                df.p2_direction_source_stem = info.source_stem or ""
                if info.source_index is not None:
                    df.p2_direction_source_offset = int(info.source_index - i)
                df.p2_direction_source_peer_stem = info.peer_stem or ""
                if info.peer_index is not None:
                    df.p2_direction_source_peer_offset = int(info.peer_index - i)
                if info.interp_angle_deg is not None:
                    df.p2_direction_interp_angle_deg = float(info.interp_angle_deg)
                df.p2_prompt_frame_idx = best_idx
                df.p2_prompt_syn_idx = initial_frame_idx
                df.p2_prompt_stem = best_stem
                df.p2_prompt_frame_offset = int(best_idx - i)
                df.p2_prompt_mode = prompt_mode
                df.p2_prompt_box_confidence = prompt_box_confidence
                df.p2_prompt_box_center_dist = prompt_box_center_dist

        prompt_desc = (
            f"box conf={prompt_box_confidence:.3f}, center_dist={prompt_box_center_dist:.1f}"
            if prompt_mode == "box" and prompt_box_confidence is not None
            else "center click"
        )
        print(f"[360] Pass 2: {len(synthetic_frames)} synthetic views rendered, "
              f"prompting on frame {best_idx} ({best_stem}, {best_count} detections, {prompt_desc}), tracking...")

        if progress_callback:
            progress_callback(
                0, len(synthetic_frames),
                f"Pass 2 — tracking ({len(synthetic_frames)} frames)...",
            )

        # Run video tracking / fallback detection
        with timer.time("p2_sam2_tracking") if timer else contextmanager(lambda: (yield))():
            tracked_masks = self._video_backend.track_sequence(
                synthetic_frames,
                initial_mask=None,
                initial_frame_idx=initial_frame_idx,
                initial_box=initial_prompt_box,
            )

        secondary_prompt_replacements: dict[int, str] = {}
        if prompt_candidates and len(synthetic_frames) > 1:
            def _best_side_prompt(before_primary: bool) -> tuple[int, str, int, float, float, np.ndarray] | None:
                eligible: list[tuple[tuple[float, float, int, int], int, str, int, float, float, np.ndarray]] = []
                for candidate in prompt_candidates:
                    _key, syn_idx, stem, frame_idx, conf, center_dist, box = candidate
                    if syn_idx == initial_frame_idx:
                        continue
                    if before_primary and frame_idx >= best_idx - 2:
                        continue
                    if (not before_primary) and frame_idx <= best_idx + 2:
                        continue
                    eligible.append(candidate)
                if not eligible:
                    return None
                eligible.sort(key=lambda item: item[0])
                _key, syn_idx, stem, frame_idx, conf, center_dist, box = eligible[0]
                return syn_idx, stem, frame_idx, conf, center_dist, box

            empty_before = [
                syn_idx for syn_idx, syn_mask in enumerate(tracked_masks)
                if (syn_mask is None or syn_mask.sum() <= 0)
                and valid_frame_indices[syn_idx] < best_idx
            ]
            empty_after = [
                syn_idx for syn_idx, syn_mask in enumerate(tracked_masks)
                if (syn_mask is None or syn_mask.sum() <= 0)
                and valid_frame_indices[syn_idx] > best_idx
            ]

            secondary_specs: list[tuple[str, list[int], tuple[int, str, int, float, float, np.ndarray]]] = []
            early_prompt = _best_side_prompt(before_primary=True)
            late_prompt = _best_side_prompt(before_primary=False)
            if empty_before and early_prompt is not None:
                secondary_specs.append(("sam2_secondary_prev", empty_before, early_prompt))
            if empty_after and late_prompt is not None:
                secondary_specs.append(("sam2_secondary_next", empty_after, late_prompt))

            n_secondary_replaced = 0
            for source_label, target_syn_indices, prompt_spec in secondary_specs:
                sec_syn_idx, sec_stem, sec_frame_idx, sec_conf, sec_center_dist, sec_box = prompt_spec
                try:
                    with timer.time("p2_sam2_tracking") if timer else contextmanager(lambda: (yield))():
                        secondary_masks = self._video_backend.track_sequence(
                            synthetic_frames,
                            initial_mask=None,
                            initial_frame_idx=sec_syn_idx,
                            initial_box=sec_box,
                        )
                except Exception:
                    continue
                for syn_idx in target_syn_indices:
                    if syn_idx >= len(secondary_masks):
                        continue
                    primary_mask = tracked_masks[syn_idx]
                    secondary_mask = secondary_masks[syn_idx]
                    if primary_mask is not None and primary_mask.sum() > 0:
                        continue
                    if secondary_mask is None or secondary_mask.sum() <= 0:
                        continue
                    tracked_masks[syn_idx] = secondary_mask
                    secondary_prompt_replacements[syn_idx] = source_label
                    n_secondary_replaced += 1
                if n_secondary_replaced > 0:
                    print(
                        f"[360] Pass 2: secondary SAM2 prompt {sec_frame_idx} ({sec_stem}, "
                        f"conf={sec_conf:.3f}, center_dist={sec_center_dist:.1f}) improved "
                        f"{n_secondary_replaced} frames"
                    )

        # Log tracking results
        n_tracked = sum(1 for m in tracked_masks if m is not None and m.sum() > 0)
        print(f"[360] Pass 2: tracked {n_tracked}/{len(tracked_masks)} frames")

        # Build reverse map for diagnostics: syn_idx → stem
        syn_idx_to_stem: dict[int, str] = {v: k for k, v in stem_to_syn_idx.items()}

        # Record initial tracking results in diagnostics
        if diag_frames is not None:
            for syn_idx, syn_mask in enumerate(tracked_masks):
                stem = syn_idx_to_stem.get(syn_idx)
                if stem is None or stem not in diag_frames:
                    continue
                df = diag_frames[stem]
                was_tracked = syn_mask is not None and syn_mask.sum() > 0
                df.p2_tracked = bool(was_tracked)
                if was_tracked:
                    df.p2_mask_pixels = int(syn_mask.sum())

        track_meta: list[dict[str, object]] | None = None
        if hasattr(self._video_backend, "last_track_meta"):
            try:
                candidate_meta = getattr(self._video_backend, "last_track_meta")
                if isinstance(candidate_meta, list):
                    track_meta = candidate_meta
            except Exception:
                track_meta = None
        if track_meta is None:
            track_meta = []
            for syn_idx, syn_mask in enumerate(tracked_masks):
                was_tracked = syn_mask is not None and syn_mask.sum() > 0
                selection_source = secondary_prompt_replacements.get(
                    syn_idx,
                    "sam2_tracked" if was_tracked else "none",
                )
                track_meta.append({
                    "selection_source": selection_source,
                    "selected_confidence": None,
                    "selected_center_dist": None,
                    "clip_padding": None,
                })
        elif secondary_prompt_replacements:
            for syn_idx, source_label in secondary_prompt_replacements.items():
                if syn_idx >= len(track_meta) or not isinstance(track_meta[syn_idx], dict):
                    continue
                track_meta[syn_idx]["selection_source"] = source_label

        # If the fallback video backend exposed per-frame selection metadata,
        # carry it into diagnostics so we can see which frames were detected
        # directly vs propagated from a neighboring synthetic box.
        if diag_frames is not None and track_meta is not None:
            try:
                if isinstance(track_meta, list):
                    for syn_idx, meta in enumerate(track_meta):
                        stem = syn_idx_to_stem.get(syn_idx)
                        if stem is None or stem not in diag_frames or not isinstance(meta, dict):
                            continue
                        df = diag_frames[stem]
                        if df.p2_box_confidence is None and meta.get("selected_confidence") is not None:
                            df.p2_box_confidence = float(meta["selected_confidence"])
                        if df.p2_box_center_dist is None and meta.get("selected_center_dist") is not None:
                            df.p2_box_center_dist = float(meta["selected_center_dist"])
                        if not df.p2_selection_source and meta.get("selection_source") is not None:
                            df.p2_selection_source = str(meta["selection_source"])
                        if df.p2_clip_padding is None and meta.get("clip_padding") is not None:
                            df.p2_clip_padding = float(meta["clip_padding"])
                        if meta.get("local_redetect_attempted") is not None:
                            df.p2_local_redetect_attempted = bool(meta["local_redetect_attempted"])
                        if meta.get("local_redetect_candidates") is not None:
                            df.p2_local_redetect_candidates = int(meta["local_redetect_candidates"])
                        if (
                            df.p2_local_redetect_window is None
                            and isinstance(meta.get("local_redetect_window"), list)
                        ):
                            df.p2_local_redetect_window = [
                                int(v) for v in meta["local_redetect_window"]
                            ]
                        if (
                            df.p2_local_redetect_selected_iou is None
                            and meta.get("local_redetect_selected_iou") is not None
                        ):
                            df.p2_local_redetect_selected_iou = float(
                                meta["local_redetect_selected_iou"]
                            )
                        if (
                            df.p2_local_redetect_selected_center_shift is None
                            and meta.get("local_redetect_selected_center_shift") is not None
                        ):
                            df.p2_local_redetect_selected_center_shift = float(
                                meta["local_redetect_selected_center_shift"]
                            )
                        if meta.get("local_redetect_replaced_propagation") is not None:
                            df.p2_local_redetect_replaced_propagation = bool(
                                meta["local_redetect_replaced_propagation"]
                            )
                        if meta.get("center_redetect_attempted") is not None:
                            df.p2_center_redetect_attempted = bool(
                                meta["center_redetect_attempted"]
                            )
                        if meta.get("center_redetect_candidates") is not None:
                            df.p2_center_redetect_candidates = int(
                                meta["center_redetect_candidates"]
                            )
                        if (
                            df.p2_center_redetect_window is None
                            and isinstance(meta.get("center_redetect_window"), list)
                        ):
                            df.p2_center_redetect_window = [
                                int(v) for v in meta["center_redetect_window"]
                            ]
                        if (
                            df.p2_center_redetect_selected_center_dist is None
                            and meta.get("center_redetect_selected_center_dist") is not None
                        ):
                            df.p2_center_redetect_selected_center_dist = float(
                                meta["center_redetect_selected_center_dist"]
                            )
                        if (
                            df.p2_center_redetect_selected_continuity_shift is None
                            and meta.get("center_redetect_selected_continuity_shift") is not None
                        ):
                            df.p2_center_redetect_selected_continuity_shift = float(
                                meta["center_redetect_selected_continuity_shift"]
                            )
                        if meta.get("center_redetect_replaced_selection") is not None:
                            df.p2_center_redetect_replaced_selection = bool(
                                meta["center_redetect_replaced_selection"]
                            )
                        if meta.get("alt_view_attempted") is not None:
                            df.p2_alt_view_attempted = bool(meta["alt_view_attempted"])
                        if meta.get("alt_view_candidates") is not None:
                            df.p2_alt_view_candidates = int(meta["alt_view_candidates"])
                        if meta.get("alt_view_raw_candidates") is not None:
                            df.p2_alt_view_raw_candidates = int(meta["alt_view_raw_candidates"])
                        if meta.get("alt_view_valid_candidates") is not None:
                            df.p2_alt_view_valid_candidates = int(meta["alt_view_valid_candidates"])
                        if (
                            df.p2_alt_view_best_raw_confidence is None
                            and meta.get("alt_view_best_raw_confidence") is not None
                        ):
                            df.p2_alt_view_best_raw_confidence = float(
                                meta["alt_view_best_raw_confidence"]
                            )
                        if (
                            df.p2_alt_view_best_raw_center_dist is None
                            and meta.get("alt_view_best_raw_center_dist") is not None
                        ):
                            df.p2_alt_view_best_raw_center_dist = float(
                                meta["alt_view_best_raw_center_dist"]
                            )
                        if (
                            df.p2_alt_view_best_raw_mask_pixels is None
                            and meta.get("alt_view_best_raw_mask_pixels") is not None
                        ):
                            df.p2_alt_view_best_raw_mask_pixels = int(
                                meta["alt_view_best_raw_mask_pixels"]
                            )
                        if (
                            df.p2_alt_view_best_confidence is None
                            and meta.get("alt_view_best_confidence") is not None
                        ):
                            df.p2_alt_view_best_confidence = float(meta["alt_view_best_confidence"])
                        if (
                            df.p2_alt_view_best_center_dist is None
                            and meta.get("alt_view_best_center_dist") is not None
                        ):
                            df.p2_alt_view_best_center_dist = float(meta["alt_view_best_center_dist"])
                        if (
                            df.p2_alt_view_best_mask_pixels is None
                            and meta.get("alt_view_best_mask_pixels") is not None
                        ):
                            df.p2_alt_view_best_mask_pixels = int(meta["alt_view_best_mask_pixels"])
                        if meta.get("alt_view_selected_source_stem") is not None:
                            df.p2_alt_view_selected_source_stem = str(
                                meta["alt_view_selected_source_stem"] or ""
                            )
                        if (
                            df.p2_alt_view_selected_source_offset is None
                            and meta.get("alt_view_selected_source_offset") is not None
                        ):
                            df.p2_alt_view_selected_source_offset = int(
                                meta["alt_view_selected_source_offset"]
                            )
                        if meta.get("alt_view_replaced_selection") is not None:
                            df.p2_alt_view_replaced_selection = bool(
                                meta["alt_view_replaced_selection"]
                            )
                        if meta.get("direction_search_attempted") is not None:
                            df.p2_direction_search_attempted = bool(
                                meta["direction_search_attempted"]
                            )
                        if meta.get("direction_search_raw_candidates") is not None:
                            df.p2_direction_search_raw_candidates = int(
                                meta["direction_search_raw_candidates"]
                            )
                        if meta.get("direction_search_valid_candidates") is not None:
                            df.p2_direction_search_valid_candidates = int(
                                meta["direction_search_valid_candidates"]
                            )
                        if (
                            df.p2_direction_search_best_raw_confidence is None
                            and meta.get("direction_search_best_raw_confidence") is not None
                        ):
                            df.p2_direction_search_best_raw_confidence = float(
                                meta["direction_search_best_raw_confidence"]
                            )
                        if (
                            df.p2_direction_search_best_raw_center_dist is None
                            and meta.get("direction_search_best_raw_center_dist") is not None
                        ):
                            df.p2_direction_search_best_raw_center_dist = float(
                                meta["direction_search_best_raw_center_dist"]
                            )
                        if (
                            df.p2_direction_search_best_raw_mask_pixels is None
                            and meta.get("direction_search_best_raw_mask_pixels") is not None
                        ):
                            df.p2_direction_search_best_raw_mask_pixels = int(
                                meta["direction_search_best_raw_mask_pixels"]
                            )
                        if (
                            df.p2_direction_search_best_raw_yaw_offset is None
                            and meta.get("direction_search_best_raw_yaw_offset") is not None
                        ):
                            df.p2_direction_search_best_raw_yaw_offset = float(
                                meta["direction_search_best_raw_yaw_offset"]
                            )
                        if (
                            df.p2_direction_search_best_raw_pitch_offset is None
                            and meta.get("direction_search_best_raw_pitch_offset") is not None
                        ):
                            df.p2_direction_search_best_raw_pitch_offset = float(
                                meta["direction_search_best_raw_pitch_offset"]
                            )
                        if (
                            df.p2_direction_search_best_confidence is None
                            and meta.get("direction_search_best_confidence") is not None
                        ):
                            df.p2_direction_search_best_confidence = float(
                                meta["direction_search_best_confidence"]
                            )
                        if (
                            df.p2_direction_search_best_center_dist is None
                            and meta.get("direction_search_best_center_dist") is not None
                        ):
                            df.p2_direction_search_best_center_dist = float(
                                meta["direction_search_best_center_dist"]
                            )
                        if (
                            df.p2_direction_search_best_mask_pixels is None
                            and meta.get("direction_search_best_mask_pixels") is not None
                        ):
                            df.p2_direction_search_best_mask_pixels = int(
                                meta["direction_search_best_mask_pixels"]
                            )
                        if (
                            df.p2_direction_search_best_yaw_offset is None
                            and meta.get("direction_search_best_yaw_offset") is not None
                        ):
                            df.p2_direction_search_best_yaw_offset = float(
                                meta["direction_search_best_yaw_offset"]
                            )
                        if (
                            df.p2_direction_search_best_pitch_offset is None
                            and meta.get("direction_search_best_pitch_offset") is not None
                        ):
                            df.p2_direction_search_best_pitch_offset = float(
                                meta["direction_search_best_pitch_offset"]
                            )
                        if meta.get("direction_search_replaced_selection") is not None:
                            df.p2_direction_search_replaced_selection = bool(
                                meta["direction_search_replaced_selection"]
                            )
                        if meta.get("dilation_applied") is not None:
                            df.p2_dilation_applied = bool(meta["dilation_applied"])
                        if df.p2_dilation_kernel is None and meta.get("dilation_kernel") is not None:
                            df.p2_dilation_kernel = int(meta["dilation_kernel"])
                        if meta.get("reprompt_applied") is not None:
                            df.p2_reprompt_applied = bool(meta["reprompt_applied"])
                        if (
                            df.p2_reprompt_gain_pixels is None
                            and meta.get("reprompt_gain_pixels") is not None
                        ):
                            df.p2_reprompt_gain_pixels = int(meta["reprompt_gain_pixels"])
                        if meta.get("completeness_applied") is not None:
                            df.p2_completeness_applied = bool(meta["completeness_applied"])
                        if (
                            df.p2_completeness_kernel is None
                            and meta.get("completeness_kernel") is not None
                        ):
                            df.p2_completeness_kernel = int(meta["completeness_kernel"])
                        if df.p2_propagation_gap is None and meta.get("propagation_gap") is not None:
                            df.p2_propagation_gap = int(meta["propagation_gap"])
            except Exception:
                pass

        # For borrowed/interpolated synthetic directions, give the current ERP
        # frame one more chance using the neighboring direct-direction views.
        # This is more targeted than general morphology tuning and is aimed at
        # lingering temporal fallback cases like Ashland 00019 / long-run 00010.
        override_bp_dirs: dict[int, np.ndarray] = {}
        n_alt_view_replaced = 0
        if self._backend is not None and track_meta is not None:
            try:
                if isinstance(track_meta, list):
                    for syn_idx, meta in enumerate(track_meta):
                        if syn_idx >= len(valid_frame_indices) or not isinstance(meta, dict):
                            continue
                        frame_idx = valid_frame_indices[syn_idx]
                        info = resolved_info[frame_idx]
                        if info.source not in ("temporal_prev", "temporal_next", "temporal_interp"):
                            continue
                        selection_source = str(meta.get("selection_source") or "")
                        current_mask = tracked_masks[syn_idx]
                        current_nonempty = current_mask is not None and current_mask.sum() > 0
                        if selection_source != "none" and not selection_source.startswith("propagated"):
                            continue
                        if info.source_index is None and info.peer_index is None:
                            continue

                        candidate_indices: list[int] = []
                        for candidate_idx in (info.source_index, info.peer_index):
                            if candidate_idx is None:
                                continue
                            if candidate_idx < 0 or candidate_idx >= len(all_dirs):
                                continue
                            candidate_dir = all_dirs[candidate_idx]
                            if candidate_dir is None:
                                continue
                            if candidate_idx not in candidate_indices:
                                candidate_indices.append(candidate_idx)
                        if not candidate_indices:
                            continue

                        stem = frame_order[frame_idx]
                        erp = cv2.imread(str(frame_files[frame_idx]))
                        if erp is None:
                            continue

                        best_alt: tuple[int, np.ndarray, np.ndarray, float | None, float | None, int, np.ndarray | None] | None = None
                        best_alt_raw: tuple[int, np.ndarray, np.ndarray, float | None, float | None, int, np.ndarray | None] | None = None
                        raw_alt_candidates = 0
                        valid_alt_candidates = 0
                        for candidate_idx in candidate_indices:
                            candidate_dir = all_dirs[candidate_idx]
                            if candidate_dir is None:
                                continue
                            R_alt = _look_at_rotation(candidate_dir)
                            with timer.time("p2_alt_view_render") if timer else contextmanager(lambda: (yield))():
                                alt_fisheye = _render_synthetic_fisheye(erp, camera, R_alt)
                            with timer.time("p2_alt_view_detect") if timer else contextmanager(lambda: (yield))():
                                alt_mask, alt_conf, alt_center_dist, alt_box = _detect_synthetic_candidate(alt_fisheye)
                            if alt_mask is None or alt_mask.sum() <= 0:
                                continue
                            raw_alt_candidates += 1
                            raw_candidate = (
                                candidate_idx,
                                candidate_dir,
                                alt_mask,
                                alt_conf,
                                alt_center_dist,
                                int(alt_mask.sum()),
                                alt_box,
                            )
                            if best_alt_raw is None:
                                best_alt_raw = raw_candidate
                            else:
                                _idx, _dir, _mask, _conf, _dist, _pixels, _box = best_alt_raw
                                if (
                                    (alt_center_dist is not None and _dist is not None and alt_center_dist < _dist - 1e-6)
                                    or (
                                        alt_center_dist is not None and _dist is not None
                                        and abs(alt_center_dist - _dist) <= 1e-6
                                        and (alt_conf or 0.0) > (_conf or 0.0)
                                    )
                                    or (
                                        alt_center_dist is not None and _dist is not None
                                        and abs(alt_center_dist - _dist) <= 1e-6
                                        and abs((alt_conf or 0.0) - (_conf or 0.0)) <= 1e-6
                                        and int(alt_mask.sum()) > _pixels
                                    )
                                    or (_dist is None and alt_center_dist is not None)
                                ):
                                    best_alt_raw = raw_candidate
                            if alt_center_dist is None or alt_center_dist > ALT_VIEW_RESCUE_MAX_CENTER_DIST:
                                continue
                            if alt_conf is not None and alt_conf < ALT_VIEW_RESCUE_MIN_CONFIDENCE:
                                continue
                            if int(alt_mask.sum()) < ALT_VIEW_RESCUE_MIN_MASK_PIXELS:
                                continue
                            valid_alt_candidates += 1
                            candidate = (
                                candidate_idx,
                                candidate_dir,
                                alt_mask,
                                alt_conf,
                                alt_center_dist,
                                int(alt_mask.sum()),
                                alt_box,
                            )
                            if best_alt is None:
                                best_alt = candidate
                            else:
                                _idx, _dir, _mask, _conf, _dist, _pixels, _box = best_alt
                                if (
                                    alt_center_dist < _dist - 1e-6
                                    or (
                                        abs(alt_center_dist - _dist) <= 1e-6
                                        and (alt_conf or 0.0) > (_conf or 0.0)
                                    )
                                    or (
                                        abs(alt_center_dist - _dist) <= 1e-6
                                        and abs((alt_conf or 0.0) - (_conf or 0.0)) <= 1e-6
                                        and int(alt_mask.sum()) > _pixels
                                    )
                                ):
                                    best_alt = candidate

                        if diag_frames is not None and stem in diag_frames:
                            df = diag_frames[stem]
                            df.p2_alt_view_attempted = True
                            df.p2_alt_view_candidates = len(candidate_indices)
                            df.p2_alt_view_raw_candidates = raw_alt_candidates
                            df.p2_alt_view_valid_candidates = valid_alt_candidates
                            if best_alt_raw is not None:
                                _idx, _dir, _mask, _conf, _dist, _pixels, _box = best_alt_raw
                                df.p2_alt_view_best_raw_confidence = _conf
                                df.p2_alt_view_best_raw_center_dist = _dist
                                df.p2_alt_view_best_raw_mask_pixels = _pixels
                            if best_alt is not None:
                                _idx, _dir, _mask, _conf, _dist, _pixels, _box = best_alt
                                df.p2_alt_view_best_confidence = _conf
                                df.p2_alt_view_best_center_dist = _dist
                                df.p2_alt_view_best_mask_pixels = _pixels

                        if best_alt is None:
                            meta["alt_view_attempted"] = True
                            meta["alt_view_candidates"] = len(candidate_indices)
                            meta["alt_view_raw_candidates"] = raw_alt_candidates
                            meta["alt_view_valid_candidates"] = valid_alt_candidates
                            if best_alt_raw is not None:
                                _idx, _dir, _mask, _conf, _dist, _pixels, _box = best_alt_raw
                                meta["alt_view_best_raw_confidence"] = _conf
                                meta["alt_view_best_raw_center_dist"] = _dist
                                meta["alt_view_best_raw_mask_pixels"] = _pixels
                            continue

                        candidate_idx, candidate_dir, alt_mask, alt_conf, alt_center_dist, _pixels, alt_box = best_alt
                        meta["alt_view_attempted"] = True
                        meta["alt_view_candidates"] = len(candidate_indices)
                        meta["alt_view_raw_candidates"] = raw_alt_candidates
                        meta["alt_view_valid_candidates"] = valid_alt_candidates
                        if best_alt_raw is not None:
                            _idx, _dir, _mask, _conf, _dist, _raw_pixels, _box = best_alt_raw
                            meta["alt_view_best_raw_confidence"] = _conf
                            meta["alt_view_best_raw_center_dist"] = _dist
                            meta["alt_view_best_raw_mask_pixels"] = _raw_pixels
                        meta["alt_view_best_confidence"] = alt_conf
                        meta["alt_view_best_center_dist"] = alt_center_dist
                        meta["alt_view_best_mask_pixels"] = _pixels
                        if not _alt_view_should_replace(
                            selection_source,
                            current_mask,
                            alt_conf,
                            alt_center_dist,
                            _pixels,
                        ):
                            continue
                        alt_mask, alt_reprompt_applied, alt_reprompt_gain_pixels = _apply_post_backend_reprompt(
                            alt_fisheye,
                            alt_mask,
                            alt_box,
                            "altview_prev" if candidate_idx < frame_idx else "altview_next",
                            alt_conf,
                            alt_center_dist,
                        )
                        alt_mask, alt_completeness_applied, alt_completeness_kernel = _apply_post_backend_completeness(
                            alt_mask,
                            alt_box,
                            "altview_prev" if candidate_idx < frame_idx else "altview_next",
                            alt_conf,
                            alt_center_dist,
                            alt_fisheye.shape,
                        )
                        tracked_masks[syn_idx] = alt_mask
                        override_bp_dirs[syn_idx] = candidate_dir
                        n_alt_view_replaced += 1

                        source_label = (
                            "altview_prev" if candidate_idx < frame_idx else "altview_next"
                        )
                        meta["selection_source"] = source_label
                        meta["selected_confidence"] = alt_conf
                        meta["selected_center_dist"] = alt_center_dist
                        meta["clip_padding"] = SYNTHETIC_PRIMARY_BOX_PADDING
                        meta["alt_view_selected_source_stem"] = frame_order[candidate_idx]
                        meta["alt_view_selected_source_offset"] = int(candidate_idx - frame_idx)
                        meta["alt_view_replaced_selection"] = True
                        meta["dilation_applied"] = False
                        meta["dilation_kernel"] = None
                        meta["reprompt_applied"] = bool(alt_reprompt_applied)
                        meta["reprompt_gain_pixels"] = alt_reprompt_gain_pixels
                        meta["completeness_applied"] = bool(alt_completeness_applied)
                        meta["completeness_kernel"] = alt_completeness_kernel

                        if diag_frames is not None and stem in diag_frames:
                            df = diag_frames[stem]
                            df.p2_tracked = True
                            df.p2_rescued = False
                            df.p2_mask_pixels = int(alt_mask.sum())
                            df.p2_box_confidence = alt_conf
                            df.p2_box_center_dist = alt_center_dist
                            df.p2_selection_source = source_label
                            df.p2_clip_padding = SYNTHETIC_PRIMARY_BOX_PADDING
                            df.p2_alt_view_attempted = True
                            df.p2_alt_view_candidates = len(candidate_indices)
                            df.p2_alt_view_selected_source_stem = frame_order[candidate_idx]
                            df.p2_alt_view_selected_source_offset = int(candidate_idx - frame_idx)
                            df.p2_alt_view_replaced_selection = True
                            df.p2_reprompt_applied = bool(alt_reprompt_applied)
                            df.p2_reprompt_gain_pixels = alt_reprompt_gain_pixels
                            df.p2_completeness_applied = bool(alt_completeness_applied)
                            df.p2_completeness_kernel = alt_completeness_kernel
            except Exception:
                pass

        if n_alt_view_replaced > 0:
            n_tracked = sum(1 for m in tracked_masks if m is not None and m.sum() > 0)
            print(
                f"[360] Pass 2: alternate-view rescue improved "
                f"{n_alt_view_replaced} borrowed-direction frames"
            )

        # For lingering direct or temporal-borrow cases that are still
        # propagated/empty after alternate-view rescue, search the current
        # direction plus a small yaw/pitch neighborhood on the current frame.
        n_direction_search_replaced = 0
        if self._backend is not None and track_meta is not None:
            try:
                if isinstance(track_meta, list):
                    for syn_idx, meta in enumerate(track_meta):
                        if syn_idx >= len(valid_frame_indices) or not isinstance(meta, dict):
                            continue
                        frame_idx = valid_frame_indices[syn_idx]
                        info = resolved_info[frame_idx]
                        if info.source not in ("direct", "temporal_prev", "temporal_next", "temporal_interp"):
                            continue
                        direction = info.direction
                        if direction is None:
                            continue

                        selection_source = str(meta.get("selection_source") or "")
                        current_mask = tracked_masks[syn_idx]
                        current_nonempty = current_mask is not None and current_mask.sum() > 0
                        if selection_source != "none" and not selection_source.startswith("propagated"):
                            continue

                        stem = frame_order[frame_idx]
                        erp = cv2.imread(str(frame_files[frame_idx]))
                        if erp is None:
                            continue

                        search_offsets = [(0.0, 0.0)] + [
                            (yaw_off, pitch_off)
                            for yaw_off in (0.0,) + DIRECTION_SEARCH_YAW_OFFSETS
                            for pitch_off in DIRECTION_SEARCH_PITCH_OFFSETS
                            if not (abs(yaw_off) <= 1e-6 and abs(pitch_off) <= 1e-6)
                        ]
                        if not search_offsets:
                            continue

                        def _search_candidates_for_direction(
                            seed_direction: np.ndarray,
                        ) -> tuple[
                            tuple[np.ndarray, np.ndarray, float | None, float | None, int, float, float, np.ndarray | None, tuple[int, int, int] | None] | None,
                            tuple[np.ndarray, np.ndarray, float | None, float | None, int, float, float, np.ndarray | None, tuple[int, int, int] | None] | None,
                            int,
                            int,
                        ]:
                            base_yaw, base_pitch = _direction_to_yaw_pitch(seed_direction)
                            local_best: tuple[np.ndarray, np.ndarray, float | None, float | None, int, float, float, np.ndarray | None, tuple[int, int, int] | None] | None = None
                            local_best_raw: tuple[np.ndarray, np.ndarray, float | None, float | None, int, float, float, np.ndarray | None, tuple[int, int, int] | None] | None = None
                            local_best_score = -1.0
                            local_best_raw_score = -1.0
                            local_raw_candidates = 0
                            local_valid_candidates = 0
                            for yaw_off, pitch_off in search_offsets:
                                cand_pitch = float(np.clip(base_pitch + pitch_off, -89.0, 89.0))
                                candidate_dir = _yaw_pitch_to_direction(base_yaw + yaw_off, cand_pitch)
                                R_search = _look_at_rotation(candidate_dir)
                                with timer.time("p2_direction_search_render") if timer else contextmanager(lambda: (yield))():
                                    search_fisheye = _render_synthetic_fisheye(erp, camera, R_search)
                                with timer.time("p2_direction_search_detect") if timer else contextmanager(lambda: (yield))():
                                    search_mask, search_conf, search_center_dist, search_box = _detect_synthetic_candidate(search_fisheye)
                                if search_mask is None or search_mask.sum() <= 0:
                                    continue

                                local_raw_candidates += 1
                                search_pixels = int(search_mask.sum())
                                raw_candidate = (
                                    candidate_dir,
                                    search_mask,
                                    search_conf,
                                    search_center_dist,
                                    search_pixels,
                                    float(yaw_off),
                                    float(cand_pitch - base_pitch),
                                    search_box,
                                    search_fisheye.shape,
                                )
                                raw_score = _direction_search_candidate_score(
                                    search_conf,
                                    search_center_dist,
                                    search_pixels,
                                    float(yaw_off),
                                    float(cand_pitch - base_pitch),
                                )
                                if local_best_raw is None:
                                    local_best_raw = raw_candidate
                                    local_best_raw_score = raw_score
                                else:
                                    _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = local_best_raw
                                    if (
                                        raw_score > local_best_raw_score + 1e-6
                                        or (
                                            abs(raw_score - local_best_raw_score) <= 1e-6
                                            and abs((search_conf or 0.0) - (_conf or 0.0)) <= 1e-6
                                            and search_pixels > _pixels
                                        )
                                    ):
                                        local_best_raw = raw_candidate
                                        local_best_raw_score = raw_score

                                if not _direction_search_is_valid_candidate(
                                    search_conf,
                                    search_center_dist,
                                    search_pixels,
                                    float(yaw_off),
                                    float(cand_pitch - base_pitch),
                                ):
                                    continue

                                local_valid_candidates += 1
                                candidate = (
                                    candidate_dir,
                                    search_mask,
                                    search_conf,
                                    search_center_dist,
                                    search_pixels,
                                    float(yaw_off),
                                    float(cand_pitch - base_pitch),
                                    search_box,
                                    search_fisheye.shape,
                                )
                                cand_score = _direction_search_candidate_score(
                                    search_conf,
                                    search_center_dist,
                                    search_pixels,
                                    float(yaw_off),
                                    float(cand_pitch - base_pitch),
                                )
                                if local_best is None:
                                    local_best = candidate
                                    local_best_score = cand_score
                                else:
                                    _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = local_best
                                    if (
                                        cand_score > local_best_score + 1e-6
                                        or (
                                            abs(cand_score - local_best_score) <= 1e-6
                                            and abs((search_conf or 0.0) - (_conf or 0.0)) <= 1e-6
                                            and search_pixels > _pixels
                                        )
                                    ):
                                        local_best = candidate
                                        local_best_score = cand_score
                            return local_best, local_best_raw, local_raw_candidates, local_valid_candidates

                        best_search, best_search_raw, raw_search_candidates, valid_search_candidates = _search_candidates_for_direction(direction)

                        alt_seed_direction: np.ndarray | None = None
                        if best_search is None and raw_search_candidates == 0:
                            alt_idx: int | None = None
                            if info.source == "temporal_prev":
                                for j in range(frame_idx + 1, len(all_dirs)):
                                    if all_dirs[j] is not None:
                                        alt_idx = j
                                        break
                            elif info.source == "temporal_next":
                                for j in range(frame_idx - 1, -1, -1):
                                    if all_dirs[j] is not None:
                                        alt_idx = j
                                        break
                            if alt_idx is not None:
                                alt_seed_direction = all_dirs[alt_idx]
                        if alt_seed_direction is not None:
                            alt_best_search, alt_best_search_raw, alt_raw_candidates, alt_valid_candidates = _search_candidates_for_direction(alt_seed_direction)
                            raw_search_candidates += alt_raw_candidates
                            valid_search_candidates += alt_valid_candidates
                            if best_search_raw is None:
                                best_search_raw = alt_best_search_raw
                            if best_search is None:
                                best_search = alt_best_search

                        if diag_frames is not None and stem in diag_frames:
                            df = diag_frames[stem]
                            df.p2_direction_search_attempted = True
                            df.p2_direction_search_raw_candidates = raw_search_candidates
                            df.p2_direction_search_valid_candidates = valid_search_candidates
                            if best_search_raw is not None:
                                _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = best_search_raw
                                df.p2_direction_search_best_raw_confidence = _conf
                                df.p2_direction_search_best_raw_center_dist = _dist
                                df.p2_direction_search_best_raw_mask_pixels = _pixels
                                df.p2_direction_search_best_raw_yaw_offset = _yaw_off
                                df.p2_direction_search_best_raw_pitch_offset = _pitch_off
                            if best_search is not None:
                                _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = best_search
                                df.p2_direction_search_best_confidence = _conf
                                df.p2_direction_search_best_center_dist = _dist
                                df.p2_direction_search_best_mask_pixels = _pixels
                                df.p2_direction_search_best_yaw_offset = _yaw_off
                                df.p2_direction_search_best_pitch_offset = _pitch_off

                        meta["direction_search_attempted"] = True
                        meta["direction_search_raw_candidates"] = raw_search_candidates
                        meta["direction_search_valid_candidates"] = valid_search_candidates
                        if best_search_raw is not None:
                            _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = best_search_raw
                            meta["direction_search_best_raw_confidence"] = _conf
                            meta["direction_search_best_raw_center_dist"] = _dist
                            meta["direction_search_best_raw_mask_pixels"] = _pixels
                            meta["direction_search_best_raw_yaw_offset"] = _yaw_off
                            meta["direction_search_best_raw_pitch_offset"] = _pitch_off
                        if best_search is not None:
                            _dir, _mask, _conf, _dist, _pixels, _yaw_off, _pitch_off, _box, _shape = best_search
                            meta["direction_search_best_confidence"] = _conf
                            meta["direction_search_best_center_dist"] = _dist
                            meta["direction_search_best_mask_pixels"] = _pixels
                            meta["direction_search_best_yaw_offset"] = _yaw_off
                            meta["direction_search_best_pitch_offset"] = _pitch_off

                        if best_search is None:
                            continue

                        current_center_dist = (
                            float(meta["selected_center_dist"])
                            if meta.get("selected_center_dist") is not None
                            else None
                        )
                        candidate_dir, search_mask, search_conf, search_center_dist, search_pixels, yaw_off, pitch_off, search_box, search_shape = best_search
                        if not _direction_search_should_replace(
                            selection_source,
                            current_mask,
                            current_center_dist,
                            search_conf,
                            search_center_dist,
                            search_pixels,
                            yaw_off,
                            pitch_off,
                        ):
                            continue
                        R_search = _look_at_rotation(candidate_dir)
                        with timer.time("p2_direction_search_render") if timer else contextmanager(lambda: (yield))():
                            search_fisheye = _render_synthetic_fisheye(erp, camera, R_search)
                        search_mask, search_reprompt_applied, search_reprompt_gain_pixels = _apply_post_backend_reprompt(
                            search_fisheye,
                            search_mask,
                            search_box,
                            "direction_search",
                            search_conf,
                            search_center_dist,
                        )
                        search_mask, search_completeness_applied, search_completeness_kernel = _apply_post_backend_completeness(
                            search_mask,
                            search_box,
                            "direction_search",
                            search_conf,
                            search_center_dist,
                            search_shape or frame.shape,
                        )

                        tracked_masks[syn_idx] = search_mask
                        override_bp_dirs[syn_idx] = candidate_dir
                        n_direction_search_replaced += 1

                        meta["selection_source"] = "direction_search"
                        meta["selected_confidence"] = search_conf
                        meta["selected_center_dist"] = search_center_dist
                        meta["clip_padding"] = SYNTHETIC_PRIMARY_BOX_PADDING
                        meta["direction_search_replaced_selection"] = True
                        meta["dilation_applied"] = False
                        meta["dilation_kernel"] = None
                        meta["reprompt_applied"] = bool(search_reprompt_applied)
                        meta["reprompt_gain_pixels"] = search_reprompt_gain_pixels
                        meta["completeness_applied"] = bool(search_completeness_applied)
                        meta["completeness_kernel"] = search_completeness_kernel

                        if diag_frames is not None and stem in diag_frames:
                            df = diag_frames[stem]
                            df.p2_tracked = True
                            df.p2_rescued = False
                            df.p2_mask_pixels = int(search_mask.sum())
                            df.p2_box_confidence = search_conf
                            df.p2_box_center_dist = search_center_dist
                            df.p2_selection_source = "direction_search"
                            df.p2_clip_padding = SYNTHETIC_PRIMARY_BOX_PADDING
                            df.p2_direction_search_attempted = True
                            df.p2_direction_search_raw_candidates = raw_search_candidates
                            df.p2_direction_search_valid_candidates = valid_search_candidates
                            df.p2_direction_search_best_confidence = search_conf
                            df.p2_direction_search_best_center_dist = search_center_dist
                            df.p2_direction_search_best_mask_pixels = search_pixels
                            df.p2_direction_search_best_yaw_offset = yaw_off
                            df.p2_direction_search_best_pitch_offset = pitch_off
                            df.p2_direction_search_replaced_selection = True
                            df.p2_reprompt_applied = bool(search_reprompt_applied)
                            df.p2_reprompt_gain_pixels = search_reprompt_gain_pixels
                            df.p2_completeness_applied = bool(search_completeness_applied)
                            df.p2_completeness_kernel = search_completeness_kernel
            except Exception:
                pass

        if n_direction_search_replaced > 0:
            n_tracked = sum(1 for m in tracked_masks if m is not None and m.sum() > 0)
            print(
                f"[360] Pass 2: direction-search rescue improved "
                f"{n_direction_search_replaced} borrowed-direction frames"
            )

        # Recover isolated SAM2 dropouts by re-running image detection on the
        # already-rendered synthetic fisheye frame instead of falling back to
        # the empty Pass 1 mask. This keeps Pass 1 as localization-only while
        # giving Pass 2 a per-frame rescue path for missed tracking frames.
        n_rescued = 0
        if self._backend is not None:
            for syn_idx, syn_mask in enumerate(tracked_masks):
                if syn_mask is None or syn_mask.sum() > 0:
                    continue
                # Run rescue detection via batch_detect_boxes for confidence data
                rescue_conf: float | None = None
                rescue_center_dist: float | None = None
                with timer.time("p2_empty_frame_rescue") if timer else contextmanager(lambda: (yield))():
                    rescued_mask = self._backend.detect_and_segment(
                        synthetic_frames[syn_idx],
                        cfg.targets,
                        detection_confidence=SYNTHETIC_FALLBACK_CONFIDENCE,
                        single_primary_box=True,
                        primary_box_mode=SYNTHETIC_PRIMARY_BOX_MODE,
                        constrain_to_primary_box=SYNTHETIC_PRIMARY_BOX_CONSTRAIN,
                        primary_box_padding=SYNTHETIC_PRIMARY_BOX_PADDING,
                    )
                    # Get box confidence + center distance if diagnostics enabled
                    if diag_frames is not None:
                        rescue_boxes = self._backend.batch_detect_boxes(
                            [synthetic_frames[syn_idx]],
                            detection_confidence=SYNTHETIC_FALLBACK_CONFIDENCE,
                        )
                        if rescue_boxes and rescue_boxes[0]:
                            boxes = [box for box, _ in rescue_boxes[0]]
                            confs = [conf for _, conf in rescue_boxes[0]]
                            selected = select_primary_person_box(
                                boxes,
                                confs,
                                synthetic_frames[syn_idx].shape,
                                mode=SYNTHETIC_PRIMARY_BOX_MODE,
                            )
                            if selected is not None:
                                best_box, best_conf = selected
                                rescue_conf = best_conf
                                bx = (float(best_box[0]) + float(best_box[2])) / 2.0
                                by = (float(best_box[1]) + float(best_box[3])) / 2.0
                                img_cx = synthetic_frames[syn_idx].shape[1] / 2.0
                                img_cy = synthetic_frames[syn_idx].shape[0] / 2.0
                                rescue_center_dist = float(np.hypot(bx - img_cx, by - img_cy))

                if rescued_mask is not None and rescued_mask.sum() > 0:
                    tracked_masks[syn_idx] = rescued_mask
                    n_rescued += 1

                    if diag_frames is not None:
                        stem = syn_idx_to_stem.get(syn_idx)
                        if stem is not None and stem in diag_frames:
                            df = diag_frames[stem]
                            df.p2_rescued = True
                            df.p2_mask_pixels = int(rescued_mask.sum())
                            df.p2_box_confidence = rescue_conf
                            df.p2_box_center_dist = rescue_center_dist

        if n_rescued > 0:
            n_tracked += n_rescued
            print(f"[360] Pass 2: rescued {n_rescued} empty tracked frames with per-frame detection")

        print(f"[360] Pass 2: backprojecting {n_tracked}/{len(tracked_masks)} non-empty synthetic masks...")

        # Backproject each synthetic mask to ERP.
        # If person direction is stable (angular spread < 10°), build
        # one shared backprojection map from the mean direction and
        # reuse it for all frames. Otherwise fall back to per-frame.
        effective_dirs = []
        for i, stem in enumerate(frame_order):
            syn_idx = stem_to_syn_idx.get(stem)
            if syn_idx is not None and syn_idx in override_bp_dirs:
                effective_dirs.append(override_bp_dirs[syn_idx])
            elif resolved_dirs[i] is not None:
                effective_dirs.append(resolved_dirs[i])

        valid_dirs = [d for d in effective_dirs if d is not None]
        spread = _direction_angular_spread(valid_dirs)
        MAX_SPREAD_FOR_SHARED_MAP = 10.0

        # Get ERP size from first available mask
        sample_stem = next(iter(stem_to_syn_idx))
        erp_h, erp_w = primary_masks[sample_stem].shape[:2]

        shared_map: _BackprojectMap | None = None
        if not override_bp_dirs and spread <= MAX_SPREAD_FOR_SHARED_MAP and len(valid_dirs) >= 2:
            mean_dir = np.mean(valid_dirs, axis=0)
            mean_dir = mean_dir / np.linalg.norm(mean_dir)
            R_mean = _look_at_rotation(mean_dir)
            with timer.time("p2_bp_map_build") if timer else contextmanager(lambda: (yield))():
                shared_map = _build_backproject_map((erp_w, erp_h), camera, R_mean)
            yaw_m, pitch_m = _direction_to_yaw_pitch(mean_dir)
            logger.debug("Backprojection: shared map (spread=%.1f°, mean yaw=%.1f° pitch=%.1f°)",
                         spread, yaw_m, pitch_m)
        else:
            if override_bp_dirs:
                logger.debug(
                    "Backprojection: per-frame (override directions used on %d frames, spread=%.1f°)",
                    len(override_bp_dirs), spread,
                )
            else:
                logger.debug("Backprojection: per-frame (spread=%.1f°)", spread)

        result: dict[str, np.ndarray] = {}
        n_backproject = len(stem_to_syn_idx)
        bp_count = 0
        for i, stem in enumerate(frame_order):
            syn_idx = stem_to_syn_idx.get(stem)
            if syn_idx is None or syn_idx >= len(tracked_masks):
                continue

            syn_mask = tracked_masks[syn_idx]
            d = override_bp_dirs.get(syn_idx, resolved_dirs[i])
            if d is None:
                continue

            if shared_map is not None:
                with timer.time("p2_bp_apply") if timer else contextmanager(lambda: (yield))():
                    erp_mask = shared_map.apply(syn_mask)
            else:
                with timer.time("p2_backproject") if timer else contextmanager(lambda: (yield))():
                    erp_mask = _backproject_fisheye_mask_to_erp(
                        syn_mask, (erp_w, erp_h), camera, R_world_from_cam=_look_at_rotation(d),
                    )
            result[stem] = erp_mask
            bp_count += 1
            print(f"[360] Pass 2: backprojecting {bp_count}/{n_backproject}")

            if progress_callback:
                progress_callback(
                    bp_count, n_backproject,
                    f"Pass 2 — backprojecting: {bp_count}/{n_backproject}",
                )

        return result

    def process_frames(
        self,
        frames_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> MaskResult:
        """Two-pass masking: primary detection → synthetic camera → postprocess.

        Owns backend teardown, runtime fallback, and retry behavior.
        If the real video backend fails, logs the error, cleans up,
        creates a fresh image backend, wraps it in FallbackVideoBackend,
        and retries Pass 2. The pipeline should not fail because SAM v2
        crashed.

        Args:
            frames_dir: Directory containing ERP frame images.
            output_dir: Directory to write ERP masks.
            progress_callback: Optional (current, total, message) callback.

        Returns:
            MaskResult with statistics.
        """
        from .backends import FallbackVideoBackend, get_video_backend

        logger.debug("process_frames starting — DETECTION_LAYOUT + authoritative Pass 2")

        if self._backend is None:
            return MaskResult(success=False, error="Not initialized")

        cfg = self.config
        if not cfg.views:
            return MaskResult(success=False, error="No views configured")

        frames_path = Path(frames_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            return MaskResult(success=False, error=f"No frames in {frames_dir}")

        n_frames = len(frame_files)
        masked_count = 0
        diagnostics_path = ""
        detection_size = None
        timer = _SubstageTimer()
        self._used_fallback_video_backend = (
            self._video_backend_name == "FallbackVideoBackend"
        )
        self._video_backend_error = ""

        try:
            # ── Phase 1: Primary detection ────────────────────────
            primary_masks: dict[str, np.ndarray] = {}
            person_directions: dict[str, np.ndarray | None] = {}
            detection_counts: dict[str, int] = {}
            frame_order: list[str] = []
            diag_frames: dict[str, _DiagFrame] = {} if cfg.enable_diagnostics else {}

            for fi, frame_file in enumerate(frame_files):
                with timer.time("p1_imread"):
                    erp = cv2.imread(str(frame_file))
                if erp is None:
                    continue

                erp_h, erp_w = erp.shape[:2]
                if detection_size is None:
                    detection_size = min(1024, erp_w // 4)

                stem = frame_file.stem
                diag = _DiagFrame(stem=stem) if cfg.enable_diagnostics else None
                erp_mask, direction, n_det = self._primary_detection(
                    erp, detection_size, timer=timer, diag=diag,
                )
                if diag is not None:
                    diag_frames[stem] = diag
                primary_masks[stem] = erp_mask
                person_directions[stem] = direction
                detection_counts[stem] = n_det
                frame_order.append(stem)

                if direction is not None:
                    yaw_d, pitch_d = _direction_to_yaw_pitch(direction)
                    print(f"[360] Pass 1: frame {fi+1}/{n_frames} — "
                          f"{n_det}/{len(DETECTION_LAYOUT)} views detected, "
                          f"direction yaw={yaw_d:.0f}° pitch={pitch_d:.0f}°")
                else:
                    print(f"[360] Pass 1: frame {fi+1}/{n_frames} — "
                          f"{n_det}/{len(DETECTION_LAYOUT)} views detected, no direction")

                if progress_callback:
                    progress_callback(
                        fi + 1, n_frames,
                        f"Pass 1: frame {fi+1}/{n_frames}",
                    )

            # ── Phase 2: Synthetic camera pass ────────────────────
            n_detected = sum(1 for d in person_directions.values() if d is not None)
            print(f"[360] Pass 1 complete: {n_detected}/{n_frames} frames with person direction")
            if cfg.enable_synthetic and self._video_backend is not None:
                # Check if any frame has a valid direction
                has_any_direction = n_detected > 0
                if has_any_direction:
                    print(f"[360] Starting Pass 2 (video tracking)")
                    try:
                        synthetic_masks = self._synthetic_pass(
                            frame_files, primary_masks,
                            person_directions, detection_counts,
                            frame_order,
                            progress_callback=progress_callback,
                            timer=timer,
                            diag_frames=diag_frames if cfg.enable_diagnostics else None,
                        )
                        # Pass 2 is authoritative: where it succeeds,
                        # replace Pass 1. Where it fails, keep Pass 1.
                        # FullCircle uses the synthetic pass as the final
                        # mask geometry — Pass 1 is only for localization.
                        n_replaced = 0
                        for stem, syn_mask in synthetic_masks.items():
                            if syn_mask.sum() > 0:
                                primary_masks[stem] = syn_mask
                                n_replaced += 1
                        print(f"[360] Pass 2 authoritative on "
                              f"{n_replaced}/{len(synthetic_masks)} frames "
                              f"(rest kept Pass 1)")
                    except Exception as exc:
                        self._video_backend_error = str(exc)
                        logger.error("Video backend failed: %s", exc)
                        print(f"[360] Pass 2 backend failed: {exc}")
                        raise RuntimeError(
                            "SAM v2 video tracking failed during masking. "
                            "Masking now requires a healthy SAM v2 runtime "
                            "instead of silently falling back."
                        ) from exc

            # ── Phase 3: Save ERP masks (no morph-close at ERP level) ──
            # FullCircle dilates per-camera masks, not the full ERP.
            # ERP-level morph close bridges false positives across the
            # sphere. Per-view dilation happens in the reframer instead.
            for si, stem in enumerate(frame_order):
                merged = primary_masks[stem]
                inverted = ((merged == 0).astype(np.uint8)) * 255
                cv2.imwrite(str(out_path / f"{stem}.png"), inverted)
                masked_count += 1

                # Record final mask stats for diagnostics
                if cfg.enable_diagnostics and stem in diag_frames:
                    df = diag_frames[stem]
                    df.final_mask_pixels = int(merged.sum())
                    if bool(df.p2_tracked):
                        df.final_source = "pass2_tracked"
                    elif df.p2_rescued:
                        df.final_source = "pass2_rescued"
                    elif df.p1_direction_yaw is None:
                        df.final_source = "no_direction"
                    elif df.final_mask_pixels > 0:
                        df.final_source = "pass1_fallback"
                    else:
                        df.final_source = "empty"

                if progress_callback:
                    progress_callback(
                        si + 1, n_frames,
                        f"Saving masks: {si+1}/{n_frames}",
                    )

            # ── Write diagnostics JSON ──
            if cfg.enable_diagnostics and diag_frames:
                try:
                    diag_list = [diag_frames[s] for s in frame_order if s in diag_frames]
                    diag_path = out_path / "masking_diagnostics.json"
                    _write_diagnostics_json(
                        diag_path,
                        diag_list,
                        backend_name=self._backend_name,
                        video_backend_name=self._video_backend_name,
                        used_fallback=self._used_fallback_video_backend,
                    )
                    diagnostics_path = str(diag_path)
                    if progress_callback:
                        progress_callback(
                            n_frames,
                            n_frames,
                            f"Mask diagnostics ready: {diagnostics_path}",
                        )
                except Exception as diag_exc:
                    logger.warning("Failed to write masking diagnostics: %s", diag_exc)
                    print(f"[360] Diagnostics write failed: {diag_exc}")

        except Exception as exc:
            import traceback
            print(f"[360] MASKING FAILED: {exc}")
            traceback.print_exc()
            logger.error("Masking pipeline failed: %s", exc)
            timer.report()
            return MaskResult(
                success=False, total_frames=n_frames,
                masked_frames=masked_count, masks_dir=str(out_path),
                diagnostics_path=diagnostics_path,
                error=str(exc),
                backend_name=self._backend_name,
                video_backend_name=self._video_backend_name,
                used_fallback_video_backend=self._used_fallback_video_backend,
                video_backend_error=self._video_backend_error,
            )

        print(f"[360] Masking complete: {masked_count}/{n_frames} frames masked, "
              f"saved to {out_path}")
        timer.report()
        return MaskResult(
            success=True, total_frames=n_frames,
            masked_frames=masked_count, masks_dir=str(out_path),
            diagnostics_path=diagnostics_path,
            backend_name=self._backend_name,
            video_backend_name=self._video_backend_name,
            used_fallback_video_backend=self._used_fallback_video_backend,
            video_backend_error=self._video_backend_error,
        )
