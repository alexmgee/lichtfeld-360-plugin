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

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import pycolmap

from .backends import get_backend, get_backend_name, MaskingBackend, VideoTrackingBackend
from .reframer import create_rotation_matrix

logger = logging.getLogger(__name__)

CUBEMAP_DIRECT_CONFIDENCE = 0.35


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


def _backproject_fisheye_mask_to_erp(
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a fisheye detection mask to ERP space.

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
    """
    erp_h: int
    erp_w: int
    valid_idx: np.ndarray    # flat indices into (erp_h * erp_w,)
    fish_px: np.ndarray      # int, fisheye x coords for valid pixels
    fish_py: np.ndarray      # int, fisheye y coords for valid pixels

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Sample a fisheye mask using the precomputed map."""
        erp_mask = np.zeros(self.erp_h * self.erp_w, dtype=np.uint8)
        erp_mask[self.valid_idx] = mask[self.fish_py, self.fish_px]
        return erp_mask.reshape(self.erp_h, self.erp_w)


def _build_backproject_map(
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> _BackprojectMap:
    """Precompute the fisheye→ERP backprojection lookup table.

    This is the expensive part of _backproject_fisheye_mask_to_erp:
    ERP grid → world rays → camera space → pycolmap.img_from_cam →
    validity checks.  The result can be reused across frames when
    the person direction (and thus R_world_from_cam) is stable.
    """
    erp_w, erp_h = erp_size
    fish_size = camera.width

    u = np.arange(erp_w, dtype=np.float64) + 0.5
    v = np.arange(erp_h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    lon = ((uu / erp_w) * 2 - 1) * np.pi
    lat = (0.5 - vv / erp_h) * np.pi

    x_w = np.cos(lat) * np.sin(lon)
    y_w = np.sin(lat)
    z_w = np.cos(lat) * np.cos(lon)
    dirs_world = np.stack([x_w.ravel(), y_w.ravel(), z_w.ravel()], axis=1)

    R_cam_from_world = R_world_from_cam.T
    dirs_cam = (R_cam_from_world @ dirs_world.T).T

    forward = dirs_cam[:, 2] > 1e-8
    if not np.any(forward):
        return _BackprojectMap(
            erp_h, erp_w,
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

    return _BackprojectMap(erp_h, erp_w, valid_idx, fish_px, fish_py)


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
    if all_directions[frame_idx] is not None:
        return all_directions[frame_idx]

    n = len(all_directions)
    max_d = max(frame_idx, n - 1 - frame_idx)

    for d in range(1, max_d + 1):
        for j in (frame_idx - d, frame_idx + d):
            if 0 <= j < n and all_directions[j] is not None:
                return all_directions[j]

    return None


def _direction_to_yaw_pitch(direction: np.ndarray) -> tuple[float, float]:
    """Convert a 3D unit direction to (yaw_deg, pitch_deg)."""
    dx, dy, dz = direction
    yaw = np.degrees(np.arctan2(dx, dz))
    pitch = np.degrees(np.arcsin(np.clip(dy, -1, 1)))
    return float(yaw), float(pitch)


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
    """Return True when at least one detection backend is importable."""
    return get_backend_name() is not None


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


@dataclass
class MaskResult:
    """Result of a masking run."""
    success: bool
    total_frames: int = 0
    masked_frames: int = 0
    masks_dir: str = ""
    error: str = ""
    backend_name: str = ""
    video_backend_name: str = ""
    used_fallback_video_backend: bool = False
    video_backend_error: str = ""


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
        from .backends import get_video_backend

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
            self._video_backend = get_video_backend(
                preference=self.config.backend_preference,
                fallback_image_backend=self._backend,
                targets=self.config.targets,
            )
            if self._video_backend is not None:
                self._video_backend_name = type(self._video_backend).__name__
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

        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        masks_path.mkdir(parents=True, exist_ok=True)

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
                    self.config.targets,
                    detection_confidence=CUBEMAP_DIRECT_CONFIDENCE,
                    single_primary_box=True,
                )

            # Convert: detection (1=detected) → COLMAP polarity (255=keep, 0=remove)
            if detection_mask.sum() > 0:
                keep_mask = ((detection_mask == 0).astype(np.uint8)) * 255
                # Per-view erosion (erode keep region = dilate remove region)
                keep_mask = cv2.erode(keep_mask, erode_kernel, iterations=1)
                masked_count += 1
            else:
                keep_mask = np.full((h, w), 255, dtype=np.uint8)
                unmasked_count += 1

            with timer.time("cubemap_write_mask"):
                cv2.imwrite(str(mask_out_path), keep_mask)

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

        total_frames = total_images // max(len(views), 1)
        return MaskResult(
            success=True,
            total_frames=total_frames,
            masked_frames=masked_count,
            masks_dir=str(masks_path),
            backend_name=self._backend_name,
        )

    def _primary_detection(
        self, erp: np.ndarray, detection_size: int,
        timer: _SubstageTimer | None = None,
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

        detection_views = DETECTION_LAYOUT
        n_views = len(detection_views)
        n_detections = 0

        for vi, (yaw, pitch, fov, view_name, flip_v) in enumerate(detection_views):
            face_img = _reframe_to_detection(
                erp, fov, yaw, pitch, detection_size, flip_v
            )

            # YOLO-only detection — boxes are enough for direction estimation
            image_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = self._backend._yolo(
                image_rgb, stream=True, verbose=False, conf=0.35,
                iou=0.6, classes=[0], agnostic_nms=False, max_det=20,
            )
            all_boxes = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for j in range(len(result.boxes)):
                    conf = float(result.boxes.conf[j])
                    if conf < 0.35:
                        continue
                    box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                    all_boxes.append(box)

            if not all_boxes:
                continue

            # Union bounding box of all detections for direction estimation
            boxes = np.array(all_boxes)
            x1 = boxes[:, 0].min()
            y1 = boxes[:, 1].min()
            x2 = boxes[:, 2].max()
            y2 = boxes[:, 3].max()
            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0
            box_area = float((x2 - x1) * (y2 - y1))

            total_px = detection_size * detection_size
            coverage_pct = 100.0 * box_area / total_px if total_px > 0 else 0.0

            n_detections += 1

            MIN_DIRECTION_COVERAGE_PCT = 5.0
            if coverage_pct >= MIN_DIRECTION_COVERAGE_PCT:
                direction = _pixel_com_to_3d_direction(
                    box_cx, box_cy, fov, yaw, pitch, detection_size, flip_v,
                )
                directions_and_weights.append((direction, box_area))

        person_direction = _compute_weighted_person_direction(
            directions_and_weights
        )

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
        resolved_dirs: list[np.ndarray | None] = []
        for i in range(len(frame_order)):
            resolved_dirs.append(_temporal_fallback_direction(i, all_dirs))

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

        # Map best_idx to synthetic frame index
        best_stem = frame_order[best_idx]
        initial_frame_idx = stem_to_syn_idx.get(best_stem, 0)

        print(f"[360] Pass 2: {len(synthetic_frames)} synthetic views rendered, "
              f"prompting on frame {best_idx} ({best_stem}, {best_count} detections), tracking...")

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
            )

        # Log tracking results
        n_tracked = sum(1 for m in tracked_masks if m is not None and m.sum() > 0)
        print(f"[360] Pass 2: tracked {n_tracked}/{len(tracked_masks)} frames, backprojecting...")

        # Backproject each synthetic mask to ERP.
        # If person direction is stable (angular spread < 10°), build
        # one shared backprojection map from the mean direction and
        # reuse it for all frames. Otherwise fall back to per-frame.
        valid_dirs = [d for d in resolved_dirs if d is not None]
        spread = _direction_angular_spread(valid_dirs)
        MAX_SPREAD_FOR_SHARED_MAP = 10.0

        # Get ERP size from first available mask
        sample_stem = next(iter(stem_to_syn_idx))
        erp_h, erp_w = primary_masks[sample_stem].shape[:2]

        shared_map: _BackprojectMap | None = None
        if spread <= MAX_SPREAD_FOR_SHARED_MAP and len(valid_dirs) >= 2:
            mean_dir = np.mean(valid_dirs, axis=0)
            mean_dir = mean_dir / np.linalg.norm(mean_dir)
            R_mean = _look_at_rotation(mean_dir)
            with timer.time("p2_bp_map_build") if timer else contextmanager(lambda: (yield))():
                shared_map = _build_backproject_map((erp_w, erp_h), camera, R_mean)
            yaw_m, pitch_m = _direction_to_yaw_pitch(mean_dir)
            logger.debug("Backprojection: shared map (spread=%.1f°, mean yaw=%.1f° pitch=%.1f°)",
                         spread, yaw_m, pitch_m)
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
            d = resolved_dirs[i]
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
        detection_size = None
        timer = _SubstageTimer()
        self._used_fallback_video_backend = False
        self._video_backend_error = ""

        try:
            # ── Phase 1: Primary detection ────────────────────────
            primary_masks: dict[str, np.ndarray] = {}
            person_directions: dict[str, np.ndarray | None] = {}
            detection_counts: dict[str, int] = {}
            frame_order: list[str] = []

            for fi, frame_file in enumerate(frame_files):
                with timer.time("p1_imread"):
                    erp = cv2.imread(str(frame_file))
                if erp is None:
                    continue

                erp_h, erp_w = erp.shape[:2]
                if detection_size is None:
                    detection_size = min(1024, erp_w // 4)

                erp_mask, direction, n_det = self._primary_detection(erp, detection_size, timer=timer)
                stem = frame_file.stem
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
                        logger.warning(
                            "Video backend failed, falling back to "
                            "per-frame detection: %s", exc,
                        )
                        print(f"[360] Pass 2 backend failed: {exc}")
                        # Clean up failed video backend
                        if self._video_backend is not None:
                            try:
                                self._video_backend.cleanup()
                            except Exception:
                                pass
                            self._video_backend = None

                        # Create fresh image backend for fallback
                        fallback_backend = get_backend(cfg.backend_preference)
                        if fallback_backend is not None:
                            fallback_backend.initialize()
                            self._video_backend = FallbackVideoBackend(
                                fallback_backend, cfg.targets,
                            )
                            self._video_backend_name = type(self._video_backend).__name__
                            self._used_fallback_video_backend = True
                            print(f"[360] Falling back to {self._video_backend_name}")
                            try:
                                synthetic_masks = self._synthetic_pass(
                                    frame_files, primary_masks,
                                    person_directions, detection_counts,
                                    frame_order,
                                    progress_callback=progress_callback,
                                    timer=timer,
                                )
                                n_replaced = 0
                                for stem, syn_mask in synthetic_masks.items():
                                    if syn_mask.sum() > 0:
                                        primary_masks[stem] = syn_mask
                                        n_replaced += 1
                                print(f"[360] Fallback Pass 2 authoritative on "
                                      f"{n_replaced}/{len(synthetic_masks)} frames")
                            finally:
                                fallback_backend.cleanup()
                                self._video_backend = None

            # ── Phase 3: Save ERP masks (no morph-close at ERP level) ──
            # FullCircle dilates per-camera masks, not the full ERP.
            # ERP-level morph close bridges false positives across the
            # sphere. Per-view dilation happens in the reframer instead.
            for si, stem in enumerate(frame_order):
                merged = primary_masks[stem]
                inverted = ((merged == 0).astype(np.uint8)) * 255
                cv2.imwrite(str(out_path / f"{stem}.png"), inverted)
                masked_count += 1

                if progress_callback:
                    progress_callback(
                        si + 1, n_frames,
                        f"Saving masks: {si+1}/{n_frames}",
                    )

        except Exception as exc:
            import traceback
            print(f"[360] MASKING FAILED: {exc}")
            traceback.print_exc()
            logger.error("Masking pipeline failed: %s", exc)
            timer.report()
            return MaskResult(
                success=False, total_frames=n_frames,
                masked_frames=masked_count, masks_dir=str(out_path),
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
            backend_name=self._backend_name,
            video_backend_name=self._video_backend_name,
            used_fallback_video_backend=self._used_fallback_video_backend,
            video_backend_error=self._video_backend_error,
        )
