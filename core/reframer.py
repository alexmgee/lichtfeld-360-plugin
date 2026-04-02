# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reframe engine — equirectangular to pinhole perspective reprojection.

Provides the low-level reprojection math (rotation matrix construction,
pixel-grid ray casting, spherical coordinate mapping) and a high-level
``Reframer`` class for batch processing with progress callbacks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .presets import Ring, ViewConfig, VIEW_PRESETS


# ---------------------------------------------------------------------------
# Reprojection math
# ---------------------------------------------------------------------------


def create_rotation_matrix(
    yaw_deg: float, pitch_deg: float, roll_deg: float = 0
) -> np.ndarray:
    """Build a 3x3 world-to-camera rotation matrix from yaw/pitch angles.

    Direct transliteration of the webapp's ``w2c(yaw, pitch)`` function.
    Rows are ``[right, up, -forward]`` in world coordinates.

    The webapp is the single source of truth for this math.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # s2d(yaw, pitch) → forward direction
    fwd = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw),
    ])

    # r = cross(fwd, [0,1,0]) — matches webapp w2c() exactly
    r = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    rl = np.linalg.norm(r)
    if rl < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / rl

    # u = cross(r, fwd)
    u = np.cross(r, fwd)

    # Return [r, u, -fwd] — identical to webapp w2c()
    return np.array([r, u, -fwd])


def reframe_view(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    out_size: int,
    mode: str = "bilinear",
) -> np.ndarray:
    """Extract a single perspective view from an equirectangular image.

    Args:
        equirect: Input equirectangular image (H, W, C) or (H, W).
        fov_deg: Field of view in degrees.
        yaw_deg: Yaw angle in degrees (0 = front).
        pitch_deg: Pitch angle in degrees (0 = horizon, +90 = up).
        out_size: Output image side length (square).
        mode: Interpolation — ``"bilinear"`` or ``"nearest"``.

    Returns:
        Perspective view as a numpy array of shape (out_size, out_size, C)
        (or (out_size, out_size) for single-channel input).
    """
    h_eq, w_eq = equirect.shape[:2]

    # --- Direct transliteration of webapp sampleERP() ---
    # Source: erp-perspective-planner.html, lines 341-357
    fov_rad = np.radians(fov_deg)
    f = (out_size / 2) / np.tan(fov_rad / 2)
    cxx = out_size / 2
    cy = out_size / 2

    R = create_rotation_matrix(yaw_deg, pitch_deg)
    # Unpack R into scalars matching webapp variable names
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    # Pixel grid
    px_arr = np.arange(out_size, dtype=np.float64)
    py_arr = np.arange(out_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px_arr, py_arr)

    crx = (px_grid - cxx) / f
    cry = -(py_grid - cy) / f

    # World-space ray:  webapp line 349
    # wx = r00*crx + r10*cry - r20
    # wy = r01*crx + r11*cry - r21
    # wz = r02*crx + r12*cry - r22
    wx = r00 * crx + r10 * cry - r20
    wy = r01 * crx + r11 * cry - r21
    wz = r02 * crx + r12 * cry - r22

    # Normalize
    l = np.sqrt(wx * wx + wy * wy + wz * wz)
    wx, wy, wz = wx / l, wy / l, wz / l

    # Spherical coordinates:  webapp line 351
    theta = np.arctan2(wx, wz)                     # longitude
    phi = np.arcsin(np.clip(wy, -1, 1))             # latitude

    # Map to equirect pixel coordinates:  webapp line 352
    u_eq = ((theta / np.pi + 1) / 2) * w_eq
    v_eq = (0.5 - phi / np.pi) * h_eq

    # Sample — then flip horizontally to convert from the webapp's
    # left-handed camera convention to right-handed output for COLMAP.
    if mode == "nearest":
        u_eq = np.round(u_eq).astype(int) % w_eq
        v_eq = np.clip(np.round(v_eq).astype(int), 0, h_eq - 1)
        return np.fliplr(equirect[v_eq, u_eq])
    else:
        map_x = u_eq.astype(np.float32) % w_eq
        map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)
        result = cv2.remap(
            equirect,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        return np.fliplr(result)


def compute_pinhole_intrinsics(fov_deg: float, crop_size: int) -> dict:
    """Compute pinhole camera intrinsics from FOV and output size."""
    f = crop_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    cx = crop_size / 2.0
    cy = crop_size / 2.0
    return {
        "model": "PINHOLE",
        "width": crop_size,
        "height": crop_size,
        "fx": round(f, 4),
        "fy": round(f, 4),
        "cx": cx,
        "cy": cy,
    }


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------


@dataclass
class ReframeResult:
    """Result of a batch reframing operation."""

    success: bool
    input_count: int
    output_count: int
    output_dir: str
    errors: List[str] = field(default_factory=list)


class Reframer:
    """High-level interface for reframing equirectangular images."""

    def __init__(self, config: Optional[ViewConfig] = None):
        from .presets import DEFAULT_PRESET
        self.config = config or VIEW_PRESETS[DEFAULT_PRESET]

    def reframe_single(
        self,
        image_path: str,
        output_dir: str,
        mask_path: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """Reframe a single equirectangular image.

        Returns:
            (list_of_output_filenames, error_message_or_None)
        """
        return _process_single_image(
            image_path, output_dir, self.config, mask_path
        )

    def reframe_batch(
        self,
        input_dir: str,
        output_dir: str,
        mask_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ReframeResult:
        """Reframe all equirectangular images in *input_dir*.

        Args:
            input_dir: Directory containing equirectangular images.
            output_dir: Destination for perspective views.
            mask_dir: Optional directory with masks (matched by stem).
            progress_callback: Called with (current, total, filename).

        Returns:
            ReframeResult summarising the operation.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = _collect_image_files(input_path)

        if not images:
            return ReframeResult(
                success=False,
                input_count=0,
                output_count=0,
                output_dir=str(output_path),
                errors=["No images found in input directory"],
            )

        # Build mask lookup
        mask_map: dict[str, str] = {}
        if mask_dir:
            mask_path_obj = Path(mask_dir)
            for m in _collect_image_files(mask_path_obj):
                mask_map[m.stem] = str(m)

        errors: List[str] = []
        total_outputs = 0

        for i, img in enumerate(images):
            m_path = mask_map.get(img.stem) if mask_dir else None
            outputs, error = _process_single_image(
                str(img), str(output_path), self.config, m_path
            )
            if error:
                errors.append(error)
            elif outputs:
                total_outputs += len(outputs)

            if progress_callback:
                progress_callback(i + 1, len(images), img.name)

        return ReframeResult(
            success=len(errors) == 0,
            input_count=len(images),
            output_count=total_outputs,
            output_dir=str(output_path),
            errors=errors,
        )

    def preview_view_positions(self) -> str:
        """Return a human-readable summary of view positions."""
        lines = [f"Total views: {self.config.total_views()}"]

        for i, ring in enumerate(self.config.rings):
            spacing = 360.0 / ring.count if ring.count > 0 else 0
            lines.append(
                f"Ring {i}: pitch={ring.pitch:+.0f} deg  "
                f"count={ring.count}  spacing={spacing:.0f} deg  "
                f"fov={ring.fov} deg"
            )

        if self.config.include_zenith:
            lines.append(f"Zenith: pitch=+90 deg  fov={self.config.zenith_fov} deg")
        if self.config.include_nadir:
            lines.append(f"Nadir: pitch=-90 deg  fov={self.config.zenith_fov} deg")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_image_files(directory: Path) -> List[Path]:
    """Collect image files once, even on case-insensitive filesystems.

    Windows globbing is case-insensitive, so querying both ``*.jpg`` and
    ``*.JPG`` can return the same file twice. Deduplicating here keeps the
    reframe counts honest and avoids processing the same panorama multiple
    times.
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    seen: set[str] = set()
    files: List[Path] = []

    for ext in extensions:
        for path in directory.glob(ext):
            norm = os.path.normcase(str(path.resolve()))
            if norm in seen:
                continue
            seen.add(norm)
            files.append(path)

    return sorted(files, key=lambda p: os.path.normcase(str(p)))


def _process_single_image(
    image_path: str,
    output_dir: str,
    config: ViewConfig,
    mask_path: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """Process one equirectangular image into all configured views."""
    equirect = cv2.imread(image_path)
    if equirect is None:
        return [], f"Failed to load {image_path}"

    mask: Optional[np.ndarray] = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return [], f"Failed to load mask {mask_path}"
        if mask.shape[:2] != equirect.shape[:2]:
            return [], (
                f"Mask dimensions {mask.shape[:2]} don't match "
                f"image {equirect.shape[:2]} for {image_path}"
            )

    stem = Path(image_path).stem
    out_root = Path(output_dir)
    output_files: List[str] = []
    mask_root = out_root.parent / "masks" if mask is not None else None

    views = config.get_all_views()
    for yaw, pitch, fov, view_name, flip_v in views:
        persp = reframe_view(
            equirect,
            fov_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            out_size=config.output_size,
        )

        if flip_v:
            persp = np.flipud(persp)

        view_dir = out_root / view_name
        view_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{stem}.jpg"
        out_path = view_dir / out_name

        cv2.imwrite(
            str(out_path), persp, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        )
        output_files.append(f"{view_name}/{out_name}")

        if mask is not None and mask_root is not None:
            mask_persp = reframe_view(
                mask,
                fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                out_size=config.output_size,
                mode="nearest",
            )
            if flip_v:
                mask_persp = np.flipud(mask_persp)
            mask_persp = (mask_persp > 0).astype(np.uint8) * 255
            mask_dir = mask_root / view_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_out = f"{stem}.png"
            cv2.imwrite(str(mask_dir / mask_out), mask_persp)

    return output_files, None
