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
    """Build a 3x3 rotation matrix from Euler angles (degrees).

    Convention: Rz(roll) @ Rx(pitch) @ Ry(yaw), applied right-to-left.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    Ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]
    )

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ]
    )

    return Rz @ Rx @ Ry


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

    # Focal length from FOV
    fov_rad = np.radians(fov_deg)
    f = (out_size / 2) / np.tan(fov_rad / 2)

    # Pixel grid centred on the optical axis
    u = np.arange(out_size) - out_size / 2
    v = np.arange(out_size) - out_size / 2
    u, v = np.meshgrid(u, v)

    # Camera-space rays
    x = u
    y = -v  # flip y for image coordinates
    z = np.full_like(u, f, dtype=np.float64)

    # Rotate into world coordinates
    xyz = np.stack([x, y, z], axis=-1)
    R = create_rotation_matrix(yaw_deg, pitch_deg, 0)
    xyz_rot = xyz @ R.T

    # Spherical coordinates
    x_r, y_r, z_r = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]
    theta = np.arctan2(x_r, z_r)  # longitude
    norm = np.linalg.norm(xyz_rot, axis=-1)
    phi = np.arcsin(np.clip(y_r / norm, -1, 1))  # latitude

    # Map to equirect pixel coordinates
    u_eq = (theta / np.pi + 1) / 2 * w_eq
    v_eq = (0.5 - phi / np.pi) * h_eq

    # Sample
    if mode == "nearest":
        u_eq = np.round(u_eq).astype(int) % w_eq
        v_eq = np.clip(np.round(v_eq).astype(int), 0, h_eq - 1)
        return equirect[v_eq, u_eq]
    else:
        map_x = u_eq.astype(np.float32) % w_eq
        map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)
        return cv2.remap(
            equirect,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )


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
        station_dirs: bool = False,
    ) -> Tuple[List[str], Optional[str]]:
        """Reframe a single equirectangular image.

        Returns:
            (list_of_output_filenames, error_message_or_None)
        """
        return _process_single_image(
            image_path, output_dir, self.config, mask_path, station_dirs
        )

    def reframe_batch(
        self,
        input_dir: str,
        output_dir: str,
        mask_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        station_dirs: bool = False,
    ) -> ReframeResult:
        """Reframe all equirectangular images in *input_dir*.

        Args:
            input_dir: Directory containing equirectangular images.
            output_dir: Destination for perspective views.
            mask_dir: Optional directory with masks (matched by stem).
            progress_callback: Called with (current, total, filename).
            station_dirs: Write per-source subdirectories + metadata.

        Returns:
            ReframeResult summarising the operation.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images: List[Path] = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

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
            for ext in extensions:
                for m in mask_path_obj.glob(ext):
                    mask_map[m.stem] = str(m)

        errors: List[str] = []
        total_outputs = 0

        for i, img in enumerate(images):
            m_path = mask_map.get(img.stem) if mask_dir else None
            outputs, error = _process_single_image(
                str(img), str(output_path), self.config, m_path, station_dirs
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


def _process_single_image(
    image_path: str,
    output_dir: str,
    config: ViewConfig,
    mask_path: Optional[str] = None,
    station_dirs: bool = False,
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

    if station_dirs:
        image_out_dir = out_root / "images" / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_out_dir = out_root

    mask_dir: Optional[Path] = None
    if mask is not None:
        if station_dirs:
            mask_dir = out_root / "masks" / stem
            mask_dir.mkdir(parents=True, exist_ok=True)
        else:
            mask_dir = out_root.parent / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

    views = config.get_all_views()
    for yaw, pitch, fov, view_name in views:
        persp = reframe_view(
            equirect,
            fov_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            out_size=config.output_size,
        )

        out_name = f"{stem}_{view_name}.jpg"
        out_path = image_out_dir / out_name

        cv2.imwrite(
            str(out_path), persp, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        )
        output_files.append(out_name)

        if mask is not None and mask_dir is not None:
            mask_persp = reframe_view(
                mask,
                fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                out_size=config.output_size,
                mode="nearest",
            )
            mask_persp = (mask_persp > 0).astype(np.uint8) * 255
            if station_dirs:
                mask_out = f"{stem}_{view_name}_mask.png"
            else:
                mask_out = f"{stem}_{view_name}.png"
            cv2.imwrite(str(mask_dir / mask_out), mask_persp)

    return output_files, None
