# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Fisheye → Pinhole reframing engine.

Extracts pinhole perspective crops from calibrated fisheye images.
Ported from reconstruction-zone (prep360/core/fisheye_reframer.py).

Usage:
    from core.fisheye_calibration import default_osmo360_calibration
    from core.fisheye_reframer import FisheyeReframer, FISHEYE_PINHOLE_PRESET

    calib = default_osmo360_calibration()
    reframer = FisheyeReframer(calib)

    results = reframer.extract_all_views(front_img, back_img, FISHEYE_PINHOLE_PRESET)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .fisheye_calibration import DualFisheyeCalibration, FisheyeCalibration

logger = logging.getLogger(__name__)


# -- View definitions -------------------------------------------------------

@dataclass
class FisheyeView:
    """A single perspective view extracted from a fisheye image."""

    name: str               # folder name, e.g. "front_ctr_lo"
    yaw_deg: float          # yaw relative to fisheye optical axis
    pitch_deg: float        # pitch relative to fisheye optical axis
    fov_deg: float          # horizontal FOV of the perspective crop
    source_lens: str        # "front" or "back"


@dataclass
class FisheyeViewConfig:
    """Configuration for all perspective views from a dual-fisheye pair."""

    views: List[FisheyeView] = field(default_factory=list)
    crop_size: int = 1920
    quality: int = 95

    def total_views(self) -> int:
        return len(self.views)

    def views_for_lens(self, lens: str) -> List[FisheyeView]:
        return [v for v in self.views if v.source_lens == lens]

    def summary(self) -> str:
        front = len(self.views_for_lens("front"))
        back = len(self.views_for_lens("back"))
        fovs = sorted({v.fov_deg for v in self.views})
        return (
            f"Views: {self.total_views()} total "
            f"({front} front, {back} back), "
            f"FOV: {', '.join(f'{f:.0f}°' for f in fovs)}, "
            f"Crop: {self.crop_size}x{self.crop_size}"
        )


# -- Default preset (from fisheye-perspective-planner.html) -----------------

# Per-lens views: (id, yaw_deg, pitch_deg) relative to lens optical axis.
# IDs become folder names: images/front_ctr_lo/, images/back_ring_ul/, etc.
_DEFAULT_VIEWS = [
    ("ctr_hi",  0,      -30),
    ("ring_l",  -55,    0),
    ("ring_ll", -35.53, 50),
    ("ring_lr", 35.53,  50),
    ("ring_r",  55,     0),
    ("ring_ur", 35.53,  -50),
    ("ring_ul", -35.53, -50),
    ("ctr_lo",  0,      30),
]

FISHEYE_PINHOLE_PRESET = FisheyeViewConfig(
    views=[
        FisheyeView(
            name=f"front_{vid}", yaw_deg=yaw, pitch_deg=pitch,
            fov_deg=90.0, source_lens="front",
        )
        for vid, yaw, pitch in _DEFAULT_VIEWS
    ] + [
        FisheyeView(
            name=f"back_{vid}", yaw_deg=yaw, pitch_deg=pitch,
            fov_deg=90.0, source_lens="back",
        )
        for vid, yaw, pitch in _DEFAULT_VIEWS
    ],
    crop_size=1920,
)


# -- Rotation math ----------------------------------------------------------

def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation matrix for virtual camera orientation.

    Yaw rotates around Y axis, pitch around X axis.
    Convention matches reframer.py's create_rotation_matrix.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float64)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float64)

    return Rx @ Ry


# -- Reframer ---------------------------------------------------------------

class FisheyeReframer:
    """Extract perspective crops from calibrated fisheye images.

    Caches remap tables per (lens, yaw, pitch, fov, crop_size) so
    repeated calls (batch processing) are fast after the first frame.
    """

    def __init__(self, calibration: DualFisheyeCalibration):
        self.calib = calibration
        self._map_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def with_defaults(cls) -> FisheyeReframer:
        from .fisheye_calibration import default_osmo360_calibration
        return cls(default_osmo360_calibration())

    def _get_lens_calib(self, lens: str) -> FisheyeCalibration:
        if lens == "front":
            return self.calib.front
        if lens == "back":
            return self.calib.back
        raise ValueError(f"Unknown lens: {lens}")

    def _build_remap_tables(
        self,
        lens_calib: FisheyeCalibration,
        source_size: Tuple[int, int],
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        crop_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build remap tables for a specific perspective view.

        Uses cv2.fisheye.initUndistortRectifyMap with a rotation
        matrix to orient the virtual camera.
        """
        f_virtual = crop_size / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        new_K = np.array([
            [f_virtual, 0, crop_size / 2.0],
            [0, f_virtual, crop_size / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

        R = _rotation_matrix(yaw_deg, pitch_deg)
        src_w, src_h = source_size
        calib_w, calib_h = lens_calib.image_size
        K = lens_calib.camera_matrix.copy()
        if (src_w, src_h) != (calib_w, calib_h):
            sx = src_w / float(calib_w)
            sy = src_h / float(calib_h)
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K=K,
            D=lens_calib.dist_coeffs,
            R=R,
            P=new_K,
            size=(crop_size, crop_size),
            m1type=cv2.CV_32FC1,
        )
        return map1, map2

    def extract_view(
        self,
        fisheye_image: np.ndarray,
        view: FisheyeView,
        crop_size: int = 1920,
        mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Extract a single perspective view from a fisheye image.

        If mask is provided, returns (crop, mask_crop) tuple.
        Otherwise returns just the crop.
        """
        lens_calib = self._get_lens_calib(view.source_lens)
        source_size = (fisheye_image.shape[1], fisheye_image.shape[0])

        cache_key = (
            view.source_lens, view.yaw_deg, view.pitch_deg,
            view.fov_deg, crop_size, source_size,
        )
        if cache_key not in self._map_cache:
            self._map_cache[cache_key] = self._build_remap_tables(
                lens_calib, source_size, view.yaw_deg, view.pitch_deg,
                view.fov_deg, crop_size,
            )

        map1, map2 = self._map_cache[cache_key]
        crop = cv2.remap(
            fisheye_image, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        if mask is not None:
            mask_crop = cv2.remap(
                mask, map1, map2,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask_crop = (mask_crop > 0).astype(np.uint8) * 255
            return crop, mask_crop

        return crop

    def extract_all_views(
        self,
        front_image: np.ndarray,
        back_image: np.ndarray,
        config: FisheyeViewConfig,
        front_mask: Optional[np.ndarray] = None,
        back_mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[FisheyeView, np.ndarray, Optional[np.ndarray]]]:
        """Extract all perspective views from a front/back fisheye pair.

        Returns list of (view, crop, mask_crop_or_None) tuples.
        """
        results = []
        for view in config.views:
            if view.source_lens == "front":
                img, msk = front_image, front_mask
            else:
                img, msk = back_image, back_mask

            if msk is not None:
                crop, mask_crop = self.extract_view(img, view, config.crop_size, msk)
            else:
                crop = self.extract_view(img, view, config.crop_size)
                mask_crop = None

            results.append((view, crop, mask_crop))

        return results
