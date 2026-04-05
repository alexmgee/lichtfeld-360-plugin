# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Equirectangular ↔ cubemap projection for masking.

Ported from Reconstruction Zone's reconstruction_pipeline.py.
Splits an ERP image into 6 undistorted perspective faces for
detection, and merges face masks back to ERP space.
"""
from __future__ import annotations

import cv2
import numpy as np


class CubemapProjection:
    """Bidirectional ERP ↔ cubemap projection."""

    FACE_DIRS = ["front", "back", "left", "right", "up", "down"]

    def __init__(
        self, face_size: int | None = None, overlap_degrees: float = 0.0
    ) -> None:
        self.face_size = face_size
        self.overlap_degrees = overlap_degrees
        half_fov = (90.0 + overlap_degrees) / 2.0
        self._grid_extent = np.tan(np.radians(half_fov))

    def equirect2cubemap(self, equirect: np.ndarray) -> dict[str, np.ndarray]:
        """Split ERP image into 6 cubemap faces."""
        h, w = equirect.shape[:2]
        fs = self.face_size or min(1024, w // 4)
        extent = self._grid_extent
        grid = np.linspace(-extent, extent, fs)
        u, v = np.meshgrid(grid, grid)
        faces = {}
        for name in self.FACE_DIRS:
            x, y, z = self._face_to_xyz(name, u, v)
            lon = np.arctan2(x, -z)
            lat = np.arctan2(y, np.sqrt(x**2 + z**2))
            map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
            faces[name] = cv2.remap(
                equirect, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        return faces

    def cubemap2equirect(
        self, face_masks: dict[str, np.ndarray], output_size: tuple[int, int]
    ) -> np.ndarray:
        """Merge 6 face masks back to ERP space.

        Uses hard face assignment (no overlap mode for v1).
        Mask values: 0/1 uint8 throughout.
        """
        w, h = output_size
        fs = self.face_size or min(1024, w // 4)
        u_eq = np.linspace(0, 1, w)
        v_eq = np.linspace(0, 1, h)
        uu, vv = np.meshgrid(u_eq, v_eq)
        lon = (uu - 0.5) * 2 * np.pi
        lat = (0.5 - vv) * np.pi
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = -np.cos(lat) * np.cos(lon)
        output = np.zeros((h, w), dtype=np.uint8)
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        for name in self.FACE_DIRS:
            face_mask = face_masks.get(name)
            if face_mask is None:
                continue
            region = self._get_face_region(name, x, y, z, abs_x, abs_y, abs_z)
            if not np.any(region):
                continue
            fu, fv = self._xyz_to_face(name, x[region], y[region], z[region])
            px = np.clip(((fu + 1) / 2 * (fs - 1)).astype(int), 0, fs - 1)
            py = np.clip(((fv + 1) / 2 * (fs - 1)).astype(int), 0, fs - 1)
            output[region] = face_mask[py, px]
        return output

    @staticmethod
    def _face_to_xyz(name, u, v):
        ones = np.ones_like(u)
        if name == "front":   return u, -v, -ones
        if name == "back":    return -u, -v, ones
        if name == "left":    return -ones, -v, -u
        if name == "right":   return ones, -v, u
        if name == "up":      return u, ones, -v
        if name == "down":    return u, -ones, v

    @staticmethod
    def _xyz_to_face(name, x, y, z):
        if name == "front":   return x / np.abs(z), -y / np.abs(z)
        if name == "back":    return -x / np.abs(z), -y / np.abs(z)
        if name == "left":    return -z / np.abs(x), -y / np.abs(x)
        if name == "right":   return z / np.abs(x), -y / np.abs(x)
        if name == "up":      return x / np.abs(y), -z / np.abs(y)
        if name == "down":    return x / np.abs(y), z / np.abs(y)

    @staticmethod
    def _face_facing(name, x, y, z):
        if name == "front":   return z < 0
        if name == "back":    return z > 0
        if name == "left":    return x < 0
        if name == "right":   return x > 0
        if name == "up":      return y > 0
        if name == "down":    return y < 0

    @staticmethod
    def _get_face_region(name, x, y, z, ax, ay, az):
        if name == "front":   return (z < 0) & (az >= ax) & (az >= ay)
        if name == "back":    return (z > 0) & (az >= ax) & (az >= ay)
        if name == "left":    return (x < 0) & (ax >= ay) & (ax >= az)
        if name == "right":   return (x > 0) & (ax >= ay) & (ax >= az)
        if name == "up":      return (y > 0) & (ay >= ax) & (ay >= az)
        if name == "down":    return (y < 0) & (ay >= ax) & (ay >= az)
