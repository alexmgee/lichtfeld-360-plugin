# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Ring-based view presets for equirectangular-to-pinhole reprojection.

Defines the configuration data structures (Ring, ViewConfig) and built-in
presets that control how an equirectangular image is decomposed into
multiple pinhole perspective views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Ring:
    """Configuration for a ring of views at a specific pitch angle.

    Args:
        pitch: Degrees from horizon (-90 to +90).
        count: Number of evenly-spaced views in this ring.
        fov: Field of view in degrees for each view.
        start_yaw: Starting yaw offset in degrees.
        flip_vertical: Mirror the rendered image top-to-bottom.
            Needed for cubemap pole faces where the standard unfolded
            layout requires a reflection, not a rotation.
    """

    pitch: float
    count: int
    fov: float = 65.0
    start_yaw: float = 0.0
    flip_vertical: bool = False

    def get_yaw_positions(self) -> List[float]:
        """Return the yaw angle for every view in this ring."""
        if self.count == 0:
            return []
        step = 360.0 / self.count
        return [self.start_yaw + i * step for i in range(self.count)]


@dataclass
class ViewConfig:
    """Full configuration describing all perspective views to extract.

    Args:
        rings: List of Ring definitions.
        include_zenith: Whether to include a top-down view (pitch=+90).
        include_nadir: Whether to include a bottom-up view (pitch=-90).
        zenith_fov: FOV for zenith/nadir views.
        output_size: Square output image side length in pixels.
        jpeg_quality: JPEG compression quality (1-100).
    """

    rings: List[Ring] = field(default_factory=list)
    include_zenith: bool = True
    include_nadir: bool = False
    zenith_fov: float = 65.0
    output_size: int = 1920
    jpeg_quality: int = 95

    def total_views(self) -> int:
        """Total number of views this configuration produces per frame."""
        count = sum(ring.count for ring in self.rings)
        if self.include_zenith:
            count += 1
        if self.include_nadir:
            count += 1
        return count

    def get_all_views(self) -> List[Tuple[float, float, float, str, bool]]:
        """Return (yaw, pitch, fov, name, flip_vertical) for every view."""
        views: List[Tuple[float, float, float, str, bool]] = []

        for ring_idx, ring in enumerate(self.rings):
            for view_idx, yaw in enumerate(ring.get_yaw_positions()):
                name = f"{ring_idx:02d}_{view_idx:02d}"
                views.append((yaw, ring.pitch, ring.fov, name, ring.flip_vertical))

        if self.include_zenith:
            views.append((0, 90, self.zenith_fov, "ZN_00", False))

        if self.include_nadir:
            views.append((0, -90, self.zenith_fov, "ND_00", False))

        return views

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "rings": [
                {
                    "pitch": r.pitch,
                    "count": r.count,
                    "fov": r.fov,
                    "start_yaw": r.start_yaw,
                    "flip_vertical": r.flip_vertical,
                }
                for r in self.rings
            ],
            "include_zenith": self.include_zenith,
            "include_nadir": self.include_nadir,
            "zenith_fov": self.zenith_fov,
            "output_size": self.output_size,
            "jpeg_quality": self.jpeg_quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ViewConfig:
        """Deserialize from a plain dictionary."""
        rings = [
            Ring(
                pitch=r["pitch"],
                count=r["count"],
                fov=r.get("fov", 65.0),
                start_yaw=r.get("start_yaw", 0.0),
                flip_vertical=r.get("flip_vertical", False),
            )
            for r in data.get("rings", [])
        ]
        return cls(
            rings=rings,
            include_zenith=data.get("include_zenith", True),
            include_nadir=data.get("include_nadir", False),
            zenith_fov=data.get("zenith_fov", 65.0),
            output_size=data.get("output_size", 1920),
            jpeg_quality=data.get("jpeg_quality", 95),
        )


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

VIEW_PRESETS: dict[str, ViewConfig] = {
    "cubemap": ViewConfig(
        rings=[
            Ring(pitch=0, count=4, fov=90),
            Ring(pitch=-90, count=1, fov=90),
            Ring(pitch=90, count=1, fov=90),
        ],
        include_zenith=False,
        include_nadir=False,
    ),
    "balanced": ViewConfig(
        rings=[
            Ring(pitch=0, count=6, fov=75),
            Ring(pitch=40, count=1, fov=75, start_yaw=30),
            Ring(pitch=-40, count=1, fov=75, start_yaw=210),
        ],
        include_zenith=True,
        zenith_fov=75,
    ),
    "standard": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=65),
            Ring(pitch=25, count=2, fov=65, start_yaw=22.5),
            Ring(pitch=-25, count=2, fov=65, start_yaw=112.5),
        ],
        include_zenith=True,
        zenith_fov=65,
    ),
    "dense": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=65),
            Ring(pitch=30, count=4, fov=65, start_yaw=22.5),
            Ring(pitch=-30, count=4, fov=65, start_yaw=67.5),
        ],
        include_zenith=True,
    ),
}

# Default preset
DEFAULT_PRESET = "cubemap"
