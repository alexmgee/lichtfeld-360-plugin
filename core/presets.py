# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
View presets for equirectangular-to-pinhole reprojection.

Defines the configuration data structures (Ring, FreeView, ViewConfig) and
built-in presets that control how an equirectangular image is decomposed
into multiple pinhole perspective views.

Views can be defined as evenly-spaced rings (Ring) or individually-placed
cameras (FreeView).  Both produce the same output format and are fully
compatible with COLMAP rig constraints.
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
class FreeView:
    """A single individually-placed camera view.

    Args:
        name: View identifier, used as the output folder name and
            COLMAP image_prefix (e.g. ``"00_00"``).
        yaw: Yaw angle in degrees (0 = front).
        pitch: Pitch angle in degrees (0 = horizon, +90 = up).
        fov: Field of view in degrees.
    """

    name: str
    yaw: float
    pitch: float
    fov: float = 65.0


@dataclass
class ViewConfig:
    """Full configuration describing all perspective views to extract.

    Views can come from three sources (emitted in this order):

    1. ``rings`` — evenly-spaced cameras at a given pitch.
    2. ``views`` — individually-placed freeform cameras.
    3. ``include_zenith`` / ``include_nadir`` — shorthand pole cameras.

    Args:
        rings: List of Ring definitions.
        views: List of individually-placed FreeView cameras.
        include_zenith: Whether to include a top-down view (pitch=+90).
        include_nadir: Whether to include a bottom-up view (pitch=-90).
        zenith_fov: FOV for zenith/nadir views.
        output_size: Square output image side length in pixels.
        jpeg_quality: JPEG compression quality (1-100).
    """

    rings: List[Ring] = field(default_factory=list)
    views: List[FreeView] = field(default_factory=list)
    include_zenith: bool = True
    include_nadir: bool = False
    zenith_fov: float = 65.0
    output_size: int = 1920
    jpeg_quality: int = 95

    def total_views(self) -> int:
        """Total number of views this configuration produces per frame."""
        count = sum(ring.count for ring in self.rings)
        count += len(self.views)
        if self.include_zenith:
            count += 1
        if self.include_nadir:
            count += 1
        return count

    def get_all_views(self) -> List[Tuple[float, float, float, str, bool]]:
        """Return (yaw, pitch, fov, name, flip_vertical) for every view."""
        result: List[Tuple[float, float, float, str, bool]] = []

        for ring_idx, ring in enumerate(self.rings):
            for view_idx, yaw in enumerate(ring.get_yaw_positions()):
                name = f"{ring_idx:02d}_{view_idx:02d}"
                result.append((yaw, ring.pitch, ring.fov, name, ring.flip_vertical))

        for fv in self.views:
            result.append((fv.yaw, fv.pitch, fv.fov, fv.name, False))

        if self.include_zenith:
            result.append((0, 90, self.zenith_fov, "ZN_00", False))

        if self.include_nadir:
            result.append((0, -90, self.zenith_fov, "ND_00", False))

        return result

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        d: dict = {
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
        if self.views:
            d["views"] = [
                {
                    "name": v.name,
                    "yaw": v.yaw,
                    "pitch": v.pitch,
                    "fov": v.fov,
                }
                for v in self.views
            ]
        return d

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
        views = [
            FreeView(
                name=v["name"],
                yaw=v["yaw"],
                pitch=v["pitch"],
                fov=v.get("fov", 65.0),
            )
            for v in data.get("views", [])
        ]
        return cls(
            rings=rings,
            views=views,
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
    "low": ViewConfig(
        views=[
            FreeView("00_00", yaw=52, pitch=16, fov=75),
            FreeView("00_01", yaw=119, pitch=-17, fov=75),
            FreeView("00_02", yaw=143, pitch=34, fov=75),
            FreeView("00_03", yaw=-131, pitch=16, fov=75),
            FreeView("00_04", yaw=-66, pitch=-14, fov=75),
            FreeView("00_05", yaw=-37, pitch=69, fov=75),
            FreeView("01_00", yaw=30, pitch=-62, fov=75),
            FreeView("02_00", yaw=210, pitch=-44, fov=75),
            FreeView("02_01", yaw=-19, pitch=0, fov=75),
        ],
        include_zenith=False,
        include_nadir=False,
    ),
    "medium": ViewConfig(
        views=[
            FreeView("00_00", yaw=159, pitch=12, fov=65),
            FreeView("00_01", yaw=88, pitch=0, fov=65),
            FreeView("00_02", yaw=-133, pitch=0, fov=65),
            FreeView("00_03", yaw=-55, pitch=0, fov=65),
            FreeView("00_04", yaw=7, pitch=0, fov=65),
            FreeView("01_00", yaw=23, pitch=30, fov=65),
            FreeView("01_01", yaw=113, pitch=30, fov=65),
            FreeView("01_02", yaw=202, pitch=30, fov=65),
            FreeView("01_03", yaw=293, pitch=30, fov=65),
            FreeView("02_00", yaw=68, pitch=-30, fov=65),
            FreeView("02_01", yaw=158, pitch=-46, fov=65),
            FreeView("02_02", yaw=248, pitch=-33, fov=65),
            FreeView("02_03", yaw=338, pitch=-46, fov=65),
            FreeView("ZN_00", yaw=0, pitch=90, fov=65),
        ],
        include_zenith=False,
        include_nadir=False,
    ),
    "high": ViewConfig(
        views=[
            FreeView("ZN_00", yaw=0, pitch=90, fov=65),
            FreeView("00_00", yaw=3, pitch=34, fov=65),
            FreeView("00_01", yaw=175, pitch=36, fov=65),
            FreeView("00_02", yaw=91, pitch=34, fov=65),
            FreeView("00_03", yaw=-89, pitch=33, fov=65),
            FreeView("02_00", yaw=133, pitch=-25, fov=65),
            FreeView("02_01", yaw=-47, pitch=-24, fov=65),
            FreeView("02_02", yaw=-139, pitch=-22, fov=65),
            FreeView("02_04", yaw=42, pitch=-26, fov=65),
            FreeView("02_05", yaw=-1, pitch=-27, fov=65),
            FreeView("02_06", yaw=91, pitch=-29, fov=65),
            FreeView("02_07", yaw=-89, pitch=-27, fov=65),
            FreeView("02_03", yaw=-180, pitch=-29, fov=65),
            FreeView("ND_00", yaw=180, pitch=-90, fov=65),
            FreeView("01_00", yaw=143, pitch=21, fov=65),
            FreeView("01_01", yaw=49, pitch=20, fov=65),
            FreeView("01_02", yaw=-47, pitch=21, fov=65),
            FreeView("01_03", yaw=-139, pitch=28, fov=65),
        ],
        include_zenith=False,
        include_nadir=False,
    ),
}

# Default preset
DEFAULT_PRESET = "cubemap"
