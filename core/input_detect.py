# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lightweight input-type detection.

Stdlib-only on purpose: the UI panel calls this while handling the
video-select event, before any heavy dependency (cv2, torch, pycolmap)
is loaded. Keeping this module import-light guarantees a broken
dependency can never prevent video selection (issues #6/#8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def detect_input_type(video_path: str) -> tuple[str, Optional[str]]:
    """Auto-detect (input_type, camera_family) from file extension.

    Returns:
        ("dual_fisheye", "dji_osmo360") for .osv (DJI Osmo 360)
        ("dual_fisheye", "insta360") for .insv (file pair OR X4/X5 single-file)
        ("erp", None) for .mp4 / .mov / other single-stream video
    """
    suffix = Path(video_path).suffix.lower()
    if suffix == ".osv":
        return "dual_fisheye", "dji_osmo360"
    if suffix == ".insv":
        return "dual_fisheye", "insta360"
    return "erp", None
