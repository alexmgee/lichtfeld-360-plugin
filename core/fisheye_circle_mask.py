"""Fisheye image circle mask generation.

Ported from reconstruction-zone prep360/core/fisheye_reframer.py.
Generates a binary validity mask for fisheye images:
  0 = valid (inside image circle)
  1 = masked (outside circle or within margin)
"""
from __future__ import annotations

import cv2
import numpy as np


def generate_fisheye_circle_mask(
    width: int,
    height: int,
    margin_percent: float = 6.0,
) -> np.ndarray:
    """Generate a binary validity mask for a fisheye image.

    The image circle fills edge-to-edge on a square sensor. The mask
    radius is::

        r_mask = min(width, height) / 2 * (1 - margin_percent / 100)

    Args:
        width:  Image width in pixels.
        height: Image height in pixels.
        margin_percent: Percentage of circle radius to trim inward
            (0 = corners only, 6 = default production value).

    Returns:
        uint8 ndarray (h, w): 0 = valid, 1 = masked.
    """
    mask = np.ones((height, width), dtype=np.uint8)
    cx, cy = width / 2.0, height / 2.0
    r_full = min(width, height) / 2.0
    r_valid = r_full * (1.0 - margin_percent / 100.0)
    cv2.circle(mask, (int(cx), int(cy)), int(r_valid), 0, thickness=-1)
    return mask
