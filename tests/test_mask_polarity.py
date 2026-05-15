"""Tests for fisheye mask polarity conversion and combination."""
import cv2
import numpy as np

from core.fisheye_circle_mask import generate_fisheye_circle_mask


def test_polarity_conversion():
    """Circle mask (0=valid, 1=masked) converts to COLMAP (0=excluded, 255=valid)."""
    circle = generate_fisheye_circle_mask(100, 100, margin_percent=0.0)
    circle_for_colmap = (1 - circle).astype(np.uint8) * 255
    # Center should be 255 (valid)
    assert circle_for_colmap[50, 50] == 255
    # Corner should be 0 (excluded)
    assert circle_for_colmap[0, 0] == 0


def test_bitwise_and_combination():
    """Combined mask is valid only where BOTH circle and operator say valid."""
    circle = generate_fisheye_circle_mask(100, 100, margin_percent=0.0)
    circle_for_colmap = (1 - circle).astype(np.uint8) * 255

    # Operator mask: top half valid, bottom half excluded
    sam_mask = np.zeros((100, 100), dtype=np.uint8)
    sam_mask[:50, :] = 255

    combined = cv2.bitwise_and(sam_mask, circle_for_colmap)

    # Center-top: both valid → 255
    assert combined[25, 50] == 255
    # Center-bottom: operator excludes → 0
    assert combined[75, 50] == 0
    # Corner-top: circle excludes → 0
    assert combined[0, 0] == 0
