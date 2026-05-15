"""Tests for fisheye circle mask generation."""
import numpy as np

from core.fisheye_circle_mask import generate_fisheye_circle_mask


def test_square_image_center_valid():
    """Center pixel of a square image should be valid (0)."""
    mask = generate_fisheye_circle_mask(100, 100, margin_percent=0.0)
    assert mask[50, 50] == 0  # center is valid


def test_square_image_corner_masked():
    """Corner pixel of a square image should be masked (1)."""
    mask = generate_fisheye_circle_mask(100, 100, margin_percent=0.0)
    assert mask[0, 0] == 1  # corner is outside circle


def test_output_shape_and_dtype():
    mask = generate_fisheye_circle_mask(3840, 3840, margin_percent=6.0)
    assert mask.shape == (3840, 3840)
    assert mask.dtype == np.uint8


def test_margin_shrinks_valid_area():
    """Higher margin should produce more masked pixels."""
    mask_0 = generate_fisheye_circle_mask(200, 200, margin_percent=0.0)
    mask_10 = generate_fisheye_circle_mask(200, 200, margin_percent=10.0)
    valid_0 = int(np.sum(mask_0 == 0))
    valid_10 = int(np.sum(mask_10 == 0))
    assert valid_10 < valid_0


def test_only_zeros_and_ones():
    mask = generate_fisheye_circle_mask(200, 200, margin_percent=6.0)
    assert set(np.unique(mask)) <= {0, 1}
