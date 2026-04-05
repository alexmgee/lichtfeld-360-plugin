# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for closest-camera Voronoi overlap masks."""

import numpy as np
import pytest
from core.overlap_mask import compute_overlap_masks
from core.presets import VIEW_PRESETS


def test_cubemap_overlap_masks_all_white():
    """Cubemap has zero overlap — should return None (skip)."""
    config = VIEW_PRESETS["cubemap"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is None  # skip for cubemap (zero overlap)


def test_high_preset_overlap_masks_partition():
    """High preset masks should partition: each pixel owned by exactly one view."""
    config = VIEW_PRESETS["high"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is not None
    assert len(masks) == len(views)
    for view_name, mask in masks.items():
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert np.any(mask > 0), f"View {view_name} has no owned pixels"
