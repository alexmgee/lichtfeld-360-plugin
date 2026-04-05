# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for masker module."""
from __future__ import annotations

import numpy as np
import pytest

from core.masker import (
    MaskConfig, MaskResult, Masker, _dilate_detection_mask,
    is_masking_available,
)


def test_is_masking_available_returns_bool():
    """is_masking_available should return a bool."""
    result = is_masking_available()
    assert isinstance(result, bool)


def test_mask_config_defaults():
    """MaskConfig should have sensible defaults."""
    cfg = MaskConfig()
    assert cfg.targets == ["person"]
    assert cfg.device == "cuda"
    assert cfg.backend_preference is None
    assert cfg.dilate_px == 2
    assert cfg.views == []


def test_mask_result_fields():
    """MaskResult should have expected fields."""
    result = MaskResult(success=True, masked_frames=5)
    assert result.success is True
    assert result.masked_frames == 5


def test_dilate_detection_mask_expands():
    """Dilation should expand detected region."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 1  # 10x10 blob
    dilated = _dilate_detection_mask(mask, 2)
    # Dilated region should be larger
    assert np.sum(dilated > 0) > np.sum(mask > 0)


def test_dilate_detection_mask_zero_px():
    """Zero dilation should return mask unchanged."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    result = _dilate_detection_mask(mask, 0)
    assert np.array_equal(result, mask)
