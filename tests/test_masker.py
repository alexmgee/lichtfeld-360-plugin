# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for masker mask decoding and inversion."""
from __future__ import annotations

import numpy as np
import pytest

from core.masker import _decode_mask, _invert_mask


def test_decode_numpy_mask():
    """Numpy array mask should be binarized to 0/255."""
    raw = np.array([[0.8, 0.0], [0.0, 0.9]])
    result = _decode_mask(raw, (2, 2))
    assert result.dtype == np.uint8
    assert result[0, 0] == 255
    assert result[0, 1] == 0
    assert result[1, 1] == 255


def test_decode_empty_returns_zeros():
    """Unknown mask format should return all zeros."""
    result = _decode_mask("garbage", (4, 4))
    assert result.shape == (4, 4)
    assert np.all(result == 0)


def test_invert_mask():
    """Inversion should flip 0<->255 for COLMAP convention."""
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    inverted = _invert_mask(mask)
    assert inverted[0, 0] == 0     # was detected (255) -> remove (0)
    assert inverted[0, 1] == 255   # was background (0) -> keep (255)


def test_invert_mask_roundtrip():
    """Double inversion should return original."""
    mask = np.array([[255, 0, 128], [0, 255, 0]], dtype=np.uint8)
    assert np.array_equal(mask, _invert_mask(_invert_mask(mask)))


def test_decode_rle_integer_counts():
    """Integer RLE counts should decode correctly."""
    rle = {"counts": [2, 2], "size": [2, 2]}
    result = _decode_mask(rle, (2, 2))
    assert result.dtype == np.uint8
    assert result[0, 0] == 0
    assert result[0, 1] == 255


def test_decode_zeros_mask():
    """All-zero numpy array should produce all-zero output."""
    raw = np.zeros((3, 3))
    result = _decode_mask(raw, (3, 3))
    assert np.all(result == 0)
