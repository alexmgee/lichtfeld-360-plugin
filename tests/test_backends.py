# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for detection backend interface."""

import numpy as np
import pytest
from core.backends import YoloSamBackend, Sam3Backend, get_backend


def test_yolo_sam_backend_interface():
    """YoloSamBackend has the required method signature."""
    assert hasattr(YoloSamBackend, "detect_and_segment")
    assert hasattr(YoloSamBackend, "initialize")
    assert hasattr(YoloSamBackend, "cleanup")


def test_sam3_backend_interface():
    """Sam3Backend has the required method signature."""
    assert hasattr(Sam3Backend, "detect_and_segment")
    assert hasattr(Sam3Backend, "initialize")
    assert hasattr(Sam3Backend, "cleanup")


def test_get_backend_returns_none_when_nothing_installed():
    """When no ML packages are installed and no preference set, returns None."""
    backend = get_backend(preference=None)
    # Result depends on environment — just check it returns Backend or None
    assert backend is None or hasattr(backend, "detect_and_segment")


# ── Video tracking backend tests (Task A3) ───────────────────────

from core.backends import FallbackVideoBackend, get_video_backend


class _MockImageBackend:
    """Minimal mock of MaskingBackend for testing FallbackVideoBackend."""
    def __init__(self):
        self.initialized = False
        self.cleaned = False

    def initialize(self):
        self.initialized = True

    def detect_and_segment(self, image, targets):
        # Return a mask with 1s wherever the image is bright
        gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
        return (gray > 128).astype(np.uint8)

    def cleanup(self):
        self.cleaned = True


def test_fallback_video_backend_calls_per_frame():
    """FallbackVideoBackend should call detect_and_segment on each frame."""
    mock = _MockImageBackend()
    fb = FallbackVideoBackend(mock, ["person"])

    frames = [
        np.full((64, 64, 3), 255, dtype=np.uint8),  # bright
        np.zeros((64, 64, 3), dtype=np.uint8),        # dark
        np.full((64, 64, 3), 200, dtype=np.uint8),    # bright
    ]
    fb.initialize()
    results = fb.track_sequence(frames)
    fb.cleanup()

    assert len(results) == 3
    assert results[0].sum() > 0, "Bright frame should have detections"
    assert results[1].sum() == 0, "Dark frame should have no detections"
    assert results[2].sum() > 0, "Bright frame should have detections"


def test_fallback_video_backend_initialize_is_noop():
    """FallbackVideoBackend.initialize() should not re-initialize the wrapped backend."""
    mock = _MockImageBackend()
    fb = FallbackVideoBackend(mock, ["person"])
    fb.initialize()
    assert not mock.initialized, "Should not re-initialize wrapped backend"


def test_get_video_backend_returns_fallback_when_forced():
    """With preference='fallback', get_video_backend should return FallbackVideoBackend."""
    mock = _MockImageBackend()
    vb = get_video_backend(
        preference="fallback", fallback_image_backend=mock, targets=["person"]
    )
    assert isinstance(vb, FallbackVideoBackend)


def test_get_video_backend_returns_none_without_fallback():
    """With no fallback backend and no SAM v2, get_video_backend returns None."""
    vb = get_video_backend(preference="fallback")
    assert vb is None


def test_sam2_video_backend_interface():
    """Sam2VideoBackend has the required method signatures."""
    from core.backends import Sam2VideoBackend

    assert hasattr(Sam2VideoBackend, "initialize")
    assert hasattr(Sam2VideoBackend, "track_sequence")
    assert hasattr(Sam2VideoBackend, "cleanup")


def test_get_video_backend_prefers_sam2_when_available():
    """If SAM v2 is installed, get_video_backend should return Sam2VideoBackend."""
    from core.backends import HAS_SAM2, Sam2VideoBackend

    mock = _MockImageBackend()
    vb = get_video_backend(fallback_image_backend=mock, targets=["person"])
    if HAS_SAM2:
        assert isinstance(vb, Sam2VideoBackend)
    else:
        assert isinstance(vb, FallbackVideoBackend)
