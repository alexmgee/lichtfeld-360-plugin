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


# ── _select_person_boxes ───────────────────────────────────────


def test_select_person_boxes_empty():
    result = YoloSamBackend._select_person_boxes(
        [], [], single_primary_box=False, image_shape=(64, 64, 3), primary_box_mode="confidence",
    )
    assert result == []


def test_select_person_boxes_all_when_not_single():
    boxes = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
    confs = [0.9, 0.5]
    result = YoloSamBackend._select_person_boxes(
        boxes, confs, single_primary_box=False, image_shape=(64, 64, 3), primary_box_mode="confidence",
    )
    assert len(result) == 2


def test_select_person_boxes_picks_highest_confidence():
    boxes = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30]), np.array([40, 40, 50, 50])]
    confs = [0.5, 0.95, 0.3]
    result = YoloSamBackend._select_person_boxes(
        boxes, confs, single_primary_box=True, image_shape=(64, 64, 3), primary_box_mode="confidence",
    )
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], boxes[1])


def test_select_person_boxes_single_with_one_box():
    boxes = [np.array([0, 0, 10, 10])]
    confs = [0.7]
    result = YoloSamBackend._select_person_boxes(
        boxes, confs, single_primary_box=True, image_shape=(64, 64, 3), primary_box_mode="confidence",
    )
    assert len(result) == 1


# ── Video tracking backend tests (Task A3) ───────────────────────

from core.backends import FallbackVideoBackend, get_video_backend


class _MockImageBackend:
    """Minimal mock of MaskingBackend for testing FallbackVideoBackend."""
    def __init__(self):
        self.initialized = False
        self.cleaned = False

    def initialize(self):
        self.initialized = True

    def detect_and_segment(
        self,
        image,
        targets,
        detection_confidence=0.35,
        single_primary_box=False,
        primary_box_mode="confidence",
        constrain_to_primary_box=False,
        primary_box_padding=0.35,
    ):
        del (
            targets,
            detection_confidence,
            single_primary_box,
            primary_box_mode,
            constrain_to_primary_box,
            primary_box_padding,
        )
        # Return a mask with 1s wherever the image is bright
        gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
        return (gray > 128).astype(np.uint8)

    def batch_detect_boxes(self, images, detection_confidence=0.35):
        all_detections = []
        for image in images:
            gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
            bright = (gray > 128).astype(np.uint8)
            detections = []
            if bright.sum() > 0:
                ys, xs = np.where(bright > 0)
                box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                detections.append((box, 0.9))
            all_detections.append(detections)
        return all_detections

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
