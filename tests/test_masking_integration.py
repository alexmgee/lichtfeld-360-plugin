# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for masking components.

Tests cubemap projection round-trip and overlap mask computation.
Does NOT require torch/YOLO/SAM — uses synthetic data.
"""
import numpy as np
import pytest
from core.cubemap_projection import CubemapProjection
from core.overlap_mask import compute_overlap_masks
from core.presets import VIEW_PRESETS


def _mock_backend_that_detects_bright(image, targets):
    """Mock backend: masks any pixel brighter than 128 as detected."""
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    return (gray > 128).astype(np.uint8)


def test_cubemap_decompose_merge_round_trip():
    """Synthetic ERP with a bright rectangle survives cubemap round-trip."""
    erp = np.zeros((200, 400, 3), dtype=np.uint8)
    erp[80:120, 180:220] = 200  # bright blob at center (front face)

    proj = CubemapProjection(face_size=64)
    faces = proj.equirect2cubemap(erp)

    face_masks = {}
    for name, face_img in faces.items():
        mask = _mock_backend_that_detects_bright(face_img, ["person"])
        face_masks[name] = mask

    assert np.any(face_masks["front"] > 0), "Front face should detect the bright blob"

    erp_mask = proj.cubemap2equirect(face_masks, (400, 200))
    center_detected = erp_mask[85:115, 185:215].mean()
    assert center_detected > 0.3, f"Center blob not detected after merge: {center_detected:.2f}"


def test_overlap_masks_reduce_mask_area():
    """Overlap masks should reduce total mask area for overlapping presets."""
    config = VIEW_PRESETS["high"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is not None

    total_white = sum(np.sum(m > 0) for m in masks.values())
    total_pixels = len(views) * 64 * 64
    ratio = total_white / total_pixels
    assert ratio < 1.0, f"Overlap masks should reduce total area, got ratio={ratio:.2f}"


def test_masker_config_accepts_views():
    """MaskConfig should accept a view list from any preset."""
    from core.masker import MaskConfig
    for preset_name, config in VIEW_PRESETS.items():
        views = config.get_all_views()
        cfg = MaskConfig(views=views)
        assert len(cfg.views) == len(views), f"Preset {preset_name}: view count mismatch"


# ── A5: Two-pass masker integration tests ────────────────────────


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
from core.masker import Masker, MaskConfig, MaskResult
from core.backends import FallbackVideoBackend


class _BrightBlobBackend:
    """Mock backend: detects bright regions (>128) as person."""

    def initialize(self):
        pass

    def detect_and_segment(self, image, targets):
        gray = np.mean(image, axis=2) if image.ndim == 3 else image.astype(float)
        return (gray > 128).astype(np.uint8)

    def cleanup(self):
        pass


def _write_test_erp_sequence(tmpdir: Path, n_frames: int = 3) -> list[Path]:
    """Write synthetic ERP frames with a bright blob at the center (forward)."""
    frames = []
    for i in range(n_frames):
        erp = np.zeros((128, 256, 3), dtype=np.uint8)
        # Bright blob at center (person at yaw=0, equator)
        erp[50:78, 115:141] = 200
        path = tmpdir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), erp)
        frames.append(path)
    return frames


def test_two_pass_masker_produces_output():
    """Two-pass masker with FallbackVideoBackend produces mask files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        masks_dir = tmpdir / "masks"
        frames_dir.mkdir()

        _write_test_erp_sequence(frames_dir, n_frames=2)

        mock_backend = _BrightBlobBackend()
        views = VIEW_PRESETS["cubemap"].get_all_views()

        cfg = MaskConfig(
            views=views,
            enable_synthetic=True,
            synthetic_size=128,
        )
        masker = Masker(cfg)
        masker._backend = mock_backend
        masker._video_backend = FallbackVideoBackend(mock_backend, ["person"])

        result = masker.process_frames(str(frames_dir), str(masks_dir))

        assert result.success, f"Masker failed: {result.error}"
        assert result.masked_frames == 2
        mask_files = list(masks_dir.iterdir())
        assert len(mask_files) == 2


def test_synthetic_pass_skipped_when_disabled():
    """With enable_synthetic=False, only Pass 1 runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        masks_dir = tmpdir / "masks"
        frames_dir.mkdir()

        _write_test_erp_sequence(frames_dir, n_frames=1)

        mock_backend = _BrightBlobBackend()
        views = VIEW_PRESETS["cubemap"].get_all_views()

        cfg = MaskConfig(
            views=views,
            enable_synthetic=False,
        )
        masker = Masker(cfg)
        masker._backend = mock_backend
        # No video backend set — should not matter since synthetic is disabled

        result = masker.process_frames(str(frames_dir), str(masks_dir))
        assert result.success


def test_no_detection_clip_skips_pass2():
    """An ERP with no detectable content should skip Pass 2 gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        masks_dir = tmpdir / "masks"
        frames_dir.mkdir()

        # Write solid black frames (nothing to detect)
        for i in range(2):
            erp = np.zeros((128, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), erp)

        mock_backend = _BrightBlobBackend()
        views = VIEW_PRESETS["cubemap"].get_all_views()

        cfg = MaskConfig(
            views=views,
            enable_synthetic=True,
            synthetic_size=128,
        )
        masker = Masker(cfg)
        masker._backend = mock_backend
        masker._video_backend = FallbackVideoBackend(mock_backend, ["person"])

        result = masker.process_frames(str(frames_dir), str(masks_dir))
        assert result.success
        assert result.masked_frames == 2
        # Masks should be all-white (255 = keep, nothing detected)
        for mf in masks_dir.iterdir():
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            assert mask.min() == 255, "No detections → mask should be all-white (keep)"


def test_video_backend_failure_falls_back():
    """If the video backend fails, process_frames should fall back gracefully."""

    class _FailingVideoBackend:
        def initialize(self):
            pass
        def track_sequence(self, frames, initial_mask=None, initial_frame_idx=0):
            raise RuntimeError("Simulated SAM v2 crash")
        def cleanup(self):
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        masks_dir = tmpdir / "masks"
        frames_dir.mkdir()

        _write_test_erp_sequence(frames_dir, n_frames=2)

        mock_backend = _BrightBlobBackend()
        views = VIEW_PRESETS["cubemap"].get_all_views()

        cfg = MaskConfig(
            views=views,
            enable_synthetic=True,
            synthetic_size=128,
        )
        masker = Masker(cfg)
        masker._backend = mock_backend
        masker._video_backend = _FailingVideoBackend()

        # Patch get_backend to return a fresh mock for the fallback path
        with patch("core.masker.get_backend", return_value=_BrightBlobBackend()):
            result = masker.process_frames(str(frames_dir), str(masks_dir))

        assert result.success, f"Should recover from video backend failure: {result.error}"
        assert result.masked_frames == 2
