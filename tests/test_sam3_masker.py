# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Sam3CubemapMasker — geometry-aware cubemap SAM 3 helper."""

import json
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
import shutil
import uuid

import cv2


def _make_erp(w=2048, h=1024):
    """Create a synthetic ERP image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_temp_workspace() -> Path:
    root = Path.cwd() / "tmp" / "pytest"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"sam3_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestSam3CubemapMaskerImport:

    def test_import(self):
        from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig, Sam3MaskerResult
        assert Sam3CubemapMasker is not None
        assert Sam3MaskerConfig is not None
        assert Sam3MaskerResult is not None


class TestSam3CubemapMaskerPipeline:

    def _make_masker_with_mock_backend(self, detect_return=None):
        """Create a Sam3CubemapMasker with a mocked backend."""
        from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

        config = Sam3MaskerConfig(
            prompts=["person"],
            confidence_threshold=0.3,
            output_size=512,
        )
        masker = Sam3CubemapMasker(config)

        mock_backend = MagicMock()
        if detect_return is not None:
            mock_backend.detect_and_segment.return_value = detect_return
        else:
            # Default: return zeros (no detection)
            def _no_detect(image, targets, **kwargs):
                h, w = image.shape[:2]
                return np.zeros((h, w), dtype=np.uint8)
            mock_backend.detect_and_segment.side_effect = _no_detect

        masker._backend = mock_backend
        masker._initialized = True
        return masker, mock_backend

    def test_process_single_frame_output_layout(self):
        """Process one ERP frame and verify masks land in {view_id}/{frame}.png."""
        from core.presets import VIEW_PRESETS

        tmp_path = _make_temp_workspace()
        try:
            erp = _make_erp()
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

            view_config = VIEW_PRESETS["cubemap"]
            masker, _ = self._make_masker_with_mock_backend()

            output_dir = tmp_path / "output"
            result = masker.process_frames(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                view_config=view_config,
            )

            assert result.success
            assert result.total_frames == 1

            masks_dir = output_dir / "masks"
            assert masks_dir.exists()

            # Cubemap preset has 6 views
            views = view_config.get_all_views()
            assert len(views) == 6
            for _, _, _, view_name, _ in views:
                view_dir = masks_dir / view_name
                assert view_dir.exists(), f"Missing mask dir for view {view_name}"
                mask_files = list(view_dir.glob("*.png"))
                assert len(mask_files) == 1, f"Expected 1 mask in {view_name}, got {len(mask_files)}"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_mask_polarity_colmap(self):
        """Output masks must be COLMAP polarity: white=keep (255), black=remove (0)."""
        from core.presets import VIEW_PRESETS

        tmp_path = _make_temp_workspace()
        try:
            erp = _make_erp()
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

            # Mock backend detects something in center of every face
            def _detect_center(image, targets, **kwargs):
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cy, cx = h // 2, w // 2
                mask[cy - 50:cy + 50, cx - 50:cx + 50] = 1
                return mask

            masker, mock = self._make_masker_with_mock_backend()
            mock.detect_and_segment.side_effect = _detect_center

            view_config = VIEW_PRESETS["cubemap"]
            output_dir = tmp_path / "output"
            masker.process_frames(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                view_config=view_config,
            )

            # Read any output mask
            masks_dir = output_dir / "masks"
            any_mask_file = next(masks_dir.rglob("*.png"))
            mask = cv2.imread(str(any_mask_file), cv2.IMREAD_GRAYSCALE)

            assert mask is not None
            # COLMAP polarity: detected region should be 0, background 255
            assert 0 in mask, "Mask should have black (removed) pixels"
            assert 255 in mask, "Mask should have white (kept) pixels"
            # Background should be majority
            assert np.sum(mask == 255) > np.sum(mask == 0)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_no_detection_produces_all_white_masks(self):
        """When SAM 3 detects nothing, all masks should be 255 (keep everything)."""
        from core.presets import VIEW_PRESETS

        tmp_path = _make_temp_workspace()
        try:
            erp = _make_erp()
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

            masker, _ = self._make_masker_with_mock_backend()  # default: no detection
            view_config = VIEW_PRESETS["cubemap"]

            output_dir = tmp_path / "output"
            result = masker.process_frames(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                view_config=view_config,
            )

            assert result.success
            assert result.masked_frames == 0

            any_mask_file = next((output_dir / "masks").rglob("*.png"))
            mask = cv2.imread(str(any_mask_file), cv2.IMREAD_GRAYSCALE)
            assert np.all(mask == 255)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_backend_called_six_times_for_cubemap(self):
        """Backend.detect_and_segment should be called once per cubemap face (6 times)."""
        from core.presets import VIEW_PRESETS

        tmp_path = _make_temp_workspace()
        try:
            erp = _make_erp()
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

            masker, mock = self._make_masker_with_mock_backend()
            view_config = VIEW_PRESETS["cubemap"]

            output_dir = tmp_path / "output"
            masker.process_frames(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                view_config=view_config,
            )

            # 6 cubemap faces × 1 frame = 6 calls
            assert mock.detect_and_segment.call_count == 6
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_diagnostics_written_when_enabled(self):
        """Diagnostics mode should write masking_diagnostics.json with frame stats."""
        from core.presets import VIEW_PRESETS
        from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

        tmp_path = _make_temp_workspace()
        try:
            erp = _make_erp()
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

            config = Sam3MaskerConfig(
                prompts=["person"],
                confidence_threshold=0.3,
                output_size=512,
                enable_diagnostics=True,
            )
            masker = Sam3CubemapMasker(config)

            mock_backend = MagicMock()

            def _detect_front_only(image, targets, **kwargs):
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[h // 4:h // 2, w // 4:w // 2] = 1
                return mask

            mock_backend.detect_and_segment.side_effect = _detect_front_only
            masker._backend = mock_backend
            masker._initialized = True

            output_dir = tmp_path / "output"
            result = masker.process_frames(
                frames_dir=str(frames_dir),
                output_dir=str(output_dir),
                view_config=VIEW_PRESETS["cubemap"],
            )

            assert result.success
            assert result.diagnostics_path

            diag_path = Path(result.diagnostics_path)
            assert diag_path.exists()

            doc = json.loads(diag_path.read_text(encoding="utf-8"))
            assert doc["mode"] == "sam3_cubemap"
            assert doc["backend"] == "MagicMock"
            assert doc["total_frames"] == 1
            assert doc["masked_frames"] == 1
            assert "summary" in doc
            assert len(doc["frames"]) == 1

            summary = doc["summary"]
            assert summary["flagged_frames"] >= 0
            assert "avg_erp_detection_coverage_pct_post" in summary
            assert "avg_face_detection_pct" in summary

            frame = doc["frames"][0]
            assert frame["frame"] == "frame_00001"
            assert frame["faces_with_detections"] == 6
            assert "erp_detection_coverage_pct_post" in frame
            assert "per_view_removed_pct" in frame
            assert isinstance(frame["flags"], list)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_fill_holes(self):
        """_fill_mask_holes should fill interior regions unreachable from edges."""
        from core.sam3_masker import Sam3CubemapMasker

        # Create a mask with a ring (interior hole)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, thickness=3)  # ring

        filled = Sam3CubemapMasker._fill_mask_holes(mask)

        # The interior of the ring should now be filled
        assert filled[50, 50] == 1


class TestCubemapFaceMapping:
    """Verify the mapping between CubemapProjection face names and plugin view IDs."""

    def test_cubemap_produces_six_faces(self):
        from core.cubemap_projection import CubemapProjection

        erp = _make_erp()
        cubemap = CubemapProjection(512)
        faces = cubemap.equirect2cubemap(erp)

        assert set(faces.keys()) == {"front", "back", "left", "right", "up", "down"}
        for name, face in faces.items():
            assert face.shape == (512, 512, 3), f"{name} has wrong shape: {face.shape}"

    def test_cubemap_preset_has_six_views(self):
        from core.presets import VIEW_PRESETS

        views = VIEW_PRESETS["cubemap"].get_all_views()
        assert len(views) == 6

        # Verify view names match expected ring-based naming
        names = [v[3] for v in views]
        assert "00_00" in names  # front (yaw=0, pitch=0)
        assert "00_01" in names  # right (yaw=90, pitch=0)
        assert "00_02" in names  # back (yaw=180, pitch=0)
        assert "00_03" in names  # left (yaw=270, pitch=0)
        assert "01_00" in names  # nadir (pitch=-90)
        assert "02_00" in names  # zenith (pitch=90)
