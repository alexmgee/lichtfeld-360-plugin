# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for pipeline orchestrator helper functions."""

import time
from pathlib import Path
from types import SimpleNamespace
import shutil
import uuid

import cv2
import numpy as np

from core.colmap_runner import ColmapResult
from core.pipeline import (
    PipelineConfig,
    PipelineJob,
    _build_runtime_view_config,
    _format_preset_signature,
)
from core.presets import VIEW_PRESETS


def test_build_runtime_view_config_cubemap():
    cfg = PipelineConfig(preset_name="cubemap", output_size=1024, jpeg_quality=90)
    vc = _build_runtime_view_config(cfg)
    assert vc.total_views() == VIEW_PRESETS["cubemap"].total_views()
    assert vc.output_size == 1024
    assert vc.jpeg_quality == 90


def test_build_runtime_view_config_default():
    cfg = PipelineConfig(preset_name="default", output_size=2048, jpeg_quality=95)
    vc = _build_runtime_view_config(cfg)
    assert vc.total_views() == VIEW_PRESETS["default"].total_views()
    assert vc.output_size == 2048


def test_build_runtime_view_config_unknown_preset_falls_back():
    cfg = PipelineConfig(preset_name="nonexistent", output_size=1920)
    vc = _build_runtime_view_config(cfg)
    # Should fall back to DEFAULT_PRESET ("default")
    assert vc.total_views() == VIEW_PRESETS["default"].total_views()


def test_build_runtime_view_config_copies_rings():
    cfg = PipelineConfig(preset_name="cubemap")
    vc = _build_runtime_view_config(cfg)
    assert len(vc.rings) == len(VIEW_PRESETS["cubemap"].rings)
    for orig, copy in zip(VIEW_PRESETS["cubemap"].rings, vc.rings):
        assert orig.count == copy.count
        assert orig.pitch == copy.pitch
        assert orig.fov == copy.fov


def test_build_runtime_view_config_copies_views():
    cfg = PipelineConfig(preset_name="default")
    vc = _build_runtime_view_config(cfg)
    assert len(vc.views) == len(VIEW_PRESETS["default"].views)


def test_format_preset_signature_cubemap():
    vc = VIEW_PRESETS["cubemap"]
    sig = _format_preset_signature("cubemap", vc)
    assert sig.startswith("cubemap | ")
    # Cubemap has 3 rings
    assert "00:" in sig
    assert "01:" in sig
    assert "02:" in sig


def test_format_preset_signature_default():
    vc = VIEW_PRESETS["default"]
    sig = _format_preset_signature("default", vc)
    assert sig.startswith("default | ")


def test_format_preset_signature_includes_zenith():
    from core.presets import ViewConfig, Ring
    vc = ViewConfig(
        rings=[Ring(pitch=0, count=4, fov=90)],
        include_zenith=True,
        zenith_fov=120,
    )
    sig = _format_preset_signature("test", vc)
    assert "ZN@90" in sig
    assert "f120" in sig


def test_format_preset_signature_includes_nadir():
    from core.presets import ViewConfig, Ring
    vc = ViewConfig(
        rings=[Ring(pitch=0, count=4, fov=90)],
        include_nadir=True,
        zenith_fov=100,
    )
    sig = _format_preset_signature("test", vc)
    assert "ND@-90" in sig


def test_sam3_route_preserves_mask_result_contract_for_cubemap_output(monkeypatch):
    tmp_root = Path.cwd() / "tmp" / "pytest"
    tmp_root.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_root / f"pipeline_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        cfg = PipelineConfig(
            video_path=str(tmp_path / "input.mp4"),
            output_dir=str(tmp_path / "out"),
            enable_masking=True,
            masking_method="sam3_cubemap",
            preset_name="cubemap",
            enable_overlap_masks=True,
            enable_diagnostics=True,
        )

        overlap_calls: list[tuple[int, int]] = []

        def fake_extract(self, video_path, output_dir, config, progress_callback=None, cancel_check=None):
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame = np.full((256, 512, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), frame)
            return SimpleNamespace(success=True, frames_extracted=1, error="")

        class FakeSam3CubemapMasker:
            def __init__(self, config):
                self.config = config

            def initialize(self):
                return None

            def process_frames(self, frames_dir, output_dir, view_config, progress_callback=None):
                masks_root = Path(output_dir) / "masks"
                for _yaw, _pitch, _fov, view_name, _flip_v in view_config.get_all_views():
                    view_dir = masks_root / view_name
                    view_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(view_dir / "frame_00001.png"),
                        np.full((128, 128), 255, dtype=np.uint8),
                    )
                return SimpleNamespace(
                    success=True,
                    total_frames=1,
                    masked_frames=1,
                    mask_dir=str(masks_root),
                    backend_name="Sam3Backend",
                    diagnostics_path=str(masks_root / "masking_diagnostics.json"),
                )

            def cleanup(self):
                return None

        def fake_reframe_batch(self, input_dir, output_dir, mask_dir=None, progress_callback=None):
            images_dir = Path(output_dir)
            for _yaw, _pitch, _fov, view_name, _flip_v in self.config.get_all_views():
                view_dir = images_dir / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(view_dir / "frame_00001.jpg"),
                    np.full((128, 128, 3), 127, dtype=np.uint8),
                )
            return SimpleNamespace(success=True, output_count=6, input_count=1, errors=[])

        def fake_compute_overlap_masks(views, output_size):
            overlap_calls.append((len(views), output_size))
            return {
                view_name: np.full((128, 128), 255, dtype=np.uint8)
                for _yaw, _pitch, _fov, view_name, _flip_v in views
            }

        class FakeColmapRunner:
            def __init__(self, *args, **kwargs):
                self.output_dir = Path(kwargs["output_dir"])

            def run(self):
                sparse_dir = self.output_dir / "sparse" / "0"
                sparse_dir.mkdir(parents=True, exist_ok=True)
                return ColmapResult(
                    success=True,
                    reconstruction_path=str(sparse_dir),
                    num_registered_images=6,
                    num_registered_frames=1,
                    num_complete_frames=1,
                    views_per_frame=6,
                )

        monkeypatch.setattr("core.pipeline.is_sam3_masking_ready", lambda: True)
        monkeypatch.setattr("core.pipeline.SharpestExtractor.extract", fake_extract)
        monkeypatch.setattr("core.sam3_masker.Sam3CubemapMasker", FakeSam3CubemapMasker)
        monkeypatch.setattr("core.pipeline.Reframer.reframe_batch", fake_reframe_batch)
        monkeypatch.setattr("core.pipeline.compute_overlap_masks", fake_compute_overlap_masks)
        monkeypatch.setattr("core.pipeline.write_rig_config", lambda *args, **kwargs: None)
        monkeypatch.setattr("core.pipeline.infer_shared_pinhole_camera_params", lambda *args, **kwargs: (None, None, None))
        monkeypatch.setattr("core.pipeline.ColmapRunner", FakeColmapRunner)

        job = PipelineJob(cfg)
        result = job._run_stages(cfg, time.time())

        assert result.success
        assert result.mask_backend_name == "Sam3Backend"
        assert result.mask_diagnostics_path.endswith("masking_diagnostics.json")
        assert overlap_calls == [(6, cfg.output_size)]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_sam3_route_supports_default_output_preset(monkeypatch):
    tmp_root = Path.cwd() / "tmp" / "pytest"
    tmp_root.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_root / f"pipeline_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        cfg = PipelineConfig(
            video_path=str(tmp_path / "input.mp4"),
            output_dir=str(tmp_path / "out"),
            enable_masking=True,
            masking_method="sam3_cubemap",
            preset_name="default",
            enable_overlap_masks=True,
            enable_diagnostics=True,
        )

        overlap_calls: list[tuple[int, int]] = []

        def fake_extract(self, video_path, output_dir, config, progress_callback=None, cancel_check=None):
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame = np.full((256, 512, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(frames_dir / "frame_00001.jpg"), frame)
            return SimpleNamespace(success=True, frames_extracted=1, error="")

        class FakeSam3CubemapMasker:
            def __init__(self, config):
                self.config = config

            def initialize(self):
                return None

            def process_frames(self, frames_dir, output_dir, view_config, progress_callback=None):
                masks_root = Path(output_dir) / "masks"
                for _yaw, _pitch, _fov, view_name, _flip_v in view_config.get_all_views():
                    view_dir = masks_root / view_name
                    view_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(view_dir / "frame_00001.png"),
                        np.full((128, 128), 255, dtype=np.uint8),
                    )
                return SimpleNamespace(
                    success=True,
                    total_frames=1,
                    masked_frames=1,
                    mask_dir=str(masks_root),
                    backend_name="Sam3Backend",
                    diagnostics_path=str(masks_root / "masking_diagnostics.json"),
                )

            def cleanup(self):
                return None

        def fake_reframe_batch(self, input_dir, output_dir, mask_dir=None, progress_callback=None):
            images_dir = Path(output_dir)
            for _yaw, _pitch, _fov, view_name, _flip_v in self.config.get_all_views():
                view_dir = images_dir / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(view_dir / "frame_00001.jpg"),
                    np.full((128, 128, 3), 127, dtype=np.uint8),
                )
            return SimpleNamespace(
                success=True,
                output_count=len(self.config.get_all_views()),
                input_count=1,
                errors=[],
            )

        def fake_compute_overlap_masks(views, output_size):
            overlap_calls.append((len(views), output_size))
            return {
                view_name: np.full((128, 128), 255, dtype=np.uint8)
                for _yaw, _pitch, _fov, view_name, _flip_v in views
            }

        class FakeColmapRunner:
            def __init__(self, *args, **kwargs):
                self.output_dir = Path(kwargs["output_dir"])

            def run(self):
                sparse_dir = self.output_dir / "sparse" / "0"
                sparse_dir.mkdir(parents=True, exist_ok=True)
                return ColmapResult(
                    success=True,
                    reconstruction_path=str(sparse_dir),
                    num_registered_images=16,
                    num_registered_frames=1,
                    num_complete_frames=1,
                    views_per_frame=16,
                )

        monkeypatch.setattr("core.pipeline.is_sam3_masking_ready", lambda: True)
        monkeypatch.setattr("core.pipeline.SharpestExtractor.extract", fake_extract)
        monkeypatch.setattr("core.sam3_masker.Sam3CubemapMasker", FakeSam3CubemapMasker)
        monkeypatch.setattr("core.pipeline.Reframer.reframe_batch", fake_reframe_batch)
        monkeypatch.setattr("core.pipeline.compute_overlap_masks", fake_compute_overlap_masks)
        monkeypatch.setattr("core.pipeline.write_rig_config", lambda *args, **kwargs: None)
        monkeypatch.setattr("core.pipeline.infer_shared_pinhole_camera_params", lambda *args, **kwargs: (None, None, None))
        monkeypatch.setattr("core.pipeline.ColmapRunner", FakeColmapRunner)

        job = PipelineJob(cfg)
        result = job._run_stages(cfg, time.time())

        assert result.success
        assert result.mask_backend_name == "Sam3Backend"
        assert result.views_per_frame == 16
        assert result.num_output_images == 16
        assert overlap_calls == [(16, cfg.output_size)]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
