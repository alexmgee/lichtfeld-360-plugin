# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
COLMAP SfM pipeline runner.

Based directly on the Lichtfeld-COLMAP-Plugin's proven patterns for running
pycolmap inside LFS (daemon thread, _try_set_attr resilience, os.fspath paths,
CameraMode, camera_mode param, dict-based reconstruction access).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _try_set_attr(obj, attr: str, value) -> bool:
    """Best-effort set for pybind option objects (older pycolmap builds may lack some fields)."""
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


@dataclass
class ColmapConfig:
    preset: str = "normal"
    camera_model: str = "PINHOLE"

    @property
    def sift_max_image_size(self) -> int:
        return 1600 if self.preset == "normal" else 1200

    @property
    def sift_max_num_features(self) -> int:
        return 2048 if self.preset == "normal" else 1536


@dataclass
class ColmapResult:
    success: bool
    reconstruction_path: str = ""
    num_registered_images: int = 0
    num_points3d: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


ProgressCallback = Callable[[str, float, str], None]
CancelCheck = Callable[[], bool]


class ColmapRunner:
    def __init__(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        rig_config_path: str | Path,
        mask_path: str | Path | None = None,
        config: Optional[ColmapConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> None:
        self._images_dir = Path(images_dir)
        self._output_dir = Path(output_dir)
        self._rig_config_path = Path(rig_config_path)
        self._mask_path = Path(mask_path) if mask_path else None
        self._config = config or ColmapConfig()
        self._on_progress = on_progress
        self._cancel_check = cancel_check

    def _progress(self, stage: str, percent: float, message: str) -> None:
        logger.info("[%s %.0f%%] %s", stage, percent * 100, message)
        if self._on_progress:
            self._on_progress(stage, percent, message)

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_check and self._cancel_check():
            raise RuntimeError("Cancelled by user")

    def run(self) -> ColmapResult:
        t0 = time.monotonic()
        try:
            import pycolmap
        except ImportError:
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error="pycolmap is not installed. Install it with: pip install pycolmap",
            )
        try:
            return self._run_pipeline(pycolmap, t0)
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            if "Cancelled" in str(exc):
                return ColmapResult(success=False, elapsed_sec=elapsed, error="Cancelled by user")
            logger.exception("COLMAP pipeline failed")
            return ColmapResult(success=False, elapsed_sec=elapsed, error=str(exc))
        except Exception as exc:
            logger.exception("COLMAP pipeline failed")
            return ColmapResult(success=False, elapsed_sec=time.monotonic() - t0, error=str(exc))

    def _run_pipeline(self, pycolmap, t0: float) -> ColmapResult:
        # File-based debug log — survives native crashes
        debug_log = self._output_dir / "colmap_debug.log"

        def _log(msg: str) -> None:
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(f"[{time.monotonic() - t0:.1f}s] {msg}\n")
                f.flush()

        # --- Setup (matches COLMAP plugin pattern) ---
        self._output_dir.mkdir(parents=True, exist_ok=True)
        _log("Pipeline started")

        sparse_dir = self._output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        database_path = os.fspath(self._output_dir / "database.db")
        image_path = os.fspath(self._images_dir)
        sparse_path = os.fspath(sparse_dir)

        # Clean stale database + WAL/SHM files (from COLMAP plugin)
        for db_file in [database_path, f"{database_path}-wal", f"{database_path}-shm"]:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

        num_threads = min(8, os.cpu_count() or 4)
        sift_max_image_size = self._config.sift_max_image_size
        sift_max_num_features = self._config.sift_max_num_features

        # ================================================
        # FEATURE EXTRACTION (from COLMAP plugin lines 643-740)
        # ================================================
        self._ensure_not_cancelled()
        _log("Step 1: Feature extraction")
        self._progress("features", 0.0, "Extracting SIFT features...")

        # CameraMode (from COLMAP plugin line 652)
        from pycolmap import CameraMode
        camera_mode = CameraMode.AUTO

        reader_opts = None
        extraction_opts = None
        extraction_gpu_requested = False

        # ImageReaderOptions (from COLMAP plugin line 661)
        if hasattr(pycolmap, "ImageReaderOptions"):
            reader_opts = pycolmap.ImageReaderOptions()
            _try_set_attr(reader_opts, "camera_model", self._config.camera_model)
            if self._mask_path and self._mask_path.is_dir():
                _try_set_attr(reader_opts, "mask_path", os.fspath(self._mask_path))
                _log(f"Step 1: Using masks from {self._mask_path}")

        # FeatureExtractionOptions or SiftExtractionOptions (from COLMAP plugin line 666)
        if hasattr(pycolmap, "FeatureExtractionOptions"):
            extraction_opts = pycolmap.FeatureExtractionOptions()
            extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
            _try_set_attr(extraction_opts, "num_threads", num_threads)
            _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
            _try_set_attr(extraction_opts, "max_num_features", sift_max_num_features)
        elif hasattr(pycolmap, "SiftExtractionOptions"):
            extraction_opts = pycolmap.SiftExtractionOptions()
            extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
            _try_set_attr(extraction_opts, "num_threads", num_threads)
            _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
            _try_set_attr(extraction_opts, "max_num_features", sift_max_num_features)

        # Build extract_kwargs (from COLMAP plugin line 684)
        extract_kwargs = dict(
            database_path=database_path,
            image_path=image_path,
            camera_mode=camera_mode,
        )

        if reader_opts is not None:
            extract_kwargs["reader_options"] = reader_opts
        else:
            extract_kwargs["camera_model"] = self._config.camera_model

        if extraction_opts is not None:
            if hasattr(pycolmap, "FeatureExtractionOptions"):
                extract_kwargs["extraction_options"] = extraction_opts
            else:
                extract_kwargs["sift_options"] = extraction_opts

        # Extract with GPU fallback (from COLMAP plugin lines 705-740)
        def _call_extract() -> None:
            try:
                pycolmap.extract_features(**extract_kwargs)
            except TypeError as e:
                fallback_kwargs = dict(extract_kwargs)
                for k in ("extraction_options", "sift_options", "reader_options", "camera_model"):
                    if k in fallback_kwargs:
                        fallback_kwargs.pop(k)
                        try:
                            pycolmap.extract_features(**fallback_kwargs)
                            return
                        except TypeError:
                            continue
                raise

        try:
            _call_extract()
        except Exception as exc:
            if extraction_gpu_requested and extraction_opts is not None and _try_set_attr(extraction_opts, "use_gpu", False):
                _log(f"Step 1: GPU failed ({exc}), retrying CPU")
                _call_extract()
            else:
                raise

        _log("Step 1: Feature extraction complete")
        self._progress("features", 1.0, "Feature extraction complete")

        # ================================================
        # FEATURE MATCHING (exhaustive — from COLMAP plugin line 808)
        # ================================================
        self._ensure_not_cancelled()
        _log("Step 2: Exhaustive matching")
        self._progress("matching", 0.0, "Matching features...")

        pycolmap.match_exhaustive(database_path=database_path)

        _log("Step 2: Matching complete")
        self._progress("matching", 1.0, "Feature matching complete")

        # ================================================
        # INCREMENTAL MAPPING (from COLMAP plugin lines 935-955)
        # ================================================
        self._ensure_not_cancelled()
        _log("Step 3: Incremental mapping")
        self._progress("mapping", 0.0, "Running incremental mapper...")

        pipeline_opts = pycolmap.IncrementalPipelineOptions()
        if hasattr(pipeline_opts, "multiple_models"):
            _try_set_attr(pipeline_opts, "multiple_models", False)
        if hasattr(pipeline_opts, "max_num_models"):
            _try_set_attr(pipeline_opts, "max_num_models", 1)

        _log("Step 3: Calling pycolmap.incremental_mapping()")

        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_path,
            output_path=sparse_path,
            options=pipeline_opts,
        )

        _log(f"Step 3: Mapping returned {len(reconstructions) if reconstructions else 0} reconstruction(s)")

        if not reconstructions:
            _log("Step 3: No reconstructions")
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error="Incremental mapping produced no reconstructions",
            )

        # Access results (from COLMAP plugin lines 952-955)
        reconstruction = next(iter(reconstructions.values()))
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)

        _log(f"Step 3: {num_images} images, {num_points} points")

        self._progress(
            "mapping", 1.0,
            f"Mapping complete: {num_images} images, {num_points} points",
        )

        _log("Done")
        elapsed = time.monotonic() - t0
        return ColmapResult(
            success=True,
            reconstruction_path=os.fspath(self._output_dir),
            num_registered_images=num_images,
            num_points3d=num_points,
            elapsed_sec=elapsed,
        )
