# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pipeline Orchestrator — wires all core modules into a single processing flow.

Stages and progress allocation:
  1. Sharpest frame extraction      (0-20%)
  2. Operator masking                (20-45%, skipped if disabled)
  3. Reframe ERP to pinhole         (45-55%)
  3.5 Closest-camera overlap masks  (55-56%, if masking enabled)
  4. Generate rig config            (56-57%)
  5. COLMAP alignment               (57-85%)
  6. Write output dataset           (85-95%)
  7. Return result                   (95-100%)

The actual import into LichtFeld (lf.load_file) happens in the panel UI,
not here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from .colmap_runner import ColmapConfig, ColmapResult, ColmapRunner
from .colmap_runner import infer_shared_pinhole_camera_params
from .input_detect import detect_input_type  # noqa: F401  (re-export)
from .masker import Masker, MaskConfig, MaskResult, is_masking_available
from .pycolmap_guard import check_loaded_pycolmap
from .setup_checks import is_sam3_masking_ready
from .overlap_mask import compute_overlap_masks
from .presets import (
    DEFAULT_PRESET,
    VIEW_PRESETS,
    ViewConfig,
    resolve_view_preset_name,
)
from .reframer import Reframer
from .rig_config import write_rig_config
from .sharpest_extractor import SharpestConfig, SharpestExtractor

logger = logging.getLogger(__name__)

# Minimum fraction of the video's reported frame count that all-frames
# extraction must produce before we treat the run as valid. A large shortfall
# usually means the source was incomplete (still being copied) or corrupt.
_ALL_FRAMES_MIN_RATIO = 0.9


def _assert_all_frames_complete(
    all_frames: bool, expected_frame_count: int, extracted: int
) -> None:
    """Fail fast when all-frames extraction produced far fewer frames than the
    video's metadata reports. No-op unless all_frames is set and the expected
    count is known (> 0)."""
    if not all_frames or expected_frame_count <= 0:
        return
    if extracted < _ALL_FRAMES_MIN_RATIO * expected_frame_count:
        raise RuntimeError(
            f"Extracted {extracted} frames but the video reports about "
            f"{expected_frame_count}. The source video may be incomplete or "
            f"corrupt. If a copy or export was still in progress, wait for it "
            f"to finish and run again."
        )


# ---------------------------------------------------------------------------
# Configuration and result data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Full configuration for the 360-camera pipeline."""

    video_path: str = ""
    output_dir: str = ""

    # Extraction
    interval: float = 2.0
    extraction_sharpness: str = "best"     # none, basic, better, best
    blur_metric: str = "tenengrad"         # tenengrad, laplacian
    scene_threshold: float = 0.3
    blur_scale_width: int = 640
    quality: int = 95
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    all_frames: bool = False               # extract every frame (no scoring/GPU)
    expected_frame_count: int = 0          # video metadata frame count; 0 = unknown (skip all-frames completeness check)

    # Masking
    enable_masking: bool = False
    masking_method: str = "fullcircle"  # "fullcircle" or "sam3_cubemap"
    mask_prompts: list[str] = field(default_factory=lambda: ["person"])
    mask_backend: Optional[str] = None  # "sam3", "yolo_sam1", or None (auto)
    enable_overlap_masks: bool = True  # Voronoi anti-overlap masks for COLMAP
    enable_diagnostics: bool = False  # Write masking_diagnostics.json alongside masks

    # Reframe
    preset_name: str = "default"
    output_size: int = 1920
    jpeg_quality: int = 95

    # COLMAP
    colmap_matcher: str = "sequential"  # "sequential", "exhaustive", "vocab_tree"
    colmap_match_budget_tier: str = "default"
    colmap_max_num_matches: Optional[int] = None

    # SIFT controls (UI-exposed via the COLMAP Preset dropdown + Advanced disclosure).
    # sift_preset: "normal" | "high" | "custom" — informational; the actual SIFT
    # values used during reconstruction come from sift_max_features / sift_max_image_size.
    # When None, the underlying ColmapConfig falls back to COLMAP's own defaults
    # (8192 features, no image-size cap).
    sift_preset: str = "normal"
    sift_max_features: Optional[int] = None
    sift_max_image_size: Optional[int] = None

    # Output mode: "pinhole" = COLMAP pinhole dataset, "erp_native" = native
    #              EQUIRECTANGULAR transforms.json,
    #              "fisheye" = OPENCV_FISHEYE COLMAP dataset (phase 1) /
    #              fisheye-native transforms.json (phase 2+)
    output_mode: str = "pinhole"
    # For output_mode == "fisheye": "native", "pinhole", or "both".
    # Default selects native fisheye output unless the user opts into pinhole crops.
    fisheye_training_output: str = "native"

    # Dual fisheye input (phase 1)
    # input_type: "erp" (default), "dual_fisheye", or "single_fisheye"
    # camera_family: "dji_osmo360" | "insta360" | None
    # source_mode: "container" (single .osv/.insv) or "split" (two pre-split videos).
    # When source_mode == "split", front_video_path / back_video_path are used and
    # the top-level video_path is ignored.
    input_type: str = "erp"
    camera_family: Optional[str] = None
    source_mode: str = "container"
    front_video_path: str = ""
    back_video_path: str = ""

    # Image folder source (Commit 2). When any of these dirs is set the
    # pipeline skips Stage 1 extraction and stages the user's images instead;
    # projection (ERP vs dual fisheye) still routes through output_mode /
    # input_type, so no separate projection field is needed.
    image_source_dir: str = ""   # single folder: ERP frames, or one-folder fisheye (front_/back_ files)
    image_front_dir: str = ""    # two-folder fisheye: front lens frames
    image_back_dir: str = ""     # two-folder fisheye: back lens frames
    mask_source: str = "generate"      # "generate" | "preexisting" | "none"
    preexisting_mask_dir: str = ""     # masks for mask_source == "preexisting"
    training_output: str = "native"    # image-folder ERP+fisheye: native | pinhole | both
                                       # (video fisheye keeps fisheye_training_output)
    keep_streams: bool = False  # retain demuxed front.mp4/back.mp4 alongside output
    keep_extracted_data: bool = True  # keep extracted frames + masks as <output>/images + <output>/masks deliverables (ERP/Fisheye Pinhole); default on so generated data is never silently discarded

    # Fisheye masking
    fisheye_circle_margin: float = 6.0  # Circle mask margin in percent

    # Rig constraint (experimental — default off)
    use_rig: bool = False

    # COLMAP 4.1 features (forwarded to ColmapConfig)
    colmap_feature_type: str = "sift"        # "sift", "aliked_n16rot", "aliked_n32"
    colmap_matcher_type: str = "bruteforce"  # "bruteforce", "lightglue"
    colmap_mapper: str = "incremental"       # "incremental", "global"
    colmap_ba_solver: str = "auto"           # "auto", "ceres", "ceres_gpu", "caspar"
    vocab_tree_path: str = ""                # path to vocab tree file (auto-resolved if empty)
    loop_detection: bool = False             # sequential matcher loop closure via vocab tree
    colmap_sequential_overlap: int = 10      # sequential matching overlap (2-20)
    colmap_guided_matching: bool = False     # epipolar-guided re-matching
    colmap_sift_affine_dsp: bool = False     # SIFT affine shape + DSP (CPU-heavy)


@dataclass
class PipelineResult:
    """Result of a completed (or failed) pipeline run."""

    success: bool
    dataset_path: str = ""
    output_mode: str = ""
    num_source_frames: int = 0
    num_output_images: int = 0
    num_aligned_cameras: int = 0
    num_registered_frames: int = 0
    num_complete_frames: int = 0
    num_partial_frames: int = 0
    views_per_frame: int = 0
    expected_images_by_view: dict[str, int] = field(default_factory=dict)
    registered_images_by_view: dict[str, int] = field(default_factory=dict)
    partial_frame_examples: list[str] = field(default_factory=list)
    dropped_frame_examples: list[str] = field(default_factory=list)
    preset_signature: str = ""
    mask_backend_name: str = ""
    video_backend_name: str = ""
    used_fallback_video_backend: bool = False
    video_backend_error: str = ""
    mask_diagnostics_path: str = ""
    gpu_extraction: bool = False
    masking_timers: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: str = ""


def _build_runtime_view_config(cfg: PipelineConfig) -> ViewConfig:
    """Resolve the active preset and apply runtime output overrides."""
    effective_preset = resolve_view_preset_name(cfg.preset_name, cfg.output_mode)
    base = VIEW_PRESETS.get(effective_preset, VIEW_PRESETS[DEFAULT_PRESET])
    return ViewConfig(
        rings=base.rings,
        views=base.views,
        include_zenith=base.include_zenith,
        include_nadir=base.include_nadir,
        zenith_fov=base.zenith_fov,
        output_size=cfg.output_size,
        jpeg_quality=cfg.jpeg_quality,
    )


def _format_preset_signature(preset_name: str, view_config: ViewConfig) -> str:
    """Summarize the effective preset geometry used for this run."""
    parts: list[str] = []
    for ring_idx, ring in enumerate(view_config.rings):
        parts.append(
            f"{ring_idx:02d}:{ring.count}x@{ring.pitch:g}"
            f"/f{ring.fov:g}/start{ring.start_yaw:g}"
        )
    if view_config.views:
        fovs = sorted({f"{view.fov:g}" for view in view_config.views})
        parts.append(f"{len(view_config.views)} freeviews/f{','.join(fovs)}")
    if view_config.include_zenith:
        parts.append(f"ZN@90/f{view_config.zenith_fov:g}")
    if view_config.include_nadir:
        parts.append(f"ND@-90/f{view_config.zenith_fov:g}")
    return f"{preset_name} | " + "; ".join(parts) if parts else preset_name


def _flatten_view_folders(parent_dir: Path) -> None:
    """Move files from view subfolders to parent, prefixing with folder name.

    ``images/front_ctr_hi/000042.jpg`` → ``images/front_ctr_hi_000042.jpg``

    Empty subfolders are removed after all files are moved.
    Idempotent: non-directory children are left untouched.
    """
    for subfolder in sorted(parent_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        view_name = subfolder.name
        for file in sorted(subfolder.iterdir()):
            if file.is_file():
                flat_name = f"{view_name}_{file.name}"
                file.rename(parent_dir / flat_name)
        try:
            subfolder.rmdir()
        except OSError:
            pass  # not empty — skip


def resolve_image_folder_masking(cfg: "PipelineConfig") -> str:
    """Resolve the image-folder mask behavior from cfg.mask_source.

    Returns "generate", "preexisting", or "none". Pre-existing masks are
    valid only on the ERP -> Pinhole path (output_mode == "pinhole"), the
    only pipeline path with a reframe_mask_dir channel to hand user masks to
    the reframer (AR-N2). Any other combination raises so the masks are never
    silently dropped. "generate" and "none" are valid for every image-folder
    mode.
    """
    source = (cfg.mask_source or "generate").lower()
    if source not in ("generate", "preexisting", "none"):
        raise ValueError(f"Unknown mask_source: {cfg.mask_source!r}")
    if source == "preexisting":
        if cfg.output_mode != "pinhole":
            raise ValueError(
                "Pre-existing masks are only supported for ERP + Pinhole "
                f"output, not output_mode={cfg.output_mode!r}."
            )
        if not cfg.preexisting_mask_dir:
            raise ValueError(
                "mask_source='preexisting' requires preexisting_mask_dir to be set."
            )
    return source


# ---------------------------------------------------------------------------
# Pipeline job
# ---------------------------------------------------------------------------


class PipelineJob:
    """Runs the full 360-camera pipeline on a background thread.

    Usage::

        job = PipelineJob(config, on_progress=..., on_complete=...)
        job.start()
        # ... poll job.stage / job.progress / job.status ...
        # ... or wait for on_complete callback ...
        job.cancel()  # request graceful cancellation

    Args:
        config: Pipeline configuration.
        on_progress: Called with ``(stage, progress_pct, status_msg)``
            whenever progress changes.  *progress_pct* is 0-100.
        on_complete: Called with a ``PipelineResult`` when the pipeline
            finishes (success or failure).
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_complete: Optional[Callable[[PipelineResult], None]] = None,
    ) -> None:
        self._config = config
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stage = ""
        self._progress = 0.0
        self._status = ""

    # -- public interface ---------------------------------------------------

    def start(self) -> None:
        """Launch the pipeline on a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request graceful cancellation."""
        with self._lock:
            self._cancelled = True

    @property
    def stage(self) -> str:
        with self._lock:
            return self._stage

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internal helpers ---------------------------------------------------

    def _check_cancel(self) -> bool:
        with self._lock:
            return self._cancelled

    def _update(self, stage: str, progress: float, status: str) -> None:
        with self._lock:
            self._stage = stage
            self._progress = progress
            self._status = status
        if self._on_progress:
            try:
                self._on_progress(stage, progress, status)
            except Exception:
                pass

    # -- main pipeline ------------------------------------------------------

    def _run(self) -> None:
        t0 = time.time()
        cfg = self._config
        result: PipelineResult

        try:
            guard_error = check_loaded_pycolmap()
            if guard_error:
                raise RuntimeError(guard_error)
            result = self._run_stages(cfg, t0)
        except Exception as exc:
            logger.exception("Pipeline failed")
            effective_preset = resolve_view_preset_name(cfg.preset_name, cfg.output_mode)
            preset_signature = _format_preset_signature(
                effective_preset,
                _build_runtime_view_config(cfg),
            )
            result = PipelineResult(
                success=False,
                preset_signature=preset_signature,
                error=str(exc),
                elapsed_sec=time.time() - t0,
            )

        if self._on_complete:
            try:
                self._on_complete(result)
            except Exception:
                logger.exception("on_complete callback raised")

    def _run_dual_fisheye_with_training_output(
        self, cfg: PipelineConfig, t0: float,
    ) -> PipelineResult:
        training_output = (cfg.fisheye_training_output or "native").lower()
        if training_output not in {"native", "pinhole", "both"}:
            training_output = "native"

        out = Path(cfg.output_dir)
        native_out = out / "native"
        native_cfg = replace(cfg, output_dir=str(native_out))
        result = self._run_fisheye_native(native_cfg, t0)
        if training_output == "native":
            return result

        images_dir = native_out / "images"
        masks_dir = native_out / "masks"
        sparse_dir = native_out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = native_out / "sparse"

        # Pinhole-only ships a single dataset in colmap/ (unified with the ERP
        # Pinhole layout: <output>/images + <output>/masks + <output>/colmap).
        # Both keeps the two-dataset native/ + pinhole/ layout.
        pinhole_out = out / ("colmap" if training_output == "pinhole" else "pinhole")
        self._update(
            "output", 96.0,
            "Exporting native-derived pinhole crops...",
        )

        from .fisheye_reframer import FISHEYE_PINHOLE_PRESET
        from .transforms_writer import write_native_propagated_transforms

        try:
            pinhole_transforms = write_native_propagated_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=pinhole_out,
                view_config=FISHEYE_PINHOLE_PRESET,
                masks_root=masks_dir if masks_dir.is_dir() else None,
                propagated_sparse_output_dir=pinhole_out / "sparse" / "0",
                log_fn=logger.info,
            )
        except Exception as exc:
            logger.exception("Native-derived pinhole export failed")
            raise RuntimeError(
                f"Native reconstruction succeeded but pinhole export failed: {exc}"
            ) from exc

        import json as _json

        pinhole_data = _json.loads(Path(pinhole_transforms).read_text())
        pinhole_frames = pinhole_data.get("frames", [])
        view_counts: dict[str, int] = {}
        for frame in pinhole_frames:
            basename = Path(frame.get("file_path", "")).name
            for view in FISHEYE_PINHOLE_PRESET.views:
                if basename.startswith(f"{view.name}_"):
                    view_counts[view.name] = view_counts.get(view.name, 0) + 1
                    break

        if training_output == "pinhole":
            import shutil

            out_resolved = out.resolve()
            native_resolved = native_out.resolve()
            if (
                native_out.name != "native"
                or native_resolved == out_resolved
                or native_resolved.parent != out_resolved
            ):
                raise RuntimeError(
                    f"Refusing to remove unsafe native output path: {native_out}"
                )
            # Always preserve the solve log alongside the shipped dataset.
            _native_log = native_out / "metadata" / "colmap_debug.log"
            if _native_log.is_file():
                shutil.move(str(_native_log), str(pinhole_out / "colmap_debug.log"))
            if cfg.keep_extracted_data:
                # Keep the extracted fisheye frames + masks as deliverables,
                # unified with ERP Pinhole: <output>/images + <output>/masks
                # (front/back preserved). Propagation already rendered its own
                # self-contained crops into colmap/, so these are free to move.
                if images_dir.is_dir():
                    shutil.move(str(images_dir), str(out / "images"))
                if masks_dir.is_dir():
                    shutil.move(str(masks_dir), str(out / "masks"))
            # native/ is now a spent intermediate (sparse/transforms/pointcloud).
            shutil.rmtree(str(native_out))
            result.dataset_path = str(pinhole_transforms)
            result.num_output_images = len(pinhole_frames)
            result.views_per_frame = FISHEYE_PINHOLE_PRESET.total_views()
            result.expected_images_by_view = {
                view.name: result.num_source_frames
                for view in FISHEYE_PINHOLE_PRESET.views
            }
            result.registered_images_by_view = {
                view.name: view_counts.get(view.name, 0)
                for view in FISHEYE_PINHOLE_PRESET.views
            }
        else:
            result.num_output_images += len(pinhole_frames)

        result.elapsed_sec = time.time() - t0
        result.preset_signature = (
            f"{result.preset_signature} | training_output={training_output}"
        )
        logger.info(
            "Native-derived pinhole export complete: %d frames at %s",
            len(pinhole_frames), pinhole_transforms,
        )
        self._update("complete", 100.0, "Fisheye output ready")
        return result

    def _run_single_fisheye_with_training_output(
        self, cfg: PipelineConfig, t0: float,
    ) -> PipelineResult:
        training_output = (cfg.fisheye_training_output or "native").lower()
        if training_output not in {"native", "pinhole", "both"}:
            training_output = "native"

        out = Path(cfg.output_dir)
        native_out = out / "native"
        native_cfg = replace(cfg, output_dir=str(native_out))
        result = self._run_single_fisheye_native(native_cfg, t0)
        if training_output == "native":
            return result

        images_dir = native_out / "images"
        masks_dir = native_out / "masks"
        sparse_dir = native_out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = native_out / "sparse"

        pinhole_out = out / ("colmap" if training_output == "pinhole" else "pinhole")
        self._update(
            "output", 96.0,
            "Exporting native-derived pinhole crops...",
        )

        from .fisheye_reframer import FISHEYE_PINHOLE_PRESET
        from .transforms_writer import write_native_propagated_transforms

        try:
            pinhole_transforms = write_native_propagated_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=pinhole_out,
                view_config=FISHEYE_PINHOLE_PRESET,
                masks_root=masks_dir if masks_dir.is_dir() else None,
                propagated_sparse_output_dir=pinhole_out / "sparse" / "0",
                lenses=("front",),
                log_fn=logger.info,
            )
        except Exception as exc:
            logger.exception("Native-derived pinhole export failed")
            raise RuntimeError(
                f"Native reconstruction succeeded but pinhole export failed: {exc}"
            ) from exc

        import json as _json

        pinhole_data = _json.loads(Path(pinhole_transforms).read_text())
        pinhole_frames = pinhole_data.get("frames", [])
        active_views = FISHEYE_PINHOLE_PRESET.views_for_lens("front")
        view_counts: dict[str, int] = {}
        for frame in pinhole_frames:
            basename = Path(frame.get("file_path", "")).name
            for view in active_views:
                if basename.startswith(f"{view.name}_"):
                    view_counts[view.name] = view_counts.get(view.name, 0) + 1
                    break

        if training_output == "pinhole":
            import shutil

            out_resolved = out.resolve()
            native_resolved = native_out.resolve()
            if (
                native_out.name != "native"
                or native_resolved == out_resolved
                or native_resolved.parent != out_resolved
            ):
                raise RuntimeError(
                    f"Refusing to remove unsafe native output path: {native_out}"
                )
            _native_log = native_out / "metadata" / "colmap_debug.log"
            if _native_log.is_file():
                shutil.move(str(_native_log), str(pinhole_out / "colmap_debug.log"))
            if cfg.keep_extracted_data:
                if images_dir.is_dir():
                    shutil.move(str(images_dir), str(out / "images"))
                if masks_dir.is_dir():
                    shutil.move(str(masks_dir), str(out / "masks"))
            shutil.rmtree(str(native_out))
            result.dataset_path = str(pinhole_transforms)
            result.num_output_images = len(pinhole_frames)
            result.views_per_frame = len(active_views)
            result.expected_images_by_view = {
                view.name: result.num_source_frames
                for view in active_views
            }
            result.registered_images_by_view = {
                view.name: view_counts.get(view.name, 0)
                for view in active_views
            }
        else:
            result.num_output_images += len(pinhole_frames)

        result.elapsed_sec = time.time() - t0
        result.preset_signature = (
            f"{result.preset_signature} | training_output={training_output}"
        )
        logger.info(
            "Native-derived pinhole export complete: %d frames at %s",
            len(pinhole_frames), pinhole_transforms,
        )
        self._update("complete", 100.0, "Fisheye output ready")
        return result

    def _run_stages(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        # Phase 1 dispatch shim — dual fisheye path is a separate leaf method
        # (Style 2 leaf-functions refactor for the ERP paths is deferred to
        # a follow-up; see spec §4.2.)
        if cfg.input_type == "dual_fisheye":
            if cfg.image_source_dir or cfg.image_front_dir:
                return self._run_image_folder_fisheye(cfg, t0)
            return self._run_dual_fisheye_with_training_output(cfg, t0)
        if cfg.input_type == "single_fisheye" and cfg.output_mode == "fisheye":
            return self._run_single_fisheye_with_training_output(cfg, t0)
        # Image-folder ERP input has its own read-in-place pipeline (all
        # artifacts under colmap/). Fisheye image folders were routed to
        # _run_image_folder_fisheye just above, so image_source_dir here is
        # always ERP. Handles both Native and Pinhole internally.
        if cfg.image_source_dir:
            return self._run_image_folder_erp(cfg, t0)

        if cfg.output_mode == "erp_native":
            return self._run_erp_native(cfg, t0)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Sub-directories, per the unified layout rule: the COLMAP dataset
        # (crops, per-view masks, sparse, database, rig config) is packaged
        # under <output>/colmap/; the extracted ERP frames and ERP masks are
        # promoted to <output>/images/ + <output>/masks/ as deliverables once
        # the run succeeds. extracted/ is a work dir that must not outlive
        # the run.
        extracted_dir = out / "extracted"
        frames_dir = extracted_dir / "frames"
        colmap_dir = out / "colmap"
        images_dir = colmap_dir / "images"

        # ===================================================================
        # Stage 1: Sharpest Frame Extraction (0-20%)
        # ===================================================================
        if cfg.extraction_sharpness == "none":
            self._update("extraction", 0.0, "Extracting frames...")
        else:
            self._update("extraction", 0.0, "Extracting sharpest frames...")

        extractor = SharpestExtractor()
        extract_config = SharpestConfig(
            interval=cfg.interval,
            extraction_sharpness=cfg.extraction_sharpness,
            blur_metric=cfg.blur_metric,
            scene_threshold=cfg.scene_threshold,
            scale_width=cfg.blur_scale_width,
            quality=cfg.quality,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            all_frames=cfg.all_frames,
        )

        def _extract_progress(cur: int, total: int, msg: str) -> None:
            pct = (cur / max(total, 1)) * 20
            self._update("extraction", pct, msg)

        extract_result = extractor.extract(
            cfg.video_path,
            str(frames_dir),
            extract_config,
            progress_callback=_extract_progress,
            cancel_check=self._check_cancel,
        )

        if not extract_result.success:
            raise RuntimeError(
                f"Frame extraction failed: {extract_result.error}"
            )

        num_source_frames = extract_result.frames_extracted
        _assert_all_frames_complete(
            cfg.all_frames, cfg.expected_frame_count, num_source_frames)

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2+3: Masking and Reframing
        #
        # The order depends on the preset:
        #   Default:  Mask (ERP) → Reframe (images + masks)
        #   Cubemap:  Reframe (images only) → Mask (direct on all faces)
        # ===================================================================
        effective_preset = resolve_view_preset_name(cfg.preset_name, cfg.output_mode)
        view_config = _build_runtime_view_config(cfg)
        preset_signature = _format_preset_signature(effective_preset, view_config)
        mask_result: Optional[MaskResult] = None
        is_cubemap = effective_preset == "cubemap"

        # ── Method-specific masking availability gate ──────────────
        if cfg.enable_masking:
            if cfg.masking_method == "sam3_cubemap":
                if not is_sam3_masking_ready():
                    raise RuntimeError(
                        "SAM 3 masking requires sam3 + weights. "
                        "Install SAM 3 via the plugin settings panel."
                    )
            else:
                # FullCircle requires the full stack including SAM v2
                if not is_masking_available():
                    raise RuntimeError(
                        "Operator masking requires the full masking stack, including "
                        "SAM v2 video tracking. Install masking before enabling it."
                    )

        # ── SAM 3 cubemap path ─────────────────────────────────────
        if (cfg.enable_masking
                and cfg.masking_method == "sam3_cubemap"):
            from .sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

            self._update("masking", 20.0, "Initializing SAM 3 masker...")

            sam3_cfg = Sam3MaskerConfig(
                prompts=cfg.mask_prompts,
                confidence_threshold=0.3,
                output_size=cfg.output_size,
                enable_diagnostics=cfg.enable_diagnostics,
            )
            sam3_masker = Sam3CubemapMasker(sam3_cfg)
            sam3_masker.initialize()

            try:
                def _sam3_progress(cur: int, total: int, msg: str) -> None:
                    pct = 20 + (cur / max(total, 1)) * 15
                    self._update("masking", pct, msg)

                sam3_result = sam3_masker.process_frames(
                    frames_dir=str(frames_dir),
                    output_dir=str(colmap_dir),
                    view_config=view_config,
                    erp_mask_dir=str(extracted_dir / "masks"),
                    progress_callback=_sam3_progress,
                )
            finally:
                sam3_masker.cleanup()

            if not sam3_result.success:
                error = getattr(sam3_result, "error", "") or "unknown error"
                raise RuntimeError(f"SAM 3 cubemap masking failed: {error}")

            # Normalize the SAM 3 helper output to the shared MaskResult contract
            # so overlap-mask generation and pipeline reporting keep working.
            mask_result = MaskResult(
                success=sam3_result.success,
                total_frames=sam3_result.total_frames,
                masked_frames=sam3_result.masked_frames,
                masks_dir=sam3_result.mask_dir,
                diagnostics_path=getattr(sam3_result, "diagnostics_path", ""),
                backend_name=getattr(sam3_result, "backend_name", "Sam3Backend"),
                masking_timers=getattr(sam3_result, "timers", {}),
            )

            if self._check_cancel():
                raise RuntimeError("Cancelled")

            # SAM 3 masker writes final per-view masks to out/masks/{view_id}/
            # using the selected output preset. Now reframe images only.
            self._update("reframe", 35.0, "Reframing to output views...")
            reframer = Reframer(view_config)

            def _reframe_progress(cur: int, total: int, filename: str) -> None:
                pct = 35 + (cur / max(total, 1)) * 15
                self._update("reframe", pct, f"Reframing {cur}/{total}: {filename}")

            reframe_result = reframer.reframe_batch(
                input_dir=str(frames_dir),
                output_dir=str(images_dir),
                mask_dir=None,
                progress_callback=_reframe_progress,
            )

            if not reframe_result.success and reframe_result.output_count == 0:
                errors = "; ".join(reframe_result.errors) if reframe_result.errors else "unknown"
                raise RuntimeError(f"Reframing failed: {errors}")

            num_output_images = reframe_result.output_count

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ── FullCircle cubemap path (unchanged) ─────────────────────
        elif is_cubemap and cfg.enable_masking and is_masking_available():
            # ── Cubemap path: reframe images first, then mask all faces directly ──

            # Stage 3 first: reframe images only (no masks)
            self._update("reframe", 20.0, "Reframing to cubemap views...")
            reframer = Reframer(view_config)

            def _reframe_progress(cur: int, total: int, filename: str) -> None:
                pct = 20 + (cur / max(total, 1)) * 10
                self._update("reframe", pct, f"Reframing {cur}/{total}: {filename}")

            reframe_result = reframer.reframe_batch(
                input_dir=str(frames_dir),
                output_dir=str(images_dir),
                mask_dir=None,  # no ERP masks for cubemap
                progress_callback=_reframe_progress,
            )

            if not reframe_result.success and reframe_result.output_count == 0:
                errors = "; ".join(reframe_result.errors) if reframe_result.errors else "unknown"
                raise RuntimeError(f"Reframing failed: {errors}")

            num_output_images = reframe_result.output_count
            logger.debug("Reframe: input_count=%d, output_count=%d",
                          reframe_result.input_count, reframe_result.output_count)

            if self._check_cancel():
                raise RuntimeError("Cancelled")

            # Stage 2: direct per-view masking on reframed images
            self._update("masking", 30.0, "Initializing masking backend...")

            # Pin backend to yolo_sam1 for FullCircle — do not trust
            # cfg.mask_backend which may auto-promote to sam3
            fc_backend = "yolo_sam1"
            mask_cfg = MaskConfig(
                targets=cfg.mask_prompts,
                output_size=cfg.output_size,
                backend_preference=fc_backend,
                views=view_config.get_all_views(),
                enable_synthetic=False,  # no synthetic pipeline for cubemap
                enable_diagnostics=cfg.enable_diagnostics,
            )
            masker = Masker(mask_cfg)
            masker.initialize()

            try:
                def _mask_progress(cur: int, total: int, msg: str) -> None:
                    pct = 30 + (cur / max(total, 1)) * 15
                    self._update("masking", pct, msg)

                # Direct per-view masking on reframed cubemap images
                cubemap_masks_dir = colmap_dir / "masks"
                mask_result = masker.process_reframed_views(
                    images_dir,
                    cubemap_masks_dir,
                    views=view_config.get_all_views(),
                    progress_callback=_mask_progress,
                )
            finally:
                masker.cleanup()

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        else:
            # ── Default path: mask ERP first, then reframe images + masks ──

            erp_masks_dir = extracted_dir / "masks"
            reframe_mask_dir: Optional[str] = None

            if cfg.enable_masking and is_masking_available():
                self._update("masking", 20.0, "Initializing masking backend...")

                # Pin backend to yolo_sam1 for FullCircle default path
                fc_backend = "yolo_sam1"
                mask_cfg = MaskConfig(
                    targets=cfg.mask_prompts,
                    output_size=cfg.output_size,
                    backend_preference=fc_backend,
                    views=view_config.get_all_views(),
                    enable_diagnostics=cfg.enable_diagnostics,
                )
                masker = Masker(mask_cfg)
                masker.initialize()

                try:
                    def _mask_progress(cur: int, total: int, msg: str) -> None:
                        pct = 20 + (cur / max(total, 1)) * 25
                        self._update("masking", pct, msg)

                    mask_result = masker.process_frames(
                        str(frames_dir),
                        str(erp_masks_dir),
                        progress_callback=_mask_progress,
                    )

                    if mask_result.success and mask_result.masked_frames > 0:
                        reframe_mask_dir = str(erp_masks_dir)
                finally:
                    masker.cleanup()

                if self._check_cancel():
                    raise RuntimeError("Cancelled")

            # Stage 3: reframe images (+ masks if available)
            self._update("reframe", 45.0, "Reframing to pinhole views...")
            reframer = Reframer(view_config)

            def _reframe_progress(cur: int, total: int, filename: str) -> None:
                pct = 45 + (cur / max(total, 1)) * 10
                self._update("reframe", pct, f"Reframing {cur}/{total}: {filename}")

            reframe_result = reframer.reframe_batch(
                input_dir=str(frames_dir),
                output_dir=str(images_dir),
                mask_dir=reframe_mask_dir,
                progress_callback=_reframe_progress,
            )

            if not reframe_result.success and reframe_result.output_count == 0:
                errors = "; ".join(reframe_result.errors) if reframe_result.errors else "unknown"
                raise RuntimeError(f"Reframing failed: {errors}")

            num_output_images = reframe_result.output_count
            logger.debug("Reframe: input_count=%d, output_count=%d",
                          reframe_result.input_count, reframe_result.output_count)

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 3.5: Closest-Camera Overlap Masks (55-56%)
        # ===================================================================
        # Voronoi masks partition overlapping regions so COLMAP doesn't
        # extract duplicate features. Written to a temporary directory
        # (extracted/colmap_masks/) that COLMAP reads from. The permanent
        # masks/ directory keeps operator-only masks for LFS training.
        colmap_masks_dir = extracted_dir / "colmap_masks"
        if colmap_masks_dir.exists():
            import shutil
            shutil.rmtree(colmap_masks_dir, ignore_errors=True)
        if cfg.enable_masking and cfg.enable_overlap_masks and mask_result and mask_result.success:
            self._update("overlap_masks", 55.0, "Computing overlap masks...")
            views = view_config.get_all_views()
            overlap_masks = compute_overlap_masks(views, cfg.output_size)
            if overlap_masks is not None:
                import cv2
                import shutil
                operator_masks_dir = colmap_dir / "masks"
                # Copy operator masks to temp dir, then AND with Voronoi
                shutil.copytree(operator_masks_dir, colmap_masks_dir)
                for view_name, voronoi_mask in overlap_masks.items():
                    view_mask_dir = colmap_masks_dir / view_name
                    if not view_mask_dir.is_dir():
                        continue
                    for mask_file in view_mask_dir.iterdir():
                        if mask_file.suffix.lower() != ".png":
                            continue
                        operator_mask = cv2.imread(
                            str(mask_file), cv2.IMREAD_GRAYSCALE
                        )
                        if operator_mask is None:
                            continue
                        combined = cv2.bitwise_and(operator_mask, voronoi_mask)
                        cv2.imwrite(str(mask_file), combined)

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 4: Generate Rig Config (56-57%)
        # ===================================================================
        self._update("rig_config", 56.0, "Generating rig configuration...")

        colmap_dir.mkdir(parents=True, exist_ok=True)
        rig_config_path = str(colmap_dir / "rig_config.json")
        write_rig_config(view_config, rig_config_path)

        # ===================================================================
        # Stage 5: COLMAP Alignment (57-85%)
        # ===================================================================
        self._update("colmap", 57.0, "Running COLMAP alignment...")

        view_fovs = [fov for _yaw, _pitch, fov, _name, _flip in view_config.get_all_views()]
        camera_params, default_focal_length_factor, _shared_fov_deg = (
            infer_shared_pinhole_camera_params(view_fovs, cfg.output_size)
        )

        colmap_config = ColmapConfig(
            camera_params=camera_params,
            default_focal_length_factor=default_focal_length_factor,
            matcher=cfg.colmap_matcher,
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=True,
            sift_max_num_features_override=cfg.sift_max_features,
            sift_max_image_size_override=cfg.sift_max_image_size,
            # COLMAP 4.1 features
            feature_type=cfg.colmap_feature_type,
            matcher_type=cfg.colmap_matcher_type,
            mapper=cfg.colmap_mapper,
            ba_solver=cfg.colmap_ba_solver,
            vocab_tree_path=cfg.vocab_tree_path or None,
            loop_detection=False,  # rig-constrained: sequential pairs sufficient
            sequential_overlap=cfg.colmap_sequential_overlap,
            guided_matching=cfg.colmap_guided_matching,
            sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
            sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
        )

        def _colmap_progress(stage: str, pct: float, msg: str) -> None:
            # ColmapRunner reports pct as 0.0-1.0 per sub-stage
            self._update("colmap", 57 + pct * 28, msg)

        # Pass mask directory to COLMAP if masks were generated.
        # Use Voronoi-combined masks (colmap_masks_dir) if available,
        # otherwise fall back to operator-only masks.
        effective_mask_path = (
            colmap_masks_dir if colmap_masks_dir.is_dir()
            else (colmap_dir / "masks"))
        runner = ColmapRunner(
            images_dir=str(images_dir),
            output_dir=str(colmap_dir),
            rig_config_path=rig_config_path,
            mask_path=str(effective_mask_path) if effective_mask_path.is_dir() else None,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )

        try:
            colmap_result = runner.run()
        finally:
            if colmap_masks_dir.is_dir():
                import shutil
                shutil.rmtree(colmap_masks_dir, ignore_errors=True)

        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

        # ===================================================================
        # Stage 6: Write Output (85-95%)
        # ===================================================================
        # Pinhole mode: COLMAP dataset already written by ColmapRunner under
        # colmap/. When "keep frames & masks" is on (default), promote the
        # extraction work products to root deliverables: ERP frames ->
        # <output>/images/, ERP masks -> <output>/masks/, the extraction
        # manifest beside them; then drop the emptied extracted/ (rmdir-only,
        # so anything unexpected keeps it). When off, discard extracted/ and
        # ship only the colmap/ dataset.
        self._update("output", 85.0, "COLMAP dataset ready")
        dataset_path = colmap_result.reconstruction_path

        import shutil

        if cfg.keep_extracted_data:
            from .frame_source import relocate_erp_frames_to_colmap

            _manifest = frames_dir / "extraction_manifest.json"
            if _manifest.is_file():
                shutil.move(str(_manifest), str(out / _manifest.name))
            relocate_erp_frames_to_colmap(frames_dir, out)  # frames -> out/images
            _erp_masks = extracted_dir / "masks"
            if _erp_masks.is_dir():
                _dest_masks = out / "masks"
                _dest_masks.mkdir(parents=True, exist_ok=True)
                for _m in sorted(_erp_masks.glob("*.png")):
                    shutil.move(str(_m), str(_dest_masks / _m.name))
            for _d in (extracted_dir / "masks", frames_dir, extracted_dir):
                try:
                    _d.rmdir()
                except OSError:
                    pass
        else:
            shutil.rmtree(extracted_dir, ignore_errors=True)

        # ===================================================================
        # Stage 7: Done (95-100%)
        # ===================================================================
        self._update("complete", 100.0, "Pipeline complete")

        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=dataset_path,
            output_mode=cfg.output_mode,
            num_source_frames=num_source_frames,
            num_output_images=num_output_images,
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=colmap_result.views_per_frame,
            expected_images_by_view=colmap_result.expected_images_by_view,
            registered_images_by_view=colmap_result.registered_images_by_view,
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=preset_signature,
            mask_backend_name=mask_result.backend_name if mask_result else "",
            video_backend_name=mask_result.video_backend_name if mask_result else "",
            used_fallback_video_backend=(
                mask_result.used_fallback_video_backend if mask_result else False
            ),
            video_backend_error=mask_result.video_backend_error if mask_result else "",
            mask_diagnostics_path=mask_result.diagnostics_path if mask_result else "",
            gpu_extraction=extract_result.gpu_accelerated,
            masking_timers=getattr(mask_result, "masking_timers", {}) if mask_result else {},
            elapsed_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Native ERP path (equirectangular COLMAP, no reframing/rig)
    # ------------------------------------------------------------------

    def _run_erp_native(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        """Video ERP pipeline: extract → optional ERP masks → native solve →
        training_output routing, under the unified layout rule.

        1 dataset → ``<output>/colmap/`` (native, or propagated pinhole);
        training=both → ``colmap/native/`` + ``colmap/pinhole/``. training=
        pinhole keeps the extracted ERP frames at ``<output>/images/`` and the
        ERP masks at ``<output>/masks/`` as deliverables (the video analog of
        image-folder's untouched source) and discards the throwaway native
        solve. Masks live inside the dataset they belong to otherwise.
        """
        import shutil

        from .frame_source import relocate_erp_frames_to_colmap
        from .transforms_writer import write_erp_native_transforms

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        extracted_dir = out / "extracted"
        frames_dir = extracted_dir / "frames"
        colmap_dir = out / "colmap"

        training = (cfg.training_output or "native").lower()
        if training not in ("native", "pinhole", "both"):
            raise ValueError(f"Unknown training_output: {cfg.training_output!r}")

        # Where the ERP masks land (same rule as the image-folder tracks):
        # native → colmap/masks (the dataset's masks; transforms.json references
        # "masks/<name>" relative to the dataset); both → colmap/native/masks;
        # pinhole → root <output>/masks/ (kept deliverable).
        if training == "native":
            masks_dir = colmap_dir / "masks"
        elif training == "both":
            masks_dir = colmap_dir / "native" / "masks"
        else:
            masks_dir = out / "masks"

        # ===================================================================
        # Stage 1: Sharpest Frame Extraction (0-30%)
        # ===================================================================
        if cfg.extraction_sharpness == "none":
            self._update("extraction", 0.0, "Extracting frames...")
        else:
            self._update("extraction", 0.0, "Extracting sharpest frames...")

        extractor = SharpestExtractor()
        extract_config = SharpestConfig(
            interval=cfg.interval,
            extraction_sharpness=cfg.extraction_sharpness,
            blur_metric=cfg.blur_metric,
            scene_threshold=cfg.scene_threshold,
            scale_width=cfg.blur_scale_width,
            quality=cfg.quality,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            all_frames=cfg.all_frames,
        )

        def _extract_progress(cur: int, total: int, msg: str) -> None:
            pct = (cur / max(total, 1)) * 30
            self._update("extraction", pct, msg)

        extract_result = extractor.extract(
            cfg.video_path,
            str(frames_dir),
            extract_config,
            progress_callback=_extract_progress,
            cancel_check=self._check_cancel,
        )

        if not extract_result.success:
            raise RuntimeError(
                f"Frame extraction failed: {extract_result.error}"
            )

        num_source_frames = extract_result.frames_extracted
        _assert_all_frames_complete(
            cfg.all_frames, cfg.expected_frame_count, num_source_frames)

        erp_width = 0
        erp_height = 0
        if extract_result.frame_paths:
            import cv2
            _first = cv2.imread(extract_result.frame_paths[0])
            if _first is not None:
                erp_height, erp_width = _first.shape[:2]
                del _first

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2: ERP masking (30-50%, optional) — written straight to the
        # track's mask destination (no extracted/masks intermediate).
        # ===================================================================
        mask_result: Optional[MaskResult] = None
        erp_masks_dir = masks_dir
        view_config = _build_runtime_view_config(cfg)

        if cfg.enable_masking:
            if cfg.masking_method == "sam3_cubemap":
                if not is_sam3_masking_ready():
                    raise RuntimeError(
                        "SAM 3 masking is not available. "
                        "Install SAM 3 from the Masking section first."
                    )
                from .sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

                self._update("masking", 30.0, "Initializing SAM 3 for ERP masking...")
                sam3_cfg = Sam3MaskerConfig(
                    prompts=cfg.mask_prompts,
                    confidence_threshold=0.3,
                    output_size=cfg.output_size,
                    enable_diagnostics=cfg.enable_diagnostics,
                )
                sam3_masker = Sam3CubemapMasker(sam3_cfg)
                sam3_masker.initialize()

                try:
                    def _sam3_progress(cur: int, total: int, msg: str) -> None:
                        pct = 30 + (cur / max(total, 1)) * 20
                        self._update("masking", pct, msg)

                    # view_config is required by Sam3CubemapMasker but only used
                    # for per-view side output; native path reads erp_mask_dir only.
                    view_config = _build_runtime_view_config(cfg)
                    sam3_result = sam3_masker.process_frames(
                        frames_dir=str(frames_dir),
                        output_dir=str(out),
                        view_config=view_config,
                        erp_mask_dir=str(erp_masks_dir),
                        progress_callback=_sam3_progress,
                        write_per_view=False,
                    )
                finally:
                    sam3_masker.cleanup()

                if not sam3_result.success:
                    error = getattr(sam3_result, "error", "") or "unknown error"
                    raise RuntimeError(f"SAM 3 ERP masking failed: {error}")

                mask_result = MaskResult(
                    success=sam3_result.success,
                    total_frames=sam3_result.total_frames,
                    masked_frames=sam3_result.masked_frames,
                    masks_dir=sam3_result.erp_mask_dir or str(erp_masks_dir),
                    diagnostics_path=getattr(sam3_result, "diagnostics_path", ""),
                    backend_name=getattr(sam3_result, "backend_name", "Sam3Backend"),
                    masking_timers=getattr(sam3_result, "timers", {}),
                )
            elif is_masking_available():
                self._update("masking", 30.0, "Initializing masking backend...")
                view_config = _build_runtime_view_config(cfg)
                mask_cfg = MaskConfig(
                    targets=cfg.mask_prompts,
                    output_size=cfg.output_size,
                    backend_preference="yolo_sam1",
                    views=view_config.get_all_views(),
                    enable_diagnostics=cfg.enable_diagnostics,
                )
                masker = Masker(mask_cfg)
                masker.initialize()

                try:
                    def _mask_progress(cur: int, total: int, msg: str) -> None:
                        pct = 30 + (cur / max(total, 1)) * 20
                        self._update("masking", pct, msg)

                    mask_result = masker.process_frames(
                        str(frames_dir),
                        str(erp_masks_dir),
                        progress_callback=_mask_progress,
                    )
                finally:
                    masker.cleanup()

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 3+: native solve + training_output routing (50-100%).
        # The solve reads extracted/frames IN PLACE (no flat-stage move);
        # _erp_native_solve relocates the frames into the kept dataset's
        # images/ only after COLMAP succeeds.
        # ===================================================================
        equirect_mask_dir = (
            str(masks_dir)
            if cfg.enable_masking and masks_dir.is_dir()
            else None
        )

        def _erp_result(dataset_path, num_output_images, preset_signature,
                        colmap_result) -> PipelineResult:
            elapsed = time.time() - t0
            return PipelineResult(
                success=True,
                dataset_path=str(dataset_path),
                output_mode=cfg.output_mode,
                num_source_frames=num_source_frames,
                num_output_images=num_output_images,
                num_aligned_cameras=colmap_result.num_registered_images,
                num_registered_frames=colmap_result.num_registered_frames,
                num_complete_frames=colmap_result.num_complete_frames,
                num_partial_frames=colmap_result.num_partial_frames,
                views_per_frame=colmap_result.views_per_frame,
                expected_images_by_view=colmap_result.expected_images_by_view,
                registered_images_by_view=colmap_result.registered_images_by_view,
                partial_frame_examples=colmap_result.partial_frame_examples,
                dropped_frame_examples=colmap_result.dropped_frame_examples,
                preset_signature=preset_signature,
                mask_backend_name=mask_result.backend_name if mask_result else "",
                video_backend_name=(
                    mask_result.video_backend_name if mask_result else ""),
                used_fallback_video_backend=(
                    mask_result.used_fallback_video_backend if mask_result else False
                ),
                video_backend_error=(
                    mask_result.video_backend_error if mask_result else ""),
                mask_diagnostics_path=(
                    mask_result.diagnostics_path if mask_result else ""),
                gpu_extraction=extract_result.gpu_accelerated,
                masking_timers=(
                    getattr(mask_result, "masking_timers", {}) if mask_result else {}),
                elapsed_sec=elapsed,
            )

        def _write_native_transforms(dataset_dir: Path, dataset_masks: Path):
            sparse_dir = dataset_dir / "sparse" / "0"
            if not sparse_dir.is_dir():
                sparse_dir = dataset_dir / "sparse"
            transforms_path = write_erp_native_transforms(
                colmap_sparse_dir=sparse_dir,
                output_dir=dataset_dir,
                masks_dir=dataset_masks if dataset_masks.is_dir() else None,
                erp_width=erp_width,
                erp_height=erp_height,
                log_fn=logger.info,
            )
            return sparse_dir, transforms_path

        def _finalize_extracted(manifest_dest: Path) -> None:
            # The extraction work dir must not outlive the run. The extractor's
            # manifest follows the frames to their final home (beside the
            # dataset's images, mirroring the fisheye paired manifest); the
            # emptied dirs are then dropped rmdir-only, so anything unexpected
            # left inside keeps them.
            manifest = frames_dir / "extraction_manifest.json"
            if manifest.is_file():
                manifest_dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(manifest), str(manifest_dest / manifest.name))
            for d in (frames_dir, extracted_dir):
                try:
                    d.rmdir()
                except OSError:
                    pass

        if training == "native":
            # Single dataset at colmap/; the extracted frames become its images.
            colmap_result, num_output_images = self._erp_native_solve(
                cfg, frames_dir, colmap_dir, equirect_mask_dir, move_frames=True)
            if not colmap_result.success:
                raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
            _finalize_extracted(colmap_dir)
            self._update("output", 90.0, "Writing native ERP transforms.json...")
            _sparse, transforms_path = _write_native_transforms(
                colmap_dir, masks_dir)
            self._update("complete", 100.0, "Native ERP export complete")
            return _erp_result(
                transforms_path, num_output_images,
                "native EQUIRECTANGULAR", colmap_result)

        if training == "both":
            # Two datasets: kept native at colmap/native/ + pinhole propagated
            # from the native solve at colmap/pinhole/.
            native_dir = colmap_dir / "native"
            colmap_result, num_output_images = self._erp_native_solve(
                cfg, frames_dir, native_dir, equirect_mask_dir, move_frames=True)
            if not colmap_result.success:
                raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
            _finalize_extracted(native_dir)
            self._update("output", 90.0, "Writing native ERP transforms.json...")
            native_sparse, transforms_path = _write_native_transforms(
                native_dir, masks_dir)
            self._erp_propagate_pinhole(
                cfg, native_dir / "images", native_sparse,
                colmap_dir / "pinhole", view_config,
                str(masks_dir) if masks_dir.is_dir() else None)
            self._update(
                "complete", 100.0,
                "Native + propagated Pinhole export complete")
            return _erp_result(
                transforms_path, num_output_images,
                "EQUIRECTANGULAR native + propagated pinhole", colmap_result)

        # training == "pinhole": ship ONLY the propagated pinhole dataset at
        # colmap/. The native solve runs in a throwaway temp; the extracted ERP
        # frames and masks are KEPT at <output>/images/ and <output>/masks/ as
        # deliverables (the video analog of image-folder's untouched source).
        temp_dir = colmap_dir / "_native_tmp"
        colmap_result, _ = self._erp_native_solve(
            cfg, frames_dir, temp_dir, equirect_mask_dir, move_frames=False)
        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
        temp_sparse = temp_dir / "sparse" / "0"
        if not temp_sparse.is_dir():
            temp_sparse = temp_dir / "sparse"
        num_output_images = self._erp_propagate_pinhole(
            cfg, frames_dir, temp_sparse, colmap_dir, view_config,
            equirect_mask_dir)
        # SM4: cancel gate immediately before removing the temp (COLMAP
        # artifacts only — frames and masks are kept deliverables).
        if self._check_cancel():
            raise RuntimeError("Cancelled")
        shutil.rmtree(temp_dir, ignore_errors=True)
        if cfg.keep_extracted_data:
            moved, _removed = relocate_erp_frames_to_colmap(frames_dir, out)
            _finalize_extracted(out)
            logger.info(
                "Kept %d extracted ERP frames at %s", moved, out / "images")
        else:
            # Lean: ship only colmap/. Drop the extracted ERP frames and the
            # source ERP masks (colmap/ carries its own propagated per-crop
            # masks). extracted/ never outlives the run.
            shutil.rmtree(extracted_dir, ignore_errors=True)
            if masks_dir.is_dir():
                shutil.rmtree(masks_dir, ignore_errors=True)
        self._update("complete", 100.0, "Propagated Pinhole export complete")
        return _erp_result(
            colmap_dir / "transforms.json", num_output_images,
            "propagated pinhole (EQUIRECTANGULAR native)", colmap_result)

    # ------------------------------------------------------------------
    # Image-folder ERP path (read-in-place; all artifacts under colmap/)
    # ------------------------------------------------------------------

    def _erp_native_solve(
        self, cfg: PipelineConfig, source: Path, dataset_dir: Path,
        equirect_mask_dir: Optional[str], *, move_frames: bool,
    ) -> tuple[object, int]:
        """Native EQUIRECTANGULAR COLMAP solve for the ERP image-folder path.

        Reads ``source`` in place (COLMAP EQUIRECTANGULAR) with ``equirect_mask_dir``
        as the mask_path, writing artifacts to ``dataset_dir``. When ``move_frames``
        (Native/Both: the source frames ARE the kept dataset), relocates the source
        frames into ``dataset_dir/images`` and removes the emptied source folder --
        ONLY after COLMAP succeeds, and only after a ``cancel_check`` gate (SM4).
        ``move_frames=False`` (training=pinhole temp solve) reads in place and never
        touches the source.

        Returns ``(colmap_result, num_output_images)``. On COLMAP failure returns the
        (unsuccessful) result and 0 without moving anything (SM3 -- no disk loss).
        """
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._update("colmap", 55.0, "Running COLMAP (EQUIRECTANGULAR)...")
        colmap_config = ColmapConfig(
            camera_model="EQUIRECTANGULAR",
            camera_mode="SINGLE",
            camera_params=None,
            matcher=cfg.colmap_matcher or "sequential",
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=False,
            refine_principal_point=False,
            refine_extra_params=False,
            sift_max_num_features_override=cfg.sift_max_features,
            sift_max_image_size_override=cfg.sift_max_image_size,
            feature_type=cfg.colmap_feature_type,
            matcher_type=cfg.colmap_matcher_type,
            mapper="incremental",
            ba_solver=cfg.colmap_ba_solver,
            vocab_tree_path=cfg.vocab_tree_path or None,
            loop_detection=cfg.loop_detection,
            sequential_overlap=cfg.colmap_sequential_overlap,
            guided_matching=cfg.colmap_guided_matching,
            sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
            sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
        )

        def _colmap_progress(stage: str, pct: float, msg: str) -> None:
            self._update("colmap", 55 + pct * 30, msg)

        runner = ColmapRunner(
            images_dir=str(source),
            output_dir=str(dataset_dir),
            rig_config_path=None,
            mask_path=equirect_mask_dir,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )
        colmap_result = runner.run()
        if not colmap_result.success:
            return colmap_result, 0

        num_output_images = 0
        if move_frames:
            # SM4: cancel gate immediately before the destructive relocate+rmdir,
            # reachable only after COLMAP success. relocate is graceful (rmdir only
            # if the emptied source has no leftovers); never an rmtree.
            if self._check_cancel():
                raise RuntimeError("Cancelled")
            from .frame_source import relocate_erp_frames_to_colmap
            self._update("output", 88.0, "Moving frames into images...")
            num_output_images, source_removed = relocate_erp_frames_to_colmap(
                source, dataset_dir)
            if not source_removed:
                logger.warning(
                    "Source folder %s not empty after moving frames; "
                    "left in place.", source)
        return colmap_result, num_output_images

    def _erp_propagate_pinhole(
        self, cfg: PipelineConfig, reframe_input_dir: Path, native_sparse_dir: Path,
        pinhole_out_dir: Path, view_config, equirect_mask_dir: Optional[str],
    ) -> int:
        """Reframe ERP frames to pinhole crops and propagate their poses from the
        native ERP solve (the ERP twin of the fisheye propagation).

        Uses the reframer's default (fliplr'd) crops: that flip and COLMAP's
        y-down equirect convention are two reflections that cancel, so the crop
        pose is a proper rotation (see ``erp_view_rotation``). SM5: the reframe
        must produce ``n_views * input_count`` crops (cv2.imwrite return codes are
        ignored, so the count is the only completeness signal) -- a short reframe
        raises BEFORE any caller cleanup runs. Returns the crop count.
        """
        from .transforms_writer import write_erp_propagated_transforms

        images_dir = pinhole_out_dir / "images"
        self._update("reframe", 90.0, "Reframing to pinhole crops...")
        reframer = Reframer(view_config)

        def _reframe_progress(cur: int, total: int, filename: str) -> None:
            self._update("reframe", 90 + (cur / max(total, 1)) * 4,
                         f"Reframing {cur}/{total}: {filename}")

        reframe_result = reframer.reframe_batch(
            input_dir=str(reframe_input_dir),
            output_dir=str(images_dir),
            mask_dir=equirect_mask_dir,
            progress_callback=_reframe_progress,
        )
        n_views = len(view_config.get_all_views())
        expected = n_views * reframe_result.input_count
        if reframe_result.output_count != expected:
            raise RuntimeError(
                f"Pinhole reframe incomplete: wrote {reframe_result.output_count} "
                f"crops, expected {expected} ({n_views} views x "
                f"{reframe_result.input_count} frames). Refusing to continue "
                "(no cleanup runs, no data deleted)."
            )

        # Flatten view subfolders to view-prefixed flat names (the fisheye
        # propagation convention). LFS's transforms loader resolves masks by
        # bare image FILENAME only (transforms.cpp: _image_name =
        # image_path.filename(); mask_path in the JSON is not consulted), so
        # same-stem crops across view subfolders make every mask lookup
        # ambiguous and the import hard-fails. Unique basenames are the
        # loader's real contract.
        pinhole_masks = pinhole_out_dir / "masks"
        _flatten_view_folders(images_dir)
        if pinhole_masks.is_dir():
            _flatten_view_folders(pinhole_masks)

        self._update("output", 95.0, "Writing propagated pinhole transforms.json...")
        write_erp_propagated_transforms(
            colmap_sparse_dir=native_sparse_dir,
            output_dir=pinhole_out_dir,
            view_config=view_config,
            output_size=cfg.output_size,
            masks_dir=pinhole_masks if pinhole_masks.is_dir() else None,
            propagated_sparse_output_dir=pinhole_out_dir / "sparse" / "0",
            log_fn=logger.info,
        )
        return reframe_result.output_count

    def _run_image_folder_erp(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        """ERP image-folder pipeline for Native and Pinhole output.

        The user's equirectangular frames are read where they sit. Every COLMAP
        artifact is written under ``<output>/colmap/``; the equirect masks (a
        kept deliverable) go to ``<output>/masks/``. Masking is uniform -- the
        equirect frames are masked first -- then Native has COLMAP read the
        source directly and, only once COLMAP succeeds, relocates the frames
        into ``colmap/images/`` and removes the emptied source folder. Pinhole
        reframes the source into ``colmap/images/<view>/`` crops (source left
        untouched) with the equirect masks projected per-view into
        ``colmap/masks/<view>/``.
        """
        import shutil

        from .frame_source import (
            assert_source_outside_output,
            assert_source_reads_safe,
            list_images_natural,
            staged_frame_stats,
            validate_mask_pairing,
        )

        out = Path(cfg.output_dir)
        source = Path(cfg.image_source_dir)
        # Guard BEFORE creating any output dirs, so a rejected run (e.g. source
        # == output root) leaves no stray colmap/ folder in the user's images.
        assert_source_reads_safe(cfg.image_source_dir, cfg.output_dir)

        out.mkdir(parents=True, exist_ok=True)
        colmap_dir = out / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        # Resolve mask behaviour (generate | preexisting | none). Unlike the
        # video paths, pre-existing masks are accepted for Native too: they are
        # the equirect masks COLMAP reads (Native) or the reframer projects
        # per-view (Pinhole).
        mask_source = (cfg.mask_source or "generate").lower()
        if mask_source not in ("generate", "preexisting", "none"):
            raise ValueError(f"Unknown mask_source: {cfg.mask_source!r}")
        if mask_source == "preexisting" and not cfg.preexisting_mask_dir:
            raise ValueError(
                "mask_source='preexisting' requires preexisting_mask_dir to be set."
            )

        is_native = cfg.output_mode == "erp_native"
        view_config = _build_runtime_view_config(cfg)

        # training_output routing (Native output only): native | pinhole | both.
        # Direct pinhole (output_mode == "pinhole") ignores it.
        training = (cfg.training_output or "native").lower()
        if is_native and training not in ("native", "pinhole", "both"):
            raise ValueError(f"Unknown training_output: {cfg.training_output!r}")

        # Where the plugin's EQUIRECT masks land, per the unified layout. Native:
        # colmap/masks (transforms.json references them "masks/<name>", relative to
        # colmap/). Both: colmap/native/masks (the kept native dataset's masks).
        # Pinhole tracks -- direct pinhole AND training=pinhole -- keep them at
        # root <output>/masks/: the equirect masks are a paid-for deliverable
        # (QA-C2 run 4); the reframer reads them there and projects the per-view
        # masks the dataset uses into colmap/masks/<view>/.
        if not is_native or training == "pinhole":
            masks_dir = out / "masks"
        elif training == "native":
            masks_dir = colmap_dir / "masks"
        else:  # training == "both"
            masks_dir = colmap_dir / "native" / "masks"

        # ===================================================================
        # Stage 1: Read the image folder (no extraction)
        # ===================================================================
        self._update("staging", 0.0, "Reading image folder...")
        num_source_frames, erp_width, erp_height = staged_frame_stats(
            cfg.image_source_dir)
        _assert_all_frames_complete(
            cfg.all_frames, cfg.expected_frame_count, num_source_frames)
        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2: Mask the equirect frames (-> colmap/masks/ for Native;
        # -> root <output>/masks/ for Pinhole, which reprojects them per-view)
        # ===================================================================
        mask_result: Optional[MaskResult] = None
        equirect_mask_dir: Optional[str] = None

        if mask_source == "preexisting":
            self._update("masking", 20.0, "Validating pre-existing masks...")
            assert_source_outside_output(cfg.preexisting_mask_dir, cfg.output_dir)
            _frames, _ = list_images_natural(source)
            _masks, _ = list_images_natural(cfg.preexisting_mask_dir)
            validate_mask_pairing(_frames, _masks)
            # Copy the user's masks into the equirect-mask dir (colmap/masks for
            # Native, root <output>/masks/ for Pinhole) so COLMAP, the reframer,
            # and the native transforms.json all read them in place.
            masks_dir.mkdir(parents=True, exist_ok=True)
            for _m in _masks:
                shutil.copy2(_m, masks_dir / _m.name)
            equirect_mask_dir = str(masks_dir)
        elif mask_source == "generate" and cfg.enable_masking:
            masks_dir.mkdir(parents=True, exist_ok=True)

            def _mask_progress(cur: int, total: int, msg: str) -> None:
                self._update("masking", 20 + (cur / max(total, 1)) * 20, msg)

            if cfg.masking_method == "sam3_cubemap":
                if not is_sam3_masking_ready():
                    raise RuntimeError(
                        "SAM 3 masking is not available. "
                        "Install SAM 3 from the Masking section first."
                    )
                from .sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

                self._update("masking", 20.0, "Masking equirect frames (SAM 3)...")
                sam3_masker = Sam3CubemapMasker(Sam3MaskerConfig(
                    prompts=cfg.mask_prompts,
                    confidence_threshold=0.3,
                    output_size=cfg.output_size,
                    enable_diagnostics=cfg.enable_diagnostics,
                ))
                sam3_masker.initialize()
                try:
                    sam3_result = sam3_masker.process_frames(
                        frames_dir=str(source),
                        output_dir=str(out),
                        view_config=view_config,
                        erp_mask_dir=str(masks_dir),
                        progress_callback=_mask_progress,
                        write_per_view=False,
                    )
                finally:
                    sam3_masker.cleanup()
                if not sam3_result.success:
                    error = getattr(sam3_result, "error", "") or "unknown error"
                    raise RuntimeError(f"SAM 3 ERP masking failed: {error}")
                mask_result = MaskResult(
                    success=sam3_result.success,
                    total_frames=sam3_result.total_frames,
                    masked_frames=sam3_result.masked_frames,
                    masks_dir=sam3_result.erp_mask_dir or str(masks_dir),
                    diagnostics_path=getattr(sam3_result, "diagnostics_path", ""),
                    backend_name=getattr(sam3_result, "backend_name", "Sam3Backend"),
                    masking_timers=getattr(sam3_result, "timers", {}),
                )
                equirect_mask_dir = str(masks_dir)
            elif is_masking_available():
                self._update("masking", 20.0, "Masking equirect frames...")
                masker = Masker(MaskConfig(
                    targets=cfg.mask_prompts,
                    output_size=cfg.output_size,
                    backend_preference="yolo_sam1",
                    views=view_config.get_all_views(),
                    enable_diagnostics=cfg.enable_diagnostics,
                ))
                masker.initialize()
                try:
                    mask_result = masker.process_frames(
                        str(source), str(masks_dir),
                        progress_callback=_mask_progress,
                    )
                finally:
                    masker.cleanup()
                if (mask_result and mask_result.success
                        and mask_result.masked_frames > 0):
                    equirect_mask_dir = str(masks_dir)
            else:
                raise RuntimeError(
                    "Operator masking requires the full masking stack "
                    "(YOLO + SAM v2). Install masking before enabling it, "
                    "or set masking to Off."
                )
            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Native vs Pinhole tails
        # ===================================================================
        if is_native:
            from .transforms_writer import write_erp_native_transforms

            def _erp_result(dataset_path, num_output_images, preset_signature,
                            colmap_result) -> PipelineResult:
                elapsed = time.time() - t0
                return PipelineResult(
                    success=True,
                    dataset_path=str(dataset_path),
                    output_mode=cfg.output_mode,
                    num_source_frames=num_source_frames,
                    num_output_images=num_output_images,
                    num_aligned_cameras=colmap_result.num_registered_images,
                    num_registered_frames=colmap_result.num_registered_frames,
                    num_complete_frames=colmap_result.num_complete_frames,
                    num_partial_frames=colmap_result.num_partial_frames,
                    views_per_frame=colmap_result.views_per_frame,
                    expected_images_by_view=colmap_result.expected_images_by_view,
                    registered_images_by_view=colmap_result.registered_images_by_view,
                    partial_frame_examples=colmap_result.partial_frame_examples,
                    dropped_frame_examples=colmap_result.dropped_frame_examples,
                    preset_signature=preset_signature,
                    mask_backend_name=mask_result.backend_name if mask_result else "",
                    video_backend_name=(
                        mask_result.video_backend_name if mask_result else ""),
                    used_fallback_video_backend=(
                        mask_result.used_fallback_video_backend if mask_result else False),
                    video_backend_error=(
                        mask_result.video_backend_error if mask_result else ""),
                    mask_diagnostics_path=(
                        mask_result.diagnostics_path if mask_result else ""),
                    gpu_extraction=False,
                    masking_timers=(
                        getattr(mask_result, "masking_timers", {}) if mask_result else {}),
                    elapsed_sec=elapsed,
                )

            def _write_native_transforms(dataset_dir: Path, dataset_masks: Path):
                sparse_dir = dataset_dir / "sparse" / "0"
                if not sparse_dir.is_dir():
                    sparse_dir = dataset_dir / "sparse"
                transforms_path = write_erp_native_transforms(
                    colmap_sparse_dir=sparse_dir,
                    output_dir=dataset_dir,
                    masks_dir=dataset_masks if dataset_masks.is_dir() else None,
                    erp_width=erp_width,
                    erp_height=erp_height,
                    log_fn=logger.info,
                )
                return sparse_dir, transforms_path

            if training == "native":
                # Single dataset at colmap/; source frames ARE the dataset.
                colmap_result, num_output_images = self._erp_native_solve(
                    cfg, source, colmap_dir, equirect_mask_dir, move_frames=True)
                if not colmap_result.success:
                    raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
                self._update("output", 90.0, "Writing native ERP transforms.json...")
                _sparse, transforms_path = _write_native_transforms(
                    colmap_dir, masks_dir)
                self._update("complete", 100.0, "Native ERP export complete")
                return _erp_result(
                    transforms_path, num_output_images,
                    "native EQUIRECTANGULAR", colmap_result)

            if training == "both":
                # Two datasets: kept native at colmap/native/ (source absorbed) +
                # pinhole propagated from the native solve at colmap/pinhole/.
                native_dir = colmap_dir / "native"
                colmap_result, num_output_images = self._erp_native_solve(
                    cfg, source, native_dir, equirect_mask_dir, move_frames=True)
                if not colmap_result.success:
                    raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
                self._update("output", 90.0, "Writing native ERP transforms.json...")
                native_sparse, transforms_path = _write_native_transforms(
                    native_dir, masks_dir)
                # Propagate un-mirrored pinhole crops from the native solve. SM5
                # completeness is enforced inside the helper (raises on shortfall).
                self._erp_propagate_pinhole(
                    cfg, native_dir / "images", native_sparse,
                    colmap_dir / "pinhole", view_config,
                    str(masks_dir) if masks_dir.is_dir() else None)
                self._update(
                    "complete", 100.0,
                    "Native + propagated Pinhole export complete")
                # Native is the primary delivered dataset (num_output_images = its
                # relocated frame count); the pinhole dataset is derived.
                return _erp_result(
                    transforms_path, num_output_images,
                    "EQUIRECTANGULAR native + propagated pinhole", colmap_result)

            # training == "pinhole": SOURCE UNTOUCHED. Solve native in a throwaway
            # TEMP (only COLMAP artifacts + generated masks live there), propagate
            # the pinhole crops to colmap/, then remove the temp. NO hoist.
            temp_dir = colmap_dir / "_native_tmp"
            colmap_result, _ = self._erp_native_solve(
                cfg, source, temp_dir, equirect_mask_dir, move_frames=False)
            if not colmap_result.success:
                raise RuntimeError(f"COLMAP failed: {colmap_result.error}")
            temp_sparse = temp_dir / "sparse" / "0"
            if not temp_sparse.is_dir():
                temp_sparse = temp_dir / "sparse"
            num_output_images = self._erp_propagate_pinhole(
                cfg, source, temp_sparse, colmap_dir, view_config,
                equirect_mask_dir)
            # SM4: cancel gate immediately before removing the temp. The temp
            # holds only COLMAP artifacts (the equirect masks live at root
            # <output>/masks/, a kept deliverable); the user's source was read
            # in place and never touched.
            if self._check_cancel():
                raise RuntimeError("Cancelled")
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._update("complete", 100.0, "Propagated Pinhole export complete")
            return _erp_result(
                colmap_dir / "transforms.json", num_output_images,
                "propagated pinhole (EQUIRECTANGULAR native)", colmap_result)

        # -- Pinhole: reframe source -> colmap/images/<view>/ crops --
        effective_preset = resolve_view_preset_name(cfg.preset_name, cfg.output_mode)
        preset_signature = _format_preset_signature(effective_preset, view_config)
        images_dir = colmap_dir / "images"

        self._update("reframe", 45.0, "Reframing to pinhole views...")
        reframer = Reframer(view_config)

        def _reframe_progress(cur: int, total: int, filename: str) -> None:
            self._update("reframe", 45 + (cur / max(total, 1)) * 10,
                         f"Reframing {cur}/{total}: {filename}")

        reframe_result = reframer.reframe_batch(
            input_dir=str(source),
            output_dir=str(images_dir),
            mask_dir=equirect_mask_dir,
            progress_callback=_reframe_progress,
        )
        if not reframe_result.success and reframe_result.output_count == 0:
            errors = ("; ".join(reframe_result.errors)
                      if reframe_result.errors else "unknown")
            raise RuntimeError(f"Reframing failed: {errors}")
        num_output_images = reframe_result.output_count
        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # Per-view masks were projected to colmap/masks/<view>/ by the reframer
        # (output_dir.parent / "masks"). Combine with Voronoi overlap masks in a
        # temp dir COLMAP reads from; the permanent colmap/masks/ is preserved.
        colmap_view_masks = colmap_dir / "masks"
        overlap_masks_dir = colmap_dir / "colmap_masks"
        if overlap_masks_dir.exists():
            shutil.rmtree(overlap_masks_dir, ignore_errors=True)
        if (cfg.enable_masking and cfg.enable_overlap_masks
                and mask_result and mask_result.success
                and colmap_view_masks.is_dir()):
            self._update("overlap_masks", 55.0, "Computing overlap masks...")
            overlap_masks = compute_overlap_masks(
                view_config.get_all_views(), cfg.output_size)
            if overlap_masks is not None:
                import cv2
                shutil.copytree(colmap_view_masks, overlap_masks_dir)
                for view_name, voronoi_mask in overlap_masks.items():
                    view_mask_dir = overlap_masks_dir / view_name
                    if not view_mask_dir.is_dir():
                        continue
                    for mask_file in view_mask_dir.iterdir():
                        if mask_file.suffix.lower() != ".png":
                            continue
                        operator_mask = cv2.imread(
                            str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if operator_mask is None:
                            continue
                        combined = cv2.bitwise_and(operator_mask, voronoi_mask)
                        cv2.imwrite(str(mask_file), combined)
            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # Rig config -> colmap/rig_config.json
        self._update("rig_config", 56.0, "Generating rig configuration...")
        rig_config_path = str(colmap_dir / "rig_config.json")
        write_rig_config(view_config, rig_config_path)

        # COLMAP alignment (pinhole)
        self._update("colmap", 57.0, "Running COLMAP alignment...")
        view_fovs = [fov for _yaw, _pitch, fov, _name, _flip
                     in view_config.get_all_views()]
        camera_params, default_focal_length_factor, _shared_fov_deg = (
            infer_shared_pinhole_camera_params(view_fovs, cfg.output_size))
        colmap_config = ColmapConfig(
            camera_params=camera_params,
            default_focal_length_factor=default_focal_length_factor,
            matcher=cfg.colmap_matcher,
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=True,
            sift_max_num_features_override=cfg.sift_max_features,
            sift_max_image_size_override=cfg.sift_max_image_size,
            feature_type=cfg.colmap_feature_type,
            matcher_type=cfg.colmap_matcher_type,
            mapper=cfg.colmap_mapper,
            ba_solver=cfg.colmap_ba_solver,
            vocab_tree_path=cfg.vocab_tree_path or None,
            loop_detection=False,
            sequential_overlap=cfg.colmap_sequential_overlap,
            guided_matching=cfg.colmap_guided_matching,
            sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
            sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
        )

        def _colmap_progress(stage: str, pct: float, msg: str) -> None:
            self._update("colmap", 57 + pct * 28, msg)

        effective_mask_path = (
            overlap_masks_dir if overlap_masks_dir.is_dir()
            else colmap_view_masks)
        runner = ColmapRunner(
            images_dir=str(images_dir),
            output_dir=str(colmap_dir),
            rig_config_path=rig_config_path,
            mask_path=str(effective_mask_path) if effective_mask_path.is_dir() else None,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )
        try:
            colmap_result = runner.run()
        finally:
            if overlap_masks_dir.is_dir():
                shutil.rmtree(overlap_masks_dir, ignore_errors=True)
        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

        self._update("complete", 100.0, "Pipeline complete")
        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=colmap_result.reconstruction_path,
            output_mode=cfg.output_mode,
            num_source_frames=num_source_frames,
            num_output_images=num_output_images,
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=colmap_result.views_per_frame,
            expected_images_by_view=colmap_result.expected_images_by_view,
            registered_images_by_view=colmap_result.registered_images_by_view,
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=preset_signature,
            mask_backend_name=mask_result.backend_name if mask_result else "",
            video_backend_name=mask_result.video_backend_name if mask_result else "",
            used_fallback_video_backend=(
                mask_result.used_fallback_video_backend if mask_result else False),
            video_backend_error=mask_result.video_backend_error if mask_result else "",
            mask_diagnostics_path=mask_result.diagnostics_path if mask_result else "",
            gpu_extraction=False,
            masking_timers=(
                getattr(mask_result, "masking_timers", {}) if mask_result else {}),
            elapsed_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Image-folder fisheye path (source untouched; artifacts under colmap/)
    # ------------------------------------------------------------------

    def _run_image_folder_fisheye(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        """Fisheye image-folder pipeline (owned): reconstruct under <output>/colmap/.

        Thin wrapper over ``core.image_folder_fisheye.run_image_folder_fisheye``,
        which owns staging (copy; source never modified), lens masking, the
        OPENCV_FISHEYE/PER_FOLDER native solve, transforms, and -- for
        pinhole/both -- native-propagated pinhole crops, all under
        the unified layout (1 dataset -> colmap/; both -> colmap/native/ +
        colmap/pinhole/). The module returns a plain dict (dodging a circular
        import); this maps it to a PipelineResult.
        """
        from .image_folder_fisheye import run_image_folder_fisheye

        result = run_image_folder_fisheye(cfg, self._update, self._check_cancel)
        if not result.get("success", False):
            raise RuntimeError(
                result.get("error") or "Image-folder fisheye pipeline failed")
        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=result.get("dataset_path", ""),
            output_mode=cfg.output_mode,
            num_source_frames=result.get("num_source_frames", 0),
            num_output_images=result.get("num_output_images", 0),
            num_aligned_cameras=result.get("num_aligned_cameras", 0),
            num_registered_frames=result.get("num_registered_frames", 0),
            num_complete_frames=result.get("num_complete_frames", 0),
            num_partial_frames=result.get("num_partial_frames", 0),
            views_per_frame=result.get("views_per_frame", 0),
            expected_images_by_view=result.get("expected_images_by_view", {}),
            registered_images_by_view=result.get("registered_images_by_view", {}),
            partial_frame_examples=result.get("partial_frame_examples", []),
            dropped_frame_examples=result.get("dropped_frame_examples", []),
            preset_signature=result.get("preset_signature", ""),
            gpu_extraction=False,
            elapsed_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Single fisheye native path
    # ------------------------------------------------------------------

    def _run_single_fisheye_native(
        self, cfg: PipelineConfig, t0: float,
    ) -> PipelineResult:
        """Single fisheye pipeline: extraction → masking → COLMAP → output."""
        import shutil

        from .fisheye_priors import infer_fisheye_camera_params

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        extracted_dir = out / "extracted"
        extracted_front_dir = extracted_dir / "front"
        images_dir = out / "images"

        # ===================================================================
        # Stage 1: Sharpest-frame extraction (0-50%)
        # ===================================================================
        print(f"\n[pipeline] Output mode: {cfg.output_mode}")
        print(f"[pipeline] Output: {out}")
        print("[pipeline] Stage 1: Single-fisheye extraction starting...")
        self._update("extraction", 0.0, "Extracting fisheye frames...")

        extractor = SharpestExtractor()
        extract_config = SharpestConfig(
            interval=cfg.interval,
            extraction_sharpness=cfg.extraction_sharpness,
            blur_metric=cfg.blur_metric,
            scene_threshold=cfg.scene_threshold,
            scale_width=cfg.blur_scale_width,
            quality=cfg.quality,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            all_frames=cfg.all_frames,
        )

        def _extract_progress(cur: int, total: int, msg: str) -> None:
            pct = (cur / max(total, 1)) * 50
            self._update("extraction", pct, msg)

        extract_result = extractor.extract(
            cfg.video_path,
            str(extracted_front_dir),
            extract_config,
            progress_callback=_extract_progress,
            cancel_check=self._check_cancel,
        )
        if not extract_result.success:
            raise RuntimeError(
                f"Single fisheye extraction failed: {extract_result.error}"
            )

        frame_count = extract_result.frames_extracted
        _assert_all_frames_complete(
            cfg.all_frames, cfg.expected_frame_count, frame_count)
        gpu_accel = getattr(extract_result, "gpu_accelerated", False)
        logger.info(
            "Extracted %d single-fisheye frames%s",
            frame_count, " (GPU)" if gpu_accel else "",
        )

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2: Fisheye masking (SAM3 + circle mask)
        # ===================================================================
        mask_enabled = cfg.enable_masking
        print(f"[pipeline] Stage 2: Masking {'enabled' if mask_enabled else 'disabled'}")
        if mask_enabled:
            self._update("masking", 20.0, "Initializing SAM 3 for fisheye masking...")

            import cv2
            import numpy as np
            from .backends import Sam3Backend
            from .fisheye_circle_mask import generate_fisheye_circle_mask

            backend = Sam3Backend(confidence_threshold=0.3)
            backend.initialize()

            try:
                masks_dir = extracted_dir / "masks"
                lens_names = ("front",)
                total_frames = sum(
                    len(list((extracted_dir / lens).glob("*.jpg")))
                    + len(list((extracted_dir / lens).glob("*.png")))
                    for lens in lens_names
                )
                frame_idx = 0
                circle_cache: dict[tuple[int, int], np.ndarray] = {}

                for lens in lens_names:
                    lens_frames_dir = extracted_dir / lens
                    lens_masks_dir = masks_dir / lens
                    lens_masks_dir.mkdir(parents=True, exist_ok=True)

                    frame_files = sorted(
                        f for f in lens_frames_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    )

                    for frame_file in frame_files:
                        if self._check_cancel():
                            raise RuntimeError("Cancelled")

                        frame_idx += 1
                        pct = 20 + (frame_idx / max(total_frames, 1)) * 25
                        self._update(
                            "masking", pct,
                            f"SAM 3 masking {lens}/{frame_file.name} "
                            f"({frame_idx}/{total_frames})",
                        )

                        image = cv2.imread(str(frame_file))
                        if image is None:
                            logger.warning("Could not read %s, skipping", frame_file)
                            continue

                        h, w = image.shape[:2]
                        detection = backend.detect_and_segment(
                            image, cfg.mask_prompts,
                        )
                        keep_mask = ((detection == 0).astype(np.uint8)) * 255

                        cache_key = (w, h)
                        if cache_key not in circle_cache:
                            circle = generate_fisheye_circle_mask(
                                w, h, margin_percent=cfg.fisheye_circle_margin,
                            )
                            circle_cache[cache_key] = (
                                (1 - circle).astype(np.uint8) * 255
                            )
                        circle_keep = circle_cache[cache_key]
                        final_mask = cv2.bitwise_and(keep_mask, circle_keep)

                        mask_out = lens_masks_dir / f"{frame_file.stem}.png"
                        cv2.imwrite(str(mask_out), final_mask)

                logger.info(
                    "Fisheye masking complete: %d frames across %d lens",
                    frame_idx, len(lens_names),
                )
                print(f"[pipeline] Masking complete: {frame_idx} frames")
            finally:
                backend.cleanup()

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 3: Staging
        # ===================================================================
        colmap_mask_path: str | None = None
        self._update("staging", 50.0, "Staging images for COLMAP...")
        images_dir.mkdir(parents=True, exist_ok=True)
        for lens in ("front",):
            src = extracted_dir / lens
            dst = images_dir / lens
            if not src.is_dir():
                raise RuntimeError(
                    f"Extraction did not produce {lens}/ subdirectory at {src}"
                )
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            src.rename(dst)

        if mask_enabled:
            output_masks_dir = out / "masks"
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            for lens in ("front",):
                src = extracted_dir / "masks" / lens
                dst = output_masks_dir / lens
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst, ignore_errors=True)
                    src.rename(dst)
            colmap_mask_path = str(output_masks_dir)

        # ===================================================================
        # Stage 5: COLMAP alignment (55-95%)
        # ===================================================================
        print("[pipeline] Stage 5: COLMAP alignment starting...")
        self._update("colmap", 55.0, "Running COLMAP (OPENCV_FISHEYE, PER_FOLDER)...")

        camera_params = infer_fisheye_camera_params(cfg.camera_family)
        has_prior = camera_params is not None
        if not has_prior:
            logger.info(
                "No calibrated prior for camera_family=%r — using fisheye-"
                "appropriate default_focal_length_factor=0.30",
                cfg.camera_family,
            )

        FISHEYE_DEFAULT_FOCAL_FACTOR = 0.30
        colmap_config = ColmapConfig(
            camera_model="OPENCV_FISHEYE",
            camera_params=camera_params,
            default_focal_length_factor=None if has_prior else FISHEYE_DEFAULT_FOCAL_FACTOR,
            matcher=cfg.colmap_matcher,
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=True,
            refine_principal_point=has_prior,
            refine_extra_params=has_prior,
            refine_sensor_from_rig=True,
            sift_max_num_features_override=cfg.sift_max_features,
            sift_max_image_size_override=cfg.sift_max_image_size,
            feature_type=cfg.colmap_feature_type,
            matcher_type=cfg.colmap_matcher_type,
            mapper=cfg.colmap_mapper,
            ba_solver=cfg.colmap_ba_solver,
            vocab_tree_path=cfg.vocab_tree_path or None,
            loop_detection=cfg.loop_detection,
            sequential_overlap=cfg.colmap_sequential_overlap,
            guided_matching=cfg.colmap_guided_matching,
            sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
            sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
        )

        def _colmap_progress(stage: str, pct: float, msg: str) -> None:
            base = 55.0
            self._update("colmap", base + pct * (95 - base), msg)

        runner = ColmapRunner(
            images_dir=str(images_dir),
            output_dir=str(out),
            rig_config_path=None,
            mask_path=colmap_mask_path,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )
        colmap_result = runner.run()
        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

        # ===================================================================
        # Stage 6: Write transforms.json + pointcloud.ply (95-100%)
        # ===================================================================
        sparse_dir = out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = out / "sparse"

        self._update("output", 95.0, "Writing fisheye transforms.json...")
        from .transforms_writer import write_fisheye_transforms
        try:
            transforms_path = write_fisheye_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=out,
                masks_dir=out / "masks",
                log_fn=logger.info,
            )
            dataset_path = str(transforms_path)
        except Exception as exc:
            logger.exception("Fisheye transforms output failed")
            raise RuntimeError(
                f"COLMAP succeeded but transforms.json export failed: {exc}"
            ) from exc

        self._update("complete", 100.0, "Fisheye reconstruction + transforms.json ready")
        self._cleanup_fisheye_output(out, extracted_dir)

        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=dataset_path,
            output_mode=cfg.output_mode,
            num_source_frames=frame_count,
            num_output_images=frame_count,
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=colmap_result.views_per_frame,
            expected_images_by_view=colmap_result.expected_images_by_view,
            registered_images_by_view=colmap_result.registered_images_by_view,
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=(
                f"single_fisheye | family={cfg.camera_family or 'unknown'}"
            ),
            gpu_extraction=gpu_accel,
            elapsed_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Dual fisheye native path (phase 1: extract → COLMAP only)
    # ------------------------------------------------------------------

    def _run_fisheye_native(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        """Phase 1 dual fisheye pipeline: paired extraction → COLMAP.

        Skips masking, rig config (use_rig=False default), and the fisheye
        transforms.json output writer. Those land in subsequent phases.
        Goal: prove the OPENCV_FISHEYE + PER_FOLDER + no-rig path produces
        a valid reconstruction on real .osv / .insv data.
        """
        import shutil

        from .fisheye_priors import infer_fisheye_camera_params

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        extracted_dir = out / "extracted"
        images_dir = out / "images"

        # ===================================================================
        # Stage 1: Paired sharpest-frame extraction (0-50%)
        # ===================================================================
        from .paired_extractor import PairedExtractorConfig, extract_dual_fisheye

        print(f"\n[pipeline] Output mode: {cfg.output_mode}")
        print(f"[pipeline] Output: {out}")
        print(f"[pipeline] Stage 1: Paired extraction starting...")
        self._update("extraction", 0.0, "Extracting paired fisheye frames...")

        # Map plugin's extraction_sharpness → paired extractor mode/scene flags.
        # "none"  → fixed interval, no scoring
        # "basic" → sharpest, no scene detection
        # "better"/"best" → sharpest + scene detection
        if cfg.extraction_sharpness == "none":
            extract_mode = "fixed"
            scene_detection = False
        else:
            extract_mode = "sharpest"
            scene_detection = cfg.extraction_sharpness in ("better", "best")

        extract_config = PairedExtractorConfig(
            mode=extract_mode,
            scoring_method=cfg.blur_metric,  # "tenengrad" or "laplacian"
            scene_detection=scene_detection,
            interval_sec=cfg.interval,
            quality=cfg.quality,
            scale_width=cfg.blur_scale_width,
            scene_threshold=cfg.scene_threshold,
            start_sec=cfg.start_sec,
            end_sec=cfg.end_sec,
            all_frames=cfg.all_frames,
        )

        def _extract_progress(cur: int, total: int, msg: str) -> None:
            pct = (cur / max(total, 1)) * 50
            self._update("extraction", pct, msg)

        if cfg.source_mode == "split":
            # User-provided pre-split front + back files (e.g. graded .mp4s
            # from Resolve). No demux step; PairedExtractor handles it
            # directly. keep_streams is ignored (no demux to keep).
            from .paired_extractor import PairedExtractor

            if not cfg.front_video_path or not cfg.back_video_path:
                raise RuntimeError(
                    "source_mode='split' requires both front_video_path "
                    "and back_video_path to be set"
                )
            extract_result = PairedExtractor().extract(
                front_video=cfg.front_video_path,
                back_video=cfg.back_video_path,
                output_dir=str(extracted_dir),
                config=extract_config,
                progress_callback=_extract_progress,
                cancel_check=self._check_cancel,
                log=lambda m: (logger.info("[paired_extract] %s", m), print(f"[paired_extract] {m}")),
            )
        else:
            # Container mode: single .osv / .insv input. extract_dual_fisheye
            # dispatches on container shape (DJI .osv, Insta360 X4/X5
            # single-file, or older Insta360 _00_/_10_ pair).
            extract_result = extract_dual_fisheye(
                cfg.video_path,
                str(extracted_dir),
                config=extract_config,
                keep_streams=cfg.keep_streams,
                progress_callback=_extract_progress,
                cancel_check=self._check_cancel,
                log=lambda m: (logger.info("[paired_extract] %s", m), print(f"[paired_extract] {m}")),
            )

        if not extract_result.success:
            raise RuntimeError(
                f"Paired fisheye extraction failed: {extract_result.error}"
            )

        num_pairs = extract_result.pair_count
        _assert_all_frames_complete(
            cfg.all_frames, cfg.expected_frame_count, num_pairs)
        gpu_accel = getattr(extract_result, 'gpu_accelerated', False)
        logger.info("Phase 1: extracted %d frame pairs%s", num_pairs,
                     " (GPU)" if gpu_accel else "")

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2: Fisheye masking (SAM3 + circle mask)
        # ===================================================================
        mask_enabled = cfg.enable_masking
        print(f"[pipeline] Stage 2: Masking {'enabled' if mask_enabled else 'disabled'}")
        if mask_enabled:
            self._update("masking", 20.0, "Initializing SAM 3 for fisheye masking...")

            import cv2
            import numpy as np
            from .backends import Sam3Backend
            from .fisheye_circle_mask import generate_fisheye_circle_mask

            backend = Sam3Backend(confidence_threshold=0.3)
            backend.initialize()

            try:
                masks_dir = extracted_dir / "masks"
                lens_names = ("front", "back")
                total_frames = sum(
                    len(list((extracted_dir / lens).glob("*.jpg")))
                    + len(list((extracted_dir / lens).glob("*.png")))
                    for lens in lens_names
                )
                frame_idx = 0
                circle_cache: dict[tuple[int, int], np.ndarray] = {}

                for lens in lens_names:
                    lens_frames_dir = extracted_dir / lens
                    lens_masks_dir = masks_dir / lens
                    lens_masks_dir.mkdir(parents=True, exist_ok=True)

                    frame_files = sorted(
                        f for f in lens_frames_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    )

                    for frame_file in frame_files:
                        if self._check_cancel():
                            raise RuntimeError("Cancelled")

                        frame_idx += 1
                        pct = 20 + (frame_idx / max(total_frames, 1)) * 25
                        self._update(
                            "masking", pct,
                            f"SAM 3 masking {lens}/{frame_file.name} "
                            f"({frame_idx}/{total_frames})",
                        )

                        image = cv2.imread(str(frame_file))
                        if image is None:
                            logger.warning("Could not read %s, skipping", frame_file)
                            continue

                        h, w = image.shape[:2]

                        # SAM3 detection: 0=background, 1=detected object
                        detection = backend.detect_and_segment(
                            image, cfg.mask_prompts,
                        )

                        # Convert to COLMAP polarity: 255=keep, 0=remove
                        keep_mask = ((detection == 0).astype(np.uint8)) * 255

                        # Fisheye circle mask (cached per resolution)
                        cache_key = (w, h)
                        if cache_key not in circle_cache:
                            circle = generate_fisheye_circle_mask(
                                w, h, margin_percent=cfg.fisheye_circle_margin,
                            )
                            # Convert: 0=valid,1=masked → 255=keep,0=remove
                            circle_cache[cache_key] = (
                                (1 - circle).astype(np.uint8) * 255
                            )
                        circle_keep = circle_cache[cache_key]

                        # Combine: pixel is valid only if BOTH masks say valid
                        final_mask = cv2.bitwise_and(keep_mask, circle_keep)

                        mask_out = lens_masks_dir / f"{frame_file.stem}.png"
                        cv2.imwrite(str(mask_out), final_mask)

                logger.info(
                    "Fisheye masking complete: %d frames across %d lenses",
                    frame_idx, len(lens_names),
                )
                print(f"[pipeline] Masking complete: {frame_idx} frames")
            finally:
                backend.cleanup()

            if self._check_cancel():
                raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 3: Staging (direct move)
        # ===================================================================
        colmap_mask_path: str | None = None
        rig_config_path_str = str(out / "rig_config.json")

        # ── Native fisheye path (direct move) ──
        self._update("staging", 50.0, "Staging images for COLMAP...")
        images_dir.mkdir(parents=True, exist_ok=True)
        for lens in ("front", "back"):
            src = extracted_dir / lens
            dst = images_dir / lens
            if not src.is_dir():
                raise RuntimeError(
                    f"Extraction did not produce {lens}/ subdirectory at {src}"
                )
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            src.rename(dst)

        if mask_enabled:
            output_masks_dir = out / "masks"
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            for lens in ("front", "back"):
                src = extracted_dir / "masks" / lens
                dst = output_masks_dir / lens
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst, ignore_errors=True)
                    src.rename(dst)
            colmap_mask_path = str(output_masks_dir)

        if cfg.use_rig:
            self._update("rig_config", 52.0, "Writing dual fisheye rig config...")
            from .rig_config import write_dual_fisheye_rig_config
            write_dual_fisheye_rig_config(rig_config_path_str)
            logger.info("Wrote dual fisheye rig config: %s", rig_config_path_str)

        # ===================================================================
        # Stage 5: COLMAP alignment (67-95%)
        # ===================================================================
        print(f"[pipeline] Stage 5: COLMAP alignment starting...")
        self._update("colmap", 55.0, "Running COLMAP (OPENCV_FISHEYE, PER_FOLDER)...")

        camera_params = infer_fisheye_camera_params(cfg.camera_family)
        has_prior = camera_params is not None
        if not has_prior:
            logger.info(
                "No calibrated prior for camera_family=%r — using fisheye-"
                "appropriate default_focal_length_factor=0.30",
                cfg.camera_family,
            )

        FISHEYE_DEFAULT_FOCAL_FACTOR = 0.30

        colmap_config = ColmapConfig(
            camera_model="OPENCV_FISHEYE",
            camera_params=camera_params,
            default_focal_length_factor=None if has_prior else FISHEYE_DEFAULT_FOCAL_FACTOR,
            matcher=cfg.colmap_matcher,
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=True,
            refine_principal_point=has_prior,
            refine_extra_params=has_prior,
            # The dual-fisheye rig (use_rig=True) carries an assumed 25mm/180°
            # transform — let BA measure it. No-op on rig-less runs.
            refine_sensor_from_rig=True,
            sift_max_num_features_override=cfg.sift_max_features,
            sift_max_image_size_override=cfg.sift_max_image_size,
            feature_type=cfg.colmap_feature_type,
            matcher_type=cfg.colmap_matcher_type,
            mapper=cfg.colmap_mapper,
            ba_solver=cfg.colmap_ba_solver,
            vocab_tree_path=cfg.vocab_tree_path or None,
            loop_detection=cfg.loop_detection,
            sequential_overlap=cfg.colmap_sequential_overlap,
            guided_matching=cfg.colmap_guided_matching,
            sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
            sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
        )

        def _colmap_progress(stage: str, pct: float, msg: str) -> None:
            base = 55.0
            self._update("colmap", base + pct * (95 - base), msg)

        runner = ColmapRunner(
            images_dir=str(images_dir),
            output_dir=str(out),
            rig_config_path=rig_config_path_str,
            mask_path=colmap_mask_path,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )

        colmap_result = runner.run()
        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

        # ===================================================================
        # Stage 6: Write transforms.json + pointcloud.ply (95-100%)
        # ===================================================================
        sparse_dir = out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = out / "sparse"

        # Native fisheye — write per-frame OPENCV_FISHEYE transforms
        self._update("output", 95.0, "Writing fisheye transforms.json...")
        from .transforms_writer import write_fisheye_transforms
        try:
            transforms_path = write_fisheye_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=out,
                masks_dir=out / "masks",
                log_fn=logger.info,
            )
            dataset_path = str(transforms_path)
        except Exception as exc:
            logger.exception("Fisheye transforms output failed")
            raise RuntimeError(
                f"COLMAP succeeded but transforms.json export failed: {exc}"
            ) from exc

        self._update("complete", 100.0, "Fisheye reconstruction + transforms.json ready")

        # Clean up intermediate files and organize metadata
        self._cleanup_fisheye_output(out, extracted_dir)

        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=dataset_path,
            output_mode=cfg.output_mode,
            num_source_frames=num_pairs,
            num_output_images=2 * num_pairs,
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=colmap_result.views_per_frame,
            expected_images_by_view=colmap_result.expected_images_by_view,
            registered_images_by_view=colmap_result.registered_images_by_view,
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=f"dual_fisheye | family={cfg.camera_family or 'unknown'}",
            gpu_extraction=gpu_accel,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def _cleanup_fisheye_output(out: Path, extracted_dir: Path) -> None:
        """Clean up intermediate files after a successful fisheye run.

        - Always removes the extracted/ working directory — it must not outlive
          a run. Any frames/masks kept for delivery are promoted to
          <output>/images + <output>/masks by the caller beforehand.
        - Moves colmap_debug.log into metadata/.
        """
        import shutil
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir, ignore_errors=True)

        # Move debug files into metadata/
        metadata_dir = out / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        for name in ("colmap_debug.log",):
            src = out / name
            if src.exists():
                dst = metadata_dir / name
                try:
                    src.rename(dst)
                except OSError:
                    pass  # cross-device or permissions — leave in place
