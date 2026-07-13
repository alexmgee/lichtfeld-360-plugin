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

    # Output mode: "pinhole" = COLMAP dataset, "erp_scaffold" = transforms.json,
    #              "fisheye" = OPENCV_FISHEYE COLMAP dataset (phase 1) /
    #              fisheye-native transforms.json (phase 2+)
    output_mode: str = "pinhole"
    # For output_mode == "fisheye": "native", "pinhole", or "both".
    # Default selects native fisheye output unless the user opts into pinhole crops.
    fisheye_training_output: str = "native"

    # Dual fisheye input (phase 1)
    # input_type: "erp" (default) or "dual_fisheye"
    # camera_family: "dji_osmo360" | "insta360" | None
    # source_mode: "container" (single .osv/.insv), "split" (two pre-split videos),
    #              or "resume" (re-run COLMAP on existing images in output_dir)
    # When source_mode == "split", front_video_path / back_video_path are used and
    # the top-level video_path is ignored. When source_mode == "resume", no video
    # is needed — the pipeline skips extraction/masking/reframing and runs COLMAP
    # directly on existing images/ in the output directory.
    input_type: str = "erp"
    camera_family: Optional[str] = None
    dual_fisheye_calibration_path: str = ""
    source_mode: str = "container"
    front_video_path: str = ""
    back_video_path: str = ""
    keep_streams: bool = False  # retain demuxed front.mp4/back.mp4 alongside output
    keep_pinhole_scaffolding: bool = False  # retain pinhole crops after ERP export
    keep_native_sparse: bool = True  # retain COLMAP sparse/ + database.db after native ERP export (native sparse is a valid EQUIRECTANGULAR dataset, unlike scaffold's)
    keep_extracted_data: bool = False  # retain raw frames, pinhole crops, masks, and scaffold data

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
        if cfg.output_mode != "fisheye":
            return self._run_fisheye_native(cfg, t0)
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

        pinhole_out = out / "pinhole"
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

    def _run_stages(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
        # Phase 1 dispatch shim — dual fisheye path is a separate leaf method
        # (Style 2 leaf-functions refactor for the ERP paths is deferred to
        # a follow-up; see spec §4.2.)
        if cfg.input_type == "dual_fisheye":
            return self._run_dual_fisheye_with_training_output(cfg, t0)

        if cfg.output_mode == "erp_native":
            return self._run_erp_native(cfg, t0)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        extracted_dir = out / "extracted"
        frames_dir = extracted_dir / "frames"
        images_dir = out / "images"
        sparse_dir = out / "sparse"

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

        # Read ERP frame dimensions from the first extracted frame.
        # Needed by ERP scaffold export for equirectangular intrinsics.
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
                    output_dir=str(out),
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
                cubemap_masks_dir = out / "masks"
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
                operator_masks_dir = out / "masks"
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

        rig_config_path = str(out / "rig_config.json")
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
        effective_mask_path = colmap_masks_dir if colmap_masks_dir.is_dir() else (out / "masks")
        runner = ColmapRunner(
            images_dir=str(images_dir),
            output_dir=str(out),
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
        if cfg.output_mode == "erp_scaffold":
            from .scaffold import (
                export_erp_scaffold, cleanup_pinhole_crops, cleanup_colmap_artifacts,
            )

            self._update("output", 85.0, "Cleaning up pinhole scaffold...")

            if erp_width == 0 or erp_height == 0:
                raise RuntimeError(
                    "ERP frame dimensions could not be determined — "
                    "cannot export ERP scaffold"
                )

            # 1. Remove pinhole images/ and masks/ so the ERP export
            #    can rename extracted/frames/ → images/.
            cleanup_pinhole_crops(
                out, keep=cfg.keep_pinhole_scaffolding, log_fn=logger.info,
            )

            self._update("output", 88.0, "Extracting rig poses...")

            # 2. Read COLMAP reconstruction, move ERP frames into
            #    images/, write transforms.json and pointcloud.ply.
            erp_masks = (extracted_dir / "masks") if cfg.enable_masking else None

            # The reference sensor's pitch determines the rig orientation
            # offset from the ERP image center.
            ref_pitch = view_config.rings[0].pitch if view_config.rings else 0.0

            transforms_path = export_erp_scaffold(
                colmap_sparse_dir=out / "sparse",
                erp_frames_dir=frames_dir,
                erp_masks_dir=erp_masks,
                output_dir=out,
                erp_width=erp_width,
                erp_height=erp_height,
                ref_pitch_deg=ref_pitch,
                log_fn=logger.info,
            )

            # 3. Delete sparse/, database.db, etc. — keeping them
            #    triggers LFS pinhole auto-detection.
            cleanup_colmap_artifacts(out, log_fn=logger.info)

            self._update("output", 95.0, "ERP scaffold export complete")
            dataset_path = str(transforms_path)

        else:
            # Pinhole mode: COLMAP dataset already written by ColmapRunner
            self._update("output", 85.0, "COLMAP dataset ready")
            dataset_path = colmap_result.reconstruction_path

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
        """ERP-native pipeline: extract → optional ERP masks → flat stage → COLMAP."""
        import shutil

        from .scaffold import cleanup_colmap_artifacts
        from .transforms_writer import write_erp_native_transforms

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        extracted_dir = out / "extracted"
        frames_dir = extracted_dir / "frames"
        images_dir = out / "images"
        masks_output_dir = out / "masks"

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
        # Stage 2: ERP masking (30-50%, optional)
        # ===================================================================
        mask_result: Optional[MaskResult] = None
        erp_masks_dir = extracted_dir / "masks"

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
        # Stage 3: Stage flat ERP frames (+ masks) (50-55%)
        # ===================================================================
        self._update("staging", 50.0, "Staging ERP frames for COLMAP...")
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for frame_file in frame_files:
            shutil.move(str(frame_file), str(images_dir / frame_file.name))

        num_output_images = len(frame_files)
        effective_colmap_mask: str | None = None

        if cfg.enable_masking and erp_masks_dir.is_dir():
            if masks_output_dir.exists():
                shutil.rmtree(masks_output_dir, ignore_errors=True)
            masks_output_dir.mkdir(parents=True, exist_ok=True)
            for mask_file in sorted(erp_masks_dir.glob("*.png")):
                shutil.move(str(mask_file), str(masks_output_dir / mask_file.name))
            effective_colmap_mask = str(masks_output_dir)

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 4: COLMAP (55-85%)
        # ===================================================================
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
            images_dir=str(images_dir),
            output_dir=str(out),
            rig_config_path=None,
            mask_path=effective_colmap_mask,
            config=colmap_config,
            on_progress=_colmap_progress,
            cancel_check=self._check_cancel,
        )

        colmap_result = runner.run()

        if not colmap_result.success:
            raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

        # ===================================================================
        # Stage 5: Write transforms.json + pointcloud.ply (85-95%)
        # ===================================================================
        self._update("output", 85.0, "Writing native ERP transforms.json...")

        sparse_dir = out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = out / "sparse"

        transforms_path = write_erp_native_transforms(
            colmap_sparse_dir=sparse_dir,
            output_dir=out,
            masks_dir=masks_output_dir if masks_output_dir.is_dir() else None,
            erp_width=erp_width,
            erp_height=erp_height,
            log_fn=logger.info,
        )

        # ===================================================================
        # Stage 6: Finalize (95-100%)
        # ===================================================================
        if not cfg.keep_native_sparse:
            cleanup_colmap_artifacts(out, log_fn=logger.info)

        self._update("complete", 100.0, "Native ERP export complete")
        dataset_path = str(transforms_path)
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
            preset_signature="native EQUIRECTANGULAR",
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
        # Resume mode: skip extraction/masking/reframing, jump to COLMAP
        # ===================================================================
        if cfg.source_mode == "resume":
            # Prefer the current colmap/ dataset layout, but allow resume from
            # older root images/ outputs.
            _colmap_images = out / "colmap" / "images"
            if _colmap_images.is_dir():
                images_dir = _colmap_images
            return self._run_fisheye_resume(cfg, t0, out, images_dir)

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
        # Stage 3: Staging (direct move OR fisheye→pinhole reframe)
        # ===================================================================
        colmap_mask_path: str | None = None
        rig_config_path_str = str(out / "rig_config.json")

        if cfg.output_mode == "fisheye_pinhole":
            # ── Fisheye → pinhole reframe path ──
            self._update("staging", 50.0, "Reframing fisheye → pinhole crops...")

            import cv2
            from .dual_fisheye_calibration_provider import (
                resolve_dual_fisheye_calibration,
            )
            from .fisheye_reframer import FisheyeReframer, FISHEYE_PINHOLE_PRESET

            calibration_resolution = resolve_dual_fisheye_calibration(
                cfg.camera_family,
                override_path=cfg.dual_fisheye_calibration_path,
                output_mode=cfg.output_mode,
            )
            calib = calibration_resolution.calibration
            logger.info(
                "Using dual-fisheye calibration: source=%s path=%s confidence=%s",
                calibration_resolution.source,
                calibration_resolution.source_path or "",
                calibration_resolution.source_confidence,
            )
            if calibration_resolution.warning:
                logger.warning(calibration_resolution.warning)
            self._write_dual_fisheye_calibration_metadata(
                out, cfg, calibration_resolution,
            )
            reframer = FisheyeReframer(calib)
            view_config = FISHEYE_PINHOLE_PRESET

            images_dir.mkdir(parents=True, exist_ok=True)
            output_masks_dir = out / "masks" if mask_enabled else None
            if output_masks_dir:
                output_masks_dir.mkdir(parents=True, exist_ok=True)

            # Create output subfolders for each virtual camera
            for view in view_config.views:
                (images_dir / view.name).mkdir(parents=True, exist_ok=True)
                if output_masks_dir:
                    (output_masks_dir / view.name).mkdir(parents=True, exist_ok=True)

            # Collect extracted frame pairs
            front_frames_dir = extracted_dir / "front"
            back_frames_dir = extracted_dir / "back"
            front_files = sorted(
                f for f in front_frames_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

            total_pairs = len(front_files)
            logger.info("Reframing %d pairs into %d views each (%d total crops)",
                        total_pairs, view_config.total_views(),
                        total_pairs * view_config.total_views())
            print(f"[reframe] Starting: {total_pairs} pairs × {view_config.total_views()} views")
            for pair_idx, front_file in enumerate(front_files):
                if self._check_cancel():
                    raise RuntimeError("Cancelled")

                back_file = back_frames_dir / front_file.name
                if not back_file.exists():
                    logger.warning("No matching back frame for %s", front_file.name)
                    continue

                pct = 50 + (pair_idx / max(total_pairs, 1)) * 15
                self._update("staging", pct,
                             f"Reframing pair {pair_idx + 1}/{total_pairs}")
                if pair_idx % 10 == 0:
                    print(f"[reframe] Pair {pair_idx + 1}/{total_pairs}")

                front_img = cv2.imread(str(front_file))
                back_img = cv2.imread(str(back_file))

                front_mask = None
                back_mask = None
                if mask_enabled:
                    fm = extracted_dir / "masks" / "front" / f"{front_file.stem}.png"
                    bm = extracted_dir / "masks" / "back" / f"{front_file.stem}.png"
                    if fm.exists():
                        front_mask = cv2.imread(str(fm), cv2.IMREAD_GRAYSCALE)
                    if bm.exists():
                        back_mask = cv2.imread(str(bm), cv2.IMREAD_GRAYSCALE)

                results = reframer.extract_all_views(
                    front_img, back_img, view_config,
                    front_mask, back_mask,
                )

                for view, crop, mask_crop in results:
                    crop_path = images_dir / view.name / front_file.name
                    cv2.imwrite(str(crop_path), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, cfg.quality])
                    if mask_crop is not None and output_masks_dir:
                        mask_path = output_masks_dir / view.name / f"{front_file.stem}.png"
                        cv2.imwrite(str(mask_path), mask_crop)

            if output_masks_dir:
                colmap_mask_path = str(output_masks_dir)

            logger.info("Reframed %d pairs into %d views (%d images total)",
                        total_pairs, view_config.total_views(),
                        total_pairs * view_config.total_views())

            # Write mini-rig config for the 2 reference views (front_ctr_hi + front_ctr_lo)
            from .fisheye_reframer import _rotation_matrix
            import json as _json
            from .rig_config import rotation_matrix_to_quaternion

            ref_view = view_config.views[0]   # front_ctr_hi
            lo_view = view_config.views[7]    # front_ctr_lo

            R_ref_m = _rotation_matrix(ref_view.yaw_deg, ref_view.pitch_deg)
            R_lo_m = _rotation_matrix(lo_view.yaw_deg, lo_view.pitch_deg)
            R_rel_m = R_lo_m @ R_ref_m.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_rel_m)

            mini_rig = [{"cameras": [
                {"image_prefix": f"{ref_view.name}/", "ref_sensor": True},
                {"image_prefix": f"{lo_view.name}/",
                 "cam_from_rig_rotation": [qw, qx, qy, qz],
                 "cam_from_rig_translation": [0.0, 0.0, 0.0]},
            ]}]
            rig_path = out / "colmap_mini_rig.json"
            rig_path.write_text(_json.dumps(mini_rig, indent=2))
            print(f"[pipeline] Mini-rig written: {rig_path}")

        else:
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
        if cfg.output_mode == "fisheye_pinhole":
            # Dual-ref COLMAP: front_ctr_hi + front_ctr_lo with mini-rig
            crop_size = view_config.crop_size
            fl = crop_size / 2.0  # 90° FOV
            camera_params = f"{fl:.1f},{fl:.1f},{crop_size/2:.1f},{crop_size/2:.1f}"

            # Copy the 2 reference view folders into a staging directory for COLMAP.
            # (Symlinks are unreliable on Windows — LFS/pycolmap can't follow them.)
            import shutil as _shutil
            colmap_images_dir = out / "colmap_input"
            if colmap_images_dir.exists():
                _shutil.rmtree(str(colmap_images_dir), ignore_errors=True)
            colmap_images_dir.mkdir(exist_ok=True)
            _shutil.copytree(str(images_dir / ref_view.name),
                             str(colmap_images_dir / ref_view.name))
            _shutil.copytree(str(images_dir / lo_view.name),
                             str(colmap_images_dir / lo_view.name))

            # Stage masks for the 2 reference views
            colmap_masks_dir = None
            if mask_enabled and output_masks_dir:
                colmap_masks_dir = out / "colmap_masks"
                if colmap_masks_dir.exists():
                    _shutil.rmtree(str(colmap_masks_dir), ignore_errors=True)
                colmap_masks_dir.mkdir(exist_ok=True)
                _shutil.copytree(str(output_masks_dir / ref_view.name),
                                 str(colmap_masks_dir / ref_view.name))
                _shutil.copytree(str(output_masks_dir / lo_view.name),
                                 str(colmap_masks_dir / lo_view.name))

            print(f"[pipeline] COLMAP config: PINHOLE, matcher={cfg.colmap_matcher}, "
                  f"dual-ref ({ref_view.name} + {lo_view.name})")
            self._update("colmap", 67.0,
                         f"Running COLMAP ({ref_view.name} + {lo_view.name})...")
            colmap_config = ColmapConfig(
                camera_model="PINHOLE",
                camera_mode="PER_FOLDER",
                camera_params=camera_params,
                matcher=cfg.colmap_matcher,
                match_budget_tier=cfg.colmap_match_budget_tier,
                max_num_matches_override=cfg.colmap_max_num_matches,
                refine_focal_length=True,
                refine_principal_point=False,
                refine_extra_params=False,
                sift_max_num_features_override=cfg.sift_max_features,
                sift_max_image_size_override=cfg.sift_max_image_size,
                feature_type=cfg.colmap_feature_type,
                matcher_type=cfg.colmap_matcher_type,
                mapper=cfg.colmap_mapper,
                ba_solver=cfg.colmap_ba_solver,
                loop_detection=cfg.loop_detection,
                sequential_overlap=cfg.colmap_sequential_overlap,
                guided_matching=cfg.colmap_guided_matching,
                sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
                sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
            )
        else:
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
            base = 67.0 if cfg.output_mode == "fisheye_pinhole" else 55.0
            self._update("colmap", base + pct * (95 - base), msg)

        if cfg.output_mode == "fisheye_pinhole":
            # Dual-ref with mini-rig
            runner = ColmapRunner(
                images_dir=str(colmap_images_dir),
                output_dir=str(out),
                rig_config_path=str(rig_path),
                mask_path=str(colmap_masks_dir) if colmap_masks_dir else None,
                config=colmap_config,
                on_progress=_colmap_progress,
                cancel_check=self._check_cancel,
            )
        else:
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

        # Clean up COLMAP staging copies
        if cfg.output_mode == "fisheye_pinhole":
            import shutil
            shutil.rmtree(str(colmap_images_dir), ignore_errors=True)
            if colmap_masks_dir:
                shutil.rmtree(str(colmap_masks_dir), ignore_errors=True)
            rig_path.unlink(missing_ok=True)

        # ===================================================================
        # Stage 6: Write transforms.json + pointcloud.ply (95-100%)
        # ===================================================================
        sparse_dir = out / "sparse" / "0"
        if not sparse_dir.is_dir():
            sparse_dir = out / "sparse"

        if cfg.output_mode == "fisheye_pinhole":
            import shutil as _shutil2

            if cfg.keep_extracted_data:
                extracted_dir.mkdir(exist_ok=True)
                _rig_img_dst = extracted_dir / "pinhole_images"
                if _rig_img_dst.exists():
                    _shutil2.rmtree(str(_rig_img_dst))
                _shutil2.copytree(str(images_dir), str(_rig_img_dst))
                if output_masks_dir and output_masks_dir.exists():
                    _rig_msk_dst = extracted_dir / "pinhole_masks"
                    if _rig_msk_dst.exists():
                        _shutil2.rmtree(str(_rig_msk_dst))
                    _shutil2.copytree(str(output_masks_dir), str(_rig_msk_dst))

            # Flatten view subfolders: images/front_ctr_hi/000042.jpg → images/front_ctr_hi_000042.jpg
            self._update("output", 93.0, "Flattening output folders...")
            _flatten_view_folders(images_dir)
            if output_masks_dir:
                _flatten_view_folders(output_masks_dir)

            self._update("output", 94.0, "Organizing colmap/ directory...")
            colmap_dir = out / "colmap"
            colmap_dir.mkdir(exist_ok=True)
            _dst_images = colmap_dir / "images"
            if _dst_images.exists():
                _shutil2.rmtree(str(_dst_images))
            _shutil2.move(str(images_dir), str(_dst_images))
            images_dir = _dst_images
            if output_masks_dir and output_masks_dir.exists():
                _dst_masks = colmap_dir / "masks"
                if _dst_masks.exists():
                    _shutil2.rmtree(str(_dst_masks))
                _shutil2.move(str(output_masks_dir), str(_dst_masks))
            # Propagate reference-view poses to all 16 views via rig geometry
            self._update("output", 95.0, "Propagating poses to all views...")
            from .transforms_writer import write_rig_propagated_transforms
            try:
                _masks_root = colmap_dir / "masks"
                transforms_path = write_rig_propagated_transforms(
                    colmap_sparse_dir=sparse_dir,
                    images_root=images_dir,
                    output_dir=colmap_dir,
                    view_config=view_config,
                    baseline_m=calib.baseline_m,
                    file_path_prefix="images",
                    masks_root=_masks_root if _masks_root.is_dir() else None,
                    mask_path_prefix="masks",
                    propagated_sparse_output_dir=colmap_dir / "sparse" / "0",
                    log_fn=logger.info,
                )
                dataset_path = str(colmap_dir)
                _shutil2.rmtree(str(out / "sparse"), ignore_errors=True)
                for _db in ("database.db", "database.db-shm", "database.db-wal"):
                    _src_db = out / _db
                    if _src_db.exists():
                        _dst_db = colmap_dir / _db
                        _dst_db.unlink(missing_ok=True)
                        _shutil2.move(str(_src_db), str(_dst_db))
            except Exception as exc:
                logger.exception("Pose propagation failed")
                raise RuntimeError(
                    f"COLMAP succeeded but pose propagation failed: {exc}"
                ) from exc
        else:
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
        self._cleanup_fisheye_output(
            out,
            extracted_dir,
            keep_extracted_data=cfg.keep_extracted_data,
        )

        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=dataset_path,
            output_mode=cfg.output_mode,
            num_source_frames=num_pairs,
            num_output_images=(
                num_pairs * view_config.total_views()
                if cfg.output_mode == "fisheye_pinhole"
                else 2 * num_pairs
            ),
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=(
                view_config.total_views()
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.views_per_frame
            ),
            expected_images_by_view=(
                {view.name: num_pairs for view in view_config.views}
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.expected_images_by_view
            ),
            registered_images_by_view=(
                {view.name: colmap_result.num_complete_frames for view in view_config.views}
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.registered_images_by_view
            ),
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=f"dual_fisheye | family={cfg.camera_family or 'unknown'}",
            gpu_extraction=gpu_accel,
            elapsed_sec=elapsed,
        )

    def _run_fisheye_resume(
        self, cfg: PipelineConfig, t0: float, out: Path, images_dir: Path,
    ) -> PipelineResult:
        """Resume mode: re-run COLMAP on existing images, skip earlier stages.

        Detects flat or subfolder pinhole crops in images/, cleans stale COLMAP
        artifacts, stages reference views, runs COLMAP, and writes transforms.
        """
        import shutil
        import cv2

        from .fisheye_reframer import (
            FISHEYE_PINHOLE_PRESET, _rotation_matrix,
        )
        from .dual_fisheye_calibration_provider import (
            resolve_dual_fisheye_calibration,
        )
        from .colmap_runner import ColmapRunner, ColmapConfig
        from .rig_config import rotation_matrix_to_quaternion
        import json as _json

        print(f"\n[pipeline] Output mode: {cfg.output_mode} (COLMAP only — resuming from existing images)")
        print(f"[pipeline] Output: {out}")

        view_config = FISHEYE_PINHOLE_PRESET
        if cfg.output_mode == "fisheye_pinhole":
            calibration_resolution = resolve_dual_fisheye_calibration(
                cfg.camera_family,
                override_path=cfg.dual_fisheye_calibration_path,
                output_mode=cfg.output_mode,
            )
            calib = calibration_resolution.calibration
            logger.info(
                "Using dual-fisheye calibration: source=%s path=%s confidence=%s",
                calibration_resolution.source,
                calibration_resolution.source_path or "",
                calibration_resolution.source_confidence,
            )
            if calibration_resolution.warning:
                logger.warning(calibration_resolution.warning)
            self._write_dual_fisheye_calibration_metadata(
                out, cfg, calibration_resolution,
            )
        else:
            calib = None
        masks_dir = out / "masks"
        if cfg.output_mode == "fisheye_pinhole" and not masks_dir.is_dir():
            colmap_masks_dir = out / "colmap" / "masks"
            if colmap_masks_dir.is_dir():
                masks_dir = colmap_masks_dir

        # Detect existing images
        if cfg.output_mode == "fisheye_pinhole":
            ref_view = view_config.views[0]
            # Try flat layout first
            flat_candidates = sorted(images_dir.glob(f"{ref_view.name}_*.jpg"))
            if flat_candidates:
                num_pairs = len(flat_candidates)
                layout = "flat"
            else:
                # Try subfolder layout
                subfolder = images_dir / ref_view.name
                if subfolder.is_dir():
                    sub_candidates = sorted(subfolder.glob("*.jpg"))
                    num_pairs = len(sub_candidates)
                    layout = "subfolder"
                else:
                    raise RuntimeError(
                        f"Resume mode: no images found in {images_dir}. "
                        f"Expected flat ({ref_view.name}_*.jpg) or subfolder "
                        f"({ref_view.name}/*.jpg) layout."
                    )
        else:
            # Fisheye native: images/front/ and images/back/
            front_dir = images_dir / "front"
            if not front_dir.is_dir():
                raise RuntimeError(
                    f"Resume mode: no images/front/ directory in {images_dir}"
                )
            num_pairs = len(sorted(front_dir.glob("*.jpg")))
            layout = "fisheye_native"

        print(f"[pipeline] Detected {num_pairs} existing frames ({layout} layout)")
        self._update("colmap", 0.0, f"Resuming from {num_pairs} existing frames...")

        # Clean stale COLMAP artifacts from root (old layout) and colmap/ (reorganized)
        for stale in ("sparse", "colmap_input", "colmap_masks",
                       "colmap_mini_rig.json", "database.db",
                       "database.db-shm", "database.db-wal",
                       "transforms.json", "pointcloud.ply"):
            p = out / stale
            if p.is_dir():
                shutil.rmtree(str(p), ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        colmap_sub = out / "colmap"
        for stale in ("sparse", "database.db", "database.db-shm", "database.db-wal"):
            p = colmap_sub / stale
            if p.is_dir():
                shutil.rmtree(str(p), ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        print("[pipeline] Cleaned stale COLMAP artifacts")

        # ── fisheye_pinhole: stage reference views + mini-rig ──
        if cfg.output_mode == "fisheye_pinhole":
            ref_view = view_config.views[0]   # front_ctr_hi
            lo_view = view_config.views[7]    # front_ctr_lo

            # Write mini-rig config
            R_ref_m = _rotation_matrix(ref_view.yaw_deg, ref_view.pitch_deg)
            R_lo_m = _rotation_matrix(lo_view.yaw_deg, lo_view.pitch_deg)
            R_rel_m = R_lo_m @ R_ref_m.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_rel_m)
            mini_rig = [{"cameras": [
                {"image_prefix": f"{ref_view.name}/", "ref_sensor": True},
                {"image_prefix": f"{lo_view.name}/",
                 "cam_from_rig_rotation": [qw, qx, qy, qz],
                 "cam_from_rig_translation": [0.0, 0.0, 0.0]},
            ]}]
            rig_path = out / "colmap_mini_rig.json"
            rig_path.write_text(_json.dumps(mini_rig, indent=2))

            # Stage reference views into subfolder layout for COLMAP PER_FOLDER
            colmap_images_dir = out / "colmap_input"
            colmap_images_dir.mkdir(exist_ok=True)
            for ref_name in [ref_view.name, lo_view.name]:
                ref_staging = colmap_images_dir / ref_name
                ref_staging.mkdir(parents=True, exist_ok=True)
                if layout == "flat":
                    flat_prefix = f"{ref_name}_"
                    for img in sorted(images_dir.glob(f"{ref_name}_*.jpg")):
                        staged_name = img.name.removeprefix(flat_prefix)
                        shutil.copy2(str(img), str(ref_staging / staged_name))
                else:
                    # subfolder: copytree
                    src = images_dir / ref_name
                    if src.is_dir():
                        shutil.copytree(str(src), str(ref_staging), dirs_exist_ok=True)

            # Stage masks for reference views (if they exist)
            colmap_masks_dir = None
            if masks_dir.is_dir():
                colmap_masks_dir = out / "colmap_masks"
                colmap_masks_dir.mkdir(exist_ok=True)
                for ref_name in [ref_view.name, lo_view.name]:
                    if layout == "flat":
                        mask_staging = colmap_masks_dir / ref_name
                        mask_staging.mkdir(parents=True, exist_ok=True)
                        flat_prefix = f"{ref_name}_"
                        for m in sorted(masks_dir.glob(f"{ref_name}_*.png")):
                            staged_name = m.name.removeprefix(flat_prefix)
                            shutil.copy2(str(m), str(mask_staging / staged_name))
                    else:
                        src = masks_dir / ref_name
                        if src.is_dir():
                            dst = colmap_masks_dir / ref_name
                            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)

            # COLMAP config
            crop_size = view_config.crop_size
            fl = crop_size / 2.0
            camera_params = f"{fl:.1f},{fl:.1f},{crop_size/2:.1f},{crop_size/2:.1f}"

            print(f"[pipeline] COLMAP config: PINHOLE, matcher={cfg.colmap_matcher}, "
                  f"dual-ref ({ref_view.name} + {lo_view.name})")
            self._update("colmap", 10.0,
                         f"Running COLMAP ({ref_view.name} + {lo_view.name})...")
            colmap_config = ColmapConfig(
                camera_model="PINHOLE",
                camera_mode="PER_FOLDER",
                camera_params=camera_params,
                matcher=cfg.colmap_matcher,
                match_budget_tier=cfg.colmap_match_budget_tier,
                max_num_matches_override=cfg.colmap_max_num_matches,
                refine_focal_length=True,
                refine_principal_point=False,
                refine_extra_params=False,
                sift_max_num_features_override=cfg.sift_max_features,
                sift_max_image_size_override=cfg.sift_max_image_size,
                feature_type=cfg.colmap_feature_type,
                matcher_type=cfg.colmap_matcher_type,
                mapper=cfg.colmap_mapper,
                ba_solver=cfg.colmap_ba_solver,
                loop_detection=cfg.loop_detection,
                sequential_overlap=cfg.colmap_sequential_overlap,
                guided_matching=cfg.colmap_guided_matching,
                sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
                sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
            )

            def _colmap_progress(stage: str, pct: float, msg: str) -> None:
                self._update("colmap", 10.0 + pct * 80, msg)

            runner = ColmapRunner(
                images_dir=str(colmap_images_dir),
                output_dir=str(out),
                rig_config_path=str(rig_path),
                mask_path=str(colmap_masks_dir) if colmap_masks_dir else None,
                config=colmap_config,
                on_progress=_colmap_progress,
                cancel_check=self._check_cancel,
            )
            colmap_result = runner.run()
            if not colmap_result.success:
                raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

            # Clean staging
            shutil.rmtree(str(colmap_images_dir), ignore_errors=True)
            if colmap_masks_dir:
                shutil.rmtree(str(colmap_masks_dir), ignore_errors=True)
            rig_path.unlink(missing_ok=True)

            # Flatten if images are still in subfolder layout
            if layout == "subfolder":
                if cfg.keep_extracted_data:
                    _resume_extracted = out / "extracted"
                    _resume_extracted.mkdir(exist_ok=True)
                    _rig_img_dst = _resume_extracted / "pinhole_images"
                    if _rig_img_dst.exists():
                        shutil.rmtree(str(_rig_img_dst))
                    shutil.copytree(str(images_dir), str(_rig_img_dst))
                    if masks_dir.is_dir():
                        _rig_msk_dst = _resume_extracted / "pinhole_masks"
                        if _rig_msk_dst.exists():
                            shutil.rmtree(str(_rig_msk_dst))
                        shutil.copytree(str(masks_dir), str(_rig_msk_dst))

                _flatten_view_folders(images_dir)
                if masks_dir.is_dir():
                    _flatten_view_folders(masks_dir)

            colmap_dir = out / "colmap"
            colmap_dir.mkdir(exist_ok=True)

            final_images_dir = colmap_dir / "images"
            if images_dir != final_images_dir:
                if final_images_dir.exists():
                    shutil.rmtree(str(final_images_dir))
                shutil.move(str(images_dir), str(final_images_dir))
                images_dir = final_images_dir

            final_masks_dir = colmap_dir / "masks"
            if masks_dir.is_dir() and masks_dir != final_masks_dir:
                if final_masks_dir.exists():
                    shutil.rmtree(str(final_masks_dir))
                shutil.move(str(masks_dir), str(final_masks_dir))
                masks_dir = final_masks_dir

            for _db in ("database.db", "database.db-shm", "database.db-wal"):
                _src_db = out / _db
                if _src_db.exists():
                    _dst_db = colmap_dir / _db
                    _dst_db.unlink(missing_ok=True)
                    shutil.move(str(_src_db), str(colmap_dir / _db))

            sparse_dir = out / "sparse" / "0"
            if not sparse_dir.is_dir():
                sparse_dir = out / "sparse"

            self._update("output", 92.0, "Propagating poses to all views...")
            from .transforms_writer import write_rig_propagated_transforms
            _masks_root = colmap_dir / "masks"
            transforms_path = write_rig_propagated_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=colmap_dir,
                view_config=view_config,
                baseline_m=calib.baseline_m,
                file_path_prefix="images",
                masks_root=_masks_root if _masks_root.is_dir() else None,
                mask_path_prefix="masks",
                propagated_sparse_output_dir=colmap_dir / "sparse" / "0",
                log_fn=logger.info,
            )
            dataset_path = str(colmap_dir)
            shutil.rmtree(str(out / "sparse"), ignore_errors=True)

        else:
            # Fisheye native resume — run COLMAP directly on images/
            from .fisheye_priors import infer_fisheye_camera_params
            camera_params = infer_fisheye_camera_params(cfg.camera_family)
            has_prior = camera_params is not None

            colmap_config = ColmapConfig(
                camera_model="OPENCV_FISHEYE",
                camera_params=camera_params,
                default_focal_length_factor=None if has_prior else 0.30,
                matcher=cfg.colmap_matcher,
                match_budget_tier=cfg.colmap_match_budget_tier,
                max_num_matches_override=cfg.colmap_max_num_matches,
                refine_focal_length=True,
                refine_principal_point=has_prior,
                refine_extra_params=has_prior,
                refine_sensor_from_rig=True,  # mirrors the primary fisheye path
                sift_max_num_features_override=cfg.sift_max_features,
                sift_max_image_size_override=cfg.sift_max_image_size,
                feature_type=cfg.colmap_feature_type,
                matcher_type=cfg.colmap_matcher_type,
                mapper=cfg.colmap_mapper,
                ba_solver=cfg.colmap_ba_solver,
                loop_detection=cfg.loop_detection,
                sequential_overlap=cfg.colmap_sequential_overlap,
                guided_matching=cfg.colmap_guided_matching,
                sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
                sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
            )

            def _colmap_progress(stage: str, pct: float, msg: str) -> None:
                self._update("colmap", 10.0 + pct * 80, msg)

            runner = ColmapRunner(
                images_dir=str(images_dir),
                output_dir=str(out),
                rig_config_path=None,
                mask_path=str(masks_dir) if masks_dir.is_dir() else None,
                config=colmap_config,
                on_progress=_colmap_progress,
                cancel_check=self._check_cancel,
            )
            colmap_result = runner.run()
            if not colmap_result.success:
                raise RuntimeError(f"COLMAP failed: {colmap_result.error}")

            sparse_dir = out / "sparse" / "0"
            if not sparse_dir.is_dir():
                sparse_dir = out / "sparse"

            self._update("output", 92.0, "Writing fisheye transforms...")
            from .transforms_writer import write_fisheye_transforms
            transforms_path = write_fisheye_transforms(
                colmap_sparse_dir=sparse_dir,
                images_root=images_dir,
                output_dir=out,
                masks_dir=masks_dir,
                log_fn=logger.info,
            )
            dataset_path = str(transforms_path)

        self._update("complete", 100.0, "Resume complete — COLMAP + transforms ready")

        elapsed = time.time() - t0
        return PipelineResult(
            success=True,
            dataset_path=dataset_path,
            output_mode=cfg.output_mode,
            num_source_frames=num_pairs,
            num_output_images=num_pairs * (view_config.total_views() if cfg.output_mode == "fisheye_pinhole" else 2),
            num_aligned_cameras=colmap_result.num_registered_images,
            num_registered_frames=colmap_result.num_registered_frames,
            num_complete_frames=colmap_result.num_complete_frames,
            num_partial_frames=colmap_result.num_partial_frames,
            views_per_frame=(
                view_config.total_views()
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.views_per_frame
            ),
            expected_images_by_view=(
                {view.name: num_pairs for view in view_config.views}
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.expected_images_by_view
            ),
            registered_images_by_view=(
                {view.name: colmap_result.num_complete_frames for view in view_config.views}
                if cfg.output_mode == "fisheye_pinhole"
                else colmap_result.registered_images_by_view
            ),
            partial_frame_examples=colmap_result.partial_frame_examples,
            dropped_frame_examples=colmap_result.dropped_frame_examples,
            preset_signature=f"dual_fisheye | family={cfg.camera_family or 'unknown'} | resume",
            gpu_extraction=False,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def _write_dual_fisheye_calibration_metadata(
        out: Path,
        cfg: PipelineConfig,
        calibration_resolution,
    ) -> None:
        """Record calibration provenance for reproducible fisheye runs."""
        import hashlib
        import json

        calibration = calibration_resolution.calibration
        source_path = calibration_resolution.source_path or ""
        source_sha256 = ""
        if source_path:
            try:
                h = hashlib.sha256()
                with Path(source_path).open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                source_sha256 = h.hexdigest()
            except OSError:
                source_sha256 = ""

        metadata = {
            "camera_family": cfg.camera_family,
            "output_mode": cfg.output_mode,
            "source": calibration_resolution.source,
            "source_path": source_path,
            "source_sha256": source_sha256,
            "source_confidence": calibration_resolution.source_confidence,
            "warning": calibration_resolution.warning,
            "camera_model": calibration.camera_model,
            "front": {
                "image_size": list(calibration.front.image_size),
                "rms_error": calibration.front.rms_error,
                "num_images_used": calibration.front.num_images_used,
                "fov_degrees": calibration.front.fov_degrees,
            },
            "back": {
                "image_size": list(calibration.back.image_size),
                "rms_error": calibration.back.rms_error,
                "num_images_used": calibration.back.num_images_used,
                "fov_degrees": calibration.back.fov_degrees,
            },
            "rig": {
                "front_rotation_deg": calibration.front_rotation_deg,
                "back_rotation_deg": calibration.back_rotation_deg,
                "baseline_m": calibration.baseline_m,
                "baseline_axis": list(calibration.baseline_axis),
            },
        }

        metadata_dir = out / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / "dual_fisheye_calibration.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _cleanup_fisheye_output(
        out: Path, extracted_dir: Path, keep_extracted_data: bool = False
    ) -> None:
        """Clean up intermediate files after a successful fisheye run.

        - Removes the extracted/ directory (demuxed streams already cleaned,
          frames staged to images/, masks staged to masks/).
        - Moves colmap_debug.log into metadata/.
        """
        import shutil
        if extracted_dir.exists():
            if not keep_extracted_data:
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
