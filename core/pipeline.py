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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .colmap_runner import ColmapConfig, ColmapResult, ColmapRunner
from .colmap_runner import infer_shared_pinhole_camera_params
from .masker import Masker, MaskConfig, MaskResult, is_masking_available
from .setup_checks import is_sam3_masking_ready
from .overlap_mask import compute_overlap_masks
from .presets import VIEW_PRESETS, ViewConfig
from .reframer import Reframer
from .rig_config import write_rig_config
from .sharpest_extractor import SharpestConfig, SharpestExtractor

logger = logging.getLogger(__name__)


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
    extraction_sharpness: str = "best"     # none, basic, best
    blur_metric: str = "tenengrad"         # tenengrad, laplacian
    scene_threshold: float = 0.3
    blur_scale_width: int = 640
    quality: int = 95
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None

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
    colmap_preset: str = "normal"
    colmap_matcher: str = "sequential"  # "sequential", "exhaustive", "vocab_tree"
    colmap_match_budget_tier: str = "default"
    colmap_max_num_matches: Optional[int] = None

    # Output mode: "pinhole" = COLMAP dataset, "erp" = transforms.json
    output_mode: str = "pinhole"


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
    elapsed_sec: float = 0.0
    error: str = ""


def _build_runtime_view_config(cfg: PipelineConfig) -> ViewConfig:
    """Resolve the active preset and apply runtime output overrides."""
    from .presets import DEFAULT_PRESET

    base = VIEW_PRESETS.get(cfg.preset_name, VIEW_PRESETS[DEFAULT_PRESET])
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
    if view_config.include_zenith:
        parts.append(f"ZN@90/f{view_config.zenith_fov:g}")
    if view_config.include_nadir:
        parts.append(f"ND@-90/f{view_config.zenith_fov:g}")
    return f"{preset_name} | " + "; ".join(parts)


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
            result = self._run_stages(cfg, t0)
        except Exception as exc:
            logger.exception("Pipeline failed")
            preset_signature = _format_preset_signature(
                cfg.preset_name,
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

    def _run_stages(self, cfg: PipelineConfig, t0: float) -> PipelineResult:
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

        if self._check_cancel():
            raise RuntimeError("Cancelled")

        # ===================================================================
        # Stage 2+3: Masking and Reframing
        #
        # The order depends on the preset:
        #   Default:  Mask (ERP) → Reframe (images + masks)
        #   Cubemap:  Reframe (images only) → Mask (direct on all faces)
        # ===================================================================
        view_config = _build_runtime_view_config(cfg)
        preset_signature = _format_preset_signature(cfg.preset_name, view_config)
        mask_result: Optional[MaskResult] = None
        is_cubemap = cfg.preset_name == "cubemap"

        # ── Method-specific masking availability gate ──────────────
        if cfg.enable_masking:
            if cfg.masking_method == "sam3_cubemap":
                if not is_sam3_masking_ready():
                    raise RuntimeError(
                        "SAM 3 masking requires sam3 + weights. "
                        "Install SAM 3 via the plugin settings panel."
                    )
                if cfg.preset_name != "cubemap":
                    raise RuntimeError(
                        "SAM 3 masking is currently cubemap-only. "
                        "Select the Cubemap preset to use SAM 3, or "
                        "switch to the FullCircle masking method."
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
                and cfg.masking_method == "sam3_cubemap"
                and is_cubemap):
            from .sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

            self._update("masking", 20.0, "Initializing SAM 3 cubemap masker...")

            sam3_cfg = Sam3MaskerConfig(
                prompts=cfg.mask_prompts,
                confidence_threshold=0.3,
                output_size=cfg.output_size,
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
            )

            if self._check_cancel():
                raise RuntimeError("Cancelled")

            # SAM 3 masker writes masks directly to out/masks/{view_id}/
            # Now reframe images only (no mask reframe needed)
            self._update("reframe", 35.0, "Reframing to cubemap views...")
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
            preset=cfg.colmap_preset,
            camera_params=camera_params,
            default_focal_length_factor=default_focal_length_factor,
            matcher=cfg.colmap_matcher,
            match_budget_tier=cfg.colmap_match_budget_tier,
            max_num_matches_override=cfg.colmap_max_num_matches,
            refine_focal_length=True,
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
        if cfg.output_mode == "erp":
            # ERP mode: write transforms.json with original ERP images
            # and camera-to-world poses derived from the COLMAP reconstruction.
            #
            # This requires:
            #   1. Reading the COLMAP sparse reconstruction
            #   2. Identifying reference sensor images (first view per station)
            #   3. Extracting their poses as the station pose
            #   4. Writing transforms.json with EQUIRECTANGULAR camera model
            #      pointing to the original ERP frames
            #
            # TODO: Implement ERP output mode.  For now, the COLMAP dataset
            #       (pinhole mode) is always available as a fallback.
            self._update("output", 85.0, "ERP output mode not yet implemented — using COLMAP dataset")
            logger.warning(
                "ERP output mode is not yet implemented. "
                "Falling back to COLMAP dataset (pinhole mode)."
            )
            dataset_path = colmap_result.reconstruction_path

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
            output_mode=cfg.output_mode if cfg.output_mode != "erp" else "pinhole",
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
            elapsed_sec=elapsed,
        )
