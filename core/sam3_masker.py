# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""SAM 3 cubemap masking — geometry-aware independent path.

Ported from reconstruction-zone's _process_equirectangular() approach.

Pipeline per ERP frame:
1. CubemapProjection.equirect2cubemap() → 6 cube faces
2. Sam3Backend.detect_and_segment() on each face (text-prompted)
3. CubemapProjection.cubemap2equirect() → merged ERP detection mask
4. Full-resolution ERP postprocess (dilation + fill-holes)
5. Convert to COLMAP polarity (white=keep, black=remove)
6. Reframe ERP keep-mask into plugin's per-view masks
"""
from __future__ import annotations

import json
import shutil
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .cubemap_projection import CubemapProjection
from .mask_diagnostics import build_mask_diagnostics_summary

logger = logging.getLogger(__name__)


def _cuda_sync() -> None:
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class _MaskingTimers:
    """Accumulated perf_counter spans for masking diagnostics (debug-gated)."""

    total: float = 0.0
    init: float = 0.0
    imread: float = 0.0
    process_single_erp: float = 0.0
    write_erp_mask: float = 0.0
    write_per_view: float = 0.0
    diagnostics: float = 0.0
    t_e2c: float = 0.0
    t_infer: float = 0.0
    t_c2e: float = 0.0
    t_post: float = 0.0
    t_bgr_pil: float = 0.0
    t_set_image: float = 0.0
    t_set_text_prompt: float = 0.0
    t_cpu_resize_merge: float = 0.0


@dataclass
class Sam3MaskerConfig:
    """Configuration for SAM 3 cubemap masking."""

    prompts: list[str] = field(default_factory=lambda: ["person", "tripod"])
    confidence_threshold: float = 0.3
    dilation_px: int = 5
    fill_holes: bool = True
    output_size: int = 1920
    face_size: int | None = None  # None = auto (min(1024, w // 4))
    enable_diagnostics: bool = False


@dataclass
class Sam3MaskerResult:
    """Result from Sam3CubemapMasker.process_frames()."""

    success: bool = False
    total_frames: int = 0
    masked_frames: int = 0
    mask_dir: str = ""
    erp_mask_dir: str = ""
    diagnostics_path: str = ""
    diagnostics_error: str = ""
    backend_name: str = ""
    timers: dict = field(default_factory=dict)


class Sam3CubemapMasker:
    """Geometry-aware SAM 3 masking via cubemap decomposition.

    Adapted from reconstruction-zone's _process_equirectangular().
    Consumes ERP frames, runs SAM 3 on cubemap faces, merges back
    to ERP, applies full-resolution postprocess, then emits the
    plugin's standard per-view masks.

    Does NOT use direction estimation, synthetic views, or video tracking.
    """

    def __init__(self, config: Sam3MaskerConfig | None = None) -> None:
        self.config = config or Sam3MaskerConfig()
        self._backend: Any = None
        self._initialized = False
        self._mask_remap_cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._mask_remap_cache_key: tuple | None = None
        self._mask_remap_views: list | None = None
        self._cubemap: CubemapProjection | None = None
        self._cubemap_key: tuple[int, int, int, float] | None = None

    def initialize(self) -> None:
        """Load Sam3Backend."""
        from .backends import Sam3Backend

        self._backend = Sam3Backend(
            confidence_threshold=self.config.confidence_threshold,
        )
        self._backend.initialize()
        self._initialized = True
        logger.info("Sam3CubemapMasker initialized")

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str,
        view_config: Any,
        erp_mask_dir: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        write_per_view: bool = True,
    ) -> Sam3MaskerResult:
        """Process extracted ERP frames through cubemap SAM 3 pipeline.

        Args:
            frames_dir: Directory of extracted ERP frames (jpg/png).
            output_dir: Root output directory. Masks written to
                output_dir/masks/{view_id}/{frame_id}.png.
            view_config: ViewConfig with preset geometry.
            erp_mask_dir: Optional directory to also persist merged ERP
                keep-masks ({frame_id}.png) before reframing.
            progress_callback: Optional (current, total, message) callback.
            write_per_view: When False, skip per-view mask reframe output
                (native ERP path only needs the ERP keep-mask).

        Returns:
            Sam3MaskerResult with statistics.
        """
        if not self._initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        diag = self.config.enable_diagnostics
        timers = _MaskingTimers() if diag else None
        infer_timing: dict[str, float] | None = {} if diag else None
        total_start = time.perf_counter() if diag else 0.0
        init_start = time.perf_counter() if diag else 0.0

        frames_path = Path(frames_dir)
        out_path = Path(output_dir)
        masks_root = out_path / "masks"
        erp_masks_root = Path(erp_mask_dir) if erp_mask_dir else None

        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            logger.warning("No frames found in %s", frames_dir)
            return Sam3MaskerResult(success=True, total_frames=0)

        result = Sam3MaskerResult(total_frames=len(frame_files))
        result.backend_name = type(self._backend).__name__ if self._backend is not None else ""
        frame_diagnostics: list[dict[str, Any]] = []
        prev_detection_mask: np.ndarray | None = None
        prev_frame_diag: dict[str, Any] | None = None
        self._mask_remap_cache = None
        self._mask_remap_cache_key = None
        self._mask_remap_views = None
        self._cubemap = None
        self._cubemap_key = None

        if diag:
            timers.init = time.perf_counter() - init_start

        for i, frame_file in enumerate(frame_files):
            if progress_callback:
                progress_callback(i, len(frame_files), f"SAM 3 masking {frame_file.name}")

            if diag:
                t0 = time.perf_counter()
            erp = cv2.imread(str(frame_file))
            if diag:
                timers.imread += time.perf_counter() - t0
            if erp is None:
                logger.warning("Could not read %s, skipping", frame_file)
                continue

            # Geometry-aware cubemap pipeline → COLMAP-polarity ERP keep-mask
            if diag:
                t0 = time.perf_counter()
            erp_keep_mask, frame_diag = self._process_single_erp(
                erp, timers=timers, infer_timing=infer_timing,
            )
            if diag:
                timers.process_single_erp += time.perf_counter() - t0

            if erp_keep_mask is not None and np.any(erp_keep_mask < 255):
                result.masked_frames += 1

            if erp_masks_root is not None:
                if diag:
                    t0 = time.perf_counter()
                self._write_erp_mask(erp_keep_mask, erp_masks_root, frame_file.stem)
                if diag:
                    timers.write_erp_mask += time.perf_counter() - t0

            # Reframe ERP keep-mask into per-view plugin masks
            if write_per_view:
                if diag:
                    t0 = time.perf_counter()
                per_view_diag = self._write_per_view_masks(
                    erp_keep_mask, view_config, masks_root, frame_file.stem,
                )
                if diag:
                    timers.write_per_view += time.perf_counter() - t0
            else:
                per_view_diag = {}
            if self.config.enable_diagnostics and write_per_view:
                if diag:
                    t0 = time.perf_counter()
                frame_diag["frame"] = frame_file.stem
                frame_diag["per_view_removed_pct"] = per_view_diag
                frame_diag["views_with_removed_pixels"] = sum(
                    1 for value in per_view_diag.values() if value > 0.0
                )
                current_detection_mask = (erp_keep_mask == 0)
                frame_diag["temporal_iou_prev_pct"] = self._mask_iou_pct(
                    current_detection_mask,
                    prev_detection_mask,
                )
                if prev_frame_diag is None:
                    frame_diag["coverage_delta_prev_pct"] = None
                    frame_diag["views_delta_prev"] = None
                    frame_diag["face_distribution_jump_pct"] = None
                    frame_diag["dominant_face_changed"] = False
                else:
                    frame_diag["coverage_delta_prev_pct"] = round(
                        abs(
                            float(frame_diag.get("erp_detection_coverage_pct_post", 0.0))
                            - float(prev_frame_diag.get("erp_detection_coverage_pct_post", 0.0))
                        ),
                        3,
                    )
                    frame_diag["views_delta_prev"] = int(
                        abs(
                            int(frame_diag.get("views_with_removed_pixels", 0))
                            - int(prev_frame_diag.get("views_with_removed_pixels", 0))
                        )
                    )
                    frame_diag["face_distribution_jump_pct"] = self._face_distribution_jump_pct(
                        frame_diag.get("face_share_pct") or {},
                        prev_frame_diag.get("face_share_pct") or {},
                    )
                    frame_diag["dominant_face_changed"] = (
                        str(frame_diag.get("dominant_face_name", "")) !=
                        str(prev_frame_diag.get("dominant_face_name", ""))
                    )
                frame_diag["flags"] = self._build_frame_flags(frame_diag)
                frame_diagnostics.append(frame_diag)
                prev_detection_mask = current_detection_mask
                prev_frame_diag = frame_diag
                if diag:
                    timers.diagnostics += time.perf_counter() - t0

        result.success = True
        result.mask_dir = str(masks_root)
        result.erp_mask_dir = str(erp_masks_root) if erp_masks_root is not None else ""

        if self.config.enable_diagnostics:
            if diag:
                t0 = time.perf_counter()
            primary_diag_path = masks_root / "masking_diagnostics.json"
            try:
                self._write_diagnostics(primary_diag_path, result, frame_diagnostics)
                if erp_masks_root is not None:
                    self._copy_diagnostics_to_erp_masks(primary_diag_path, erp_masks_root)
                result.diagnostics_path = str(primary_diag_path)
                if progress_callback:
                    progress_callback(
                        len(frame_files),
                        len(frame_files),
                        f"Mask diagnostics ready: {result.diagnostics_path}",
                    )
            except Exception as exc:
                result.diagnostics_error = str(exc)
                logger.warning("Failed to write SAM 3 diagnostics: %s", exc)
                fallback_written = self._write_fallback_diagnostics(
                    primary_diag_path,
                    result,
                    frame_diagnostics,
                    str(exc),
                )
                if fallback_written and erp_masks_root is not None:
                    self._copy_diagnostics_to_erp_masks(primary_diag_path, erp_masks_root)
                if fallback_written:
                    result.diagnostics_path = str(primary_diag_path)
            if diag:
                timers.diagnostics += time.perf_counter() - t0
                timers.t_bgr_pil = infer_timing.get("t_bgr_pil", 0.0)
                timers.t_set_image = infer_timing.get("t_set_image", 0.0)
                timers.t_set_text_prompt = infer_timing.get("t_set_text_prompt", 0.0)
                timers.t_cpu_resize_merge = infer_timing.get("t_cpu_resize_merge", 0.0)
                timers.total = time.perf_counter() - total_start
                result.timers = asdict(timers)
                self._log_masking_timer_report(timers)

        if progress_callback:
            progress_callback(len(frame_files), len(frame_files), "SAM 3 masking complete")

        logger.info(
            "Sam3CubemapMasker: %d/%d frames masked",
            result.masked_frames, result.total_frames,
        )
        return result

    @staticmethod
    def _log_masking_timer_report(timers: _MaskingTimers) -> None:
        """Emit a compact reconciliation breakdown (debug-gated)."""
        outer_sum = (
            timers.init
            + timers.imread
            + timers.process_single_erp
            + timers.write_erp_mask
            + timers.write_per_view
            + timers.diagnostics
        )
        residual = timers.total - outer_sum
        logger.info(
            "SAM3 masking timer report (s): "
            "total=%.3f init=%.3f imread=%.3f process_single_erp=%.3f "
            "write_erp_mask=%.3f write_per_view=%.3f diagnostics=%.3f "
            "outer_sum=%.3f residual=%.3f | "
            "inner: t_e2c=%.3f t_infer=%.3f t_c2e=%.3f t_post=%.3f | "
            "nested: t_bgr_pil=%.3f t_set_image=%.3f t_set_text_prompt=%.3f "
            "t_cpu_resize_merge=%.3f",
            timers.total,
            timers.init,
            timers.imread,
            timers.process_single_erp,
            timers.write_erp_mask,
            timers.write_per_view,
            timers.diagnostics,
            outer_sum,
            residual,
            timers.t_e2c,
            timers.t_infer,
            timers.t_c2e,
            timers.t_post,
            timers.t_bgr_pil,
            timers.t_set_image,
            timers.t_set_text_prompt,
            timers.t_cpu_resize_merge,
        )

    def _get_cubemap(self, w: int, h: int, face_size: int) -> CubemapProjection:
        """Reuse one CubemapProjection across frames; invalidate on size change."""
        overlap_degrees = 0.0
        key = (w, h, face_size, overlap_degrees)
        if self._cubemap is None or self._cubemap_key != key:
            self._cubemap = CubemapProjection(face_size, overlap_degrees=overlap_degrees)
            self._cubemap_key = key
        return self._cubemap

    def _process_single_erp(
        self,
        erp: np.ndarray,
        *,
        timers: _MaskingTimers | None = None,
        infer_timing: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process one ERP frame. Returns COLMAP-polarity mask (255=keep, 0=remove).

        Follows reconstruction-zone's geometry-aware cubemap pipeline:
        1. ERP → 6 cubemap faces
        2. SAM 3 per-face detection (0/1 uint8 masks)
        3. Merge face masks → ERP detection mask
        4. Full-resolution postprocess (dilation + fill-holes)
        5. Convert to COLMAP polarity
        """
        h, w = erp.shape[:2]
        face_size = self.config.face_size or min(1024, w // 4)
        cubemap = self._get_cubemap(w, h, face_size)
        diagnostics: dict[str, Any] = {
            "source_size": {"width": int(w), "height": int(h)},
            "face_size": int(face_size),
        }

        # 1. ERP → cubemap faces
        if timers is not None:
            t0 = time.perf_counter()
        faces = cubemap.equirect2cubemap(erp)
        if timers is not None:
            timers.t_e2c += time.perf_counter() - t0

        # 2. SAM 3 per-face detection
        face_masks: dict[str, np.ndarray] = {}
        total_detections = 0
        face_coverage_pct: dict[str, float] = {}
        if timers is not None:
            t0 = time.perf_counter()
        for face_name, face_img in faces.items():
            detection = self._backend.detect_and_segment(
                face_img,
                targets=self.config.prompts,
                detection_confidence=self.config.confidence_threshold,
                timing=infer_timing,
            )
            face_masks[face_name] = detection
            face_coverage_pct[face_name] = self._coverage_pct(detection > 0)
            if detection.sum() > 0:
                total_detections += 1
                logger.debug("  %s: detected", face_name)
        if timers is not None:
            timers.t_infer += time.perf_counter() - t0

        # 3. Merge face masks → ERP detection mask (0/1 uint8)
        if timers is not None:
            t0 = time.perf_counter()
        erp_detection = cubemap.cubemap2equirect(face_masks, (w, h))
        if timers is not None:
            timers.t_c2e += time.perf_counter() - t0

        coverage_pre = self._coverage_pct(erp_detection > 0)
        logger.info(
            "Cubemap merge: %d/6 faces with detections, %.1f%% ERP coverage",
            total_detections, coverage_pre,
        )

        # 4. Full-resolution ERP postprocess (dilation + fill-holes)
        #    This is where reconstruction-zone gets quality benefit — gaps
        #    that are invisible at 1024px face level become bridgeable at
        #    full ERP resolution.
        if timers is not None:
            t0 = time.perf_counter()
        erp_detection = self._postprocess_erp_mask(erp_detection)
        if timers is not None:
            timers.t_post += time.perf_counter() - t0
        coverage_post = self._coverage_pct(erp_detection > 0)

        # 5. Convert: detection (1=detected) → COLMAP polarity (255=keep, 0=remove)
        erp_keep_mask = ((erp_detection == 0).astype(np.uint8)) * 255

        diagnostics["faces_with_detections"] = int(total_detections)
        diagnostics["face_detection_pct"] = face_coverage_pct
        diagnostics["erp_detection_coverage_pct_pre"] = coverage_pre
        diagnostics["erp_detection_coverage_pct_post"] = coverage_post
        diagnostics.update(self._component_metrics(erp_detection))
        diagnostics.update(self._face_distribution_metrics(face_coverage_pct))

        return erp_keep_mask, diagnostics

    def _postprocess_erp_mask(self, mask: np.ndarray) -> np.ndarray:
        """Full-resolution ERP postprocess: dilation then fill-holes.

        Adapted from reconstruction-zone's postprocess_mask(final=True).
        Runs at full ERP resolution where gaps are large enough to bridge.
        """
        # Dilation
        if self.config.dilation_px > 0:
            k = self.config.dilation_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Fill interior holes
        if self.config.fill_holes:
            mask = self._fill_mask_holes(mask)

        # Morphological cleanup — remove small isolated artifacts
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)

        return mask

    @staticmethod
    def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
        """Fill interior holes in a binary mask via flood-fill.

        Adapted from reconstruction-zone's _fill_mask_holes().
        1. Morphological close to bridge narrow channels.
        2. Flood-fill from padded border to find reachable exterior.
        3. Any interior region the flood can't reach is a hole — OR back in.
        """
        h, w = mask.shape[:2]

        # Close kernel: ~0.4% of image width, clamped to 15-51px
        close_k = max(15, min(51, int(w * 0.004)))
        if close_k % 2 == 0:
            close_k += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # Flood-fill from padded border
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = closed
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 1)
        exterior = flood[1:-1, 1:-1]

        # Interior holes = not in mask AND not reachable from exterior
        holes = ((closed == 0) & (exterior == 0)).astype(np.uint8)
        return np.maximum(mask, holes)

    @staticmethod
    def _coverage_pct(mask: np.ndarray) -> float:
        """Percent of pixels that are active in a boolean/binary mask."""
        if mask.size == 0:
            return 0.0
        active = np.count_nonzero(mask)
        return round(float(active / mask.size * 100.0), 3)

    @staticmethod
    def _component_metrics(mask: np.ndarray) -> dict[str, float]:
        """Describe connected-component structure of the ERP detection mask."""
        binary = (mask > 0).astype(np.uint8)
        active = int(np.count_nonzero(binary))
        if active == 0:
            return {
                "component_count": 0,
                "largest_component_share_pct": 0.0,
                "secondary_component_share_pct": 0.0,
                "tiny_component_count": 0,
                "tiny_component_share_pct": 0.0,
            }

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        component_sizes = sorted(
            (
                int(stats[idx, cv2.CC_STAT_AREA])
                for idx in range(1, num_labels)
            ),
            reverse=True,
        )
        largest = component_sizes[0] if component_sizes else 0
        secondary = sum(component_sizes[1:]) if len(component_sizes) > 1 else 0
        tiny_threshold = max(64, int(active * 0.02))
        tiny_sizes = [size for size in component_sizes[1:] if size <= tiny_threshold]
        return {
            "component_count": int(len(component_sizes)),
            "largest_component_share_pct": round(float(largest / active * 100.0), 3),
            "secondary_component_share_pct": round(float(secondary / active * 100.0), 3),
            "tiny_component_count": int(len(tiny_sizes)),
            "tiny_component_share_pct": round(float(sum(tiny_sizes) / active * 100.0), 3),
        }

    @staticmethod
    def _face_distribution_metrics(face_detection_pct: dict[str, float]) -> dict[str, Any]:
        """Describe how detections are distributed across cubemap faces."""
        total = sum(float(value) for value in face_detection_pct.values())
        if total <= 0.0:
            return {
                "face_share_pct": {
                    str(name): 0.0 for name in face_detection_pct.keys()
                },
                "dominant_face_name": "",
                "dominant_face_share_pct": 0.0,
                "down_face_share_pct": 0.0,
            }

        face_share_pct = {
            str(name): round(float(value / total * 100.0), 3)
            for name, value in face_detection_pct.items()
        }
        dominant_face_name = max(
            face_share_pct.items(),
            key=lambda item: (float(item[1]), item[0]),
        )[0]
        return {
            "face_share_pct": face_share_pct,
            "dominant_face_name": dominant_face_name,
            "dominant_face_share_pct": float(face_share_pct.get(dominant_face_name, 0.0)),
            "down_face_share_pct": float(face_share_pct.get("down", 0.0)),
        }

    @staticmethod
    def _mask_iou_pct(current_mask: np.ndarray, previous_mask: np.ndarray | None) -> float | None:
        """IoU between current and previous ERP detection masks as a percent."""
        if previous_mask is None:
            return None
        current = current_mask.astype(bool)
        previous = previous_mask.astype(bool)
        union = np.logical_or(current, previous)
        union_count = int(np.count_nonzero(union))
        if union_count == 0:
            return 100.0
        intersection = np.logical_and(current, previous)
        return round(float(np.count_nonzero(intersection) / union_count * 100.0), 3)

    @staticmethod
    def _face_distribution_jump_pct(
        current_shares: dict[str, float],
        previous_shares: dict[str, float],
    ) -> float | None:
        """Half-L1 distance between consecutive face-share distributions."""
        if not previous_shares:
            return None
        face_names = sorted(set(current_shares.keys()) | set(previous_shares.keys()))
        total_delta = sum(
            abs(float(current_shares.get(name, 0.0)) - float(previous_shares.get(name, 0.0)))
            for name in face_names
        )
        return round(float(total_delta / 2.0), 3)

    @staticmethod
    def _build_frame_flags(frame_diag: dict[str, Any]) -> list[str]:
        """Flag suspicious or notable frame outcomes for quick triage."""
        flags: list[str] = []
        if int(frame_diag.get("faces_with_detections", 0)) == 0:
            flags.append("no_face_detections")
        if float(frame_diag.get("erp_detection_coverage_pct_post", 0.0)) >= 60.0:
            flags.append("high_erp_detection_coverage")
        per_view = frame_diag.get("per_view_removed_pct", {})
        if any(float(value) >= 50.0 for value in per_view.values()):
            flags.append("heavy_removed_view")
        return flags

    def _write_diagnostics(
        self,
        path: Path,
        result: Sam3MaskerResult,
        frames: list[dict[str, Any]],
    ) -> None:
        """Write a JSON diagnostics summary for SAM 3 cubemap masking."""
        doc = {
            "version": 3,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "sam3_cubemap",
            "backend": result.backend_name,
            "prompts": list(self.config.prompts),
            "confidence_threshold": float(self.config.confidence_threshold),
            "dilation_px": int(self.config.dilation_px),
            "fill_holes": bool(self.config.fill_holes),
            "output_size": int(self.config.output_size),
            "face_size": int(self.config.face_size) if self.config.face_size is not None else None,
            "total_frames": int(result.total_frames),
            "masked_frames": int(result.masked_frames),
            "frames_with_face_detections": sum(
                1 for frame in frames if int(frame.get("faces_with_detections", 0)) > 0
            ),
            "summary": build_mask_diagnostics_summary(frames),
            "frames": frames,
        }
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        logger.info("SAM 3 diagnostics written to %s", path)

    @staticmethod
    def _copy_diagnostics_to_erp_masks(primary_path: Path, erp_masks_root: Path) -> None:
        """Mirror diagnostics beside ERP masks for easier run inspection."""
        erp_masks_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary_path, erp_masks_root / primary_path.name)

    @staticmethod
    def _write_fallback_diagnostics(
        path: Path,
        result: Sam3MaskerResult,
        frames: list[dict[str, Any]],
        error_text: str,
    ) -> bool:
        """Best-effort fallback so diagnostics failures stay visible on disk."""
        fallback_doc = {
            "version": 3,
            "mode": "sam3_cubemap",
            "backend": result.backend_name,
            "total_frames": int(result.total_frames),
            "masked_frames": int(result.masked_frames),
            "frames_written": len(frames),
            "summary": build_mask_diagnostics_summary(frames),
            "frames": frames,
            "error": error_text,
            "partial": True,
        }
        try:
            path.write_text(json.dumps(fallback_doc, indent=2), encoding="utf-8")
            logger.warning("Wrote fallback SAM 3 diagnostics to %s", path)
            return True
        except Exception as fallback_exc:
            logger.warning("Failed to write fallback SAM 3 diagnostics: %s", fallback_exc)
            return False

    def _ensure_mask_remap_cache(
        self,
        erp_keep_mask: np.ndarray,
        view_config: Any,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list]:
        """Build or reuse remap tables for per-view mask reframing."""
        from .reframer import _build_reframe_remap

        h_eq, w_eq = erp_keep_mask.shape[:2]
        views = view_config.get_all_views()
        cache_key = (
            w_eq,
            h_eq,
            tuple((yaw, pitch, fov, name, flip_v) for yaw, pitch, fov, name, flip_v in views),
        )
        if self._mask_remap_cache_key != cache_key:
            self._mask_remap_cache = [
                _build_reframe_remap(
                    fov, yaw, pitch, self.config.output_size, w_eq, h_eq,
                )
                for yaw, pitch, fov, _name, _flip_v in views
            ]
            self._mask_remap_views = views
            self._mask_remap_cache_key = cache_key
        return self._mask_remap_cache, self._mask_remap_views

    def _write_per_view_masks(
        self,
        erp_keep_mask: np.ndarray,
        view_config: Any,
        masks_root: Path,
        frame_stem: str,
    ) -> dict[str, float]:
        """Reframe ERP keep-mask into per-view plugin masks.

        Uses cached remap tables (same geometry as Reframer.reframe_batch)
        with INTER_NEAREST for binary masks.
        """
        from .reframer import _apply_reframe_remap

        remap_tables, views = self._ensure_mask_remap_cache(erp_keep_mask, view_config)
        per_view_removed_pct: dict[str, float] = {}

        for (map_x, map_y), (yaw, pitch, fov, view_name, flip_v) in zip(
            remap_tables, views, strict=True,
        ):
            view_dir = masks_root / view_name
            view_dir.mkdir(parents=True, exist_ok=True)

            pinhole_mask = _apply_reframe_remap(
                erp_keep_mask, map_x, map_y, mode="nearest",
            )
            if flip_v:
                pinhole_mask = np.flipud(pinhole_mask)

            mask_path = view_dir / f"{frame_stem}.png"
            cv2.imwrite(str(mask_path), pinhole_mask)
            per_view_removed_pct[view_name] = self._coverage_pct(pinhole_mask == 0)

        return per_view_removed_pct

    @staticmethod
    def _write_erp_mask(
        erp_keep_mask: np.ndarray,
        erp_masks_root: Path,
        frame_stem: str,
    ) -> None:
        """Persist the merged ERP keep-mask for artifact parity/debugging."""
        erp_masks_root.mkdir(parents=True, exist_ok=True)
        mask_path = erp_masks_root / f"{frame_stem}.png"
        cv2.imwrite(str(mask_path), erp_keep_mask)

    def cleanup(self) -> None:
        """Release SAM 3 backend resources."""
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None
        self._initialized = False
        logger.info("Sam3CubemapMasker cleaned up")
