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
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .cubemap_projection import CubemapProjection
from .mask_diagnostics import build_mask_diagnostics_summary

logger = logging.getLogger(__name__)


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
    backend_name: str = ""


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

        Returns:
            Sam3MaskerResult with statistics.
        """
        if not self._initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

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

        for i, frame_file in enumerate(frame_files):
            if progress_callback:
                progress_callback(i, len(frame_files), f"SAM 3 masking {frame_file.name}")

            erp = cv2.imread(str(frame_file))
            if erp is None:
                logger.warning("Could not read %s, skipping", frame_file)
                continue

            # Geometry-aware cubemap pipeline → COLMAP-polarity ERP keep-mask
            erp_keep_mask, frame_diag = self._process_single_erp(erp)

            if erp_keep_mask is not None and np.any(erp_keep_mask < 255):
                result.masked_frames += 1

            if erp_masks_root is not None:
                self._write_erp_mask(erp_keep_mask, erp_masks_root, frame_file.stem)

            # Reframe ERP keep-mask into per-view plugin masks
            per_view_diag = self._write_per_view_masks(
                erp_keep_mask, view_config, masks_root, frame_file.stem,
            )
            if self.config.enable_diagnostics:
                frame_diag["frame"] = frame_file.stem
                frame_diag["per_view_removed_pct"] = per_view_diag
                frame_diag["views_with_removed_pixels"] = sum(
                    1 for value in per_view_diag.values() if value > 0.0
                )
                frame_diag["flags"] = self._build_frame_flags(frame_diag)
                frame_diagnostics.append(frame_diag)

        result.success = True
        result.mask_dir = str(masks_root)
        result.erp_mask_dir = str(erp_masks_root) if erp_masks_root is not None else ""

        if self.config.enable_diagnostics:
            try:
                diag_path = masks_root / "masking_diagnostics.json"
                self._write_diagnostics(diag_path, result, frame_diagnostics)
                result.diagnostics_path = str(diag_path)
                if progress_callback:
                    progress_callback(
                        len(frame_files),
                        len(frame_files),
                        f"Mask diagnostics ready: {result.diagnostics_path}",
                    )
            except Exception as exc:
                logger.warning("Failed to write SAM 3 diagnostics: %s", exc)

        if progress_callback:
            progress_callback(len(frame_files), len(frame_files), "SAM 3 masking complete")

        logger.info(
            "Sam3CubemapMasker: %d/%d frames masked",
            result.masked_frames, result.total_frames,
        )
        return result

    def _process_single_erp(self, erp: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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
        cubemap = CubemapProjection(face_size)
        diagnostics: dict[str, Any] = {
            "source_size": {"width": int(w), "height": int(h)},
            "face_size": int(face_size),
        }

        # 1. ERP → cubemap faces
        faces = cubemap.equirect2cubemap(erp)

        # 2. SAM 3 per-face detection
        face_masks: dict[str, np.ndarray] = {}
        total_detections = 0
        face_coverage_pct: dict[str, float] = {}
        for face_name, face_img in faces.items():
            detection = self._backend.detect_and_segment(
                face_img,
                targets=self.config.prompts,
                detection_confidence=self.config.confidence_threshold,
            )
            face_masks[face_name] = detection
            face_coverage_pct[face_name] = self._coverage_pct(detection > 0)
            if detection.sum() > 0:
                total_detections += 1
                logger.debug("  %s: detected", face_name)

        # 3. Merge face masks → ERP detection mask (0/1 uint8)
        erp_detection = cubemap.cubemap2equirect(face_masks, (w, h))

        coverage_pre = self._coverage_pct(erp_detection > 0)
        logger.info(
            "Cubemap merge: %d/6 faces with detections, %.1f%% ERP coverage",
            total_detections, coverage_pre,
        )

        # 4. Full-resolution ERP postprocess (dilation + fill-holes)
        #    This is where reconstruction-zone gets quality benefit — gaps
        #    that are invisible at 1024px face level become bridgeable at
        #    full ERP resolution.
        erp_detection = self._postprocess_erp_mask(erp_detection)
        coverage_post = self._coverage_pct(erp_detection > 0)

        # 5. Convert: detection (1=detected) → COLMAP polarity (255=keep, 0=remove)
        erp_keep_mask = ((erp_detection == 0).astype(np.uint8)) * 255

        diagnostics["faces_with_detections"] = int(total_detections)
        diagnostics["face_detection_pct"] = face_coverage_pct
        diagnostics["erp_detection_coverage_pct_pre"] = coverage_pre
        diagnostics["erp_detection_coverage_pct_post"] = coverage_post

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
            "version": 1,
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

    def _write_per_view_masks(
        self,
        erp_keep_mask: np.ndarray,
        view_config: Any,
        masks_root: Path,
        frame_stem: str,
    ) -> dict[str, float]:
        """Reframe ERP keep-mask into per-view plugin masks.

        Uses the standalone reframe_view() with mode="nearest" for binary masks.
        ViewConfig.get_all_views() returns (yaw, pitch, fov, name, flip_vertical).
        """
        from .reframer import reframe_view

        views = view_config.get_all_views()
        per_view_removed_pct: dict[str, float] = {}

        for yaw, pitch, fov, view_name, flip_v in views:
            view_dir = masks_root / view_name
            view_dir.mkdir(parents=True, exist_ok=True)

            pinhole_mask = reframe_view(
                erp_keep_mask, fov, yaw, pitch,
                self.config.output_size, mode="nearest",
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
