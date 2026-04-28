# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Paired sharpest-frame extraction from dual fisheye sources.

Ported from reconstruction-zone (`prep360/core/paired_split_video_extractor.py`)
with two additions:
  1. Embedded `_detect_scene_change` (the source imported it from
     SharpestExtractor; the plugin's SharpestExtractor doesn't expose it,
     so we keep the implementation local).
  2. New `extract_dual_fisheye()` orchestrator that handles all three
     supported input shapes:
       - DJI .osv               → demux via OSVHandler, then extract pair
       - Insta360 .insv (X4/X5) → demux via OSVHandler, then extract pair
       - Insta360 .insv (older) → file-pair detection, then extract pair

Pair selection uses `min(front_score, back_score)` so both lenses are
guaranteed sharp — a frame can't be picked just because one side is
crisp while the other is blurry/occluded/smudged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2

from .osv_handler import OSVHandler


# ---------------------------------------------------------------------------
# Insta360 .insv pair detection
# ---------------------------------------------------------------------------

def detect_insv_pair(path: Path) -> Optional[Tuple[Path, Path]]:
    """Detect the front/rear sibling for an older Insta360 .insv file.

    Older Insta360 cameras (ONE X / X2 / X3) ship dual fisheye as TWO
    files in the same directory:
        VID_..._00_NNN.insv  ← front lens
        VID_..._10_NNN.insv  ← rear lens

    Args:
        path: Path to either file in the pair (front or rear).

    Returns:
        (front_path, rear_path) if the pair exists; None if `path` is a
        single-file container (X4/X5) or the sibling is missing.
    """
    name = path.name
    if "_00_" in name:
        front = path
        rear = path.with_name(name.replace("_00_", "_10_"))
    elif "_10_" in name:
        rear = path
        front = path.with_name(name.replace("_10_", "_00_"))
    else:
        return None

    if front.exists() and rear.exists():
        return front, rear
    return None


# ---------------------------------------------------------------------------
# Configuration / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PairedExtractorConfig:
    """Configuration for paired sharpest-frame extraction."""

    mode: str = "sharpest"               # "fixed" or "sharpest"
    scoring_method: str = "tenengrad"    # "laplacian" or "tenengrad"
    scene_detection: bool = True
    interval_sec: float = 2.0
    quality: int = 95
    output_format: str = "jpg"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    scene_threshold: float = 0.3
    scale_width: int = 1920


@dataclass
class PairedExtractorResult:
    """Result of paired extraction."""

    success: bool
    pair_count: int = 0
    output_dir: str = ""
    front_paths: List[str] = field(default_factory=list)
    back_paths: List[str] = field(default_factory=list)
    selected_times: List[float] = field(default_factory=list)
    source_front_frames: List[int] = field(default_factory=list)
    source_back_frames: List[int] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Sharpness + scene-change scoring (cv2 only)
# ---------------------------------------------------------------------------

def _laplacian_sharpness(frame, analysis_width: int = 1920) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > analysis_width > 0:
        resized_height = max(1, int(round(height * (analysis_width / width))))
        gray = cv2.resize(gray, (analysis_width, resized_height), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad_sharpness(frame, analysis_width: int = 1920) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > analysis_width > 0:
        resized_height = max(1, int(round(height * (analysis_width / width))))
        gray = cv2.resize(gray, (analysis_width, resized_height), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def _detect_scene_change(prev_bgr, curr_bgr, threshold: float) -> bool:
    """HSV histogram-correlation scene detector. Embedded from
    reconstruction-zone's SharpestExtractor (the plugin's version of
    SharpestExtractor doesn't expose this).

    `threshold` in [0, 1]; higher = more sensitive. Mapped to
    correlation_threshold = 1 - threshold.
    """
    corr_threshold = 1.0 - threshold
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    return corr < corr_threshold


# ---------------------------------------------------------------------------
# PairedExtractor
# ---------------------------------------------------------------------------

class PairedExtractor:
    """Extract paired front/back frames from two video files.

    Use `extract()` directly if you already have two video files (e.g.,
    after demuxing a .osv or after locating an .insv pair). Use
    `extract_dual_fisheye()` to dispatch from a single container path.
    """

    @staticmethod
    def _format_clock(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _open_video(path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Could not read FPS/frame count for {path}")
        return cap, fps, frame_count

    @staticmethod
    def _shared_stream_info(front_fps: float, back_fps: float,
                            front_count: int, back_count: int) -> Tuple[float, int]:
        fps_delta = abs(front_fps - back_fps)
        if fps_delta > 0.05:
            raise RuntimeError(
                f"Front/back FPS mismatch too large: {front_fps:.3f} vs {back_fps:.3f}"
            )
        return min(front_fps, back_fps), min(front_count, back_count)

    @staticmethod
    def _ensure_empty_output(path: Path) -> None:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty paired extraction folder: {path}"
            )

    @staticmethod
    def _write_pair_image(path: Path, frame, config: PairedExtractorConfig) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = config.output_format.lower()
        if ext in ("jpg", "jpeg"):
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(config.quality)])
        elif ext == "png":
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(path), frame)

    @staticmethod
    def _fixed_frame_indices(start_frame: int, end_frame: int, window_size: int) -> List[int]:
        return list(range(start_frame, end_frame, window_size))

    @staticmethod
    def _pair_sharpness_sort_key(entry: dict) -> Tuple[float, float, float]:
        """Rank pairs by min(front, back) so the worse lens still passes;
        tiebreakers: average score, then -|imbalance|."""
        min_sharp = min(entry["front_score"], entry["back_score"])
        avg_sharp = (entry["front_score"] + entry["back_score"]) * 0.5
        imbalance = abs(entry["front_score"] - entry["back_score"])
        return (min_sharp, avg_sharp, -imbalance)

    @staticmethod
    def _split_pair_chunk_at_scenes(chunk: List[dict]) -> List[List[dict]]:
        if not chunk:
            return []
        sub_chunks: List[List[dict]] = []
        current: List[dict] = []
        for entry in chunk:
            if entry.get("scene_change") and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    def _select_from_paired_entries(
        self,
        paired_entries: List[dict],
        window_size: int,
        scene_aware: bool,
        log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[int], List[float], List[float]]:
        def _log(msg):
            if log:
                log(msg)

        selected_indices: List[int] = []
        front_scores: List[float] = []
        back_scores: List[float] = []
        current_chunk: List[dict] = []
        current_chunk_index: Optional[int] = None
        total_chunks = 0

        def _flush_chunk(chunk_entries: List[dict]) -> None:
            nonlocal total_chunks
            if not chunk_entries:
                return
            total_chunks += 1
            sub_chunks = (
                self._split_pair_chunk_at_scenes(chunk_entries)
                if scene_aware else
                [chunk_entries]
            )
            for sub_chunk in sub_chunks:
                winner = max(sub_chunk, key=self._pair_sharpness_sort_key)
                selected_indices.append(int(winner["absolute_frame"]))
                front_scores.append(float(winner["front_score"]))
                back_scores.append(float(winner["back_score"]))

        for entry in paired_entries:
            chunk_index = int(entry["relative_frame"]) // window_size
            if current_chunk_index is None:
                current_chunk_index = chunk_index
            if chunk_index != current_chunk_index:
                _flush_chunk(current_chunk)
                current_chunk = []
                current_chunk_index = chunk_index
            current_chunk.append(entry)

        _flush_chunk(current_chunk)
        _log(f"  Selection: {len(selected_indices)} winners from {total_chunks} chunks")
        return selected_indices, front_scores, back_scores

    def _select_sharpest_frame_indices(
        self,
        front_video: str,
        back_video: str,
        start_frame: int,
        end_frame: int,
        window_size: int,
        shared_fps: float,
        config: PairedExtractorConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        _log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[int], List[float], List[float]]:
        if _log is None:
            _log = lambda _msg: None

        scene_aware = config.scene_detection
        score_fn = (
            _tenengrad_sharpness if config.scoring_method == "tenengrad"
            else _laplacian_sharpness
        )

        front_cap = back_cap = None
        try:
            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            total_to_score = max(1, end_frame - start_frame)
            total_duration = total_to_score / shared_fps if shared_fps > 0 else 0.0
            paired_entries: List[dict] = []
            prev_front_small = None
            prev_back_small = None

            method_label = config.scoring_method.title()

            for offset, frame_idx in enumerate(range(start_frame, end_frame), start=1):
                if cancel_check and cancel_check():
                    raise RuntimeError("Cancelled")

                ok_front, front_frame = front_cap.read()
                ok_back, back_frame = back_cap.read()
                if not ok_front or not ok_back:
                    break

                front_score = score_fn(front_frame, config.scale_width)
                back_score = score_fn(back_frame, config.scale_width)

                is_scene = False
                if scene_aware:
                    h, w = front_frame.shape[:2]
                    if config.scale_width and w > config.scale_width:
                        scale = config.scale_width / w
                        new_h = int(h * scale)
                        front_small = cv2.resize(front_frame, (config.scale_width, new_h))
                        back_small = cv2.resize(back_frame, (config.scale_width, new_h))
                    else:
                        front_small = front_frame
                        back_small = back_frame

                    if prev_front_small is not None:
                        front_scene = _detect_scene_change(
                            prev_front_small, front_small, config.scene_threshold)
                        back_scene = _detect_scene_change(
                            prev_back_small, back_small, config.scene_threshold)
                        is_scene = front_scene or back_scene

                    prev_front_small = front_small
                    prev_back_small = back_small

                paired_entries.append({
                    "relative_frame": frame_idx - start_frame,
                    "absolute_frame": frame_idx,
                    "front_score": front_score,
                    "back_score": back_score,
                    "scene_change": is_scene,
                })

                if progress_callback and (offset == 1 or offset == total_to_score or offset % 30 == 0):
                    raw_pct = min(offset / total_to_score, 1.0)
                    elapsed_sec = offset / shared_fps if shared_fps > 0 else 0.0
                    msg = (
                        "Analyzing pair timeline: "
                        f"{self._format_clock(elapsed_sec)} / {self._format_clock(total_duration)} "
                        f"({int(raw_pct * 100)}%) [{method_label}]"
                    )
                    progress_callback(int(raw_pct * 80), 100, msg)

            if paired_entries:
                front_vals = [e["front_score"] for e in paired_entries]
                back_vals = [e["back_score"] for e in paired_entries]
                imbalances = [abs(e["front_score"] - e["back_score"]) for e in paired_entries]
                _log(f"  {config.scoring_method.title()} scoring: {len(paired_entries)} frame pairs")
                _log(f"  Front sharpness: min={min(front_vals):.1f} max={max(front_vals):.1f} mean={sum(front_vals)/len(front_vals):.1f}")
                _log(f"  Back sharpness:  min={min(back_vals):.1f} max={max(back_vals):.1f} mean={sum(back_vals)/len(back_vals):.1f}")
                _log(f"  Imbalance: min={min(imbalances):.1f} max={max(imbalances):.1f} mean={sum(imbalances)/len(imbalances):.1f}")
                if scene_aware:
                    scene_count = sum(1 for e in paired_entries if e["scene_change"])
                    _log(f"  Scene changes detected: {scene_count}")

            if not paired_entries:
                raise RuntimeError("Pair analysis found no overlapping frames")

            if progress_callback:
                label = "Selecting scene-aware pair winners..." if scene_aware else "Selecting pair winners..."
                progress_callback(82, 100, label)

            return self._select_from_paired_entries(
                paired_entries, window_size,
                scene_aware=scene_aware, log=_log,
            )
        finally:
            if front_cap is not None:
                front_cap.release()
            if back_cap is not None:
                back_cap.release()

    def _extract_selected_pairs(
        self,
        front_cap, back_cap,
        frame_indices: List[int],
        start_frame: int,
        end_frame: int,
        shared_fps: float,
        front_out: Path,
        back_out: Path,
        config: PairedExtractorConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[str], List[str], List[float], List[int], List[int]]:
        frame_set = set(frame_indices)
        front_paths: List[str] = []
        back_paths: List[str] = []
        selected_times: List[float] = []
        source_front_frames: List[int] = []
        source_back_frames: List[int] = []

        total_selected = len(frame_indices)
        written = 0

        for frame_idx in range(start_frame, end_frame):
            if cancel_check and cancel_check():
                raise RuntimeError("Cancelled")

            ok_front, front_frame = front_cap.read()
            ok_back, back_frame = back_cap.read()
            if not ok_front or not ok_back:
                break

            if frame_idx not in frame_set:
                continue

            written += 1
            frame_id = f"{written:06d}"
            front_path = front_out / f"{frame_id}.{config.output_format.lower()}"
            back_path = back_out / f"{frame_id}.{config.output_format.lower()}"
            self._write_pair_image(front_path, front_frame, config)
            self._write_pair_image(back_path, back_frame, config)
            front_paths.append(str(front_path))
            back_paths.append(str(back_path))
            selected_times.append(frame_idx / shared_fps)
            source_front_frames.append(frame_idx)
            source_back_frames.append(frame_idx)

            if progress_callback and (written == 1 or written == total_selected or written % 10 == 0):
                progress_callback(
                    written, total_selected,
                    f"extracting {written}/{total_selected} selected pairs",
                )

            if written >= total_selected:
                break

        return front_paths, back_paths, selected_times, source_front_frames, source_back_frames

    def extract(
        self,
        front_video: str,
        back_video: str,
        output_dir: str,
        config: Optional[PairedExtractorConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> PairedExtractorResult:
        """Extract paired frames from two video files.

        Output layout (in `output_dir`):
            front/000001.jpg, front/000002.jpg, ...
            back/000001.jpg,  back/000002.jpg,  ...
            paired_extraction_manifest.json

        (Note: this is one level shallower than the reconstruction-zone
        version, which used `front/frames/` and `back/frames/`. We use
        `front/` and `back/` directly so they map cleanly into COLMAP's
        `images/front/` + `images/back/` layout for PER_FOLDER mode.)
        """
        _log = log or (lambda _msg: None)
        if config is None:
            config = PairedExtractorConfig()

        mode = config.mode.lower()
        if mode not in {"fixed", "sharpest"}:
            return PairedExtractorResult(
                success=False, output_dir=output_dir,
                error=f"Unsupported paired extraction mode: {config.mode}",
            )

        out_root = Path(output_dir)
        front_out = out_root / "front"
        back_out = out_root / "back"
        self._ensure_empty_output(front_out)
        self._ensure_empty_output(back_out)
        manifest_path = out_root / "paired_extraction_manifest.json"
        if manifest_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing paired extraction manifest: {manifest_path}"
            )
        front_out.mkdir(parents=True, exist_ok=True)
        back_out.mkdir(parents=True, exist_ok=True)

        front_cap = back_cap = None
        try:
            front_cap, front_fps, front_count = self._open_video(front_video)
            back_cap, back_fps, back_count = self._open_video(back_video)
            shared_fps, shared_count = self._shared_stream_info(
                front_fps, back_fps, front_count, back_count,
            )

            start_frame = max(0, int(round((config.start_sec or 0.0) * shared_fps)))
            end_frame = shared_count
            if config.end_sec is not None:
                end_frame = min(end_frame, int(round(config.end_sec * shared_fps)))
            if end_frame <= start_frame:
                return PairedExtractorResult(
                    success=False, output_dir=str(out_root),
                    error="Invalid time range for paired extraction",
                )

            window_size = max(1, int(round(config.interval_sec * shared_fps)))

            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if mode == "fixed":
                selected_indices = self._fixed_frame_indices(start_frame, end_frame, window_size)
                selected_front_scores: List[float] = []
                selected_back_scores: List[float] = []
            else:
                selected_indices, selected_front_scores, selected_back_scores = (
                    self._select_sharpest_frame_indices(
                        front_video, back_video,
                        start_frame, end_frame, window_size, shared_fps,
                        config,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        _log=_log,
                    )
                )

            if not selected_indices:
                return PairedExtractorResult(
                    success=False, output_dir=str(out_root),
                    error="No shared frame pairs were selected",
                )

            front_cap.release()
            back_cap.release()
            front_cap = back_cap = None

            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if progress_callback:
                progress_callback(
                    0, len(selected_indices),
                    f"extracting 0/{len(selected_indices)} selected pairs",
                )

            front_paths, back_paths, selected_times, src_front, src_back = (
                self._extract_selected_pairs(
                    front_cap, back_cap, selected_indices,
                    start_frame, end_frame, shared_fps,
                    front_out, back_out, config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            )

            # Manifest
            selection_method = (
                f"{config.scoring_method}_scene_aware_pair"
                if mode == "sharpest" and config.scene_detection else
                f"{config.scoring_method}_pair" if mode == "sharpest" else
                "fixed_interval_pair"
            )
            manifest = {
                "schema_version": 1,
                "dataset_type": "paired_split_frames",
                "front_video": str(Path(front_video).resolve()),
                "back_video": str(Path(back_video).resolve()),
                "mode": mode,
                "scoring_method": config.scoring_method if mode == "sharpest" else None,
                "scene_detection": config.scene_detection if mode == "sharpest" else None,
                "interval_sec": config.interval_sec,
                "fps": shared_fps,
                "selection_method": selection_method,
                "pairs": [],
            }
            front_metric_values = (
                selected_front_scores if mode == "sharpest" else [None] * len(front_paths)
            )
            back_metric_values = (
                selected_back_scores if mode == "sharpest" else [None] * len(front_paths)
            )
            for index, (fp, bp, t, sf, sb, fm, bm) in enumerate(
                zip(front_paths, back_paths, selected_times, src_front, src_back,
                    front_metric_values, back_metric_values),
                start=1,
            ):
                manifest["pairs"].append({
                    "pair_index": index,
                    "frame_id": f"{index:06d}",
                    "front_image": Path(fp).relative_to(out_root).as_posix(),
                    "back_image": Path(bp).relative_to(out_root).as_posix(),
                    "time_sec": round(t, 3),
                    "source_front_frame": sf,
                    "source_back_frame": sb,
                    "score_kind": f"{config.scoring_method}_sharpness" if mode == "sharpest" else None,
                    "front_score": fm,
                    "back_score": bm,
                    "pair_score": (
                        min(fm, bm) if mode == "sharpest" and fm is not None and bm is not None
                        else None
                    ),
                })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            return PairedExtractorResult(
                success=True, pair_count=len(front_paths),
                output_dir=str(out_root),
                front_paths=front_paths, back_paths=back_paths,
                selected_times=selected_times,
                source_front_frames=src_front, source_back_frames=src_back,
            )
        except Exception as exc:
            return PairedExtractorResult(
                success=False, output_dir=str(out_root), error=str(exc),
            )
        finally:
            if front_cap is not None:
                front_cap.release()
            if back_cap is not None:
                back_cap.release()


# ---------------------------------------------------------------------------
# High-level orchestrator: dispatch on container shape
# ---------------------------------------------------------------------------

def extract_dual_fisheye(
    input_path: str,
    output_dir: str,
    config: Optional[PairedExtractorConfig] = None,
    keep_streams: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> PairedExtractorResult:
    """Extract paired frames from any supported dual fisheye source.

    Dispatches on the container shape:
      - DJI .osv               → demux via ffmpeg, then extract pair
      - Insta360 .insv (X4/X5) → demux via ffmpeg, then extract pair
      - Insta360 .insv (older) → file-pair detection, then extract pair

    Args:
        input_path: Path to a .osv or .insv file. For older Insta360,
            either the `_00_` or `_10_` file works — the sibling is
            located by filename pattern.
        output_dir: Where to write extracted frames + manifest.
            Layout: <output_dir>/front/, <output_dir>/back/,
            <output_dir>/paired_extraction_manifest.json.
        config: Extraction settings (defaults to PairedExtractorConfig()).
        keep_streams: If True, keep demuxed front.mp4/back.mp4 alongside
            the output. If False (default), they're deleted after frame
            extraction. Ignored for older Insta360 (no demux involved).
        progress_callback: fn(current, total, message).
        cancel_check: fn() -> bool, polled between frames.
        log: fn(message), for verbose status lines.

    Returns:
        PairedExtractorResult.
    """
    _log = log or (lambda _msg: None)
    src = Path(input_path)
    if not src.exists():
        return PairedExtractorResult(
            success=False, output_dir=output_dir,
            error=f"Input file not found: {input_path}",
        )

    # Older Insta360 .insv pair?
    pair = detect_insv_pair(src) if src.suffix.lower() == ".insv" else None
    if pair is not None:
        front_video, rear_video = pair
        _log(f"Detected Insta360 file pair: front={front_video.name}, rear={rear_video.name}")
        return PairedExtractor().extract(
            front_video=str(front_video),
            back_video=str(rear_video),
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            log=log,
        )

    # Single-file container: .osv or single-file .insv (X4/X5).
    # Demux to temp directory.
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    streams_dir = out_root / "_demuxed_streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    handler = OSVHandler()
    _log(f"Demuxing {src.name} via ffmpeg stream copy...")
    if progress_callback:
        progress_callback(0, 100, f"Demuxing {src.name}...")
    try:
        front_video, rear_video = handler.demux_streams(
            str(src), str(streams_dir), streams="both",
        )
    except Exception as exc:
        return PairedExtractorResult(
            success=False, output_dir=output_dir,
            error=f"Demux failed: {exc}",
        )

    if not front_video or not rear_video:
        return PairedExtractorResult(
            success=False, output_dir=output_dir,
            error="Demux returned empty front/back paths",
        )

    _log(f"Demuxed: front={Path(front_video).name}, rear={Path(rear_video).name}")

    try:
        result = PairedExtractor().extract(
            front_video=front_video,
            back_video=rear_video,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            log=log,
        )
    finally:
        if not keep_streams:
            shutil.rmtree(streams_dir, ignore_errors=True)
            _log("Cleaned up demuxed streams")
        else:
            _log(f"Demuxed streams retained at {streams_dir}")

    return result
