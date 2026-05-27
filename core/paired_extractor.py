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
import math
import shutil
import subprocess
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
    gpu_accelerated: bool = False


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

    _gpu_ok: Optional[bool] = None

    @classmethod
    def _gpu_available(cls) -> bool:
        """Check if CUDA OpenCV with cudacodec and required ops is available."""
        if cls._gpu_ok is None:
            try:
                cls._gpu_ok = (
                    cv2.cuda.getCudaEnabledDeviceCount() > 0
                    and callable(getattr(cv2.cudacodec, 'createVideoReader', None))
                    and callable(getattr(cv2.cudacodec, 'VideoReaderInitParams', None))
                    and callable(getattr(cv2.cuda, 'createSobelFilter', None))
                    and callable(getattr(cv2.cuda, 'createLaplacianFilter', None))
                    and callable(getattr(cv2.cuda, 'resize', None))
                    and callable(getattr(cv2.cuda, 'cvtColor', None))
                    and callable(getattr(cv2.cuda, 'sum', None))
                    and callable(getattr(cv2.cuda, 'multiply', None))
                    and callable(getattr(cv2.cuda, 'add', None))
                    and hasattr(cv2.cuda, 'GpuMat')
                )
            except Exception:
                cls._gpu_ok = False
        return cls._gpu_ok

    @staticmethod
    def _frame_range_for_seconds(
        fps: float,
        total_frame_count: int,
        start_sec: Optional[float],
        end_sec: Optional[float],
    ) -> Tuple[float, int, int]:
        """Convert a user time range to an exclusive frame range."""
        range_start_sec = max(0.0, float(start_sec or 0.0))
        start_frame = max(0, int(math.ceil(range_start_sec * fps - 1e-9)))
        if end_sec is None:
            end_frame = total_frame_count
        else:
            range_end_sec = max(range_start_sec, float(end_sec))
            end_frame = min(
                total_frame_count,
                int(math.ceil(range_end_sec * fps - 1e-9)),
            )
        return range_start_sec, start_frame, end_frame

    @staticmethod
    def _time_window_index(
        frame_idx: int,
        fps: float,
        range_start_sec: float,
        interval_sec: float,
    ) -> int:
        """Return the user-time interval window containing frame_idx."""
        interval = max(float(interval_sec), 1e-9)
        frame_time = frame_idx / fps
        elapsed = max(0.0, frame_time - range_start_sec)
        return int(math.floor((elapsed + 1e-9) / interval))

    @staticmethod
    def _nvdec_failure_hint(media_summary: str) -> str:
        """Return a concise hint for common non-NVDEC media formats."""
        summary = media_summary.lower()
        if any(token in summary for token in ("dnxhd", "dnxhr", "vc3", "avdh")):
            return (
                "DNxHD/DNxHR is not supported by NVDEC; CPU fallback is expected. "
                "Use HEVC/H.265 or H.264 for GPU extraction."
            )
        if "prores" in summary or "apch" in summary or "apcn" in summary:
            return (
                "ProRes is not supported by NVDEC; CPU fallback is expected. "
                "Use HEVC/H.265 or H.264 for GPU extraction."
            )
        if media_summary:
            return (
                "NVDEC may not support this codec, profile, chroma format, "
                "container, or resolution on this GPU."
            )
        return "NVDEC could not open this stream; CPU fallback is expected."

    def _media_summary_for_gpu_failure(self, video_path: str) -> str:
        """Return a compact video summary for explaining GPU decode fallback."""
        try:
            from .sharpest_extractor import _resolve_ffmpeg_binaries, _SUBPROCESS_FLAGS
            _, ffprobe = _resolve_ffmpeg_binaries()
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries",
                    "stream=codec_name,codec_long_name,profile,codec_tag_string,"
                    "pix_fmt,width,height,avg_frame_rate,r_frame_rate",
                    "-of", "json",
                    str(video_path),
                ],
                capture_output=True, text=True, check=True, timeout=10,
                **_SUBPROCESS_FLAGS,
            )
            data = json.loads(proc.stdout)
            streams = data.get("streams") or []
            if streams:
                stream = streams[0]
                codec = stream.get("codec_name") or "unknown codec"
                profile = stream.get("profile")
                tag = stream.get("codec_tag_string")
                pix_fmt = stream.get("pix_fmt")
                width = stream.get("width")
                height = stream.get("height")
                rate = (
                    stream.get("avg_frame_rate")
                    or stream.get("r_frame_rate")
                    or "unknown fps"
                )
                codec_bits = []
                if profile and profile not in ("unknown", "N/A"):
                    codec_bits.append(str(profile))
                codec_bits.append(str(codec))
                if tag and tag not in ("unknown", "N/A"):
                    codec_bits.append(str(tag))
                parts = [" / ".join(codec_bits)]
                if pix_fmt:
                    parts.append(str(pix_fmt))
                if width and height:
                    parts.append(f"{width}x{height}")
                if rate and rate != "0/0":
                    parts.append(f"{rate} fps")
                return ", ".join(parts)
        except Exception:
            pass
        return ""

    def _gpu_reader_failure_messages(
        self,
        video_path: str,
        stream_label: str = "",
        exc: Optional[BaseException] = None,
    ) -> List[str]:
        """Build user-facing log lines for cudacodec reader failures."""
        prefix = f"{stream_label} " if stream_label else ""
        messages = [f"  GPU: {prefix}cudacodec reader failed"]
        summary = self._media_summary_for_gpu_failure(video_path)
        if summary:
            messages.append(f"       media: {summary}")
        messages.append(f"       note: {self._nvdec_failure_hint(summary)}")
        if exc is not None:
            detail = str(exc).strip().splitlines()
            if detail:
                messages.append(f"       OpenCV: {detail[0]}")
        return messages

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

    # -- GPU streaming paired extraction ------------------------------------

    def _extract_sharpest_gpu(
        self,
        front_video: str,
        back_video: str,
        out_root: Path,
        front_out: Path,
        back_out: Path,
        start_frame: int,
        end_frame: int,
        range_start_sec: float,
        shared_fps: float,
        config: PairedExtractorConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        _log: Optional[Callable[[str], None]] = None,
    ) -> Optional[PairedExtractorResult]:
        """GPU streaming single-pass paired extraction.

        Scores and writes winners directly from GPU memory to avoid
        GPU/CPU decoder frame-index misalignment.

        Returns PairedExtractorResult on success or cancellation.
        Returns None on GPU init/runtime failure (caller falls back to CPU).
        """
        import numpy as np
        import time

        if _log is None:
            _log = lambda _msg: None

        total_frames = end_frame - start_frame
        scene_aware = config.scene_detection

        # Open both cudacodec readers with firstFrameIdx
        params = cv2.cudacodec.VideoReaderInitParams()
        if start_frame > 0:
            params.firstFrameIdx = start_frame
        try:
            front_reader = cv2.cudacodec.createVideoReader(front_video, params=params)
        except (cv2.error, TypeError) as e:
            for msg in self._gpu_reader_failure_messages(front_video, "front", e):
                _log(msg)
            return None
        try:
            back_reader = cv2.cudacodec.createVideoReader(back_video, params=params)
        except (cv2.error, TypeError) as e:
            for msg in self._gpu_reader_failure_messages(back_video, "back", e):
                _log(msg)
            return None

        gpu_written_paths: List[str] = []

        try:
            # Read first frame pair for format detection
            ret_f, first_front = front_reader.nextFrame()
            ret_b, first_back = back_reader.nextFrame()
            if not ret_f or not ret_b:
                _log("  GPU: could not read first frame pair")
                return None

            # Format detection — validate BOTH streams
            f_ch, b_ch = first_front.channels(), first_back.channels()
            f_depth, b_depth = first_front.type() & 7, first_back.type() & 7

            if f_ch != b_ch or f_depth != b_depth:
                _log(f"  GPU: front/back format mismatch "
                     f"(front: {f_ch}ch depth={f_depth}, back: {b_ch}ch depth={b_depth})")
                return None

            channels = f_ch
            is_16bit = (f_depth == 2)
            is_8bit = (f_depth == 0)

            if not (is_8bit or is_16bit):
                _log(f"  GPU: unsupported frame depth (depth={f_depth})")
                return None
            if channels not in (3, 4):
                _log(f"  GPU: unsupported frame format ({channels} channels)")
                return None

            gray_code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
            bgr_code = cv2.COLOR_BGRA2BGR if channels == 4 else None

            _log(f"  GPU: paired NVDEC decode, {channels}ch "
                 f"{'uint16' if is_16bit else 'uint8'}, "
                 f"scoring: {config.scoring_method}, scene detection: {scene_aware}")

            # Pre-create GPU filters
            if config.scoring_method == "tenengrad":
                sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)
            else:
                lap_filter = cv2.cuda.createLaplacianFilter(cv2.CV_32FC1, cv2.CV_32FC1, ksize=1)

            # -- Helper functions --

            def _prepare_gray(gpu_frm):
                w, h = gpu_frm.size()
                if config.scale_width and w > config.scale_width:
                    gpu_small = cv2.cuda.resize(
                        gpu_frm, (config.scale_width, int(h * config.scale_width / w)))
                else:
                    gpu_small = gpu_frm
                gpu_gray = cv2.cuda.cvtColor(gpu_small, gray_code)
                if is_16bit:
                    gpu_8u = cv2.cuda.GpuMat(gpu_gray.size(), cv2.CV_8UC1)
                    gpu_gray.convertTo(cv2.CV_8UC1, gpu_8u, alpha=1.0 / 256.0)
                else:
                    gpu_8u = gpu_gray
                return gpu_8u

            def _score_tenengrad(gpu_8u):
                gx = sobel_x.apply(gpu_8u)
                gy = sobel_y.apply(gpu_8u)
                gx2 = cv2.cuda.multiply(gx, gx)
                gy2 = cv2.cuda.multiply(gy, gy)
                energy = cv2.cuda.add(gx2, gy2)
                s = cv2.cuda.sum(energy)
                n = energy.size()[0] * energy.size()[1]
                return s[0] / n

            def _score_laplacian(gpu_8u):
                gpu_32f = cv2.cuda.GpuMat(gpu_8u.size(), cv2.CV_32FC1)
                gpu_8u.convertTo(cv2.CV_32FC1, gpu_32f)
                dst = lap_filter.apply(gpu_32f)
                sq = cv2.cuda.multiply(dst, dst)
                sum_sq = cv2.cuda.sum(sq)
                sum_val = cv2.cuda.sum(dst)
                n = dst.size()[0] * dst.size()[1]
                mean = sum_val[0] / n
                return sum_sq[0] / n - mean * mean

            score_fn = _score_tenengrad if config.scoring_method == "tenengrad" else _score_laplacian

            def _prepare_scene_bgr(gpu_frm):
                w, h = gpu_frm.size()
                scene_w = min(480, w)
                scene_h = int(h * scene_w / w)
                gpu_small = cv2.cuda.resize(gpu_frm, (scene_w, scene_h))
                if channels == 4:
                    gpu_bgr = cv2.cuda.cvtColor(gpu_small, cv2.COLOR_BGRA2BGR)
                else:
                    gpu_bgr = gpu_small
                if is_16bit:
                    gpu_bgr_8u = cv2.cuda.GpuMat(gpu_bgr.size(), cv2.CV_8UC3)
                    gpu_bgr.convertTo(cv2.CV_8UC3, gpu_bgr_8u, alpha=1.0 / 256.0)
                    return gpu_bgr_8u.download()
                return gpu_bgr.download()

            def _save_winner_gpu(gpu_frm, filepath):
                frame = gpu_frm.download()
                if is_16bit:
                    frame = (frame >> 8).astype(np.uint8)
                if bgr_code is not None:
                    frame = cv2.cvtColor(frame, bgr_code)
                ext = config.output_format
                if ext in ("jpg", "jpeg"):
                    cv2.imwrite(str(filepath), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, int(config.quality)])
                else:
                    cv2.imwrite(str(filepath), frame)

            # -- Output tracking --

            ext = config.output_format
            winners_written = 0
            front_paths: List[str] = []
            back_paths: List[str] = []
            selected_times: List[float] = []
            source_front_frames: List[int] = []
            source_back_frames: List[int] = []
            selected_front_scores: List[float] = []
            selected_back_scores: List[float] = []

            def _write_pair_winner(best_f_gpu, best_b_gpu, frame_num, f_score, b_score):
                nonlocal winners_written
                winners_written += 1
                idx_str = f"{winners_written:06d}"
                f_path = front_out / f"{idx_str}.{ext}"
                b_path = back_out / f"{idx_str}.{ext}"
                _save_winner_gpu(best_f_gpu, f_path)
                _save_winner_gpu(best_b_gpu, b_path)
                front_paths.append(str(f_path))
                back_paths.append(str(b_path))
                gpu_written_paths.extend([str(f_path), str(b_path)])
                selected_times.append(round(frame_num / shared_fps, 3))
                source_front_frames.append(frame_num)
                source_back_frames.append(frame_num)
                selected_front_scores.append(f_score)
                selected_back_scores.append(b_score)

            # -- Streaming single-pass loop --

            t_start = time.perf_counter()
            total_scored = 0
            scene_count = 0

            best_key = None
            best_front_gpu = None
            best_back_gpu = None
            best_frame_num = -1
            best_f_score = 0.0
            best_b_score = 0.0
            prev_front_scene = None
            prev_back_scene = None
            current_window_idx: Optional[int] = None

            def _flush_best():
                if best_front_gpu is not None:
                    _write_pair_winner(best_front_gpu, best_back_gpu,
                                       best_frame_num, best_f_score, best_b_score)

            def _process_pair(f_gpu, b_gpu, frame_num, relative_idx):
                nonlocal best_key, best_front_gpu, best_back_gpu, best_frame_num
                nonlocal best_f_score, best_b_score, scene_count, total_scored
                nonlocal prev_front_scene, prev_back_scene
                nonlocal current_window_idx

                f_score = score_fn(_prepare_gray(f_gpu))
                b_score = score_fn(_prepare_gray(b_gpu))
                total_scored += 1
                window_idx = self._time_window_index(
                    frame_num, shared_fps, range_start_sec, config.interval_sec)

                is_scene = False
                if scene_aware:
                    f_scene_bgr = _prepare_scene_bgr(f_gpu)
                    b_scene_bgr = _prepare_scene_bgr(b_gpu)
                    if prev_front_scene is not None:
                        f_sc = _detect_scene_change(
                            prev_front_scene, f_scene_bgr, config.scene_threshold)
                        b_sc = _detect_scene_change(
                            prev_back_scene, b_scene_bgr, config.scene_threshold)
                        is_scene = f_sc or b_sc
                    prev_front_scene = f_scene_bgr
                    prev_back_scene = b_scene_bgr

                if current_window_idx is None:
                    current_window_idx = window_idx
                elif window_idx != current_window_idx:
                    _flush_best()
                    best_key = None
                    best_front_gpu = None
                    best_back_gpu = None
                    best_frame_num = -1
                    current_window_idx = window_idx

                if is_scene:
                    scene_count += 1
                    _flush_best()
                    best_key = None
                    best_front_gpu = None
                    best_back_gpu = None
                    best_frame_num = -1

                pair_key = self._pair_sharpness_sort_key({
                    "front_score": f_score, "back_score": b_score,
                })
                if best_key is None or pair_key > best_key:
                    best_key = pair_key
                    best_front_gpu = f_gpu.clone()
                    best_back_gpu = b_gpu.clone()
                    best_frame_num = frame_num
                    best_f_score = f_score
                    best_b_score = b_score

                if total_frames > 0 and relative_idx % max(1, total_frames // 100) == 0:
                    pct = int(relative_idx / total_frames * 80)
                    if progress_callback:
                        progress_callback(pct, 100,
                            f"Analyzing pair (GPU): {int(relative_idx / total_frames * 100)}% "
                            f"({relative_idx}/{total_frames})")

            # Check cancel before first-frame processing
            if cancel_check and cancel_check():
                raise RuntimeError("Cancelled")

            # Process first pair (already read for format detection)
            _process_pair(first_front, first_back, start_frame, 0)

            # Continue with remaining frames
            for frame_idx in range(start_frame + 1, end_frame):
                if cancel_check and cancel_check():
                    for p in gpu_written_paths:
                        Path(p).unlink(missing_ok=True)
                    raise RuntimeError("Cancelled")

                ret_f, f_gpu = front_reader.nextFrame()
                ret_b, b_gpu = back_reader.nextFrame()
                if not ret_f or not ret_b:
                    break

                relative_idx = frame_idx - start_frame
                _process_pair(f_gpu, b_gpu, frame_idx, relative_idx)

            # Final flush (inside try — cv2.error here still falls back)
            _flush_best()

        except RuntimeError:
            raise  # re-raise cancel — do NOT catch as GPU error

        except cv2.error as e:
            _log(f"  GPU error during paired processing: {e}")
            for p in gpu_written_paths:
                Path(p).unlink(missing_ok=True)
            return None

        elapsed = time.perf_counter() - t_start
        _log(f"  GPU paired {config.scoring_method.title()} scoring: "
             f"{total_scored} frame pairs ({elapsed:.1f}s)")
        if scene_aware:
            _log(f"  Scene changes detected: {scene_count}")
        _log(f"  Winners: {winners_written} pairs extracted")

        if winners_written == 0:
            return PairedExtractorResult(
                success=False,
                output_dir=str(out_root),
                error="No pairs selected (GPU path)",
            )

        # Build manifest
        scoring_method = config.scoring_method
        selection_method = (
            f"{scoring_method}_scene_aware_pair"
            if scene_aware else f"{scoring_method}_pair"
        )
        manifest = {
            "schema_version": 1,
            "dataset_type": "paired_split_frames",
            "front_video": str(Path(front_video).resolve()),
            "back_video": str(Path(back_video).resolve()),
            "mode": "sharpest",
            "scoring_method": scoring_method,
            "scene_detection": scene_aware,
            "interval_sec": config.interval_sec,
            "fps": shared_fps,
            "selection_method": selection_method,
            "gpu_accelerated": True,
            "pairs": [],
        }
        for index, (fp, bp, t, sf, sb, fs, bs) in enumerate(
            zip(front_paths, back_paths, selected_times,
                source_front_frames, source_back_frames,
                selected_front_scores, selected_back_scores),
            start=1,
        ):
            manifest["pairs"].append({
                "pair_index": index,
                "frame_id": f"{index:06d}",
                "front_image": Path(fp).relative_to(out_root).as_posix(),
                "back_image": Path(bp).relative_to(out_root).as_posix(),
                "time_sec": t,
                "source_front_frame": sf,
                "source_back_frame": sb,
                "score_kind": f"{scoring_method}_sharpness",
                "front_score": fs,
                "back_score": bs,
                "pair_score": min(fs, bs),
            })

        manifest_path = out_root / "paired_extraction_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

        return PairedExtractorResult(
            success=True,
            pair_count=winners_written,
            output_dir=str(out_root),
            front_paths=front_paths,
            back_paths=back_paths,
            selected_times=selected_times,
            source_front_frames=source_front_frames,
            source_back_frames=source_back_frames,
            gpu_accelerated=True,
        )

    def _extract_fixed_ffmpeg(
        self,
        front_video: str,
        back_video: str,
        out_root: Path,
        front_out: Path,
        back_out: Path,
        start_frame: int,
        end_frame: int,
        window_size: int,
        shared_fps: float,
        config: PairedExtractorConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        _log: Optional[Callable[[str], None]] = None,
    ) -> PairedExtractorResult:
        """Fixed-interval extraction via FFmpeg per-frame seeks.

        For each selected frame, seeks directly to the target timestamp using
        ``ffmpeg -ss <t> -i <video> -frames:v 1 <output>``. This avoids
        decoding every frame in the video (NVDEC sequential) and avoids the
        CPU HEVC 10-bit hang (cv2.VideoCapture).
        """
        import time

        if _log is None:
            _log = lambda _msg: None

        from .sharpest_extractor import _resolve_ffmpeg_binaries, _SUBPROCESS_FLAGS
        ffmpeg, _ = _resolve_ffmpeg_binaries()

        selected_indices = self._fixed_frame_indices(start_frame, end_frame, window_size)
        total_selected = len(selected_indices)

        _log(f"  FFmpeg seek: {total_selected} pairs from "
             f"{end_frame - start_frame} frames ({shared_fps:.1f} fps)")

        ext = config.output_format
        # JPEG quality → FFmpeg qscale (1=best, 31=worst)
        qscale_args = []
        if ext in ("jpg", "jpeg"):
            qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
            qscale_args = ["-qscale:v", str(qscale)]

        written_paths: List[str] = []
        front_paths: List[str] = []
        back_paths: List[str] = []
        selected_times: List[float] = []
        source_front_frames: List[int] = []
        source_back_frames: List[int] = []
        winners_written = 0
        skipped = 0

        t_start = time.perf_counter()

        for i, frame_idx in enumerate(selected_indices):
            if cancel_check and cancel_check():
                for p in written_paths:
                    Path(p).unlink(missing_ok=True)
                raise RuntimeError("Cancelled")

            ts = frame_idx / shared_fps
            pair_num = winners_written + 1
            idx_str = f"{pair_num:06d}"
            f_path = front_out / f"{idx_str}.{ext}"
            b_path = back_out / f"{idx_str}.{ext}"

            # Extract front frame
            cmd_front = [
                ffmpeg, "-hide_banner", "-y",
                "-ss", f"{ts:.3f}",
                "-i", front_video,
                "-frames:v", "1",
                *qscale_args,
                str(f_path),
            ]
            subprocess.run(cmd_front, capture_output=True, **_SUBPROCESS_FLAGS)

            # Extract back frame at same timestamp
            cmd_back = [
                ffmpeg, "-hide_banner", "-y",
                "-ss", f"{ts:.3f}",
                "-i", back_video,
                "-frames:v", "1",
                *qscale_args,
                str(b_path),
            ]
            subprocess.run(cmd_back, capture_output=True, **_SUBPROCESS_FLAGS)

            # Verify both files were written
            if not f_path.exists() or not b_path.exists():
                _log(f"  FFmpeg: skipped pair at frame {frame_idx} "
                     f"(t={ts:.3f}s) — output missing")
                # Clean up partial
                f_path.unlink(missing_ok=True)
                b_path.unlink(missing_ok=True)
                skipped += 1
                continue

            winners_written += 1
            front_paths.append(str(f_path))
            back_paths.append(str(b_path))
            written_paths.extend([str(f_path), str(b_path)])
            selected_times.append(round(ts, 3))
            source_front_frames.append(frame_idx)
            source_back_frames.append(frame_idx)

            if progress_callback:
                pct = int((i + 1) / total_selected * 80)
                progress_callback(
                    pct, 100,
                    f"Extracting: {winners_written}/{total_selected} pairs")

        elapsed = time.perf_counter() - t_start
        _log(f"  FFmpeg fixed-interval: {winners_written} pairs extracted ({elapsed:.1f}s)"
             + (f", {skipped} skipped" if skipped else ""))

        if winners_written == 0:
            return PairedExtractorResult(
                success=False, output_dir=str(out_root),
                error="No pairs extracted (FFmpeg fixed path)",
            )

        # Manifest
        manifest = {
            "schema_version": 1,
            "dataset_type": "paired_split_frames",
            "front_video": str(Path(front_video).resolve()),
            "back_video": str(Path(back_video).resolve()),
            "mode": "fixed",
            "interval_sec": config.interval_sec,
            "fps": shared_fps,
            "selection_method": "fixed_interval_pair_ffmpeg",
            "pairs": [],
        }
        for index, (fp, bp, t, sf, sb) in enumerate(
            zip(front_paths, back_paths, selected_times,
                source_front_frames, source_back_frames),
            start=1,
        ):
            manifest["pairs"].append({
                "pair_index": index,
                "frame_id": f"{index:06d}",
                "front_image": Path(fp).relative_to(out_root).as_posix(),
                "back_image": Path(bp).relative_to(out_root).as_posix(),
                "time_sec": t,
                "source_front_frame": sf,
                "source_back_frame": sb,
            })

        manifest_path = out_root / "paired_extraction_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

        return PairedExtractorResult(
            success=True,
            pair_count=winners_written,
            output_dir=str(out_root),
            front_paths=front_paths,
            back_paths=back_paths,
            selected_times=selected_times,
            source_front_frames=source_front_frames,
            source_back_frames=source_back_frames,
            gpu_accelerated=False,
        )

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
            range_start_sec = max(0.0, float(config.start_sec or 0.0))

            # Release CPU captures before GPU/FFmpeg path
            front_cap.release()
            back_cap.release()
            front_cap = back_cap = None

            # Fixed mode: FFmpeg per-frame seeks (fast sparse access, no HEVC hang)
            if mode == "fixed":
                return self._extract_fixed_ffmpeg(
                    front_video, back_video, out_root, front_out, back_out,
                    start_frame, end_frame, window_size, shared_fps, config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    _log=_log,
                )

            # Sharpest mode: try GPU (NVDEC sequential decode for scoring every frame)
            if mode == "sharpest" and self._gpu_available():
                gpu_result = self._extract_sharpest_gpu(
                    front_video, back_video, out_root, front_out, back_out,
                    start_frame, end_frame, range_start_sec, shared_fps, config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    _log=_log,
                )
                if gpu_result is not None:
                    return gpu_result
                _log("  Falling back to CPU paired extraction")

            # Re-open CPU captures for CPU path
            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)

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
            error=str(exc),
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
