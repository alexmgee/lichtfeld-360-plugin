# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Sharpest Frame Extractor Module

Extracts the sharpest frame from each time-interval chunk of a video
using a two-pass pipeline: FFmpeg for scene detection, OpenCV for
sharpness scoring (Tenengrad or Laplacian).

Scene-aware chunking: FFmpeg's ``select='gte(scene,0)'`` filter
populates ``lavfi.scene_score``.  When a score exceeds
``scene_threshold``, the interval chunk is split so both sides of the
transition get a representative sharp frame.

Algorithm (Basic/Best modes):
  1. FFmpeg scene detection pass -> per-frame scene scores
  2. OpenCV scoring pass -> per-frame sharpness scores
  3. Divide frames into interval chunks; split at scene boundaries
  4. Pick highest-scoring frame per (sub-)chunk
  5. Extract only those frames with ffmpeg seeks
"""

import bisect
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ffmpeg / ffprobe binary discovery
# ---------------------------------------------------------------------------
try:
    from static_ffmpeg import run as _sfr

    _FFMPEG, _FFPROBE = _sfr.get_or_fetch_platform_executables_else_raise()
except ImportError:
    import shutil

    _FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
    _FFPROBE = shutil.which("ffprobe") or "ffprobe"

# Hide the console window that subprocess.Popen creates on Windows.
_SUBPROCESS_FLAGS: Dict[str, Any] = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)

MANIFEST_FILENAME = "extraction_manifest.json"


# -- scene score merging ----------------------------------------------------

def _merge_scene_scores(
    scored_frames: List[Tuple[float, float]],
    scene_data: List[Tuple[int, float]],
    analysis_fps: float,
    start_sec: float,
) -> List[Tuple[float, float, float]]:
    """Merge FFmpeg scene scores into scored frame data.

    Uses bisect for O(n log m) instead of O(n*m).

    Args:
        scored_frames: (timestamp_sec, sharpness_score) from OpenCV pass.
        scene_data: (frame_number, scene_score) from FFmpeg pass.
        analysis_fps: FPS used in FFmpeg scene detection pass.
        start_sec: Video start offset.

    Returns:
        List of (timestamp_sec, sharpness_score, scene_score) tuples.
    """
    if not scene_data:
        return [(ts, score, 0.0) for ts, score in scored_frames]

    scene_times = [start_sec + (frame_num / analysis_fps) for frame_num, _ in scene_data]
    scene_scores = [score for _, score in scene_data]
    tolerance = (1.0 / analysis_fps) * 1.5

    merged = []
    for ts, sharpness in scored_frames:
        idx = bisect.bisect_left(scene_times, ts)
        best_scene = 0.0
        best_dist = float("inf")

        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(scene_times):
                dist = abs(ts - scene_times[candidate_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_scene = scene_scores[candidate_idx]

        if best_dist > tolerance:
            best_scene = 0.0
        merged.append((ts, sharpness, best_scene))

    return merged


# -- data classes -----------------------------------------------------------

@dataclass
class SharpestConfig:
    """Configuration for sharpest-frame extraction.

    Extraction sharpness levels (``extraction_sharpness``):
      - ``"none"``:  extract at fixed intervals, no analysis
      - ``"basic"``: ~10 candidates per interval, OpenCV scoring
      - ``"best"``:  score every frame in the video
    """
    interval: float = 2.0            # seconds between selections
    extraction_sharpness: str = "best"  # none, basic, best
    blur_metric: str = "tenengrad"   # tenengrad, laplacian
    scene_threshold: float = 0.3     # scene-change score to split chunks
    scale_width: int = 640           # resolution for blur analysis
    quality: int = 95                # JPEG quality (1-100)
    output_format: str = "jpg"       # jpg or png
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclass
class SharpestResult:
    """Result of sharpest-frame extraction."""
    success: bool
    total_frames_analyzed: int = 0
    frames_extracted: int = 0
    output_dir: str = ""
    frame_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None


# -- main class -------------------------------------------------------------

class SharpestExtractor:
    """Extract the sharpest frame per interval from a video."""

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ):
        self.ffmpeg_path = ffmpeg_path or _FFMPEG
        self.ffprobe_path = ffprobe_path or _FFPROBE

    # -- public API ---------------------------------------------------------

    def extract(
        self,
        video_path: str,
        output_dir: str,
        config: Optional[SharpestConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
    ) -> SharpestResult:
        """Full pipeline: scene detect -> score -> select -> extract.

        Args:
            video_path:  Path to input video file.
            output_dir:  Directory for extracted frames.
            config:      Extraction settings (defaults to SharpestConfig()).
            progress_callback: Called with ``(current, total, message)``.
            cancel_check: Called to check if the operation should be cancelled.
            prefix_source: Prefix output filenames with the video stem.

        Returns:
            SharpestResult with extraction details.
        """
        if config is None:
            config = SharpestConfig()

        video = Path(video_path)
        out = Path(output_dir)

        if not video.exists():
            return SharpestResult(success=False, error=f"Video not found: {video}")

        out.mkdir(parents=True, exist_ok=True)

        eq = config.extraction_sharpness

        # None: extract at fixed intervals, no analysis
        if eq == "none":
            return self._extract_interval_only(
                str(video), str(out), config,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                prefix_source=prefix_source,
            )

        # Basic / Best: two-pass pipeline (scene detect + OpenCV scoring)
        return self._extract_scored(
            str(video), str(out), config,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            prefix_source=prefix_source,
        )

    # -- interval-only extraction -------------------------------------------

    def _extract_interval_only(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
    ) -> SharpestResult:
        """Extract frames at fixed intervals directly from video (no analysis)."""
        import cv2

        out = Path(output_dir)
        stem = Path(video_path).stem + "_" if prefix_source else ""
        start = config.start_sec or 0.0
        end = config.end_sec

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return SharpestResult(success=False, error=f"Cannot open {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total_frames / fps if fps > 0 else 0
            if end is None:
                end = duration

            timestamps = []
            t = start
            while t < end:
                timestamps.append(t)
                t += config.interval

            frame_paths: List[str] = []
            for i, ts in enumerate(timestamps):
                if cancel_check and cancel_check():
                    return SharpestResult(success=False, error="Cancelled")

                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                out_name = f"{stem}{i + 1:05d}.jpg"
                cv2.imwrite(str(out / out_name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                frame_paths.append(str(out / out_name))

                if progress_callback:
                    pct = (i + 1) / len(timestamps) * 100
                    progress_callback(int(pct), 100,
                                      f"Extracting {i + 1}/{len(timestamps)}")

            self._write_manifest(
                out,
                frame_paths,
                Path(video_path),
                config,
                timestamps,
                fps,
                start,
                extraction_mode="interval",
            )

            return SharpestResult(
                success=True,
                total_frames_analyzed=0,
                frames_extracted=len(frame_paths),
                output_dir=str(out),
                frame_paths=frame_paths,
            )
        finally:
            cap.release()

    # -- scored extraction (Basic / Best) -----------------------------------

    def _score_frame(self, frame: "np.ndarray", config: SharpestConfig) -> float:
        """Score a BGR frame for sharpness using the configured metric."""
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        if config.scale_width > 0 and w > config.scale_width:
            scale = config.scale_width / w
            frame = cv2.resize(frame, (config.scale_width, int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if config.blur_metric == "laplacian":
            return self._score_laplacian(gray)
        return self._score_tenengrad(gray)

    def _score_all_frames(
        self,
        cap: "cv2.VideoCapture",
        config: SharpestConfig,
        start_sec: float,
        end_sec: float,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[Tuple[float, float]]:
        """Score every frame sequentially. Returns list of (timestamp_sec, score)."""
        import cv2

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        scored: List[Tuple[float, float]] = []
        total_expected = int((end_sec - start_sec) * fps)
        frame_idx = 0

        while True:
            if cancel_check and cancel_check():
                return scored

            # Capture timestamp BEFORE read — read() advances the position
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts > end_sec:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            score = self._score_frame(frame, config)
            scored.append((ts, score))
            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, max(total_expected, 1),
                                  f"Scoring frame {frame_idx}")

        return scored

    def _score_candidate_frames(
        self,
        cap: "cv2.VideoCapture",
        config: SharpestConfig,
        start_sec: float,
        end_sec: float,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[Tuple[float, float]]:
        """Score ~5fps evenly-spaced candidates per interval. Returns (timestamp, score)."""
        import cv2

        # Sample at ~5fps, giving ~10 candidates for the default 2s interval.
        sample_fps = 5.0
        scored: List[Tuple[float, float]] = []

        # Build all candidate timestamps
        all_timestamps: List[float] = []
        t = start_sec
        while t < end_sec:
            window_end = min(t + config.interval, end_sec)
            n_candidates = max(1, round((window_end - t) * sample_fps))
            step = (window_end - t) / n_candidates
            for j in range(n_candidates):
                ts = t + step * j
                if ts < end_sec:
                    all_timestamps.append(ts)
            t += config.interval

        for i, ts in enumerate(all_timestamps):
            if cancel_check and cancel_check():
                return scored

            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            score = self._score_frame(frame, config)
            scored.append((ts, score))

            if progress_callback and i % 10 == 0:
                progress_callback(i, len(all_timestamps),
                                  f"Scoring candidate {i + 1}/{len(all_timestamps)}")

        return scored

    def _extract_scored(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
    ) -> SharpestResult:
        """Two-pass extraction: FFmpeg scene detect + OpenCV sharpness scoring."""
        import cv2

        out = Path(output_dir)
        stem = Path(video_path).stem + "_" if prefix_source else ""
        start = config.start_sec or 0.0

        def _progress(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        # -- Probe video properties --
        native_fps = self._probe_fps(video_path)
        if native_fps <= 0:
            return SharpestResult(success=False, error="Could not determine video FPS")
        duration_sec = self._probe_duration(video_path)

        is_best = config.extraction_sharpness == "best"

        # Analysis FPS: Basic samples ~5fps, Best uses native
        if is_best:
            analysis_fps = native_fps
        else:
            analysis_fps = min(5.0 / max(config.interval, 0.1), native_fps)

        chunk_size = max(1, round(analysis_fps * config.interval))

        # -- Pass 1: FFmpeg scene detection (0-40%) --
        _progress(0, 100, "Detecting scene changes...")
        metadata_path = None
        try:
            metadata_path = Path(tempfile.mktemp(suffix="_scene.txt",
                                                  dir=output_dir))
            ok, err = self._run_scene_detect(
                video_path, metadata_path, config,
                analysis_fps=analysis_fps if not is_best else None,
                progress_callback=lambda c, t, m: _progress(
                    int(c * 40 / max(t, 1)), 100, m),
                duration_sec=duration_sec,
                cancel_check=cancel_check,
            )
            if not ok:
                return SharpestResult(success=False,
                                      error=f"Scene detection failed: {err}")

            scene_data = self._parse_scene_metadata(metadata_path)
        finally:
            if metadata_path and metadata_path.exists():
                metadata_path.unlink(missing_ok=True)

        if cancel_check and cancel_check():
            return SharpestResult(success=False, error="Cancelled")

        # -- Pass 2: OpenCV scoring (40-85%) --
        _progress(40, 100, "Scoring frames for sharpness...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return SharpestResult(success=False,
                                  error=f"Cannot open {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total_frames / fps if fps > 0 else 0
            end = config.end_sec or duration

            if is_best:
                scored_frames = self._score_all_frames(
                    cap, config, start, end,
                    progress_callback=lambda c, t, m: _progress(
                        40 + int(c * 45 / max(t, 1)), 100, m),
                    cancel_check=cancel_check,
                )
            else:
                scored_frames = self._score_candidate_frames(
                    cap, config, start, end,
                    progress_callback=lambda c, t, m: _progress(
                        40 + int(c * 45 / max(t, 1)), 100, m),
                    cancel_check=cancel_check,
                )
        finally:
            cap.release()

        if not scored_frames:
            return SharpestResult(success=False, error="No frames scored")

        if cancel_check and cancel_check():
            return SharpestResult(success=False, error="Cancelled")

        # -- Select best per chunk (scene-aware) --
        _progress(85, 100, "Selecting sharpest frames...")

        # analysis_fps is always a valid float here
        scored_with_scenes = _merge_scene_scores(
            scored_frames, scene_data, analysis_fps, start)

        winner_timestamps = self._parse_best_frames(
            scored_with_scenes, chunk_size, config.scene_threshold)

        if not winner_timestamps:
            return SharpestResult(
                success=False,
                total_frames_analyzed=len(scored_frames),
                error="No frames selected",
            )

        # -- Extract winners at full quality (85-100%) --
        _progress(85, 100, f"Extracting {len(winner_timestamps)} sharpest frames...")
        frame_paths = self._extract_at_timestamps(
            video_path, winner_timestamps, output_dir, config,
            stem=stem,
            progress_callback=lambda c, t, m: _progress(
                85 + int(c * 15 / max(t, 1)), 100, m),
            cancel_check=cancel_check,
        )

        self._write_manifest(out, frame_paths, Path(video_path), config,
                             winner_timestamps, analysis_fps, start)

        return SharpestResult(
            success=True,
            total_frames_analyzed=len(scored_frames),
            frames_extracted=len(frame_paths),
            output_dir=output_dir,
            frame_paths=frame_paths,
        )

    # -- timestamp-based extraction -----------------------------------------

    def _extract_at_timestamps(
        self,
        video_path: str,
        timestamps: List[float],
        output_dir: str,
        config: SharpestConfig,
        stem: str = "",
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        """Extract frames at specific timestamps using ffmpeg seeks."""
        out = Path(output_dir)
        ext = config.output_format
        frame_paths: List[str] = []

        for i, ts in enumerate(timestamps):
            if cancel_check and cancel_check():
                return frame_paths

            out_name = f"{stem}{i + 1:05d}.{ext}"
            out_path = out / out_name

            cmd = [
                self.ffmpeg_path, "-hide_banner", "-y",
                "-ss", f"{ts:.3f}",
                "-i", video_path,
                "-frames:v", "1",
            ]
            if ext in ("jpg", "jpeg"):
                qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
                cmd.extend(["-qscale:v", str(qscale)])
            cmd.append(str(out_path))

            subprocess.run(cmd, capture_output=True, **_SUBPROCESS_FLAGS)

            if out_path.exists():
                frame_paths.append(str(out_path))

            if progress_callback:
                pct = 85 + (i + 1) / max(len(timestamps), 1) * 15
                progress_callback(int(pct), 100, f"Extracting {i + 1}/{len(timestamps)}")

        return frame_paths

    # -- internals ----------------------------------------------------------

    def _probe_fps(self, video_path: str) -> float:
        """Get video FPS via ffprobe."""
        cmd = [
            self.ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return 0.0
        try:
            data = json.loads(result.stdout)
            rate = data["streams"][0]["r_frame_rate"]
            num, den = rate.split("/")
            return float(num) / float(den)
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            return 0.0

    def _probe_duration(self, video_path: str) -> float:
        """Get video duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return 0.0
        try:
            data = json.loads(result.stdout)
            # Try stream duration first, fall back to format duration
            dur = data.get("streams", [{}])[0].get("duration")
            if not dur:
                dur = data.get("format", {}).get("duration")
            return float(dur) if dur else 0.0
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.0

    def _run_scene_detect(
        self,
        video_path: str,
        metadata_path: Path,
        config: SharpestConfig,
        analysis_fps: Optional[float] = None,
        progress_callback: Optional[Callable] = None,
        duration_sec: float = 0.0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """Run FFmpeg scene detection only (no blurdetect), write per-frame metadata.

        Args:
            analysis_fps: If set, subsample with fps filter before analysis.
                Pass None for Best mode (native fps, no fps filter).
        """
        filters = []
        if analysis_fps is not None:
            filters.append(f"fps={analysis_fps:.4f}")
        filters.append(f"scale={config.scale_width}:-1")
        filters.append(f"select='gte(scene\\,0)'")
        filters.append(f"metadata=print:file={metadata_path.name}")
        vf = ",".join(filters)

        cmd = [
            self.ffmpeg_path, "-hide_banner", "-y",
            "-progress", "pipe:1", "-nostats",
        ]
        if config.start_sec is not None:
            cmd.extend(["-ss", str(config.start_sec)])
        cmd.extend(["-i", video_path])
        if config.end_sec is not None:
            cmd.extend(["-to", str(config.end_sec)])
        cmd.extend(["-vf", vf, "-an", "-f", "null", "-"])

        eff_duration = duration_sec
        if config.start_sec or config.end_sec:
            start = config.start_sec or 0.0
            end = config.end_sec or duration_sec
            eff_duration = max(0.0, end - start)

        def _fmt_time(secs):
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        total_display = _fmt_time(eff_duration) if eff_duration > 0 else ""

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(metadata_path.parent), **_SUBPROCESS_FLAGS,
        )

        last_pct = -1
        current_time_us = 0
        current_speed = ""
        for line in proc.stdout:
            if cancel_check and cancel_check():
                proc.terminate()
                proc.wait()
                return False, "cancelled"
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    current_time_us = int(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("speed="):
                current_speed = line.split("=", 1)[1].strip()
            elif line.startswith("progress="):
                if not progress_callback:
                    continue
                elapsed_sec = current_time_us / 1_000_000
                elapsed_display = _fmt_time(elapsed_sec)

                if eff_duration > 0:
                    raw_pct = min(elapsed_sec / eff_duration, 1.0)
                    pct = int(raw_pct * 100)
                    if pct == last_pct:
                        continue
                    last_pct = pct
                    speed_str = f" [{current_speed}]" if current_speed and current_speed != "N/A" else ""
                    msg = f"Scene detect: {elapsed_display} / {total_display} ({pct}%){speed_str}"
                    progress_callback(pct, 100, msg)
                else:
                    msg = f"Scene detect: {elapsed_display}"
                    progress_callback(0, 0, msg)

        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            return False, err[-500:] if err else "unknown error"
        return True, ""

    @staticmethod
    def _parse_scene_metadata(metadata_path: Path) -> List[Tuple[int, float]]:
        """Parse scene-only metadata file into (frame_number, scene_score) tuples."""
        pat_frame = re.compile(r"frame:(\d+)")
        pat_scene = re.compile(r"lavfi\.scene_score=([0-9.]+)")

        frame_data: List[Tuple[int, float]] = []
        current_frame = -1
        current_scene = 0.0

        try:
            lines = metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            return []

        for line in lines:
            line = line.strip()
            m = pat_frame.search(line)
            if m:
                if current_frame >= 0:
                    frame_data.append((current_frame, current_scene))
                current_frame = int(m.group(1))
                current_scene = 0.0
                continue
            m = pat_scene.search(line)
            if m and current_frame >= 0:
                try:
                    current_scene = float(m.group(1))
                except ValueError:
                    pass

        if current_frame >= 0:
            frame_data.append((current_frame, current_scene))

        return frame_data

    @staticmethod
    def _parse_best_frames(
        frame_data: List[Tuple[int, float, float]],
        chunk_size: int,
        scene_threshold: float = 0.3,
    ) -> List[int]:
        """Select the sharpest frame per chunk with scene-aware splitting.

        Args:
            frame_data: List of ``(frame_number, blur_score, scene_score)`` tuples.
            chunk_size: Number of frames per interval chunk.
            scene_threshold: Scene-change score threshold to split chunks.

        Returns:
            List of frame numbers (one per chunk/sub-chunk).

        Algorithm:
            1. Divide *frame_data* into interval-based chunks of *chunk_size*.
            2. If a frame inside a chunk has ``scene_score >= threshold``,
               split the chunk at that boundary.
            3. Pick the lowest-blur frame from each (sub-)chunk.
        """
        if not frame_data:
            return []

        best: List[int] = []
        for i in range(0, len(frame_data), chunk_size):
            chunk = frame_data[i: i + chunk_size]
            sub_chunks = SharpestExtractor._split_at_scenes(chunk, scene_threshold)
            for sc in sub_chunks:
                winner = max(sc, key=lambda x: x[1])
                best.append(winner[0])

        return best

    @staticmethod
    def _split_at_scenes(
        chunk: List[Tuple[int, float, float]],
        threshold: float,
    ) -> List[List[Tuple[int, float, float]]]:
        """Split a chunk into sub-chunks at scene boundaries."""
        if not chunk:
            return []
        # threshold <= 0 means no scene splitting
        if threshold <= 0:
            return [chunk]
        sub_chunks: List[List[Tuple[int, float, float]]] = []
        current: List[Tuple[int, float, float]] = []
        for entry in chunk:
            # Scene change -> flush current sub-chunk, start new one
            if entry[2] >= threshold and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    @staticmethod
    def _score_tenengrad(gray: "np.ndarray") -> float:
        """Sobel gradient energy — higher = sharper. Noise-robust via implicit Gaussian."""
        import cv2
        import numpy as np
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(sx**2 + sy**2))

    @staticmethod
    def _score_laplacian(gray: "np.ndarray") -> float:
        """Laplacian variance — higher = sharper. Gaussian pre-blur for noise reduction."""
        import cv2
        blurred = cv2.GaussianBlur(gray, (0, 0), 0.7)
        return float(cv2.Laplacian(blurred, cv2.CV_64F).var())

    def _extract_frames(
        self,
        video_path: str,
        frame_numbers: List[int],
        output_dir: str,
        config: SharpestConfig,
        stem: str = "",
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        """Extract only the selected frames."""
        out = Path(output_dir)
        ext = config.output_format
        pattern = str(out / f"{stem}%05d.{ext}")
        total_frames = len(frame_numbers)

        select_expr = "+".join(f"eq(n\\,{f})" for f in frame_numbers)
        vf = f"select='{select_expr}'"

        # On Windows the select expression can exceed the command-line
        # length limit (~8 KB for cmd.exe / 32 KB for CreateProcess).
        # Fall back to a temporary filter_script file when that happens.
        filter_script_path = None
        use_filter_script = len(vf) > 7000

        if use_filter_script:
            filter_script_path = Path(tempfile.mktemp(
                suffix="_select.txt", dir=str(out)))
            filter_script_path.write_text(
                f"select='{select_expr}'", encoding="utf-8")

        cmd = [
            self.ffmpeg_path, "-hide_banner", "-y",
            "-progress", "pipe:1", "-nostats",
            "-i", video_path,
        ]
        if use_filter_script:
            cmd.extend(["-filter_script:v", str(filter_script_path)])
        else:
            cmd.extend(["-vf", vf])
        cmd.append("-vsync")
        cmd.append("0")

        if ext in ("jpg", "jpeg"):
            qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
            cmd.extend(["-qscale:v", str(qscale)])
        elif ext == "png":
            cmd.extend(["-compression_level", "6"])

        cmd.append(pattern)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, **_SUBPROCESS_FLAGS)

            last_pct = -1
            current_frame = 0
            for line in proc.stdout:
                if cancel_check and cancel_check():
                    proc.terminate()
                    proc.wait()
                    return []
                line = line.strip()
                if line.startswith("frame="):
                    try:
                        current_frame = int(line.split("=", 1)[1])
                    except ValueError:
                        pass
                elif line.startswith("progress=") and progress_callback:
                    # Extraction occupies 85-100% of overall progress
                    if total_frames > 0:
                        raw_pct = min(current_frame / total_frames, 1.0)
                        pct = 85 + int(raw_pct * 15)
                        if pct != last_pct:
                            last_pct = pct
                            progress_callback(
                                pct, 100,
                                f"Extracting: {current_frame}/{total_frames} frames")

            proc.wait()
            if proc.returncode != 0:
                return []
        finally:
            if filter_script_path and filter_script_path.exists():
                filter_script_path.unlink()

        # Collect extracted files
        return sorted(str(p) for p in out.glob(f"{stem}*.{ext}"))

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        frame_paths: List[str],
        video_path: Path,
        config: SharpestConfig,
        timestamps: List[float],
        fps: float,
        start_sec: float,
        extraction_mode: str = "sharpest",
    ):
        """Write extraction manifest mapping frames to source video timestamps."""
        manifest = {
            "video": str(video_path.absolute()),
            "video_stem": video_path.stem,
            "extraction_mode": extraction_mode,
            "extraction_tier": config.extraction_sharpness,
            "blur_metric": config.blur_metric,
            "interval": config.interval,
            "start_sec": start_sec,
            "end_sec": config.end_sec,
            "fps": fps,
            "frames": [],
        }

        for i, path in enumerate(frame_paths):
            filename = Path(path).name
            time_sec = round(timestamps[i], 3) if i < len(timestamps) else 0.0
            manifest["frames"].append({
                "filename": filename,
                "index": i + 1,
                "time_sec": time_sec,
            })

        manifest_path = output_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
