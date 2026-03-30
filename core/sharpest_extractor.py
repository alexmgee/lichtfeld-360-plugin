# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Sharpest Frame Extractor Module

Extracts the sharpest frame from each time-interval chunk of a video
using FFmpeg's blurdetect filter.  Instead of extracting all frames and
then discarding blurry ones, this analyses blur on every frame *first*
and only extracts the winners.

Scene-aware chunking: a ``select='gte(scene,0)'`` filter in the same
pass populates ``lavfi.scene_score`` for free.  When a score exceeds
``scene_threshold``, the interval chunk is split so both sides of the
transition get a representative sharp frame.

Algorithm (Normal/Maximum modes):
  1. Run ffmpeg blurdetect + scene scoring -> per-frame metadata
  2. Divide frames into interval chunks; split at scene boundaries
  3. Pick lowest-blur frame per (sub-)chunk
  4. Extract only those frames with ffmpeg select filter
"""

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


# -- data classes -----------------------------------------------------------

@dataclass
class SharpestConfig:
    """Configuration for sharpest-frame extraction.

    Extraction sharpness levels (``extraction_sharpness``):
      - ``"none"``:    extract at fixed intervals, no analysis
      - ``"fast"``:    extract 3× candidates, Laplacian score, keep sharpest
      - ``"normal"``:  blurdetect on subsampled candidates (~5× FPS)
      - ``"maximum"``: blurdetect on every frame in the video
    """
    interval: float = 2.0            # seconds between selections
    extraction_sharpness: str = "normal"  # none, fast, normal, maximum
    scene_threshold: float = 0.3     # scene-change score to split chunks
    scale_width: int = 640           # resolution for blur analysis
    block_size: int = 32             # blurdetect block dimensions
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
        """Full pipeline: probe fps -> blurdetect -> parse -> extract.

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

        def _progress(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        eq = config.extraction_sharpness

        # None: extract at fixed intervals, no analysis
        if eq == "none":
            return self._extract_interval_only(
                str(video), str(out), config,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                prefix_source=prefix_source,
            )

        # Fast: extract 3× candidates, Laplacian score, keep sharpest per interval
        if eq == "fast":
            return self._extract_fast_blur(
                str(video), str(out), config,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                prefix_source=prefix_source,
            )

        # Normal / Maximum: blurdetect-based analysis
        # Normal subsamples to ~5× target FPS; Maximum analyzes every frame
        _progress(0, 100, "Probing video...")
        native_fps = self._probe_fps(str(video))
        if native_fps <= 0:
            return SharpestResult(success=False, error="Could not determine video FPS")
        duration_sec = self._probe_duration(str(video))

        # Normal mode: subsample to ~5 candidates per interval
        # Maximum mode: analyze at native fps (all frames)
        if eq == "normal":
            analysis_fps = min(5.0 / max(config.interval, 0.1), native_fps)
        else:
            analysis_fps = native_fps

        chunk_size = max(1, round(analysis_fps * config.interval))

        _progress(0, 100, f"Analyzing ({int(analysis_fps)} fps, {chunk_size}/chunk)...")
        metadata_path = None
        try:
            metadata_path = Path(tempfile.mktemp(suffix="_blurdetect.txt",
                                                  dir=str(out)))
            ok, err = self._run_blurdetect(
                str(video), metadata_path, config,
                analysis_fps=analysis_fps if eq == "normal" else None,
                progress_callback=progress_callback,
                duration_sec=duration_sec,
                cancel_check=cancel_check,
            )
            if not ok:
                return SharpestResult(success=False, error=f"blurdetect failed: {err}")

            # Parse and select best frames
            _progress(85, 100, "Selecting sharpest frames...")
            frame_data = self._parse_metadata_file(metadata_path)
            best_frames = self._parse_best_frames(
                frame_data, chunk_size, config.scene_threshold,
            )
            total_analyzed = len(frame_data)

            if not best_frames:
                return SharpestResult(
                    success=False,
                    total_frames_analyzed=total_analyzed,
                    error="No frames selected (blurdetect returned no data)",
                )

            if cancel_check and cancel_check():
                return SharpestResult(success=False, error="Cancelled")

            # Extract winners — use timestamps for subsampled mode
            stem = video.stem + "_" if prefix_source else ""
            _progress(85, 100, f"Extracting {len(best_frames)} sharpest frames...")

            if eq == "normal":
                # Frame numbers are in the subsampled stream; convert to timestamps
                start_offset = config.start_sec or 0.0
                timestamps = [start_offset + (f / analysis_fps) for f in best_frames]
                frame_paths = self._extract_at_timestamps(
                    str(video), timestamps, str(out), config,
                    stem=stem,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            else:
                frame_paths = self._extract_frames(
                    str(video), best_frames, str(out), config,
                    stem=stem,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )

            start = config.start_sec or 0.0
            self._write_manifest(out, frame_paths, video, config, best_frames, analysis_fps, start)

            return SharpestResult(
                success=True,
                total_frames_analyzed=total_analyzed,
                frames_extracted=len(frame_paths),
                output_dir=str(out),
                frame_paths=frame_paths,
            )

        finally:
            if metadata_path and metadata_path.exists():
                metadata_path.unlink(missing_ok=True)

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

            return SharpestResult(
                success=True,
                total_frames_analyzed=0,
                frames_extracted=len(frame_paths),
                output_dir=str(out),
                frame_paths=frame_paths,
            )
        finally:
            cap.release()

    # -- fast blur extraction -----------------------------------------------

    def _extract_fast_blur(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
    ) -> SharpestResult:
        """Read 3× candidates per interval from video, score with Laplacian,
        save only the sharpest.  No intermediate files — all in memory."""
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

            # Build interval windows: each has 3 candidate timestamps
            candidates_per_interval = 3
            intervals: List[List[float]] = []
            t = start
            while t < end:
                window_end = min(t + config.interval, end)
                step = (window_end - t) / candidates_per_interval
                candidates = [t + step * j for j in range(candidates_per_interval)]
                intervals.append(candidates)
                t += config.interval

            frame_paths: List[str] = []
            total_scored = 0

            for i, candidate_times in enumerate(intervals):
                if cancel_check and cancel_check():
                    return SharpestResult(success=False, error="Cancelled")

                # Read and score each candidate in memory
                best_frame = None
                best_score = -1.0

                for ts in candidate_times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    total_scored += 1

                    if score > best_score:
                        best_score = score
                        best_frame = frame

                # Save winner
                if best_frame is not None:
                    out_name = f"{stem}{len(frame_paths) + 1:05d}.jpg"
                    cv2.imwrite(str(out / out_name), best_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                    frame_paths.append(str(out / out_name))

                if progress_callback:
                    pct = (i + 1) / len(intervals) * 100
                    progress_callback(int(pct), 100,
                                      f"Scoring {i + 1}/{len(intervals)} intervals")

            return SharpestResult(
                success=True,
                total_frames_analyzed=total_scored,
                frames_extracted=len(frame_paths),
                output_dir=str(out),
                frame_paths=frame_paths,
            )
        finally:
            cap.release()

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

    def _run_blurdetect(
        self,
        video_path: str,
        metadata_path: Path,
        config: SharpestConfig,
        analysis_fps: Optional[float] = None,
        progress_callback: Optional[Callable] = None,
        duration_sec: float = 0.0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """Run ffmpeg blurdetect + scene scoring, write per-frame metadata.

        Args:
            analysis_fps: If set, subsample with fps filter before analysis.
                This dramatically reduces frames analyzed for Normal mode.
        """
        # Build filter chain
        filters = []
        if analysis_fps is not None:
            filters.append(f"fps={analysis_fps:.4f}")
        filters.append(f"scale={config.scale_width}:-1")
        filters.append(f"select='gte(scene\\,0)'")
        filters.append(
            f"blurdetect=block_width={config.block_size}"
            f":block_height={config.block_size}"
        )
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

        # Effective duration for percentage calculation
        eff_duration = duration_sec
        if config.start_sec or config.end_sec:
            start = config.start_sec or 0.0
            end = config.end_sec or duration_sec
            eff_duration = max(0.0, end - start)

        # Format duration as MM:SS for display
        def _fmt_time(secs):
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        total_display = _fmt_time(eff_duration) if eff_duration > 0 else ""

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(metadata_path.parent), **_SUBPROCESS_FLAGS,
        )

        # Parse ffmpeg -progress output for time/speed updates
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
                # Each progress block ends with "progress=continue" or "progress=end"
                # -- emit a callback once per block with accumulated values
                if not progress_callback:
                    continue
                elapsed_sec = current_time_us / 1_000_000
                elapsed_display = _fmt_time(elapsed_sec)

                if eff_duration > 0:
                    # Blurdetect occupies 0-85% of overall progress
                    raw_pct = min(elapsed_sec / eff_duration, 1.0)
                    pct = int(raw_pct * 85)
                    if pct == last_pct:
                        continue
                    last_pct = pct
                    speed_str = f" [{current_speed}]" if current_speed and current_speed != "N/A" else ""
                    msg = f"Analyzing: {elapsed_display} / {total_display} ({int(raw_pct * 100)}%){speed_str}"
                    progress_callback(pct, 100, msg)
                else:
                    msg = f"Analyzing: {elapsed_display}"
                    progress_callback(0, 0, msg)

        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            return False, err[-500:] if err else "unknown error"
        return True, ""

    @staticmethod
    def _parse_metadata_file(metadata_path: Path) -> List[Tuple[int, float, float]]:
        """Parse blurdetect metadata file into a list of (frame, blur, scene_score) tuples."""
        pat_frame = re.compile(r"frame:(\d+)")
        pat_blur = re.compile(r"lavfi\.blur=([0-9.]+)")
        pat_scene = re.compile(r"lavfi\.scene_score=([0-9.]+)")

        frame_data: List[Tuple[int, float, float]] = []
        current_frame = -1
        current_blur = -1.0
        current_scene = 0.0

        try:
            lines = metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            return []

        for line in lines:
            line = line.strip()
            m = pat_frame.search(line)
            if m:
                # Emit previous frame if complete
                if current_frame >= 0 and current_blur >= 0:
                    frame_data.append((current_frame, current_blur, current_scene))
                current_frame = int(m.group(1))
                current_blur = -1.0
                current_scene = 0.0
                continue
            m = pat_blur.search(line)
            if m and current_frame >= 0:
                try:
                    current_blur = float(m.group(1))
                except ValueError:
                    pass
                continue
            m = pat_scene.search(line)
            if m and current_frame >= 0:
                try:
                    current_scene = float(m.group(1))
                except ValueError:
                    pass

        # Don't forget the last frame
        if current_frame >= 0 and current_blur >= 0:
            frame_data.append((current_frame, current_blur, current_scene))

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
                winner = min(sc, key=lambda x: x[1])
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
        frame_numbers: List[int],
        fps: float,
        start_sec: float,
    ):
        """Write extraction manifest mapping frames to source video timestamps.

        Uses exact frame numbers and FPS for precise timestamp computation.
        """
        manifest = {
            "video": str(video_path.absolute()),
            "video_stem": video_path.stem,
            "extraction_mode": "sharpest",
            "interval": config.interval,
            "start_sec": start_sec,
            "end_sec": config.end_sec,
            "fps": fps,
            "frames": [],
        }

        for i, path in enumerate(frame_paths):
            filename = Path(path).name
            # frame_numbers[i] is the source video frame index
            frame_num = frame_numbers[i] if i < len(frame_numbers) else 0
            time_sec = round(start_sec + frame_num / fps, 3)
            manifest["frames"].append({
                "filename": filename,
                "index": i + 1,
                "source_frame": frame_num,
                "time_sec": time_sec,
            })

        manifest_path = output_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
