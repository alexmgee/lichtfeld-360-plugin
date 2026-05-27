# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Dual Fisheye Container Handler — DJI .osv and Insta360 .insv

Ported from reconstruction-zone (`prep360/core/osv.py`) with minimal
adjustments to use the plugin's ffmpeg/ffprobe discovery.

Container layouts (confirmed via ffprobe on real captures):

DJI Osmo 360 (.osv):
    Stream 0: HEVC 3840x3840 50fps 10-bit  — fisheye lens 0 (back)
    Stream 1: HEVC 3840x3840 50fps 10-bit  — fisheye lens 1 (front)
    Stream 2: AAC stereo 48kHz             — audio
    Stream 3-6: djmd / dbgi telemetry
    Stream 7: MJPEG thumbnail (attached_pic)
    Container: MOV/MP4, encoder tag "Osmo 360".

Insta360 X4/X5 (.insv, single-file dual-track):
    Stream 0: HEVC 3840x3840              — fisheye lens 0 (back)
    Stream 1: HEVC 3840x3840              — fisheye lens 1 (front)
    Stream 2: AAC                          — audio
    Container: MOV/MP4, handler "INS.HVC". No encoder tag.

Older Insta360 (ONE X / X2 / X3) ships dual fisheye as a *file pair*
(VID_..._00_NNN.insv = front, VID_..._10_NNN.insv = rear). That format
is handled by `paired_extractor.detect_insv_pair`, not by this module.

Both supported formats: stream 0 = back, stream 1 = front.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Hide the console window that subprocess.Popen creates on Windows.
_SUBPROCESS_FLAGS: Dict[str, Any] = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)

# Lazy ffmpeg/ffprobe discovery — reuse the plugin's existing helper so
# OSVHandler picks up the same binaries SharpestExtractor uses.
_FFMPEG: Optional[str] = None
_FFPROBE: Optional[str] = None


def _resolve_ffmpeg_binaries() -> Tuple[str, str]:
    """Resolve ffmpeg/ffprobe binaries, preferring static_ffmpeg if present."""
    global _FFMPEG, _FFPROBE
    if _FFMPEG and _FFPROBE:
        return _FFMPEG, _FFPROBE

    try:
        from static_ffmpeg import run as _sfr

        _FFMPEG, _FFPROBE = _sfr.get_or_fetch_platform_executables_else_raise()
    except ImportError:
        import shutil

        _FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
        _FFPROBE = shutil.which("ffprobe") or "ffprobe"

    return _FFMPEG, _FFPROBE


@dataclass
class OSVStreamInfo:
    """Metadata for a single stream within a dual fisheye container."""
    index: int
    codec_type: str         # "video", "audio", "data"
    codec_name: str         # "hevc", "aac", "mjpeg", or codec_tag_string for data
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bit_depth: int = 0
    bitrate: int = 0
    handler_name: str = ""
    is_default: bool = False
    is_thumbnail: bool = False


@dataclass
class OSVInfo:
    """Parsed metadata from a dual fisheye container."""
    path: str
    filename: str
    duration: float
    total_size: int
    encoder: str
    creation_time: str

    front_stream: int
    back_stream: int
    width: int
    height: int
    fps: float
    codec: str
    bit_depth: int
    frame_count: int

    has_metadata: bool = False
    has_gyro: bool = False
    has_audio: bool = False
    has_thumbnail: bool = False

    streams: List[OSVStreamInfo] = field(default_factory=list)

    @property
    def total_bitrate_mbps(self) -> float:
        return (self.total_size * 8) / (self.duration * 1_000_000) if self.duration else 0

    def summary(self) -> str:
        lines = [
            f"File: {self.filename}",
            f"Encoder: {self.encoder}",
            f"Duration: {self.duration:.1f}s ({self._format_duration()})",
            f"Fisheye: {self.width}x{self.height} {self.fps:.0f}fps {self.bit_depth}-bit {self.codec}",
            f"Streams: front={self.front_stream}, back={self.back_stream}",
            f"Frames: {self.frame_count} per stream",
            f"Size: {self.total_size / (1024**3):.2f} GB ({self.total_bitrate_mbps:.0f} Mbps)",
        ]
        extras = []
        if self.has_metadata:
            extras.append("metadata")
        if self.has_gyro:
            extras.append("gyro")
        if self.has_audio:
            extras.append("audio")
        if extras:
            lines.append(f"Data: {', '.join(extras)}")
        return "\n".join(lines)

    def _format_duration(self) -> str:
        total = int(self.duration)
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class OSVHandler:
    """Probe and demux DJI .osv / Insta360 .insv (single-file) containers."""

    def __init__(
        self,
        ffprobe_path: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        resolved_ffmpeg, resolved_ffprobe = _resolve_ffmpeg_binaries()
        self.ffmpeg_path = ffmpeg_path or resolved_ffmpeg
        self.ffprobe_path = ffprobe_path or resolved_ffprobe

    def probe(self, osv_path: str) -> OSVInfo:
        """Probe a dual fisheye container and return its stream layout."""
        path = Path(osv_path)
        if not path.exists():
            raise FileNotFoundError(f"Container file not found: {osv_path}")

        raw = self._run_ffprobe(osv_path)
        return self._parse_osv(raw, path)

    def demux_streams(
        self,
        osv_path: str,
        output_dir: str,
        streams: str = "both",
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract raw fisheye video streams via ffmpeg stream copy.

        Args:
            osv_path: Path to the container file.
            output_dir: Directory for extracted MP4 files.
            streams: "front", "back", or "both".

        Returns:
            (front_path, back_path) — paths to extracted MP4 files.
            Either may be None if not requested.
        """
        info = self.probe(osv_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = Path(osv_path).stem
        front_path = None
        back_path = None

        if streams in ("front", "both"):
            front_path = str(out / f"{stem}_front.mp4")
            self._demux_one_stream(osv_path, info.front_stream, front_path)

        if streams in ("back", "both"):
            back_path = str(out / f"{stem}_back.mp4")
            self._demux_one_stream(osv_path, info.back_stream, back_path)

        return front_path, back_path

    # --- Internal methods ---

    def _run_ffprobe(self, path: str) -> Dict[str, Any]:
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                **_SUBPROCESS_FLAGS,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}") from e

    def _parse_osv(self, raw: Dict[str, Any], path: Path) -> OSVInfo:
        fmt = raw.get("format", {})
        raw_streams = raw.get("streams", [])

        # Validate it's a dual-fisheye container (.osv or .insv).
        encoder = fmt.get("tags", {}).get("encoder", "")
        suffix = path.suffix.lower()
        known_suffix = suffix in (".osv", ".insv")
        known_encoder = "osmo" in encoder.lower() or "360" in encoder.lower()
        if not known_suffix and not known_encoder:
            format_name = fmt.get("format_name", "")
            raise ValueError(
                "not a recognized dual-fisheye container. "
                "Fisheye (Pinhole) mode requires the original .insv or .osv file "
                "from your camera, not a re-exported or re-encoded .mp4. "
                "If your camera saved this as .mp4, try renaming the file to .insv "
                "and loading it again."
            )

        # Parse all streams.
        parsed_streams: List[OSVStreamInfo] = []
        hevc_streams: List[OSVStreamInfo] = []

        for s in raw_streams:
            codec_type = s.get("codec_type", "unknown")
            codec_name = s.get("codec_name", s.get("codec_tag_string", "unknown"))
            index = s.get("index", 0)

            tags = s.get("tags", {})
            disp = s.get("disposition", {})

            si = OSVStreamInfo(
                index=index,
                codec_type=codec_type,
                codec_name=codec_name,
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                handler_name=tags.get("handler_name", ""),
                is_default=bool(disp.get("default", 0)),
                is_thumbnail=bool(disp.get("attached_pic", 0)),
            )

            if codec_type == "video":
                fps_str = s.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    si.fps = float(num) / float(den) if float(den) else 0
                else:
                    si.fps = float(fps_str)

            pix_fmt = s.get("pix_fmt", "")
            if "10" in pix_fmt:
                si.bit_depth = 10
            elif "12" in pix_fmt:
                si.bit_depth = 12
            elif pix_fmt:
                si.bit_depth = 8

            br = s.get("bit_rate")
            if br:
                si.bitrate = int(br)

            parsed_streams.append(si)

            if codec_type == "video" and codec_name == "hevc":
                hevc_streams.append(si)

        if len(hevc_streams) < 2:
            raise ValueError(
                f"Expected 2 HEVC video streams in dual fisheye container, "
                f"found {len(hevc_streams)}. (Older Insta360 ONE X/X2/X3 "
                f"captures ship as a file pair instead — see "
                f"paired_extractor.detect_insv_pair.)"
            )

        # Stream 0 = back (lower index, often disposition.default=1),
        # stream 1 = front. Verified empirically against DuckbillStudio
        # convention and ffprobe output on real captures.
        back_stream = hevc_streams[0]
        front_stream = hevc_streams[1]

        has_metadata = any(
            s.codec_name == "djmd" or "CAM meta" in s.handler_name
            for s in parsed_streams
        )
        has_gyro = any(
            s.codec_name == "dbgi" or "dbgi" in s.handler_name
            for s in parsed_streams
        )
        has_audio = any(s.codec_type == "audio" for s in parsed_streams)
        has_thumbnail = any(s.is_thumbnail for s in parsed_streams)

        duration = float(fmt.get("duration", 0))
        size = int(fmt.get("size", 0))
        creation_time = fmt.get("tags", {}).get("creation_time", "")

        return OSVInfo(
            path=str(path.absolute()),
            filename=path.name,
            duration=duration,
            total_size=size,
            encoder=encoder,
            creation_time=creation_time,
            front_stream=front_stream.index,
            back_stream=back_stream.index,
            width=front_stream.width,
            height=front_stream.height,
            fps=front_stream.fps,
            codec=front_stream.codec_name,
            bit_depth=front_stream.bit_depth,
            frame_count=int(duration * front_stream.fps) if duration else 0,
            has_metadata=has_metadata,
            has_gyro=has_gyro,
            has_audio=has_audio,
            has_thumbnail=has_thumbnail,
            streams=parsed_streams,
        )

    def _demux_one_stream(self, osv_path: str, stream_idx: int, output_path: str) -> None:
        """Demux a single video stream using ffmpeg stream copy."""
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", osv_path,
            "-map", f"0:{stream_idx}",
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to demux stream {stream_idx}: {result.stderr}"
            )
