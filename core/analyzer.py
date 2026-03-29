# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Video Analyzer Module

Analyze video files to extract metadata and recommend extraction parameters.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val: object) -> float:
    """Parse a float from ffprobe output, returning 0.0 for None/'N/A'/garbage."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: object) -> int:
    """Parse an int from ffprobe output, returning 0 for None/'N/A'/garbage."""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Known log formats and their recommended LUTs
# ---------------------------------------------------------------------------
LOG_FORMATS: Dict[str, Dict[str, Any]] = {
    "dlog": {
        "name": "D-Log M",
        "cameras": ["DJI"],
        "lut": "DJI_DLog_M_to_Rec709.cube",
    },
    "ilog": {
        "name": "I-Log",
        "cameras": ["Insta360"],
        "lut": "Insta360_ILog_to_Rec709.cube",
    },
    "protune": {
        "name": "Protune Flat",
        "cameras": ["GoPro"],
        "lut": "GoPro_Protune_to_Rec709.cube",
    },
    "vlog": {
        "name": "V-Log",
        "cameras": ["Panasonic"],
        "lut": "VLog_to_Rec709.cube",
    },
}


# ---------------------------------------------------------------------------
# VideoInfo
# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    """Video metadata and analysis results."""

    path: str
    filename: str
    format: str
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    bitrate: Optional[int] = None
    pixel_format: Optional[str] = None

    # Derived properties
    is_erp: bool = False
    is_log_format: bool = False
    detected_log_type: Optional[str] = None

    # Recommendations
    recommended_interval: float = 2.0
    recommended_lut: Optional[str] = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_equirectangular(width: int, height: int) -> bool:
        """Return *True* if *width* x *height* is approximately 2:1 (ERP)."""
        if height == 0:
            return False
        return abs(width / height - 2.0) < 0.05

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "format": self.format,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "bitrate": self.bitrate,
            "pixel_format": self.pixel_format,
            "is_equirectangular": self.is_erp,
            "is_log_format": self.is_log_format,
            "detected_log_type": self.detected_log_type,
            "recommended_interval": self.recommended_interval,
            "recommended_lut": self.recommended_lut,
        }


# ---------------------------------------------------------------------------
# VideoAnalyzer
# ---------------------------------------------------------------------------
class VideoAnalyzer:
    """Analyze video files for 360-degree processing."""

    def __init__(self, ffprobe_path: Optional[str] = None):
        self.ffprobe_path: str = ffprobe_path or _FFPROBE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, video_path: str) -> VideoInfo:
        """Analyze a video file and return metadata with recommendations.

        Args:
            video_path: Path to video file.

        Returns:
            VideoInfo with metadata and recommendations.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If ffprobe fails or returns unparseable output.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        raw_data = self._run_ffprobe(video_path)
        info = self._parse_metadata(raw_data, path)

        self._detect_360(info)
        self._detect_log_format(info, path)
        self._generate_recommendations(info)

        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON output."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration,nb_frames,codec_name,pix_fmt,bit_rate",
            "-show_entries",
            "format=duration,size,bit_rate,format_name",
            "-of", "json",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, **_SUBPROCESS_FLAGS
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffprobe failed: {exc.stderr}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse ffprobe output: {exc}") from exc

    def _parse_metadata(self, data: Dict[str, Any], path: Path) -> VideoInfo:
        """Parse ffprobe JSON output into a *VideoInfo*."""
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        # Frame rate — may be "30/1" or "29.97"
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        # Duration — .mov/.mkv often lack stream-level duration
        duration = (
            _safe_float(stream.get("duration"))
            or _safe_float(fmt.get("duration"))
            or 0.0
        )

        # Frame count — .mov/.mkv often lack nb_frames
        frame_count = _safe_int(stream.get("nb_frames"))
        if not frame_count:
            frame_count = int(duration * fps) if duration else 0

        bitrate = _safe_int(stream.get("bit_rate")) or _safe_int(fmt.get("bit_rate")) or None

        return VideoInfo(
            path=str(path.absolute()),
            filename=path.name,
            format=fmt.get("format_name", path.suffix[1:]),
            codec=stream.get("codec_name", "unknown"),
            width=int(stream.get("width", 0)),
            height=int(stream.get("height", 0)),
            fps=fps,
            duration_seconds=duration,
            frame_count=frame_count,
            bitrate=bitrate,
            pixel_format=stream.get("pix_fmt"),
        )

    @staticmethod
    def _detect_360(info: VideoInfo) -> None:
        """Detect if video is equirectangular 360-degree."""
        info.is_erp = VideoInfo.is_equirectangular(info.width, info.height)

    @staticmethod
    def _detect_log_format(info: VideoInfo, path: Path) -> None:
        """Detect if video uses a log colour profile (heuristic: filename/ext)."""
        filename_lower = path.name.lower()

        log_hints = {
            "dlog": ["dlog", "d-log", "dji"],
            "ilog": ["ilog", "i-log", "insta360", "insv"],
            "protune": ["protune", "gopro", "flat"],
            "vlog": ["vlog", "v-log"],
        }
        for log_type, hints in log_hints.items():
            if any(hint in filename_lower for hint in hints):
                info.is_log_format = True
                info.detected_log_type = log_type
                info.recommended_lut = LOG_FORMATS[log_type]["lut"]
                return

        ext = path.suffix.lower()
        if ext == ".insv":
            info.is_log_format = True
            info.detected_log_type = "ilog"
            info.recommended_lut = LOG_FORMATS["ilog"]["lut"]
        elif ext == ".360":
            info.is_log_format = True
            info.detected_log_type = "protune"
            info.recommended_lut = LOG_FORMATS["protune"]["lut"]

    @staticmethod
    def _generate_recommendations(info: VideoInfo) -> None:
        """Set recommended extraction interval based on resolution."""
        if info.is_erp:
            if info.width >= 7680:  # 8K
                info.recommended_interval = 2.0
            elif info.width >= 5760:  # 6K
                info.recommended_interval = 1.5
            else:  # 4K and below
                info.recommended_interval = 1.0
        else:
            info.recommended_interval = 0.5

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_duration_formatted(info: VideoInfo) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        total_seconds = int(info.duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def estimate_frame_count(info: VideoInfo, interval: float) -> int:
        """Estimate number of frames that will be extracted at *interval*."""
        if interval <= 0:
            return 0
        return int(info.duration_seconds / interval)
