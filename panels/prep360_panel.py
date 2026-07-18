# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Plugin panel — full UI with data model, threading, and import."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import lichtfeld as lf

try:
    from lfs_plugins import ScrubFieldController, ScrubFieldSpec
except ImportError:
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec

from ..core.analyzer import VideoAnalyzer, VideoInfo
from ..core.colmap_runner import MATCH_BUDGETS
from ..core.mask_diagnostics import (
    format_mask_diagnostics_overview,
    get_mask_diagnostics_summary,
    load_mask_diagnostics_document,
)
from ..core.setup_checks import (
    MaskingSetupState, Sam3SetupReport, check_masking_setup, check_sam3_setup,
    forget_hf_token, install_default_tier, install_premium_tier, install_video_tracking,
    make_sam3_install_failure_report, verify_hf_token_detailed,
)
from ..core.output_mode import OUTPUT_MODES, output_mode_index
from ..core.presets import DEFAULT_PRESET, VIEW_PRESETS, resolve_view_preset_name

if TYPE_CHECKING:
    from ..core.pipeline import PipelineConfig, PipelineJob, PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Win32 native file dialog (Windows only)
# ---------------------------------------------------------------------------
# LichtFeld's lf.ui.open_video_file_dialog() takes no parameters, so its
# accepted extensions are baked in (no .osv / .insv). Tkinter is not
# available in the LFS-bundled Python (vcpkg build doesn't include
# _tkinter). Solution: drive comdlg32.GetOpenFileNameW directly via ctypes.
# ctypes is part of stdlib core and always available.
#
# We use c_void_p for the filter and file buffers so embedded NULs (which
# the Win32 filter format requires between label/pattern pairs and at the
# end) survive the boundary — c_wchar_p / LPWSTR truncate at the first NUL.

if os.name == "nt":
    class _OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", ctypes.c_void_p),
            ("lpstrCustomFilter", ctypes.c_void_p),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", ctypes.c_void_p),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", ctypes.c_void_p),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    _OFN_FILEMUSTEXIST = 0x00001000
    _OFN_PATHMUSTEXIST = 0x00000800
    _OFN_HIDEREADONLY = 0x00000004
    _OFN_EXPLORER     = 0x00080000
    _OFN_NOCHANGEDIR  = 0x00000008  # don't change CWD on selection


def _open_file_via_win32(title: str, filters: list[tuple[str, str]]) -> str:
    """Open a Windows native Open-File dialog with arbitrary extension filters.

    `filters` is a list of (display_label, semicolon_pattern) tuples in the
    order they should appear, e.g.
        [("Video files", "*.mp4;*.osv;*.insv"), ("All files", "*.*")]

    Returns the selected absolute path, or "" if the user cancelled.

    Raises:
        OSError if not on Windows (caller should handle / fall back).
        ctypes errors if comdlg32 isn't loadable.
    """
    if os.name != "nt":
        raise OSError("Win32 file dialog requires Windows")

    # Win32 filter format: pairs of NUL-terminated strings, ending in
    # an extra NUL: "Display1\0pattern1\0Display2\0pattern2\0\0"
    filter_text = ""
    for label, pattern in filters:
        filter_text += f"{label}\x00{pattern}\x00"
    filter_text += "\x00"  # final terminator (create_unicode_buffer adds one more)

    filter_buf = ctypes.create_unicode_buffer(filter_text, len(filter_text) + 1)
    file_buf = ctypes.create_unicode_buffer(32768)

    ofn = _OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(ofn)
    ofn.hwndOwner = None
    ofn.lpstrFilter = ctypes.addressof(filter_buf)
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.addressof(file_buf)
    ofn.nMaxFile = 32768
    ofn.lpstrTitle = title
    ofn.Flags = (
        _OFN_FILEMUSTEXIST
        | _OFN_PATHMUSTEXIST
        | _OFN_HIDEREADONLY
        | _OFN_EXPLORER
        | _OFN_NOCHANGEDIR
    )

    comdlg32 = ctypes.windll.comdlg32
    comdlg32.GetOpenFileNameW.argtypes = [ctypes.POINTER(_OPENFILENAMEW)]
    comdlg32.GetOpenFileNameW.restype = wintypes.BOOL

    if not comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        # User cancelled, OR an error occurred. Use CommDlgExtendedError to
        # distinguish — code 0 means cancelled, anything else is an error.
        comdlg32.CommDlgExtendedError.argtypes = []
        comdlg32.CommDlgExtendedError.restype = wintypes.DWORD
        err_code = comdlg32.CommDlgExtendedError()
        if err_code != 0:
            raise OSError(f"GetOpenFileNameW failed with code 0x{err_code:04X}")
        return ""

    return file_buf.value


_PANEL_SPACE_MAIN = getattr(
    lf.ui.PanelSpace,
    "MAIN_PANEL_TAB",
    getattr(lf.ui.PanelSpace, "FLOATING", None),
)
_PANEL_HEIGHT_CONTENT = getattr(
    lf.ui.PanelHeightMode,
    "CONTENT",
    getattr(lf.ui.PanelHeightMode, "FILL", None),
)

# ---------------------------------------------------------------------------
# Preset ordering — indices match the <select> in the RML
# ---------------------------------------------------------------------------

PRESET_NAMES = [
    "cubemap",
    "low",
    "medium",
    "high",
    "ultra",
]

PRESET_LABELS = [
    "Cubemap (6 views)",
    "Low (12 views)",
    "Medium (16 views)",
    "High (20 views)",
    "Ultra (24 views)",
]

COLMAP_MATCHERS = ["sequential", "exhaustive"]

# Bundled faiss-format vocab trees keyed by feature type
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_VOCAB_TREE_BUNDLED = {
    "sift": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words256K.bin",
    "aliked_n16rot": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
    "aliked_n32": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin",
}

# COLMAP 4.1 feature controls
FEATURE_TYPES = ["sift", "aliked_n16rot", "aliked_n32"]
MATCHER_TYPES = ["bruteforce", "lightglue"]
MAPPERS = ["incremental", "global"]
BA_SOLVERS = ["auto", "caspar", "ceres_gpu", "ceres"]
MATCH_BUDGET_TIERS = ["fast", "balanced", "default", "high", "custom"]

OUTPUT_SIZES = [960, 1280, 1536, 1920, 2400, 3072, 3840]

# COLMAP / SIFT preset matrix exposed via the Output Quality panel.
# Each entry maps (output_mode, preset_name) -> (max_features, max_image_size, max_matches).
# Pinhole/ERP "High" bumps DSLR detail meaningfully; ERP High matches Fisheye
# Normal because the extended Crop Size dropdown lets ERP render at native
# fisheye-equivalent resolution. Fisheye baselines are anchored at the
# integration-report-recommended 8192/3840 (this is the value that fixed the
# Insta360 .insv alignment failure in commit 131808d).
SIFT_PRESETS = ["normal", "high", "custom"]
SIFT_PRESET_LABELS = ["Normal", "High", "Custom"]
SIFT_NORMAL_IDX, SIFT_HIGH_IDX, SIFT_CUSTOM_IDX = 0, 1, 2
SIFT_PRESET_MATRIX: dict[tuple[str, str], tuple[int, int, int]] = {
    ("erp_native", "normal"): (8192, 3840, 32768),
    ("erp_native", "high"):   (16384, 5760, 65536),
    ("pinhole", "normal"): (2048, 1664, 32768),
    ("pinhole", "high"):   (4096, 2432, 65536),
    ("fisheye", "normal"):         (8192, 3840, 32768),
    ("fisheye", "high"):           (16384, 3840, 65536),
}

# Default max_num_features per extractor type (from COLMAP docs).
# Used by _set_feature_type_idx to reset max_features when switching extractors,
# and by _set_sift_preset_idx / _resnap_sift_for_mode to cap values for non-SIFT.
_FEATURE_MAX_DEFAULTS = {
    "sift": 8192,
    "aliked_n16rot": 2048,
    "aliked_n32": 2048,
}

# OUTPUT_MODES is imported from core.output_mode (the single source of truth):
# ("erp_native", "pinhole", "fisheye"). Labels stay local because the mapping
# module is deliberately kept free of UI concerns.
OUTPUT_MODE_LABELS = [
    "ERP", "ERP (Pinhole)", "Fisheye",
]
ERP_NATIVE_OUTPUT_MODE_IDX = OUTPUT_MODES.index("erp_native")
PINHOLE_OUTPUT_MODE_IDX = OUTPUT_MODES.index("pinhole")
FISHEYE_OUTPUT_MODE_IDX = OUTPUT_MODES.index("fisheye")
_FISHEYE_MODES = {FISHEYE_OUTPUT_MODE_IDX}

# Processing axis: the collapsed Output Mode dropdown (Native / Pinhole),
# orthogonal to projection ("erp" / "fisheye"). The derived output_mode is
# output_mode_string(projection, processing).
NATIVE_PROCESSING_IDX = 0
PINHOLE_PROCESSING_IDX = 1

FISHEYE_TRAINING_OUTPUTS = ["native", "pinhole", "both"]
FISHEYE_TRAINING_OUTPUT_LABELS = [
    "Native (fisheye)", "Pinhole", "Both",
]

# Image-folder training output (PipelineConfig.training_output). Serves BOTH
# projections (ERP + fisheye), so the labels stay projection-neutral -- unlike
# the video-only fisheye control above, which says "Native (fisheye)".
IMAGE_TRAINING_OUTPUTS = ["native", "pinhole", "both"]
IMAGE_TRAINING_OUTPUT_LABELS = ["Native", "Pinhole", "Both"]

# Source modes for dual fisheye input (fisheye output mode only)
SOURCE_MODES = ["container", "split", "image_folder"]
SOURCE_MODE_LABELS = ["Single file", "Two files (split)"]

# Camera families for split mode (auto-detected for container; manual for split)
CAMERA_FAMILIES = ["dji_osmo360", "insta360"]
CAMERA_FAMILY_LABELS = ["DJI Osmo 360", "Insta360"]

# Scrub field specifications — keys MUST match the data-value attribute in the RML
SCRUB_FIELD_SPECS = {
    "extract_fps_str": ScrubFieldSpec(
        min_value=0.1, max_value=5.0, step=0.1, fmt="%.1f", data_type=float,
    ),
    "jpeg_quality_str": ScrubFieldSpec(
        min_value=50.0, max_value=100.0, step=1.0, fmt="%d", data_type=int,
    ),
    "colmap_max_matches_str": ScrubFieldSpec(
        min_value=1024.0, max_value=80000.0, step=128.0, fmt="%d", data_type=int,
    ),
    "sift_max_features_str": ScrubFieldSpec(
        min_value=1024.0, max_value=32768.0, step=1024.0, fmt="%d", data_type=int,
    ),
    "sift_max_image_size_str": ScrubFieldSpec(
        min_value=1024.0, max_value=8192.0, step=128.0, fmt="%d", data_type=int,
    ),
    "sequential_overlap_str": ScrubFieldSpec(
        min_value=2.0, max_value=20.0, step=1.0, fmt="%d", data_type=int,
    ),
}

# Extraction sharpness tiers
# - None: no analysis, fixed-interval extraction
# - Basic: ~10 candidates per interval, OpenCV scoring
# - Better: score every frame with lighter analysis settings
# - Best: score every frame with the most thorough analysis
EXTRACT_SHARPNESS_PRESETS = [
    {"label": "None",  "scale_width": 0,   "scene_threshold": 0.0},
    {"label": "Basic", "scale_width": 640, "scene_threshold": 0.3},
    {"label": "Better", "scale_width": 512, "scene_threshold": 0.3},
    {"label": "Best",  "scale_width": 640, "scene_threshold": 0.3},
]

# Blur metric options
BLUR_METRICS = [
    {"label": "Tenengrad", "value": "tenengrad"},
    {"label": "Laplacian", "value": "laplacian"},
]

# Human-readable coverage descriptions per preset
COVERAGE_DESCRIPTIONS = {
    "cubemap": "4 horizon, 1 top, 1 bottom",
    "low": "12 Fibonacci-spiral views, poles to poles",
    "medium": "8+8 two-ring layout at ±35°, no poles",
    "high": "20 Fibonacci-spiral views, poles to poles",
    "ultra": "24 Fibonacci-spiral views, poles to poles",
}

# ---------------------------------------------------------------------------
# Section management
# ---------------------------------------------------------------------------

SECTIONS = ["extraction", "masking", "reframe", "quality"]


class Plugin360Panel(lf.ui.Panel):
    id = "plugin360.main"
    label = "360 Plugin"
    space = _PANEL_SPACE_MAIN
    order = 10100
    template = str(Path(__file__).resolve().with_name("prep360_panel.rml"))
    height_mode = _PANEL_HEIGHT_CONTENT
    update_interval_ms = 100

    def __init__(self):
        super().__init__()

        # Data model handle
        self._handle = None
        self._doc = None

        # Section collapse state (all start expanded)
        self._collapsed: set[str] = set()

        # Video state
        self._video_loaded: bool = False
        self._video_path: str = ""
        self._video_info: Optional[VideoInfo] = None
        self._video_info_text: str = ""

        # Extraction
        self._extract_fps: float = 1.0
        self._extract_sharpness_idx: int = 1  # default: Basic
        self._blur_metric_idx: int = 0        # default: Tenengrad
        self._extract_all_frames: bool = False

        # Masking
        self._setup_state: MaskingSetupState = MaskingSetupState()
        self._sam3_setup_report: Sam3SetupReport = Sam3SetupReport(
            token_status="unknown",
            access_status="unknown",
            runtime_status="unknown",
            weights_status="unknown",
            overall_stage="needs_access",
            message="Checking local SAM 3 setup...",
            next_action="",
        )
        self._masking_method_idx: int = 1  # SAM 3 only in the UI
        self._masking_available: bool = False
        self._enable_masking: bool = False
        self._enable_diagnostics: bool = False
        self._mask_prompts_str: str = "person"
        self._hf_token_input: str = ""
        self._hf_verify_text: str = self._sam3_setup_report.message
        self._hf_verify_ok: bool = False
        self._install_busy: bool = False
        self._install_button_text: str = "Repair Default Masking"

        # GPU extraction opt-in (row under the Frame Extraction header).
        # State comes from core.gpu_extraction_install; refreshed in the
        # setup probe thread, mutated by the gpu_* event handlers.
        self._gpu_state: str = "disabled"
        self._gpu_hw_present: bool = False
        self._gpu_probe_done: bool = False
        self._gpu_busy: bool = False
        self._gpu_progress_text: str = ""
        self._gpu_notice_text: str = ""
        self._gpu_error_detail: str = ""
        self._sam3_check_button_text: str = "Check Setup"
        self._sam3_edit_token: bool = False
        self._sam3_notice_text: str = ""
        self._sam3_hf_help_key: str = ""

        # Reframe
        self._preset_idx: int = PRESET_NAMES.index(DEFAULT_PRESET)
        self._output_size_idx: int = 3  # index into OUTPUT_SIZES, default 1920
        self._output_size: int = OUTPUT_SIZES[3]
        self._jpeg_quality: int = 97
        self._colmap_matcher_idx: int = 1  # default: exhaustive
        self._match_budget_idx: int = 2  # default
        self._colmap_max_num_matches: int = MATCH_BUDGETS["default"]

        # SIFT controls (Output Quality section). Default tracks the
        # current Output Mode via SIFT_PRESET_MATRIX; preset Normal at init
        # since output_mode_idx defaults to 0 (Pinhole) below.
        # _sift_last_non_custom_preset_idx is the Normal/High target the Reset
        # button restores to when preset has been auto-flipped to Custom by a
        # slider drag. Preserved across mode switches.
        self._sift_preset_idx: int = SIFT_NORMAL_IDX
        self._sift_last_non_custom_preset_idx: int = SIFT_NORMAL_IDX
        self._sift_advanced_expanded: bool = False
        self._sift_max_features: int = SIFT_PRESET_MATRIX[("pinhole", "normal")][0]
        self._sift_max_image_size: int = SIFT_PRESET_MATRIX[("pinhole", "normal")][1]

        # Output — two orthogonal axes. _output_mode_idx is a derived,
        # read-only property computed from these (see below).
        self._projection: str = "erp"  # "erp" | "fisheye"
        # Projection-first input menu (Plan 03 / Option B): False = Initial
        # state ([Equirectangular]/[Fisheye] buttons); True = that projection's
        # source menu. Gates ONLY the input-menu markup — _projection stays
        # non-null ("provisional ERP" in Initial) so the ~24 read sites and the
        # derived _output_mode_idx keep working unchanged (plan D1).
        self._projection_chosen: bool = False
        self._fisheye_single_lens: bool = False
        self._processing_idx: int = PINHOLE_PROCESSING_IDX  # default: ERP (Pinhole)
        self._output_path: str = ""

        # Dual fisheye state (only relevant when output_mode == "fisheye")
        # Auto-detected camera family from .osv/.insv extension when in container
        # mode; user-selected via dropdown in split mode.
        self._source_mode_idx: int = 0  # 0=container, 1=split, 2=image_folder
        self._camera_family_idx: int = 0  # 0=dji_osmo360, 1=insta360
        self._camera_family_detected: Optional[str] = None  # set by detect_input_type()
        self._front_video_path: str = ""
        self._back_video_path: str = ""
        self._keep_streams: bool = False
        self._keep_extracted_data: bool = True  # keep frames & masks by default
        self._fisheye_training_output_idx: int = 0  # native by default
        self._training_output_idx: int = 0  # image-folder: native by default
        self._fisheye_circle_margin: float = 6.0

        # Image-folder input state (source_mode_idx == 2). Projection reuses
        # the canonical self._projection (AR-2 — no separate projection field).
        self._image_dir: str = ""        # ERP: folder of equirectangular frames
        self._fisheye_dir: str = ""      # one-folder fisheye (front_/back_ files)
        self._front_dir: str = ""        # two-folder fisheye: front lens frames
        self._back_dir: str = ""         # two-folder fisheye: back lens frames
        self._fisheye_folder_mode: str = "one"  # "one" | "two"
        self._mask_source: str = "generate"     # "generate" | "preexisting" | "none"
        self._preexisting_mask_dir: str = ""

        # COLMAP 4.1 controls
        self._feature_type_idx: int = 0   # SIFT
        self._matcher_type_idx: int = 0   # Bruteforce
        self._mapper_idx: int = 0         # Incremental
        self._ba_solver_idx: int = 0      # Auto (hybrid)
        self._loop_closure_enabled: bool = False
        self._sequential_overlap: int = 10
        self._guided_matching_enabled: bool = False
        self._sift_affine_dsp_enabled: bool = False

        # Processing state
        self._is_processing: bool = False
        self._processing_stage: str = ""
        self._processing_status: str = ""
        self._processing_progress: float = 0.0
        self._processing_log_lines: list[str] = []
        self._error_message: str = ""
        self._completion_summary: str = ""
        self._completion_report: str = ""
        self._import_after: bool = False

        # Timing accumulator (tracks stage transitions from progress callbacks)
        self._timing_stages: dict[str, dict] = {}
        self._timing_current_stage: str = ""
        self._timing_t0: float = 0.0
        self._timing_colmap_substage: str = ""
        self._timing_colmap_substage_t0: float = 0.0

        # Threading
        self._job: Optional[PipelineJob] = None
        self._pending_result: Optional[PipelineResult] = None
        self._pending_lock = threading.Lock()
        self._setup_probe_started: bool = False
        self._setup_probe_pending: Optional[
            tuple[MaskingSetupState, Sam3SetupReport, bool]
        ] = None
        self._setup_probe_lock = threading.Lock()

        # Scrub field controller
        self._scrub_fields = ScrubFieldController(
            specs=SCRUB_FIELD_SPECS,
            get_value=self._get_scrub_value,
            set_value=self._set_scrub_value,
        )

        # UI signature for change detection
        self._last_sig: Optional[tuple] = None

    # ── Draw (unused — RmlUI template handles rendering) ──────

    def draw(self, ui):
        del ui

    # ── Data model binding ────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("plugin360")
        if model is None:
            return

        # -- Video state --
        model.bind_func("show_no_video", lambda: not self._video_loaded)
        model.bind_func("show_video_loaded", lambda: self._video_loaded)
        model.bind_func("video_info_text", lambda: self._video_info_text)
        model.bind_func("video_path_text", lambda: self._video_path or "No file selected")

        # -- Extraction (range sliders need two-way bindings for data-value) --
        model.bind("extract_fps_str", lambda: f"{self._extract_fps:.1f}", self._set_extract_fps)
        model.bind("extract_sharpness_idx", lambda: str(self._extract_sharpness_idx), self._set_extract_sharpness)
        model.bind("blur_metric_idx", lambda: str(self._blur_metric_idx), self._set_blur_metric)
        model.bind("extract_all_frames", lambda: self._extract_all_frames, self._set_extract_all_frames)
        model.bind_func("extract_controls_disabled", lambda: self._extract_all_frames)
        model.bind_func("est_frames_text", self._get_est_frames_text)
        model.bind_func("gpu_indicator_text", self._get_gpu_indicator_text)

        # -- GPU extraction opt-in row --
        model.bind_func("show_gpu_row", self._show_gpu_row)
        model.bind_func("gpu_row_text", self._get_gpu_row_text)
        model.bind_func("show_gpu_row_text", lambda: bool(self._get_gpu_row_text()))
        model.bind_func("gpu_row_button_text", self._get_gpu_row_button_text)
        model.bind_func("show_gpu_row_button", self._show_gpu_row_button)
        model.bind_func("gpu_busy", lambda: self._gpu_busy)
        model.bind_func("show_gpu_notice", lambda: bool(self._gpu_notice_text))
        model.bind_func("gpu_notice_text", lambda: self._gpu_notice_text)
        model.bind_func("show_gpu_copy_details", self._show_gpu_copy_details)

        # -- Masking --
        model.bind("enable_masking", lambda: self._enable_masking, self._set_enable_masking)
        model.bind("mask_prompts_str", lambda: self._mask_prompts_str, self._set_mask_prompts)
        model.bind("hf_token_input", lambda: self._hf_token_input, self._set_hf_token_input)
        model.bind_func("hf_verify_text", lambda: self._hf_verify_text)
        model.bind_func("masking_available", lambda: self._masking_available)
        # SAM 3 conditional states
        model.bind_func(
            "show_masking_sam3_setup",
            lambda: not self._setup_state.sam3_ready
            and not (self._source_mode_idx == 2 and self._mask_source != "generate"),
        )
        model.bind_func("show_masking_sam3_ready", lambda: self._setup_state.sam3_ready)
        model.bind_func("sam3_intro_text", self._get_sam3_intro_text)
        model.bind_func("show_sam3_support_text", self._show_sam3_support_text)
        model.bind_func("sam3_support_text", self._get_sam3_support_text)
        model.bind_func("sam3_status_message", self._get_sam3_status_message)
        model.bind_func("show_sam3_status_message", self._show_sam3_status_message)
        model.bind_func("show_sam3_notice", lambda: bool(self._sam3_notice_text))
        model.bind_func("sam3_notice_text", lambda: self._sam3_notice_text)
        model.bind_func("sam3_hf_help_text", self._get_sam3_hf_help_text)
        model.bind_func("sam3_token_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.token_status))
        model.bind_func("sam3_access_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.access_status))
        model.bind_func("sam3_runtime_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.runtime_status))
        model.bind_func("sam3_weights_status_text", self._get_sam3_weights_status_text)
        model.bind_func("show_sam3_cached_weights_note", self._show_sam3_cached_weights_note)
        model.bind_func("sam3_cached_weights_note_text", self._get_sam3_cached_weights_note_text)
        model.bind_func("sam3_check_button_text", lambda: self._sam3_check_button_text)
        model.bind_func("sam3_install_button_text", self._get_sam3_install_button_text)
        model.bind_func("sam3_install_disabled", self._get_sam3_install_disabled)
        model.bind_func("show_sam3_install_button", self._show_sam3_install_button)
        model.bind_func("show_sam3_check_button", self._show_sam3_check_button)
        model.bind_func("show_sam3_hf_heading", self._show_sam3_hf_heading)
        model.bind_func("show_sam3_hf_inline_actions", self._show_sam3_hf_inline_actions)
        model.bind_func("show_sam3_external_actions", self._show_sam3_external_actions)
        model.bind_func("show_sam3_signup_button", self._show_sam3_signup_button)
        model.bind_func("show_sam3_request_access_button", self._show_sam3_request_access_button)
        model.bind_func("show_sam3_tokens_button", self._show_sam3_tokens_button)
        model.bind_func("show_sam3_local_actions", self._show_sam3_local_actions)
        model.bind_func("show_sam3_saved_token_notice", self._show_sam3_saved_token_notice)
        model.bind_func("sam3_saved_token_text", self._get_sam3_saved_token_text)
        model.bind_func("show_sam3_token_editor", self._show_sam3_token_editor)
        model.bind_func("show_sam3_token_actions", self._show_sam3_token_actions)
        model.bind_func("show_sam3_token_actions_row", self._show_sam3_token_actions_row)
        model.bind_func("show_sam3_token_editor_message", self._show_sam3_token_editor_message)
        model.bind_func("show_hf_verify_text", self._show_hf_verify_text)
        model.bind_func("sam3_change_token_button_text", self._get_sam3_change_token_button_text)
        model.bind_func("install_button_text", lambda: self._install_button_text)
        model.bind_func("install_busy", lambda: self._install_busy)

        # -- Reframe --
        model.bind("preset_idx", lambda: str(self._preset_idx), self._set_preset)
        # User-selectable preset dropdown only applies to Pinhole mode.
        # Fisheye mode forces OPENCV_FISHEYE PER_FOLDER (no view rig).
        model.bind_func("show_preset_select", lambda: self._output_mode_idx == PINHOLE_OUTPUT_MODE_IDX)
        model.bind_func("show_fisheye_preset", lambda: self._output_mode_idx == FISHEYE_OUTPUT_MODE_IDX)  # native fisheye only
        model.bind_func("fisheye_preset_text", self._get_fisheye_preset_text)
        model.bind_func("coverage_text", self._get_coverage_text)
        model.bind_func("total_output_text", self._get_total_output_text)
        model.bind("output_size_idx", lambda: str(self._output_size_idx), self._set_output_size_idx)
        model.bind("jpeg_quality_str", lambda: str(self._jpeg_quality), self._set_jpeg_quality)
        model.bind("colmap_matcher_idx", lambda: str(self._colmap_matcher_idx), self._set_colmap_matcher)
        model.bind("match_budget_idx", lambda: str(self._match_budget_idx), self._set_match_budget_idx)
        model.bind("colmap_max_matches_str", lambda: str(self._colmap_max_num_matches), self._set_colmap_max_matches)
        model.bind_func("match_budget_text", self._get_match_budget_text)

        # Loop closure detection
        model.bind("loop_closure_enabled", lambda: self._loop_closure_enabled, self._set_loop_closure)
        model.bind_func("show_loop_closure_tree", lambda: self._loop_closure_enabled)
        # ERP (Pinhole) hardcodes loop_detection=False (rig-constrained
        # sequential pairs suffice — pipeline.py); grey the checkbox out there
        # so it doesn't accept a setting the solve ignores. All other modes
        # (ERP Native, both fisheye paths) forward the value.
        model.bind_func("loop_closure_disabled",
                        lambda: self._output_mode_idx == PINHOLE_OUTPUT_MODE_IDX)
        model.bind_func("vocab_tree_status", self._get_vocab_tree_status)

        # Advanced quality toggles (default off)
        model.bind("guided_matching_enabled", lambda: self._guided_matching_enabled, self._set_guided_matching)
        model.bind("sift_affine_dsp_enabled", lambda: self._sift_affine_dsp_enabled, self._set_sift_affine_dsp)

        # COLMAP 4.1 controls
        model.bind("feature_type_idx", lambda: str(self._feature_type_idx), self._set_feature_type_idx)
        model.bind("matcher_type_idx", lambda: str(self._matcher_type_idx), self._set_matcher_type_idx)
        model.bind("mapper_idx", lambda: str(self._mapper_idx), self._set_mapper_idx)
        model.bind("ba_solver_idx", lambda: str(self._ba_solver_idx), self._set_ba_solver_idx)
        model.bind_func("show_ba_solver_note", lambda: True)
        model.bind_func("ba_solver_note", self._get_ba_solver_note)
        model.bind_func("mapper_note", self._get_mapper_note)
        model.bind_func("show_mapper_note", lambda: bool(self._get_mapper_note()))
        # ERP-native is spherical (EQUIRECTANGULAR); GLOMAP is upstream-wired
        # for perspective cameras only, so the mapper is always Incremental
        # there. Hide the choice rather than offer an option the pipeline
        # ignores.
        model.bind_func(
            "show_mapper_select",
            lambda: self._output_mode_idx != ERP_NATIVE_OUTPUT_MODE_IDX,
        )

        # Sequential overlap (visible when Matcher = Sequential)
        model.bind("sequential_overlap_str", lambda: str(self._sequential_overlap), self._set_sequential_overlap)
        model.bind_func("show_sequential_overlap", lambda: self._colmap_matcher_idx == 0)

        # Fisheye circle margin (visible when fisheye + masking enabled)
        model.bind("fisheye_circle_margin_str", lambda: f"{self._fisheye_circle_margin:.0f}", self._set_fisheye_circle_margin)
        model.bind_func("show_fisheye_circle_margin", lambda: self._output_mode_idx in _FISHEYE_MODES and self._enable_masking)

        # SIFT controls (COLMAP Preset dropdown + Advanced disclosure)
        model.bind("sift_preset_idx", lambda: str(self._sift_preset_idx), self._set_sift_preset_idx)
        model.bind("sift_max_features_str", lambda: str(self._sift_max_features), self._set_sift_max_features)
        model.bind("sift_max_image_size_str", lambda: str(self._sift_max_image_size), self._set_sift_max_image_size)
        model.bind_func("show_sift_advanced", lambda: self._sift_advanced_expanded)
        model.bind_func("show_sift_reset", self._show_sift_reset)
        model.bind_func("sift_preset_label", self._get_sift_preset_reset_label)
        model.bind_func("sift_advanced_arrow", lambda: "▼" if self._sift_advanced_expanded else "▶")

        # -- Output --
        # The Output Mode dropdown is now the 2-option processing axis
        # (Native / Pinhole); projection is orthogonal and auto-detected.
        model.bind("processing_idx", lambda: str(self._processing_idx), self._set_processing_idx)
        # Fisheye has no Native/Pinhole processing axis (its output is chosen by
        # the Training output selector), so hide the Output Mode dropdown for it.
        model.bind_func("show_processing_dropdown", lambda: self._projection != "fisheye")
        model.bind(
            "fisheye_training_output_idx",
            lambda: str(self._fisheye_training_output_idx),
            self._set_fisheye_training_output,
        )
        model.bind(
            "training_output_idx",
            lambda: str(self._training_output_idx),
            self._set_training_output,
        )
        model.bind_func(
            # training_output (native|pinhole|both) applies to image-folder
            # Native on BOTH projections, and to video ERP Native (the video
            # fisheye path keeps its own fisheye_training_output control).
            "show_image_training_output",
            lambda: (
                (self._source_mode_idx == 2
                 and self._processing_idx == NATIVE_PROCESSING_IDX)
                or (self._source_mode_idx != 2
                    and self._output_mode_idx == ERP_NATIVE_OUTPUT_MODE_IDX)
            ),
        )
        model.bind_func(
            # Video fisheye keeps its own field (fisheye_training_output).
            # EXCLUDE image-folder here: otherwise an image-folder fisheye run
            # would show two dropdowns bound to different config fields.
            "show_fisheye_training_output",
            lambda: (self._output_mode_idx == FISHEYE_OUTPUT_MODE_IDX
                     and self._source_mode_idx != 2),
        )
        model.bind_func(
            "show_output_mode_note",
            lambda: self._output_mode_idx in (
                ERP_NATIVE_OUTPUT_MODE_IDX,
                FISHEYE_OUTPUT_MODE_IDX,
            ),
        )
        model.bind_func(
            "show_crop_size",
            lambda: self._output_mode_idx not in {
                ERP_NATIVE_OUTPUT_MODE_IDX,
                FISHEYE_OUTPUT_MODE_IDX,
            },
        )

        # ── Fisheye-specific bindings ──
        # Split mode is determined by direct user choice (the "Front + Back"
        # button on the empty state) rather than nested under Output Mode.
        # output_mode_idx still flips to Fisheye automatically when split is
        # active or when the user loads a .osv/.insv container.
        is_fisheye = lambda: self._output_mode_idx in _FISHEYE_MODES
        is_split = lambda: self._source_mode_idx == 1
        is_image_folder = lambda: self._source_mode_idx == 2
        is_fisheye_container = lambda: is_fisheye() and not is_split() and not is_image_folder()

        # Projection-first input menu (Plan 03 / Option B). Initial shows the
        # two projection buttons; after a choice the source menu renders, and
        # once a source is active (video loaded / split / image folder) the
        # source buttons yield to that source's block. "‹ Back to projection"
        # is the single reset affordance (plan D3a).
        empty_state = lambda: not self._video_loaded and not is_split() and not is_image_folder()
        model.bind_func("show_projection_choice", lambda: not self._projection_chosen)
        model.bind_func("show_source_menu", lambda: self._projection_chosen)
        model.bind_func("show_source_buttons", empty_state)
        model.bind_func("show_erp_sources", lambda: self._projection == "erp")
        model.bind_func("show_fisheye_sources", lambda: self._projection == "fisheye")
        model.bind_event("select_projection_erp", self._on_select_projection_erp)
        model.bind_event("select_projection_fisheye", self._on_select_projection_fisheye)
        model.bind_event("back_to_projection", self._on_back_to_projection)
        model.bind_func("show_image_folder_mode", is_image_folder)
        # Single-file picker visible when video loaded (regular flow) or
        # when nothing loaded and not in split mode (so the "Select 360°
        # Video" half of the empty state still works).
        model.bind_func("show_split_pickers", is_split)
        model.bind_func("show_single_picker", lambda: not is_split() and not is_image_folder())
        model.bind_func("show_camera_family_select", is_split)
        model.bind_func("show_camera_family_detected", lambda: is_fisheye_container() and self._camera_family_detected is not None)
        model.bind_func("show_keep_streams", is_fisheye_container)
        model.bind("camera_family_idx", lambda: str(self._camera_family_idx), self._set_camera_family)
        model.bind("front_video_path", lambda: self._front_video_path or "(not set)", self._set_front_video_path_noop)
        model.bind("back_video_path", lambda: self._back_video_path or "(not set)", self._set_back_video_path_noop)
        model.bind_func("front_video_path_text", lambda: self._front_video_path or "(not set)")
        model.bind_func("back_video_path_text", lambda: self._back_video_path or "(not set)")
        model.bind_func("camera_family_detected_text", self._get_camera_family_detected_text)
        model.bind("keep_streams", lambda: self._keep_streams, self._set_keep_streams)
        model.bind("keep_extracted_data", lambda: self._keep_extracted_data, self._set_keep_extracted_data)
        # "Keep frames & masks" — retain the extracted source frames + masks as
        # <output>/images + <output>/masks deliverables next to the pinhole
        # dataset. Shown for the VIDEO runs that produce a pinhole dataset with
        # otherwise-discarded source frames: ERP (Pinhole), native Fisheye with
        # Training output = Pinhole, and ERP (Native) with Training output =
        # Pinhole. Image folders read the user's own frames in place, so there
        # is nothing to discard there.
        model.bind_func(
            "show_keep_extracted_data",
            lambda: (
                self._source_mode_idx != 2
                and (
                    self._output_mode_idx == PINHOLE_OUTPUT_MODE_IDX
                    or (
                        self._output_mode_idx == FISHEYE_OUTPUT_MODE_IDX
                        and self._get_fisheye_training_output() == "pinhole"
                    )
                    or (
                        self._output_mode_idx == ERP_NATIVE_OUTPUT_MODE_IDX
                        and self._get_training_output() == "pinhole"
                    )
                )
            ),
        )
        model.bind_event("select_front_video", self._on_select_front_video)
        model.bind_event("select_back_video", self._on_select_back_video)
        model.bind_event("clear_front_video", self._on_clear_front_video)
        model.bind_event("clear_back_video", self._on_clear_back_video)
        model.bind_event("enter_split_mode", self._on_enter_split_mode)
        # ── Image-folder input mode (source_mode_idx == 2) ──
        model.bind_event("enter_image_folder_mode", self._on_enter_image_folder_mode)
        # R1: Folders is an inline two-way control rather than a dropdown (the
        # dropdown overlapped its neighbouring row). Same _fisheye_folder_mode
        # state; the active button is driven by data-class-is-active.
        model.bind_func("folder_mode_is_one",
                        lambda: self._fisheye_folder_mode == "one")
        model.bind_func("folder_mode_is_two",
                        lambda: self._fisheye_folder_mode != "one")
        model.bind_event("set_folder_mode_one", self._on_set_folder_mode_one)
        model.bind_event("set_folder_mode_two", self._on_set_folder_mode_two)
        model.bind("mask_source_idx",
                   lambda: str(("generate", "preexisting", "none").index(self._mask_source)),
                   self._set_mask_source)
        model.bind_func("show_image_erp_picker",
                        lambda: is_image_folder() and self._projection == "erp")
        model.bind_func("show_image_fisheye_controls",
                        lambda: is_image_folder() and self._projection == "fisheye")
        model.bind_func("show_image_fisheye_one",
                        lambda: is_image_folder() and self._projection == "fisheye"
                        and self._fisheye_folder_mode == "one")
        model.bind_func("show_image_fisheye_two",
                        lambda: is_image_folder() and self._projection == "fisheye"
                        and self._fisheye_folder_mode == "two")
        # Pre-existing masks only for ERP + Pinhole (AR-N2)
        model.bind_func("show_preexisting_mask_option",
                        lambda: is_image_folder() and self._get_output_mode() == "pinhole")
        model.bind_func("show_masks_generate_none",
                        lambda: is_image_folder() and self._get_output_mode() != "pinhole")
        model.bind_func("show_preexisting_mask_picker",
                        lambda: is_image_folder() and self._mask_source == "preexisting")
        model.bind_func("image_dir_text", lambda: self._image_dir or "(not set)")
        model.bind_func("fisheye_dir_text", lambda: self._fisheye_dir or "(not set)")
        model.bind_func("front_dir_text", lambda: self._front_dir or "(not set)")
        model.bind_func("back_dir_text", lambda: self._back_dir or "(not set)")
        model.bind_func("preexisting_mask_dir_text",
                        lambda: self._preexisting_mask_dir or "(not set)")
        model.bind_event("select_image_dir", self._on_select_image_dir)
        model.bind_event("clear_image_dir", self._on_clear_image_dir)
        model.bind_event("select_fisheye_dir", self._on_select_fisheye_dir)
        model.bind_event("clear_fisheye_dir", self._on_clear_fisheye_dir)
        model.bind_event("select_front_dir", self._on_select_front_dir)
        model.bind_event("clear_front_dir", self._on_clear_front_dir)
        model.bind_event("select_back_dir", self._on_select_back_dir)
        model.bind_event("clear_back_dir", self._on_clear_back_dir)
        model.bind_event("select_preexisting_mask_dir", self._on_select_preexisting_mask_dir)
        model.bind_event("clear_preexisting_mask_dir", self._on_clear_preexisting_mask_dir)
        model.bind_event("toggle_sift_advanced", self._on_toggle_sift_advanced)
        model.bind_event("reset_sift_to_preset", self._on_reset_sift_to_preset)
        model.bind_func("output_mode_note", self._get_output_mode_note)
        model.bind_func("output_path_display", lambda: self._output_path or "(not set)")
        model.bind_func("dataset_summary_text", self._get_dataset_summary)

        # -- Processing --
        model.bind_func("show_processing", lambda: self._is_processing)
        model.bind_func("show_idle", lambda: not self._is_processing)
        # Post-run summary: visible after a run completes (success or failure)
        # until the user starts another run (which clears _completion_summary).
        model.bind_func(
            "show_completion",
            lambda: not self._is_processing and bool(self._completion_summary),
        )
        model.bind_func("completion_summary_text", lambda: self._completion_summary)
        model.bind_func("completion_report_text", lambda: self._completion_report)
        model.bind_event("clear_completion_report", self._on_clear_completion_report)
        model.bind_func("processing_stage_text", lambda: self._processing_stage)
        model.bind_func("processing_status_text", lambda: self._processing_status)
        model.bind_func("processing_progress_value", lambda: f"{self._processing_progress / 100:.4f}")
        model.bind_func("processing_progress_pct", lambda: f"{self._processing_progress:.1f}%")
        model.bind_func("processing_recent_text", self._get_processing_recent_text)
        model.bind_func("show_error", lambda: bool(self._error_message))
        model.bind_func("error_text", lambda: self._error_message)
        # -- Events --
        model.bind_event("select_video", self._on_select_video)
        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("run_pipeline", self._on_run_pipeline)
        model.bind_event("run_pipeline_only", self._on_run_pipeline_only)
        model.bind_event("cancel_pipeline", self._on_cancel)
        model.bind_event("toggle_section", self._on_toggle_section)
        # Masking setup events
        model.bind_event("install_masking_deps", self._on_install_default_tier)
        model.bind_event("install_video_tracking", self._on_install_video_tracking)
        # SAM 3 / HuggingFace setup events
        model.bind_event("install_premium_tier", self._on_install_premium_tier)
        model.bind_event("open_hf_signup", self._on_open_hf_signup)
        model.bind_event("open_hf_model", self._on_open_hf_model)
        model.bind_event("open_hf_tokens", self._on_open_hf_tokens)
        model.bind_event("verify_hf_token", self._on_verify_hf_token)
        model.bind_event("check_sam3_setup", self._on_check_sam3_setup)
        model.bind_event("toggle_sam3_token_editor", self._on_toggle_sam3_token_editor)
        model.bind_event("forget_hf_token", self._on_forget_hf_token)
        model.bind_event("set_sam3_hf_help", self._on_set_sam3_hf_help)
        model.bind_event("clear_sam3_hf_help", self._on_clear_sam3_hf_help)
        # GPU extraction opt-in events
        model.bind_event("gpu_primary_action", self._on_gpu_primary_action)
        model.bind_event("gpu_copy_details", self._on_gpu_copy_details)
        model.bind_func("show_video_tracking_install",
                         lambda: False)

        self._handle = model.get_handle()

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        self._doc = doc
        self._scrub_fields.mount(doc)
        self._start_setup_probe(auto_enable=True)
        # Set GPU indicator class (cached, won't change at runtime)
        gpu_el = doc.get_element_by_id("gpu-indicator")
        if gpu_el:
            from ..core.sharpest_extractor import SharpestExtractor
            cls = "gpu-ready" if SharpestExtractor._gpu_available() else "gpu-cpu-only"
            gpu_el.set_class(cls, True)

    def on_unmount(self, doc):
        self._scrub_fields.unmount()
        self._doc = None

    def on_update(self, doc):
        if not self._handle:
            return False

        dirty = self._consume_pending_result()
        if self._consume_pending_setup_probe():
            dirty = True
        if self._scrub_fields.sync_all():
            dirty = True

        # Masking setup polling — disabled pending python3.dll fix

        sig = self._ui_signature()
        if sig != self._last_sig:
            self._last_sig = sig
            dirty = True

        if dirty:
            self._handle.dirty_all()

        return dirty

    # ── UI signature for change detection ─────────────────────

    def _ui_signature(self):
        return (
            self._video_loaded,
            self._video_path,
            self._source_mode_idx,
            self._camera_family_idx,
            self._camera_family_detected,
            self._front_video_path,
            self._back_video_path,
            self._keep_streams,
            self._keep_extracted_data,
            self._extract_fps,
            self._extract_sharpness_idx,
            self._blur_metric_idx,
            self._extract_all_frames,
            self._masking_method_idx,
            self._masking_available,
            self._enable_masking,
            self._enable_diagnostics,
            self._mask_prompts_str,
            self._hf_token_input,
            self._hf_verify_text,
            self._install_busy,
            self._install_button_text,
            self._gpu_state,
            self._gpu_hw_present,
            self._gpu_probe_done,
            self._gpu_busy,
            self._gpu_progress_text,
            self._gpu_notice_text,
            self._sam3_check_button_text,
            self._sam3_notice_text,
            self._sam3_hf_help_key,
            (
                self._sam3_setup_report.token_status,
                self._sam3_setup_report.access_status,
                self._sam3_setup_report.runtime_status,
                self._sam3_setup_report.weights_status,
                self._sam3_setup_report.overall_stage,
                self._sam3_setup_report.message,
                self._sam3_setup_report.next_action,
                self._sam3_setup_report.detail,
            ),
            self._preset_idx,
            self._output_mode_idx,
            self._fisheye_training_output_idx,
            self._output_size_idx,
            self._jpeg_quality,
            self._colmap_matcher_idx,
            self._match_budget_idx,
            self._colmap_max_num_matches,
            self._sift_preset_idx,
            self._sift_last_non_custom_preset_idx,
            self._sift_advanced_expanded,
            self._sift_max_features,
            self._sift_max_image_size,
            self._feature_type_idx,
            self._matcher_type_idx,
            self._mapper_idx,
            self._ba_solver_idx,
            self._fisheye_circle_margin,
            self._loop_closure_enabled,
            self._guided_matching_enabled,
            self._sift_affine_dsp_enabled,
            self._output_path,
            self._is_processing,
            self._processing_stage,
            self._processing_progress,
            tuple(self._processing_log_lines),
            self._error_message,
            self._completion_summary,
            self._completion_report,
        )

    def _append_processing_log(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self._processing_log_lines and self._processing_log_lines[-1] == line:
            return
        self._processing_log_lines.append(line)
        if len(self._processing_log_lines) > 14:
            self._processing_log_lines = self._processing_log_lines[-14:]

    # ── Computed text helpers ─────────────────────────────────

    def _get_selected_preset_name(self) -> str:
        if 0 <= self._preset_idx < len(PRESET_NAMES):
            return PRESET_NAMES[self._preset_idx]
        return DEFAULT_PRESET

    @property
    def _processing(self) -> str:
        """Processing axis as a string: "native" | "pinhole"."""
        return "native" if self._processing_idx == NATIVE_PROCESSING_IDX else "pinhole"

    @property
    def _output_mode_idx(self) -> int:
        """Derived, read-only index into OUTPUT_MODES.

        Computed from the two orthogonal axes (_projection, _processing).
        The ~two dozen read sites keep working unchanged; the four former
        write sites now set _projection / _processing_idx instead.

        Fisheye has no processing axis — its output is chosen by the Training
        output selector — so processing is pinned to "native" for fisheye and
        the retired ("fisheye", "pinhole") combo is never derived.
        """
        processing = "native" if self._projection == "fisheye" else self._processing
        return output_mode_index(self._projection, processing)

    def _get_output_mode(self) -> str:
        if 0 <= self._output_mode_idx < len(OUTPUT_MODES):
            return OUTPUT_MODES[self._output_mode_idx]
        return "pinhole"

    def _get_fisheye_training_output(self) -> str:
        if 0 <= self._fisheye_training_output_idx < len(FISHEYE_TRAINING_OUTPUTS):
            return FISHEYE_TRAINING_OUTPUTS[self._fisheye_training_output_idx]
        return "native"

    def _get_fisheye_training_output_label(self) -> str:
        if 0 <= self._fisheye_training_output_idx < len(FISHEYE_TRAINING_OUTPUT_LABELS):
            return FISHEYE_TRAINING_OUTPUT_LABELS[self._fisheye_training_output_idx]
        return FISHEYE_TRAINING_OUTPUT_LABELS[0]

    def _get_training_output(self) -> str:
        """Image-folder training output (native | pinhole | both)."""
        if 0 <= self._training_output_idx < len(IMAGE_TRAINING_OUTPUTS):
            return IMAGE_TRAINING_OUTPUTS[self._training_output_idx]
        return "native"

    def _get_training_output_label(self) -> str:
        if 0 <= self._training_output_idx < len(IMAGE_TRAINING_OUTPUT_LABELS):
            return IMAGE_TRAINING_OUTPUT_LABELS[self._training_output_idx]
        return IMAGE_TRAINING_OUTPUT_LABELS[0]

    def _get_current_view_config(self):
        preset_name = resolve_view_preset_name(
            self._get_selected_preset_name(),
            self._get_output_mode(),
        )
        return VIEW_PRESETS.get(preset_name, VIEW_PRESETS[DEFAULT_PRESET])

    def _get_fisheye_preset_text(self) -> str:
        return "OPENCV_FISHEYE × PER_FOLDER, no rig (forced)"

    def _get_camera_family_detected_text(self) -> str:
        if self._camera_family_detected is None:
            return ""
        family_to_label = dict(zip(CAMERA_FAMILIES, CAMERA_FAMILY_LABELS))
        label = family_to_label.get(self._camera_family_detected, self._camera_family_detected)
        return f"Camera family detected: {label}"

    @staticmethod
    def _format_sam3_status(status: str) -> str:
        labels = {
            "missing": "Missing",
            "saved": "Saved",
            "verified": "Verified",
            "invalid": "Invalid",
            "unknown": "Unknown",
            "pending": "Pending Approval",
            "granted": "Granted",
            "network_error": "Network Error",
            "installed": "Installed",
            "broken": "Broken",
            "present": "Present",
            "failed": "Failed",
        }
        return labels.get(status, status.replace("_", " ").title())

    def _get_sam3_intro_text(self) -> str:
        if self._sam3_setup_report.access_status == "granted" or self._setup_state.sam3_ready:
            return "This plugin uses SAM 3 for masking."
        return (
            "This plugin uses SAM 3 for masking. SAM 3 is a gated model that requires "
            "approval before downloading, and setup requires a user token. Hover over "
            "the HuggingFace buttons for details."
        )

    def _get_sam3_support_text(self) -> str:
        return "This plugin will function without masking enabled."

    def _show_sam3_support_text(self) -> bool:
        return not (
            self._sam3_setup_report.access_status == "granted" or self._setup_state.sam3_ready
        )

    def _get_sam3_status_message(self) -> str:
        if (
            self._sam3_setup_report.overall_stage == "needs_token"
            and self._sam3_setup_report.token_status == "missing"
        ):
            return ""
        if (
            self._sam3_setup_report.access_status == "granted"
            and self._sam3_setup_report.overall_stage in {"ready_to_install", "needs_weights"}
        ):
            return "SAM 3 access is approved. Click 'Install SAM 3' to add the model."
        return self._sam3_setup_report.message

    def _get_sam3_hf_help_text(self) -> str:
        help_text = {
            "signup": "Make a new HuggingFace account.",
            "request": "Apply for access to SAM 3. Approval can take 30 minutes to several hours.",
            "tokens": (
                "Create a new token, select Read for the Token Type, and name it whatever "
                "you like. Input the token below and click Verify Access."
            ),
        }
        if self._sam3_hf_help_key in help_text:
            return help_text[self._sam3_hf_help_key]
        return ""

    def _get_sam3_saved_token_text(self) -> str:
        return "A HuggingFace token is stored locally on this machine."

    def _get_sam3_weights_status_text(self) -> str:
        if (
            self._sam3_setup_report.weights_status == "present"
            and self._sam3_setup_report.access_status != "granted"
        ):
            return "Cached Locally"
        return self._format_sam3_status(self._sam3_setup_report.weights_status)

    def _show_sam3_cached_weights_note(self) -> bool:
        return (
            self._sam3_setup_report.weights_status == "present"
            and self._sam3_setup_report.access_status != "granted"
        )

    def _get_sam3_cached_weights_note_text(self) -> str:
        return "SAM 3 files are already cached on this machine from an earlier setup."

    def _get_sam3_install_button_text(self) -> str:
        if self._install_busy:
            return self._install_button_text
        stage = self._sam3_setup_report.overall_stage
        if stage == "needs_weights":
            return "Download Weights"
        if stage == "error":
            return "Retry Install"
        return "Install SAM 3"

    def _get_sam3_install_disabled(self) -> bool:
        if self._install_busy:
            return True
        return self._sam3_setup_report.overall_stage not in {
            "ready_to_install",
            "needs_weights",
            "error",
        }

    def _show_sam3_install_button(self) -> bool:
        return self._sam3_setup_report.overall_stage in {
            "ready_to_install",
            "needs_weights",
            "error",
        }

    def _show_sam3_check_button(self) -> bool:
        return (
            self._sam3_setup_report.token_status == "invalid"
            or self._sam3_setup_report.access_status in {"pending", "network_error"}
            or self._sam3_setup_report.runtime_status in {"installed", "broken"}
        )

    def _show_sam3_hf_heading(self) -> bool:
        return self._show_sam3_external_actions() or self._show_sam3_token_actions()

    def _show_sam3_hf_inline_actions(self) -> bool:
        return self._show_sam3_token_actions() and not self._show_sam3_external_actions()

    def _show_sam3_external_actions(self) -> bool:
        return (
            self._show_sam3_signup_button()
            or self._show_sam3_request_access_button()
            or self._show_sam3_tokens_button()
        )

    def _show_sam3_signup_button(self) -> bool:
        return (
            self._show_sam3_token_editor()
            and not self._setup_state.has_token
            and self._sam3_setup_report.access_status != "granted"
        )

    def _show_sam3_request_access_button(self) -> bool:
        return (
            self._show_sam3_token_editor()
            and self._sam3_setup_report.access_status != "granted"
        )

    def _show_sam3_tokens_button(self) -> bool:
        return self._show_sam3_token_editor()

    def _show_sam3_local_actions(self) -> bool:
        return (
            self._show_sam3_token_actions()
            or self._show_sam3_token_editor()
            or self._show_sam3_check_button()
            or self._show_sam3_install_button()
        )

    def _show_sam3_saved_token_notice(self) -> bool:
        return False

    def _show_sam3_token_editor(self) -> bool:
        return (
            self._sam3_edit_token
            or not self._setup_state.has_token
            or self._sam3_setup_report.token_status in {"missing", "invalid"}
        )

    def _show_sam3_status_message(self) -> bool:
        return bool(self._get_sam3_status_message())

    def _show_sam3_token_actions(self) -> bool:
        return self._setup_state.has_token or self._sam3_edit_token

    def _show_sam3_token_actions_row(self) -> bool:
        return self._show_sam3_token_actions() and not self._show_sam3_hf_inline_actions()

    def _show_hf_verify_text(self) -> bool:
        if not self._hf_verify_text:
            return False
        return self._hf_verify_text != "SAM 3 setup needs a HuggingFace token."

    def _show_sam3_token_editor_message(self) -> bool:
        return self._show_sam3_token_editor() and self._show_hf_verify_text()

    def _get_sam3_change_token_button_text(self) -> str:
        if self._sam3_edit_token:
            return "Cancel Token Change"
        return "Change Token"

    def _get_est_frames_text(self) -> str:
        if not self._video_info:
            return "Select a video source"
        if self._extract_all_frames:
            n = self._video_info.frame_count
            if not n:
                n = round(self._video_info.fps * self._video_info.duration_seconds)
            return f"All frames: ~{n}"
        interval = 1.0 / max(0.1, self._extract_fps)
        base = VideoAnalyzer.estimate_frame_count(self._video_info, interval)
        preset = EXTRACT_SHARPNESS_PRESETS[self._extract_sharpness_idx]
        if preset["scene_threshold"] > 0:
            extra = int(base * 0.2)
            return f"Estimated frames   ~{base}\u2013{base + extra}"
        return f"Estimated frames   ~{base}"

    def _get_gpu_indicator_text(self) -> str:
        from ..core.sharpest_extractor import SharpestExtractor
        if SharpestExtractor._gpu_available():
            return "GPU"
        return "CPU"

    def _get_coverage_text(self) -> str:
        if self._get_output_mode() == "fisheye":
            training_output = self._get_fisheye_training_output()
            if training_output == "pinhole":
                return "8 pinhole crops per registered fisheye lens"
            if training_output == "both":
                return "Native fisheye plus 8 pinhole crops per registered lens"
            return "2 fisheye images per pair (front + back, native resolution)"
        name = resolve_view_preset_name(
            self._get_selected_preset_name(),
            self._get_output_mode(),
        )
        return COVERAGE_DESCRIPTIONS.get(name, "")

    def _get_total_output_text(self) -> str:
        output_mode = self._get_output_mode()
        if output_mode == "fisheye":
            training_output = self._get_fisheye_training_output()
            if not self._video_info:
                if training_output == "pinhole":
                    return "16 pinhole crops per complete pair"
                if training_output == "both":
                    return "2 native fisheye images + 16 pinhole crops per complete pair"
                return "2 fisheye images per pair (front + back)"
            _interval_fish = 1.0 / max(0.1, self._extract_fps)
            _pairs_fish = VideoAnalyzer.estimate_frame_count(self._video_info, _interval_fish)
            if self._fisheye_single_lens:
                # One lens: 1 native frame and 8 propagated crops per frame.
                if training_output == "pinhole":
                    return f"~{_pairs_fish} frames x 8 = {8 * _pairs_fish:,} pinhole crops"
                if training_output == "both":
                    return f"~{_pairs_fish} frames x 9 = {9 * _pairs_fish:,} total images"
                return f"~{_pairs_fish:,} fisheye frames"
            if training_output == "pinhole":
                return f"~{_pairs_fish} pairs x 16 = {16 * _pairs_fish:,} pinhole crops"
            if training_output == "both":
                return f"~{_pairs_fish} pairs x 18 = {18 * _pairs_fish:,} total images"
            return f"~{_pairs_fish} pairs x 2 = {2 * _pairs_fish:,} fisheye images"
        vc = self._get_current_view_config()
        views = vc.total_views()
        if not self._video_info:
            if output_mode == "erp_native":
                return "1 ERP image per frame"
            return f"{views} views per frame"
        interval = 1.0 / max(0.1, self._extract_fps)
        frames = VideoAnalyzer.estimate_frame_count(self._video_info, interval)
        total = views * frames
        if output_mode == "erp_native":
            training = self._get_training_output()
            if training == "both":
                return (
                    f"~{frames:,} ERP images + {views} \u00d7 {frames} "
                    f"= {total:,} propagated crops"
                )
            if training == "pinhole":
                return f"{views} \u00d7 {frames} = {total:,} propagated crops"
            return f"~{frames:,} ERP images"
        return f"{views} views \u00d7 {frames} frames = {total:,} images"

    def _get_dataset_summary(self) -> str:
        output_mode = self._get_output_mode()
        if self._source_mode_idx == 2:
            # Image-folder mode (AR-N4): summarise the folder selection.
            proc = "Pinhole" if self._processing_idx == PINHOLE_PROCESSING_IDX else "Native"
            if self._processing_idx == NATIVE_PROCESSING_IDX:
                # Native gains the training-output choice; spell out what the
                # run actually delivers (two datasets for "Both").
                training = self._get_training_output()
                if training == "both":
                    proc = "Native + propagated Pinhole"
                elif training == "pinhole":
                    proc = "Propagated Pinhole"
            if self._projection == "erp":
                where = self._image_dir or "(no folder selected)"
                return f"Image folder (Equirectangular) | {proc} | {where}"
            if self._fisheye_folder_mode == "one":
                where = self._fisheye_dir or "(no folder selected)"
                mode = "one folder"
            else:
                front = self._front_dir or "(front not set)"
                back = self._back_dir or "(back not set)"
                where = f"{front} + {back}"
                mode = "two folders"
            fam = (
                CAMERA_FAMILY_LABELS[self._camera_family_idx]
                if 0 <= self._camera_family_idx < len(CAMERA_FAMILY_LABELS)
                else "unknown"
            )
            return f"Image folder (Fisheye, {mode}) | {fam} | {proc} | {where}"
        if output_mode == "fisheye":
            family = self._camera_family_detected
            if family is None and 0 <= self._camera_family_idx < len(CAMERA_FAMILIES):
                family = CAMERA_FAMILIES[self._camera_family_idx]
            family_label = (
                CAMERA_FAMILY_LABELS[CAMERA_FAMILIES.index(family)]
                if family in CAMERA_FAMILIES else (family or "unknown")
            )
            if self._source_mode_idx == 1:
                if self._front_video_path and self._back_video_path:
                    return (
                        f"Fisheye (split) | {family_label} | "
                        f"{self._get_fisheye_training_output_label()} | "
                        "OPENCV_FISHEYE \u00d7 2 cameras"
                    )
                return "Fisheye (split) \u2014 front + back not yet selected"
            if not self._video_loaded:
                return "No video loaded"
            interval = 1.0 / max(0.1, self._extract_fps)
            pairs = (
                VideoAnalyzer.estimate_frame_count(self._video_info, interval)
                if self._video_info else 0
            )
            if self._fisheye_single_lens:
                # No camera family for a bare single-lens file \u2014 don't let the
                # dropdown's default leak into the summary.
                return (
                    "Fisheye (single lens) | "
                    f"{self._get_fisheye_training_output_label()} | "
                    "OPENCV_FISHEYE \u00d7 1 camera | "
                    f"~{pairs:,} frames"
                )
            return (
                f"Fisheye | {family_label} | "
                f"{self._get_fisheye_training_output_label()} | "
                "OPENCV_FISHEYE \u00d7 2 cameras | "
                f"~{pairs:,} pairs"
            )

        if not self._video_loaded:
            return "No video loaded"
        vc = self._get_current_view_config()
        views = vc.total_views()
        interval = 1.0 / max(0.1, self._extract_fps)
        frames = VideoAnalyzer.estimate_frame_count(self._video_info, interval) if self._video_info else 0
        total = views * frames
        if output_mode == "erp_native":
            return f"ERP (Native) | {frames:,} ERP frames | EQUIRECTANGULAR"
        return f"Pinhole (COLMAP) | {total:,} images | {self._output_size}px"

    def _get_match_budget_text(self) -> str:
        return (
            f"Up to {self._colmap_max_num_matches:,} matches per image pair. "
            "Higher limits preserve more correspondences but cost more time and memory."
        )

    def _get_processing_recent_text(self) -> str:
        """Return a compact recent-activity view for the live processing card."""
        lines = [
            line for line in self._processing_log_lines
            if line
            and line != self._processing_stage
            and line != self._processing_status
        ]
        if not lines:
            return "Waiting for pipeline updates..."
        return "\n".join(lines[-6:])

    def _get_matcher_and_tier(self) -> tuple[str, str]:
        matcher = (
            COLMAP_MATCHERS[self._colmap_matcher_idx]
            if 0 <= self._colmap_matcher_idx < len(COLMAP_MATCHERS)
            else "sequential"
        )
        tier = (
            MATCH_BUDGET_TIERS[self._match_budget_idx]
            if 0 <= self._match_budget_idx < len(MATCH_BUDGET_TIERS)
            else "custom"
        )
        return matcher, tier

    def _resolve_import_target(self, dataset_path: str) -> tuple[str, str]:
        """Normalize a finished dataset path for LichtFeld's import flow.

        LichtFeld's own retained import panel first runs ``detect_dataset_info``
        and then imports the detected dataset base path with an output folder at
        ``base_path/output``. Mirroring that here avoids relying on the raw
        pipeline return path and keeps Process & Import aligned with the app's
        normal dataset workflow.
        """
        info = lf.detect_dataset_info(dataset_path)
        if not info:
            raise RuntimeError(f"Could not detect dataset info for: {dataset_path}")

        base_path = str(info.base_path).strip()
        images_path = str(info.images_path).strip()
        sparse_path = str(info.sparse_path).strip()

        if not base_path:
            raise RuntimeError(f"Dataset base path is empty for: {dataset_path}")
        if not images_path:
            raise RuntimeError(f"Dataset images path is missing for: {dataset_path}")
        if not sparse_path:
            raise RuntimeError(f"Dataset sparse path is missing for: {dataset_path}")

        output_path = str(Path(base_path) / "output")
        return base_path, output_path

    def _resolve_erp_import_target(self, transforms_path: str) -> tuple[str, str]:
        """Resolve a transforms.json-based dataset import target.

        Used for ERP native (camera_model=EQUIRECTANGULAR) output, which
        produces transforms.json at the dataset root. Fisheye native is
        imported by DIRECTORY instead (see _resolve_transforms_directory_import_target)
        because its front/back masks collide under BlenderLoader's
        bare-filename matching; see the call site in the import block.

        Passes the transforms.json file path directly (not the directory)
        so LFS's BlenderLoader picks it up instead of ColmapLoader. When
        the directory is passed and sparse/ also exists, ColmapLoader wins
        (registered first) and loads pinhole crop cameras instead of
        the ERP cameras declared in transforms.json.
        """
        transforms_file = Path(transforms_path)
        if transforms_file.suffix.lower() != ".json":
            raise RuntimeError(f"Dataset path must be a transforms.json file: {transforms_path}")
        if not transforms_file.is_file():
            raise RuntimeError(f"transforms.json not found: {transforms_path}")
        dataset_dir = transforms_file.parent
        output_path = str(dataset_dir / "output")
        return str(transforms_file), output_path

    def _resolve_transforms_directory_import_target(self, dataset_path: str) -> tuple[str, str]:
        """Resolve a transforms dataset that must retain its directory as base."""
        dataset_dir = Path(dataset_path)
        if dataset_dir.suffix.lower() == ".json":
            dataset_dir = dataset_dir.parent
        transforms_file = dataset_dir / "transforms.json"
        if not transforms_file.is_file():
            raise RuntimeError(f"transforms.json not found: {transforms_file}")
        output_path = str(dataset_dir / "output")
        return str(dataset_dir), output_path

    def _build_stage_timing_lines(
        self,
        stage_names: dict[str, str],
        timing_dict: dict[str, dict],
        total: float,
    ) -> tuple[list[str], list[str]]:
        parts: list[str] = []
        lines: list[str] = []

        for key, entry in timing_dict.items():
            label = stage_names.get(key, key)
            elapsed = entry["elapsed_sec"]
            pct = (elapsed / total * 100) if total > 0 else 0
            rate_str = ""
            if "rate_per_sec" in entry:
                rate_str = f" | {entry['rate_per_sec']:.1f} items/sec"
            parts.append(f"{label} {elapsed:.1f}s")
            lines.append(f"{label:<16s} {elapsed:7.1f}s  ({pct:4.0f}%){rate_str}")

            for sub, sub_elapsed in entry.get("substeps", {}).items():
                sub_pct = (sub_elapsed / elapsed * 100) if elapsed > 0 else 0
                lines.append(f"  {sub:<14s} {sub_elapsed:7.1f}s  ({sub_pct:4.0f}%)")

        lines.append(f"{'TOTAL':<16s} {total:7.1f}s")
        return parts, lines

    def _count_output_images(self, directory: Path) -> int:
        if not directory.is_dir():
            return 0
        return sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def _load_mask_diagnostics_summary(self, path: str) -> dict | None:
        """Best-effort load of a mask diagnostics summary for reporting."""
        doc = load_mask_diagnostics_document(path)
        return get_mask_diagnostics_summary(doc)

    def _append_mask_diagnostics_lines(self, lines: list[str], path: str) -> dict | None:
        """Append human-readable masking diagnostics lines to a report."""
        if not path:
            return None

        lines.append(f"Mask diagnostics: {path}")
        doc = load_mask_diagnostics_document(path)
        if not doc:
            lines.append("  Mask diagnostics summary: unavailable")
            return None

        for line in format_mask_diagnostics_overview(doc):
            lines.append(f"  {line}")

        return get_mask_diagnostics_summary(doc)

    def _build_failure_report(
        self,
        result: PipelineResult,
        stage_names: dict[str, str],
        timing_dict: dict[str, dict],
        total: float,
    ) -> tuple[str, str]:
        parts, lines = self._build_stage_timing_lines(stage_names, timing_dict, total)
        matcher, tier = self._get_matcher_and_tier()
        lines.append(f"Matcher: {matcher}")
        lines.append(f"Match limit: {tier}")
        lines.append(f"Max. matches: {self._colmap_max_num_matches:,} per pair")
        if result.preset_signature:
            lines.append(f"Preset geometry: {result.preset_signature}")
        if result.mask_backend_name:
            lines.append(f"Mask backend: {result.mask_backend_name}")
        if result.video_backend_name:
            suffix = " (fallback used)" if result.used_fallback_video_backend else ""
            lines.append(f"Video backend: {result.video_backend_name}{suffix}")
        if result.video_backend_error:
            lines.append(f"Video backend error: {result.video_backend_error}")
        if result.mask_diagnostics_path:
            self._append_mask_diagnostics_lines(lines, result.mask_diagnostics_path)

        output_root = Path(self._output_path) if self._output_path else None
        if output_root:
            source_frames = self._count_output_images(output_root / "extracted" / "frames")
            output_images = self._count_output_images(output_root / "images")
            if source_frames > 0:
                lines.append(f"Frames extracted: {source_frames}")
            if output_images > 0:
                lines.append(f"Images written: {output_images}")
            lines.append(f"Output path: {output_root}")
            log_path = output_root / "metadata" / "colmap_debug.log"
            if not log_path.is_file():
                log_path = output_root / "colmap_debug.log"
            if log_path.is_file():
                lines.append(f"COLMAP log: {log_path}")

        error_text = result.error or "Pipeline failed"
        lines.append(f"Error: {error_text}")

        summary = f"Failed after {total:.1f}s - {error_text}"
        if parts:
            summary += f" ({', '.join(parts)})"
        return summary, "\n".join(lines)

    def _build_minimal_failure_report(
        self,
        result: PipelineResult,
        total: float,
        report_error: str = "",
    ) -> tuple[str, str]:
        error_text = result.error or "Pipeline failed"
        lines = [
            f"TOTAL            {total:7.1f}s",
        ]

        matcher, tier = self._get_matcher_and_tier()
        lines.append(f"Matcher: {matcher}")
        lines.append(f"Match limit: {tier}")
        lines.append(f"Max. matches: {self._colmap_max_num_matches:,} per pair")
        if result.preset_signature:
            lines.append(f"Preset geometry: {result.preset_signature}")
        if result.mask_diagnostics_path:
            self._append_mask_diagnostics_lines(lines, result.mask_diagnostics_path)

        output_root = Path(self._output_path) if self._output_path else None
        if output_root:
            try:
                source_frames = self._count_output_images(output_root / "extracted" / "frames")
                output_images = self._count_output_images(output_root / "images")
                if source_frames > 0:
                    lines.append(f"Frames extracted: {source_frames}")
                if output_images > 0:
                    lines.append(f"Images written: {output_images}")
                lines.append(f"Output path: {output_root}")
                log_path = output_root / "colmap_debug.log"
                if log_path.is_file():
                    lines.append(f"COLMAP log: {log_path}")
            except Exception as exc:
                lines.append(f"Diagnostics scan failed: {exc}")

        lines.append(f"Error: {error_text}")
        if report_error:
            lines.append(f"Report generation error: {report_error}")

        summary = f"Failed after {total:.1f}s - {error_text}"
        return summary, "\n".join(lines)

    def _write_failure_timing_json(
        self,
        result: PipelineResult,
        total: float,
        timing_dict: dict[str, dict],
        matcher: str,
        tier: str,
        report_error: str = "",
    ) -> None:
        if not self._output_path:
            return

        metadata_dir = Path(self._output_path) / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        timing_path = metadata_dir / "timing.json"
        mask_diagnostics_summary = self._load_mask_diagnostics_summary(
            result.mask_diagnostics_path
        )
        timing_output = {
            "success": False,
            "total_sec": round(total, 3),
            "stages": timing_dict,
            "result": {
                "source_frames": result.num_source_frames,
                "output_images": result.num_output_images,
                "aligned_cameras": result.num_aligned_cameras,
                "matcher": matcher,
                "match_budget_tier": tier,
                "max_num_matches": self._colmap_max_num_matches,
                "preset_geometry": result.preset_signature,
                "views_per_frame": result.views_per_frame,
                "registered_frames": result.num_registered_frames,
                "complete_rig_frames": result.num_complete_frames,
                "partial_rig_frames": result.num_partial_frames,
                "dropped_rig_frames": max(
                    result.num_source_frames - result.num_registered_frames, 0
                ),
                "registered_images_by_view": result.registered_images_by_view,
                "expected_images_by_view": result.expected_images_by_view,
                "partial_frame_examples": result.partial_frame_examples,
                "dropped_frame_examples": result.dropped_frame_examples,
                "mask_backend": result.mask_backend_name,
                "mask_diagnostics_path": result.mask_diagnostics_path,
                "mask_diagnostics_summary": mask_diagnostics_summary,
                "video_backend": result.video_backend_name,
                "used_fallback_video_backend": result.used_fallback_video_backend,
                "video_backend_error": result.video_backend_error,
                "error": self._error_message,
            },
        }
        if report_error:
            timing_output["report_error"] = report_error
        timing_path.write_text(json.dumps(timing_output, indent=2))

    def _build_completion_report(
        self,
        result: PipelineResult,
        stage_names: dict[str, str],
        timing_dict: dict[str, dict],
        total: float,
    ) -> tuple[str, str]:
        parts, lines = self._build_stage_timing_lines(stage_names, timing_dict, total)
        matcher, tier = self._get_matcher_and_tier()
        lines.append(f"Matcher: {matcher}")
        lines.append(f"Match limit: {tier}")
        lines.append(f"Max. matches: {self._colmap_max_num_matches:,} per pair")
        if result.preset_signature:
            lines.append(f"Preset geometry: {result.preset_signature}")
        if result.mask_backend_name:
            lines.append(f"Mask backend: {result.mask_backend_name}")
        if result.video_backend_name:
            suffix = " (fallback used)" if result.used_fallback_video_backend else ""
            lines.append(f"Video backend: {result.video_backend_name}{suffix}")
        if result.video_backend_error:
            lines.append(f"Video backend error: {result.video_backend_error}")
        if result.mask_diagnostics_path:
            self._append_mask_diagnostics_lines(lines, result.mask_diagnostics_path)

        if result.views_per_frame > 0:
            dropped_frames = max(result.num_source_frames - result.num_registered_frames, 0)
            # Image-folder runs read frames the user supplied; nothing is
            # extracted from a video, so don't report them as such.
            frames_label = (
                "Frames read" if self._source_mode_idx == 2 else "Frames extracted"
            )
            lines.append(f"{frames_label}: {result.num_source_frames}")
            lines.append(f"Views per frame: {result.views_per_frame}")
            lines.append(f"Images written: {result.num_output_images}")
            lines.append(
                f"Registered frames: {result.num_registered_frames}/{result.num_source_frames}"
            )
            # "rig" only describes multi-view frames; single-view runs
            # (single-lens fisheye, ERP native) have no rig to speak of.
            rig_label = "rig frames" if result.views_per_frame > 1 else "frames"
            lines.append(f"Complete {rig_label}: {result.num_complete_frames}")
            lines.append(f"Dropped {rig_label}: {dropped_frames}")
            if result.dropped_frame_examples:
                lines.append(
                    f"Dropped examples: {', '.join(result.dropped_frame_examples)}"
                )
            if result.num_partial_frames > 0:
                lines.append(f"Partial {rig_label}: {result.num_partial_frames}")
                if result.partial_frame_examples:
                    lines.append(
                        f"Examples: {', '.join(result.partial_frame_examples)}"
                    )
            lines.append(f"Registered images: {result.num_aligned_cameras}")
            if result.expected_images_by_view:
                lines.append("Per-view registration:")
                for view_name, expected in result.expected_images_by_view.items():
                    registered = result.registered_images_by_view.get(view_name, 0)
                    lines.append(f"  {view_name:<14s} {registered:>3d}/{expected:<3d}")
        else:
            lines.append(
                f"Frames: {result.num_source_frames} -> Images: "
                f"{result.num_output_images} -> Aligned: {result.num_aligned_cameras}"
            )

        if result.gpu_extraction:
            lines.append("Extraction: GPU (NVDEC + CUDA scoring)")
        lines.append(f"Output path:\n{result.dataset_path}")

        summary = f"Completed in {total:.1f}s - {', '.join(parts)}"
        return summary, "\n".join(lines)

    # ── Setters (called by data model on user input) ──────────

    def _set_extract_fps(self, val):
        try:
            v = float(val)
            if 0.1 <= v <= 5.0:
                self._extract_fps = v
        except (ValueError, TypeError):
            pass

    def _set_extract_sharpness(self, val):
        try:
            v = int(float(val))
            if 0 <= v < len(EXTRACT_SHARPNESS_PRESETS):
                self._extract_sharpness_idx = v
        except (ValueError, TypeError):
            pass

    def _set_blur_metric(self, val):
        try:
            v = int(float(val))
            if 0 <= v < len(BLUR_METRICS):
                self._blur_metric_idx = v
        except (ValueError, TypeError):
            pass

    def _set_extract_all_frames(self, val):
        if isinstance(val, bool):
            self._extract_all_frames = val
        else:
            self._extract_all_frames = str(val).lower() in ("true", "1", "yes", "on")

    def _set_jpeg_quality(self, val):
        try:
            v = int(float(val))
            if 50 <= v <= 100:
                self._jpeg_quality = v
        except (ValueError, TypeError):
            pass

    def _set_enable_masking(self, val):
        self._enable_masking = bool(val) and self._masking_available

    def _set_mask_prompts(self, val):
        self._mask_prompts_str = str(val)

    def _selected_masking_ready(self) -> bool:
        return self._setup_state.sam3_ready

    def _refresh_masking_availability(self, *, auto_enable: bool = False) -> None:
        ready = self._selected_masking_ready()
        self._masking_available = ready
        if auto_enable:
            self._enable_masking = ready
        else:
            self._enable_masking = self._enable_masking and ready

    def _start_setup_probe(self, *, auto_enable: bool = False) -> None:
        if self._setup_probe_started:
            return
        self._setup_probe_started = True

        def _probe():
            try:
                state = check_masking_setup()
                report = check_sam3_setup(setup_state=state)
                with self._setup_probe_lock:
                    self._setup_probe_pending = (state, report, auto_enable)
            except Exception:
                logger.exception("Initial SAM 3 setup probe failed")
            try:
                # GPU extraction opt-in state. detect_runtime_state() also
                # persists active -> reverted when the installed cv2 is no
                # longer the CUDA build (Task 3.2 wiring). Cheap + cv2-free.
                from ..core import gpu_extraction_install as _gpu

                self._gpu_state = _gpu.detect_runtime_state()
                self._gpu_error_detail = _gpu.get_state().get("detail", "")
                self._gpu_hw_present = _gpu.gpu_hardware_present()
                self._gpu_probe_done = True
            except Exception:
                logger.exception("GPU extraction state probe failed")
            finally:
                if self._handle:
                    self._handle.dirty_all()

        threading.Thread(target=_probe, daemon=True).start()

    def _consume_pending_setup_probe(self) -> bool:
        with self._setup_probe_lock:
            pending = self._setup_probe_pending
            self._setup_probe_pending = None

        if pending is None:
            return False

        state, report, auto_enable = pending
        self._setup_state = state
        self._sam3_setup_report = report
        self._hf_verify_text = report.message
        self._hf_verify_ok = report.access_status == "granted"
        self._refresh_masking_availability(auto_enable=auto_enable)
        return True

    def _sync_setup_state(
        self,
        *,
        force_access_check: bool = False,
        auto_enable: bool = False,
    ) -> None:
        state = check_masking_setup()
        self._setup_state = state
        self._sam3_setup_report = check_sam3_setup(
            setup_state=state,
            force_access_check=force_access_check,
        )
        self._hf_verify_text = self._sam3_setup_report.message
        self._hf_verify_ok = self._sam3_setup_report.access_status == "granted"
        if not self._setup_state.has_token:
            self._sam3_edit_token = False
        self._refresh_masking_availability(auto_enable=auto_enable)

    def _set_hf_token_input(self, val):
        self._hf_token_input = str(val)

    # ── Masking setup event handlers ─────────────────────────

    def _on_open_hf_signup(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/join")

    def _on_open_hf_model(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/facebook/sam3")

    def _on_open_hf_tokens(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/settings/tokens")

    def _on_set_sam3_hf_help(self, handle, event, args):
        del handle, event
        key = str(args[0]) if args else ""
        if key == self._sam3_hf_help_key:
            return
        self._sam3_hf_help_key = key
        if self._handle:
            self._handle.dirty_all()

    def _on_clear_sam3_hf_help(self, handle, event, args):
        del handle, event, args
        if not self._sam3_hf_help_key:
            return
        self._sam3_hf_help_key = ""
        if self._handle:
            self._handle.dirty_all()

    def _on_verify_hf_token(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._sam3_notice_text = ""
        token = self._hf_token_input.strip()
        if not token:
            self._sam3_setup_report = verify_hf_token_detailed("")
            self._hf_verify_text = self._sam3_setup_report.message
            self._hf_verify_ok = False
            if self._handle:
                self._handle.dirty_all()
            return
        self._install_busy = True
        self._sam3_check_button_text = "Verifying..."
        self._hf_verify_text = "Verifying HuggingFace access..."
        if self._handle:
            self._handle.dirty_all()

        def _verify():
            report = verify_hf_token_detailed(token)
            self._hf_token_input = ""
            self._setup_state = check_masking_setup()
            self._sam3_setup_report = report
            self._hf_verify_text = report.message
            self._hf_verify_ok = report.access_status == "granted"
            self._sam3_edit_token = report.access_status != "granted"
            self._refresh_masking_availability()
            self._install_busy = False
            self._sam3_check_button_text = "Re-check Setup"
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_verify, daemon=True).start()

    def _on_check_sam3_setup(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._sam3_notice_text = ""
        self._install_busy = True
        self._sam3_check_button_text = "Checking..."
        if self._handle:
            self._handle.dirty_all()

        def _check():
            state = check_masking_setup()
            report = check_sam3_setup(
                setup_state=state,
                force_access_check=True,
            )
            self._setup_state = state
            self._sam3_setup_report = report
            self._hf_verify_text = report.message
            self._hf_verify_ok = report.access_status == "granted"
            self._refresh_masking_availability()
            self._install_busy = False
            self._sam3_check_button_text = "Re-check Setup"
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_check, daemon=True).start()

    def _on_toggle_sam3_token_editor(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._sam3_edit_token = not self._sam3_edit_token
        if not self._sam3_edit_token:
            self._hf_token_input = ""
            self._hf_verify_text = self._sam3_setup_report.message
        self._sam3_notice_text = ""
        if self._handle:
            self._handle.dirty_all()

    def _on_forget_hf_token(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._install_busy = True
        self._hf_verify_text = "Removing saved HuggingFace token..."
        if self._handle:
            self._handle.dirty_all()

        def _forget():
            ok = forget_hf_token()
            self._install_busy = False
            if ok:
                self._hf_token_input = ""
                self._sam3_edit_token = True
                self._sam3_check_button_text = "Check Setup"
                self._sync_setup_state()
                self._sam3_notice_text = "Saved HuggingFace token removed."
                if self._sam3_setup_report.weights_status == "present":
                    self._sam3_notice_text += " Cached SAM 3 files were kept on this machine."
                self._hf_verify_text = self._sam3_setup_report.message
                self._hf_verify_ok = False
            else:
                self._hf_verify_text = "Could not remove the saved HuggingFace token."
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_forget, daemon=True).start()

    def _on_install_default_tier(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._install_busy = True
        self._install_button_text = "Repairing Default Masking..."
        if self._handle:
            self._handle.dirty_all()

        def _install():
            def _progress(msg):
                self._install_button_text = msg
                if self._handle:
                    self._handle.dirty_all()

            ok = install_default_tier(on_output=_progress)
            self._install_busy = False
            self._sync_setup_state(auto_enable=ok)
            if ok:
                self._install_button_text = "Default Masking Repaired"
            else:
                self._install_button_text = "Repair failed — retry"
                self._enable_masking = False
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_install, daemon=True).start()

    def _on_install_video_tracking(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._install_busy = True
        self._install_button_text = "Repairing SAM v2 runtime..."
        if self._handle:
            self._handle.dirty_all()

        def _install():
            def _progress(msg):
                self._install_button_text = msg
                if self._handle:
                    self._handle.dirty_all()

            ok = install_video_tracking(on_output=_progress)
            self._install_busy = False
            self._sync_setup_state(auto_enable=ok)
            if ok:
                self._install_button_text = "SAM v2 runtime repaired"
            else:
                self._install_button_text = "Repair failed — retry"
                self._enable_masking = False
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_install, daemon=True).start()

    def _on_install_premium_tier(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._sam3_notice_text = ""
        self._install_busy = True
        self._install_button_text = "Installing SAM 3..."
        if self._handle:
            self._handle.dirty_all()

        def _install():
            last_progress = ""

            def _progress(msg):
                nonlocal last_progress
                last_progress = msg
                self._install_button_text = msg
                if self._handle:
                    self._handle.dirty_all()

            ok = install_premium_tier(on_output=_progress)
            self._install_busy = False
            state = check_masking_setup()
            self._setup_state = state
            if ok:
                self._sam3_setup_report = check_sam3_setup(setup_state=state)
                self._hf_verify_text = self._sam3_setup_report.message
                self._hf_verify_ok = self._sam3_setup_report.access_status == "granted"
                self._refresh_masking_availability(auto_enable=True)
                self._install_button_text = "SAM 3 installed"
            else:
                if state.has_sam3:
                    # Runtime installed OK; only the weights download failed.
                    # Keep stage accurate ("needs_weights") so button shows
                    # "Download Weights" and users can retry cleanly.
                    self._sam3_setup_report = check_sam3_setup(setup_state=state)
                    self._sam3_notice_text = (
                        last_progress or "SAM 3 weight download failed. Check your connection and try again."
                    )
                    self._hf_verify_text = self._sam3_setup_report.message
                    self._hf_verify_ok = False
                    self._refresh_masking_availability()
                else:
                    self._sam3_setup_report = make_sam3_install_failure_report(
                        last_progress or "SAM 3 install appears incomplete.",
                        setup_state=state,
                    )
                    self._hf_verify_text = self._sam3_setup_report.message
                    self._hf_verify_ok = False
                    self._refresh_masking_availability()
                    self._enable_masking = False
                self._install_button_text = "Retry Install"
            self._sam3_check_button_text = "Re-check Setup"
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_install, daemon=True).start()

    # ── GPU extraction opt-in ─────────────────────────────────

    _GPU_DOWNLOAD_LABEL = "Enable GPU Extraction (~1.2 GB download)"

    def _show_gpu_row(self) -> bool:
        if not self._gpu_probe_done:
            return False
        if self._gpu_state == "active":
            return False  # the "GPU" indicator conveys it; no row needed
        if self._gpu_state == "disabled":
            return self._gpu_hw_present  # offer only to NVIDIA machines
        return True  # enable_staged / error / reverted

    def _get_gpu_row_text(self) -> str:
        if self._gpu_busy:
            return self._gpu_progress_text
        state = self._gpu_state
        if state == "enable_staged":
            return "Restart LichtFeld Studio to activate GPU extraction."
        if state == "error":
            detail = self._gpu_error_detail
            if "WinError 5" in detail or "Access is denied" in detail:
                return ("Another program is using the graphics files. "
                        "Click Retry, then restart LichtFeld Studio.")
            return "GPU setup failed: %s" % detail
        if state == "reverted":
            return "GPU extraction was turned off."
        # disabled: the Enable button is the prompt (no explanatory text).
        return ""

    def _get_gpu_row_button_text(self) -> str:
        return {
            "disabled": self._GPU_DOWNLOAD_LABEL,
            "enable_staged": "Cancel",
            "error": "Retry",
            "reverted": "Re-enable",
        }.get(self._gpu_state, "")

    def _show_gpu_row_button(self) -> bool:
        return bool(self._get_gpu_row_button_text())

    def _show_gpu_copy_details(self) -> bool:
        return self._gpu_state == "error" and not self._gpu_busy

    def _on_gpu_primary_action(self, handle, event, args):
        del handle, event, args
        if self._gpu_busy:
            return
        state = self._gpu_state
        if state == "enable_staged":
            # Cancel: delete staging, return to disabled. A mis-click never
            # forces a restart.
            from ..core import gpu_extraction_install as gpu

            self._gpu_state = gpu.cancel_staged()
            if self._handle:
                self._handle.dirty_all()
        elif state in ("disabled", "error", "reverted"):
            # Enable / Retry / Re-enable: stage_enable is cache-cheap on a
            # retry (no re-download). The button itself is the prompt.
            self._start_gpu_enable()

    def _start_gpu_enable(self) -> None:
        if self._gpu_busy:
            return
        self._gpu_busy = True
        self._gpu_notice_text = ""
        self._gpu_progress_text = "Starting download..."
        if self._handle:
            self._handle.dirty_all()

        def _work():
            from ..core import gpu_extraction_install as gpu

            last = ""

            def _progress(msg):
                nonlocal last
                last = msg
                self._gpu_progress_text = msg
                if self._handle:
                    self._handle.dirty_all()

            ok = gpu.stage_enable(on_output=_progress)
            state = gpu.get_state()
            self._gpu_state = state["state"]
            self._gpu_error_detail = state.get("detail", "")
            if not ok:
                # stage_enable leaves state untouched on failure; keep the
                # failure line visible next to the Enable button.
                self._gpu_notice_text = last or "GPU setup failed."
            self._gpu_busy = False
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_work, daemon=True).start()

    def _on_gpu_copy_details(self, handle, event, args):
        del handle, event, args
        try:
            from ..core import gpu_extraction_install as gpu

            text = gpu.diagnostics_text()
            import subprocess

            subprocess.run(["clip"], input=text, text=True, check=False)
        except OSError:
            logger.exception("Copying GPU diagnostics failed")

    def _set_preset(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(PRESET_NAMES):
                self._preset_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_output_size_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(OUTPUT_SIZES):
                self._output_size_idx = idx
                self._output_size = OUTPUT_SIZES[idx]
        except (ValueError, TypeError):
            pass

    def _set_colmap_matcher(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(COLMAP_MATCHERS):
                self._colmap_matcher_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_match_budget_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(MATCH_BUDGET_TIERS):
                self._match_budget_idx = idx
                tier = MATCH_BUDGET_TIERS[idx]
                if tier != "custom":
                    self._colmap_max_num_matches = MATCH_BUDGETS[tier]
        except (ValueError, TypeError):
            pass

    def _set_colmap_max_matches(self, val):
        try:
            max_matches = int(float(val))
            max_matches = max(1024, min(80000, max_matches))
            self._colmap_max_num_matches = max_matches

            matched_tier = None
            for idx, tier in enumerate(MATCH_BUDGET_TIERS):
                if tier == "custom":
                    continue
                if MATCH_BUDGETS[tier] == max_matches:
                    matched_tier = idx
                    break

            if matched_tier is None:
                self._match_budget_idx = len(MATCH_BUDGET_TIERS) - 1
            else:
                self._match_budget_idx = matched_tier
            self._recompute_sift_preset_from_values()
        except (ValueError, TypeError):
            pass

    # ── SIFT controls ─────────────────────────────────────────

    def _get_sift_mode_key(self) -> str:
        """Return the SIFT_PRESET_MATRIX key for the current output_mode_idx."""
        if 0 <= self._output_mode_idx < len(OUTPUT_MODES):
            return OUTPUT_MODES[self._output_mode_idx]
        return "pinhole"

    def _set_sift_preset_idx(self, val):
        try:
            idx = int(val)
            if not (0 <= idx < len(SIFT_PRESETS)):
                return
            self._sift_preset_idx = idx
            if idx != SIFT_CUSTOM_IDX:
                # Normal or High: snap slider values to the new preset row and
                # remember this as the "Reset to X" target.
                self._sift_last_non_custom_preset_idx = idx
                mode_key = self._get_sift_mode_key()
                preset_name = SIFT_PRESETS[idx]
                feat, size, matches = SIFT_PRESET_MATRIX[(mode_key, preset_name)]
                self._sift_max_features = feat
                self._sift_max_image_size = size
                self._colmap_max_num_matches = matches
                # If the current extractor is not SIFT, cap max_features at
                # the extractor's documented default (e.g. 2048 for ALIKED).
                cur_feat_type = FEATURE_TYPES[self._feature_type_idx]
                feat_cap = _FEATURE_MAX_DEFAULTS.get(cur_feat_type)
                if feat_cap is not None and self._sift_max_features > feat_cap:
                    self._sift_max_features = feat_cap
            # Refresh sister bindings (max_features_str / max_image_size_str
            # plus show_sift_reset / sift_preset_label) so the slider widgets
            # actually re-read their data-value after the snap.
            if self._handle:
                self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _set_sift_max_features(self, val):
        try:
            v = int(float(val))
            v = max(1024, min(32768, v))
            if v == self._sift_max_features:
                return
            self._sift_max_features = v
            self._recompute_sift_preset_from_values()
            if self._handle:
                self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _set_sift_max_image_size(self, val):
        try:
            v = int(float(val))
            v = max(1024, min(4096, v))
            if v == self._sift_max_image_size:
                return
            self._sift_max_image_size = v
            self._recompute_sift_preset_from_values()
            if self._handle:
                self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _recompute_sift_preset_from_values(self) -> None:
        """Auto-flip _sift_preset_idx based on whether (max_features,
        max_image_size) exactly matches a Normal/High row of the active
        output mode. Editing back to a preset value flips Custom → preset.
        """
        mode_key = self._get_sift_mode_key()
        cur = (self._sift_max_features, self._sift_max_image_size, self._colmap_max_num_matches)
        if cur == SIFT_PRESET_MATRIX.get((mode_key, "normal")):
            self._sift_preset_idx = SIFT_NORMAL_IDX
            self._sift_last_non_custom_preset_idx = SIFT_NORMAL_IDX
        elif cur == SIFT_PRESET_MATRIX.get((mode_key, "high")):
            self._sift_preset_idx = SIFT_HIGH_IDX
            self._sift_last_non_custom_preset_idx = SIFT_HIGH_IDX
        else:
            # Don't update last_non_custom — preserves the Reset target.
            self._sift_preset_idx = SIFT_CUSTOM_IDX

    def _resnap_sift_for_mode(self, force: bool = False) -> None:
        """Called when output_mode changes. If user is on Normal/High,
        re-snap slider values to the new mode's matrix row. If Custom,
        leave values alone — Custom values persist across mode switches.

        Args:
            force: If True, snap to the last non-custom preset even if
                the current preset is Custom. Used on auto-detect to
                ensure fisheye defaults override stale initial values.
        """
        if self._sift_preset_idx == SIFT_CUSTOM_IDX and not force:
            return
        if force and self._sift_preset_idx == SIFT_CUSTOM_IDX:
            self._sift_preset_idx = self._sift_last_non_custom_preset_idx
        mode_key = self._get_sift_mode_key()
        preset_name = SIFT_PRESETS[self._sift_preset_idx]
        feat, size, matches = SIFT_PRESET_MATRIX[(mode_key, preset_name)]
        self._sift_max_features = feat
        self._sift_max_image_size = size
        self._colmap_max_num_matches = matches
        # Cap max_features for non-SIFT extractors
        cur_feat_type = FEATURE_TYPES[self._feature_type_idx]
        feat_cap = _FEATURE_MAX_DEFAULTS.get(cur_feat_type)
        if feat_cap is not None and self._sift_max_features > feat_cap:
            self._sift_max_features = feat_cap
        # Preset index must reflect actual capped values — without this call,
        # an extractor cap can leave _sift_preset_idx claiming "Normal" while
        # the actual feat value no longer matches that preset row.
        self._recompute_sift_preset_from_values()

    def _show_sift_reset(self) -> bool:
        """Reset pill is visible only when sliders are exposed AND the user
        has dragged values away from a preset (preset == Custom).
        """
        return self._sift_advanced_expanded and self._sift_preset_idx == SIFT_CUSTOM_IDX

    def _get_sift_preset_reset_label(self) -> str:
        """Label for the Reset button — the Normal/High target it restores to."""
        idx = self._sift_last_non_custom_preset_idx
        if 0 <= idx < len(SIFT_PRESET_LABELS):
            return SIFT_PRESET_LABELS[idx]
        return SIFT_PRESET_LABELS[SIFT_NORMAL_IDX]

    def _on_toggle_sift_advanced(self, handle, event, args):
        del handle, event, args
        self._sift_advanced_expanded = not self._sift_advanced_expanded
        if self._handle:
            self._handle.dirty_all()

    def _on_reset_sift_to_preset(self, handle, event, args):
        del handle, event, args
        target_idx = self._sift_last_non_custom_preset_idx
        if not (0 <= target_idx < len(SIFT_PRESETS)) or target_idx == SIFT_CUSTOM_IDX:
            target_idx = SIFT_NORMAL_IDX
        mode_key = self._get_sift_mode_key()
        preset_name = SIFT_PRESETS[target_idx]
        feat, size, matches = SIFT_PRESET_MATRIX[(mode_key, preset_name)]
        self._sift_max_features = feat
        self._sift_max_image_size = size
        self._colmap_max_num_matches = matches
        self._sift_preset_idx = target_idx
        if self._handle:
            self._handle.dirty_all()

    # ── COLMAP 4.1 controls ──────────────────────────────────

    def _set_feature_type_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(FEATURE_TYPES):
                old_type = FEATURE_TYPES[self._feature_type_idx]
                self._feature_type_idx = idx
                new_type = FEATURE_TYPES[idx]
                # Reset max_features to new extractor's default if user hasn't
                # manually adjusted (i.e., current value matches old default)
                old_default = _FEATURE_MAX_DEFAULTS.get(old_type, 8192)
                if self._sift_max_features == old_default:
                    new_default = _FEATURE_MAX_DEFAULTS.get(new_type, 8192)
                    self._sift_max_features = new_default
                if self._handle:
                    self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _set_matcher_type_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(MATCHER_TYPES):
                self._matcher_type_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_mapper_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(MAPPERS):
                self._mapper_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_ba_solver_idx(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(BA_SOLVERS):
                self._ba_solver_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_loop_closure(self, val):
        self._loop_closure_enabled = bool(val)
        if self._handle:
            self._handle.dirty_all()

    def _set_guided_matching(self, val):
        self._guided_matching_enabled = bool(val)
        if self._handle:
            self._handle.dirty_all()

    def _set_sift_affine_dsp(self, val):
        self._sift_affine_dsp_enabled = bool(val)
        if self._handle:
            self._handle.dirty_all()

    def _set_sequential_overlap(self, val):
        try:
            v = int(float(val))
            self._sequential_overlap = max(2, min(20, v))
        except (ValueError, TypeError):
            pass

    def _get_ba_solver_note(self) -> str:
        solver = BA_SOLVERS[self._ba_solver_idx] if 0 <= self._ba_solver_idx < len(BA_SOLVERS) else "auto"
        output_mode = self._get_output_mode()
        # Effective mapper: erp_native forces incremental regardless of the
        # dropdown (pipeline._erp_native_solve hardcodes it), so the note
        # must not describe GLOMAP there.
        is_glomap = self._mapper_idx == 1 and output_mode != "erp_native"
        # Caspar is only eligible for PINHOLE/SIMPLE_RADIAL; fisheye
        # (OPENCV_FISHEYE) and erp_native (EQUIRECTANGULAR) always route
        # global BA to Ceres-GPU on model grounds.
        caspar_eligible_model = output_mode == "pinhole"

        if solver == "ceres":
            return "Ceres CPU solver (no GPU — for debugging/comparison)."

        # GLOMAP has a single BA option (no separate local/global).
        if is_glomap:
            if solver == "caspar" and caspar_eligible_model:
                return "GLOMAP BA runs on Ceres-GPU (Caspar excluded here: converges prematurely under GLOMAP)."
            if solver == "auto" and caspar_eligible_model:
                return "GLOMAP BA runs on Ceres-GPU (Caspar excluded under GLOMAP)."
            return "GLOMAP BA runs on Ceres-GPU (cuDSS)."

        # Incremental mapper (local + global BA are distinct)
        if solver == "auto":
            if caspar_eligible_model:
                return "Hybrid: Ceres-GPU local BA + Caspar global BA."
            model = "OPENCV_FISHEYE" if output_mode == "fisheye" else "EQUIRECTANGULAR"
            return f"Hybrid: Ceres-GPU on both local and global BA ({model})."
        if solver == "caspar":
            if caspar_eligible_model:
                return "Caspar GPU BA on both local and global (fast, PINHOLE only)."
            model = "OPENCV_FISHEYE" if output_mode == "fisheye" else "EQUIRECTANGULAR"
            return f"{model} not supported by Caspar — will fall back to Ceres-GPU."
        # ceres_gpu
        return "Ceres with cuDSS GPU acceleration on both local and global BA."

    def _get_mapper_note(self) -> str:
        """Warning when GLOMAP is selected in a rig-dependent output mode."""
        if self._mapper_idx != 1:  # not GLOMAP
            return ""
        mode = self._get_output_mode()
        if mode == "fisheye":
            return ""  # fisheye native doesn't use rig constraints
        # erp_native never reaches here: the Mapper dropdown is hidden in that
        # mode (show_mapper_select), so GLOMAP can't be selected.
        return ("GLOMAP lacks rig constraints — cameras will drift relative to "
                "each other. Use Incremental mapper for rig-dependent modes.")

    def _set_fisheye_circle_margin(self, val):
        try:
            v = float(val)
            if 0 <= v <= 15:
                self._fisheye_circle_margin = v
        except (ValueError, TypeError):
            pass

    def _get_vocab_tree_status(self) -> str:
        """Status text showing which bundled vocab tree will be used."""
        feat = FEATURE_TYPES[self._feature_type_idx]
        tree = _VOCAB_TREE_BUNDLED.get(feat)
        if tree and tree.exists():
            return tree.name
        for p in _VOCAB_TREE_BUNDLED.values():
            if p.exists():
                return f"{p.name} (fallback)"
        return "Not available — no bundled tree found"

    def _apply_mode_side_effects(self, force_sift: bool = False) -> None:
        """Apply everything that must follow an output-mode change, driven by
        the derived mode string (not a raw idx). Shared by the Output Mode
        dropdown and the auto-detect transitions (split entry / dual-fisheye
        load) so all three paths stay in agreement.

        force_sift is passed through to _resnap_sift_for_mode: the dropdown
        passes False (respects a user's Custom SIFT slider state); auto-detect
        transitions pass True (deliberately replace stale Custom values so the
        detected projection's defaults win).
        """
        mode = self._get_output_mode()
        # Snap SIFT to the new mode's preset row (no-op if Custom and not forced).
        self._resnap_sift_for_mode(force=force_sift)
        # Force incremental mapper for rig-dependent modes (GLOMAP lacks the
        # constant_rigs constraint). Native fisheye is the only exempt mode.
        if mode != "fisheye" and self._mapper_idx != 0:
            self._mapper_idx = 0
        # Refresh bindings after the changes.
        if self._handle:
            self._handle.dirty_all()

    def _set_processing_idx(self, val):
        """Bound to the collapsed Native/Pinhole Output Mode dropdown. Sets the
        processing axis; projection is orthogonal and auto-detected."""
        try:
            idx = int(val)
            if idx in (NATIVE_PROCESSING_IDX, PINHOLE_PROCESSING_IDX):
                self._processing_idx = idx
                self._apply_mode_side_effects(force_sift=False)
        except (ValueError, TypeError):
            pass

    def _set_fisheye_training_output(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(FISHEYE_TRAINING_OUTPUTS):
                self._fisheye_training_output_idx = idx
                if self._handle:
                    self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _set_training_output(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(IMAGE_TRAINING_OUTPUTS):
                self._training_output_idx = idx
                if self._handle:
                    self._handle.dirty_all()
        except (ValueError, TypeError):
            pass

    def _get_output_mode_note(self) -> str:
        if self._output_mode_idx == ERP_NATIVE_OUTPUT_MODE_IDX:
            return (
                "Reconstructs the equirectangular frames directly with COLMAP's "
                "native equirectangular camera model — faster and simpler, with "
                "no pinhole reframing to align. A Pinhole or Both training "
                "output still derives pinhole crops from that native solve."
            )
        if self._output_mode_idx == FISHEYE_OUTPUT_MODE_IDX:
            if self._fisheye_single_lens:
                return (
                    "Outputs raw fisheye frames aligned via COLMAP's "
                    "OPENCV_FISHEYE camera model — one lens, one camera, "
                    "calibrated by bundle adjustment."
                )
            return (
                "Outputs raw fisheye frames aligned via COLMAP's OPENCV_FISHEYE "
                "camera model with PER_FOLDER mode. Each lens (front/back) is "
                "calibrated independently by bundle adjustment."
            )
        return ""

    def _set_output_path(self, val):
        self._output_path = str(val)

    # ── Dual fisheye state setters ────────────────────────────

    def _set_source_mode(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(SOURCE_MODES):
                self._source_mode_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_camera_family(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(CAMERA_FAMILIES):
                self._camera_family_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_keep_streams(self, val):
        if isinstance(val, bool):
            self._keep_streams = val
        else:
            self._keep_streams = str(val).lower() in ("true", "1", "yes", "on")

    def _set_keep_extracted_data(self, val):
        if isinstance(val, bool):
            self._keep_extracted_data = val
        else:
            self._keep_extracted_data = str(val).lower() in ("true", "1", "yes", "on")

    def _set_front_video_path_noop(self, val):
        # Path is set by _on_select_front_video, not by user typing.
        # This setter exists so model.bind() has a setter callback.
        del val

    def _set_back_video_path_noop(self, val):
        del val

    # ── Section toggle ────────────────────────────────────────

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])

        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        if self._doc:
            content = self._doc.get_element_by_id(f"sec-{name}")
            arrow = self._doc.get_element_by_id(f"arrow-{name}")
            if content:
                if expanding:
                    content.set_class("collapsed", False)
                    if arrow:
                        arrow.set_class("is-expanded", True)
                else:
                    content.set_class("collapsed", True)
                    if arrow:
                        arrow.set_class("is-expanded", False)

    # ── Scrub field callbacks ─────────────────────────────────

    def _get_scrub_value(self, prop: str) -> float:
        if prop == "extract_fps_str":
            return float(self._extract_fps)
        if prop == "jpeg_quality_str":
            return float(self._jpeg_quality)
        if prop == "colmap_max_matches_str":
            return float(self._colmap_max_num_matches)
        if prop == "sift_max_features_str":
            return float(self._sift_max_features)
        if prop == "sift_max_image_size_str":
            return float(self._sift_max_image_size)
        if prop == "sequential_overlap_str":
            return float(self._sequential_overlap)
        raise KeyError(prop)

    def _set_scrub_value(self, prop: str, value: float) -> None:
        if prop == "extract_fps_str":
            self._extract_fps = max(0.1, min(5.0, float(value)))
        elif prop == "jpeg_quality_str":
            self._jpeg_quality = max(50, min(100, int(value)))
        elif prop == "colmap_max_matches_str":
            self._set_colmap_max_matches(value)
        elif prop == "sift_max_features_str":
            self._set_sift_max_features(value)
        elif prop == "sift_max_image_size_str":
            self._set_sift_max_image_size(value)
        elif prop == "sequential_overlap_str":
            self._sequential_overlap = max(2, min(20, int(value)))
        else:
            raise KeyError(prop)

    # ── Video selection ───────────────────────────────────────

    @staticmethod
    def _open_video_file_dialog_ext(title: str = "Select Video") -> str:
        """File dialog accepting .mp4 / .mov / .osv / .insv / .360.

        LichtFeld's built-in `lf.ui.open_video_file_dialog()` doesn't accept
        an extension filter — `.osv` and `.insv` aren't in its allowed list,
        so dual-fisheye captures can't be selected through it.

        Tkinter isn't available in LichtFeld's embedded Python (vcpkg build
        doesn't include `_tkinter`), so we go straight to the Win32 common
        dialog via comdlg32.GetOpenFileNameW (ctypes). On non-Windows or if
        that fails for any reason, we degrade to LichtFeld's dialog.
        """
        if os.name == "nt":
            try:
                return _open_file_via_win32(
                    title=title,
                    filters=[
                        ("All supported video",
                         "*.mp4;*.mov;*.osv;*.insv;*.avi;*.mkv;*.360"),
                        ("Equirectangular video", "*.mp4;*.mov;*.avi;*.mkv"),
                        ("DJI Osmo 360 (.osv)", "*.osv"),
                        ("Insta360 (.insv)", "*.insv"),
                        ("DJI 360 series (.360)", "*.360"),
                        ("All files", "*.*"),
                    ],
                )
            except Exception as exc:
                logger.warning(
                    "Win32 GetOpenFileNameW failed (%s); falling back to "
                    "LichtFeld dialog (no .osv/.insv support).", exc,
                )
        return lf.ui.open_video_file_dialog()

    def _on_select_video(self, handle, event, args):
        del handle, event, args
        path = self._open_video_file_dialog_ext("Select 360° Video")
        if not path:
            return
        self._load_video(path)

    # ── Projection-first input menu (Plan 03 / Option B) ──────────

    def _on_select_projection_erp(self, handle, event, args):
        del handle, event, args
        self._select_projection("erp")

    def _on_select_projection_fisheye(self, handle, event, args):
        del handle, event, args
        self._select_projection("fisheye")

    def _select_projection(self, projection: str):
        """Explicit projection choice — the first step of the input menu.

        Replaces the old auto-detect (extension / 2:1 aspect guess) as the
        thing that decides ERP vs fisheye. _projection stays the canonical
        field the rest of the panel reads (plan D1).
        """
        self._projection = projection
        self._projection_chosen = True
        if projection == "fisheye":
            # Processing is moot for fisheye (output is chosen by Training
            # output); mirror the old fisheye transitions.
            self._processing_idx = PINHOLE_PROCESSING_IDX
        # Transplanted from the retired image-folder projection dropdown
        # (AR-N2): reset a stale pre-existing mask choice when the mode
        # leaves ERP+Pinhole.
        if self._get_output_mode() != "pinhole" and self._mask_source == "preexisting":
            self._mask_source = "generate"
        self._error_message = ""
        self._apply_mode_side_effects(force_sift=True)

    def _on_back_to_projection(self, handle, event, args):
        """The single reset affordance (plan D3a): clear ALL source state and
        return to the Initial [Equirectangular]/[Fisheye] step."""
        del handle, event, args
        self._projection_chosen = False
        # Deterministic Initial state (matches the on-load mockup):
        # provisional ERP + Pinhole processing until the next explicit choice.
        self._projection = "erp"
        self._processing_idx = PINHOLE_PROCESSING_IDX
        self._source_mode_idx = 0  # "container"
        # Single-file state
        self._video_loaded = False
        self._video_path = ""
        self._video_info = None
        self._video_info_text = ""
        self._camera_family_detected = None
        self._fisheye_single_lens = False
        # Split state
        self._front_video_path = ""
        self._back_video_path = ""
        # Image-folder state
        self._image_dir = ""
        self._fisheye_dir = ""
        self._front_dir = ""
        self._back_dir = ""
        self._preexisting_mask_dir = ""
        self._error_message = ""
        self._apply_mode_side_effects(force_sift=True)

    def _on_enter_split_mode(self, handle, event, args):
        """Switch to fisheye + split-files input mode without opening any
        dialog. The user clicks Choose on each picker separately afterward.
        """
        del handle, event, args
        # Auto-detect transition → fisheye projection, Pinhole processing
        # (processing is moot for fisheye; output is chosen by Training output).
        self._projection = "fisheye"
        self._processing_idx = PINHOLE_PROCESSING_IDX
        self._source_mode_idx = 1  # "split"
        # Reset any prior single-file state so the UI doesn't show stale info.
        self._video_loaded = False
        self._video_path = ""
        self._video_info = None
        self._video_info_text = ""
        self._camera_family_detected = None
        self._fisheye_single_lens = False
        self._error_message = ""
        # Forced SIFT resnap + sequential matcher + incremental mapper + dirty
        # bindings, all via the shared helper (called last so dirty_all sees
        # the final state).
        self._apply_mode_side_effects(force_sift=True)

    # ── Image-folder input mode (source_mode_idx == 2) ──────────

    def _on_enter_image_folder_mode(self, handle, event, args):
        """Switch to image-folder input: the user supplies a folder of
        already-extracted frames instead of a video (AR-N1)."""
        del handle, event, args
        self._source_mode_idx = 2  # "image_folder"
        # Clear stale single-file / split state.
        self._video_loaded = False
        self._video_path = ""
        self._video_info = None
        self._video_info_text = ""
        self._front_video_path = ""
        self._back_video_path = ""
        self._camera_family_detected = None
        self._fisheye_single_lens = False
        self._error_message = ""
        # Projection was chosen upstream in the input menu (plan D6) — do NOT
        # reset it here.
        self._apply_mode_side_effects(force_sift=True)

    def _on_set_folder_mode_one(self, handle, event, args):
        """Folders toggle: one shared folder of front_/back_ files."""
        self._set_fisheye_folder_mode("0")

    def _on_set_folder_mode_two(self, handle, event, args):
        """Folders toggle: separate front + back folders."""
        self._set_fisheye_folder_mode("1")

    def _set_fisheye_folder_mode(self, val):
        """One (single folder of front_/back_ files) vs Two (front + back)."""
        try:
            idx = int(val)
        except (TypeError, ValueError):
            return
        self._fisheye_folder_mode = "one" if idx == 0 else "two"
        if self._handle:
            self._handle.dirty_all()

    def _set_mask_source(self, val):
        try:
            idx = int(val)
        except (TypeError, ValueError):
            return
        self._mask_source = (
            ("generate", "preexisting", "none")[idx] if 0 <= idx < 3 else "generate"
        )
        if self._handle:
            self._handle.dirty_all()

    def _pick_folder_into(self, attr: str, title: str, set_output: bool):
        """Open a folder dialog and store the result in self.<attr>."""
        path = lf.ui.open_folder_dialog(title=title, start_dir=getattr(self, attr) or "")
        if not path:
            return
        setattr(self, attr, path)
        if set_output and not self._output_path:
            # Default Output Path to the PARENT of the chosen frames folder:
            # picking Folder/Images yields Folder/ -- the supported
            # source-in-a-subfolder layout (the run writes Folder/colmap/,
            # Folder/masks/, Folder/metadata/ beside the source). Never
            # overwrites an Output Path the user already set.
            self._output_path = str(Path(path).parent)
        if self._handle:
            self._handle.dirty_all()

    def _clear_attr(self, attr: str):
        setattr(self, attr, "")
        if self._handle:
            self._handle.dirty_all()

    def _on_select_image_dir(self, handle, event, args):
        del handle, event, args
        self._pick_folder_into("_image_dir", "Select Image Folder", set_output=True)

    def _on_clear_image_dir(self, handle, event, args):
        del handle, event, args
        self._clear_attr("_image_dir")

    def _on_select_fisheye_dir(self, handle, event, args):
        del handle, event, args
        self._pick_folder_into("_fisheye_dir", "Select Fisheye Frames Folder", set_output=True)

    def _on_clear_fisheye_dir(self, handle, event, args):
        del handle, event, args
        self._clear_attr("_fisheye_dir")

    def _on_select_front_dir(self, handle, event, args):
        del handle, event, args
        self._pick_folder_into("_front_dir", "Select Front Lens Frames Folder", set_output=True)

    def _on_clear_front_dir(self, handle, event, args):
        del handle, event, args
        self._clear_attr("_front_dir")

    def _on_select_back_dir(self, handle, event, args):
        del handle, event, args
        self._pick_folder_into("_back_dir", "Select Back Lens Frames Folder", set_output=False)

    def _on_clear_back_dir(self, handle, event, args):
        del handle, event, args
        self._clear_attr("_back_dir")

    def _on_select_preexisting_mask_dir(self, handle, event, args):
        del handle, event, args
        self._pick_folder_into("_preexisting_mask_dir", "Select Pre-existing Masks Folder", set_output=False)

    def _on_clear_preexisting_mask_dir(self, handle, event, args):
        del handle, event, args
        self._clear_attr("_preexisting_mask_dir")

    def _load_video(self, path: str):
        self._error_message = ""
        # Import from the stdlib-only module — never from core.pipeline,
        # which pulls the full cv2/torch/pycolmap stack. A broken heavy
        # dependency must not be able to break video selection (#6/#8).
        from ..core.input_detect import detect_input_type

        input_type, family = detect_input_type(path)
        info = None
        if self._projection == "fisheye" and input_type == "dual_fisheye":
            self._fisheye_single_lens = False
            self._processing_idx = PINHOLE_PROCESSING_IDX
            self._source_mode_idx = 0  # container mode by default
            self._camera_family_detected = family
            if family in CAMERA_FAMILIES:
                self._camera_family_idx = CAMERA_FAMILIES.index(family)
            self._apply_mode_side_effects(force_sift=True)
        elif (
            self._projection == "fisheye"
            and (input_type, family) == ("erp", None)
        ):
            try:
                info = VideoAnalyzer().analyze(path)
            except Exception as exc:
                logger.error("Failed to analyze video: %s", exc)
                self._error_message = f"Video analysis failed: {exc}"
                if self._handle:
                    self._handle.dirty_all()
                return
            if info.is_erp:
                self._error_message = (
                    "This looks like a 2:1 equirectangular video — go ‹ Back "
                    "and choose Equirectangular."
                )
                if self._handle:
                    self._handle.dirty_all()
                return
            # 1:1 gate: the circle mask assumes a centered, frame-inscribed
            # image circle, which only holds for square sensors (Osmo /
            # Insta360). Non-square fisheye needs circle auto-fit + preview
            # (follow-up plan item) before it can mask correctly.
            if abs(info.width / max(info.height, 1) - 1.0) > 0.02:
                self._error_message = (
                    "Single-lens fisheye currently requires 1:1 video "
                    "(e.g. 3840×3840) — non-square fisheye is planned."
                )
                if self._handle:
                    self._handle.dirty_all()
                return
            self._fisheye_single_lens = True
            self._source_mode_idx = 0
            self._camera_family_detected = None
            self._processing_idx = PINHOLE_PROCESSING_IDX
            self._apply_mode_side_effects(force_sift=True)
        elif input_type == "dual_fisheye":
            self._error_message = (
                "This is a dual-fisheye container — go ‹ Back and choose "
                "Fisheye → Video."
            )
            if self._handle:
                self._handle.dirty_all()
            return
        else:
            self._camera_family_detected = None

        try:
            if info is None:
                analyzer = VideoAnalyzer()
                info = analyzer.analyze(path)
            # ERP content gate (plan D5b): the user chose Equirectangular, so
            # a non-2:1 video (e.g. a 1:1 single-lens fisheye .mp4) would
            # silently produce a wrong EQUIRECTANGULAR solve. Refuse
            # atomically BEFORE any loaded state is set.
            if self._projection == "erp" and not info.is_erp:
                self._error_message = (
                    "This video isn't 2:1 equirectangular — if it's a "
                    "fisheye capture, go ‹ Back and choose Fisheye."
                )
                if self._handle:
                    self._handle.dirty_all()
                return
            self._video_info = info
            self._video_path = path
            self._video_loaded = True

            # Build info text
            duration_str = VideoAnalyzer.get_duration_formatted(info)
            erp_tag = " (ERP)" if info.is_erp else ""
            self._video_info_text = (
                f"{info.width}\u00d7{info.height}{erp_tag} | "
                f"{info.fps:.1f} fps | {duration_str}"
            )

            # Apply recommended extract rate (convert interval → FPS)
            self._extract_fps = round(1.0 / max(0.1, info.recommended_interval), 1)

        except Exception as exc:
            logger.error("Failed to analyze video: %s", exc)
            self._error_message = f"Video analysis failed: {exc}"

        if self._handle:
            self._handle.dirty_all()

    def _on_clear_completion_report(self, handle, event, args):
        del handle, event, args
        self._completion_summary = ""
        self._completion_report = ""
        if self._handle:
            self._handle.dirty_all()

    # ── Split-mode video selection (fisheye + source_mode=split) ──

    def _probe_split_lens_info(self) -> None:
        """Populate ``_video_info`` from a fisheye split-video lens so the Frame
        Extraction estimate works in two-file mode. Front and back lenses are
        synchronized, so the front lens (or back, if only it is set) is
        representative. Clears when neither lens is set. Leaves the FPS the user
        may have set untouched.
        """
        lens = self._front_video_path or self._back_video_path
        if not lens:
            self._video_info = None
            return
        try:
            self._video_info = VideoAnalyzer().analyze(lens)
        except Exception as exc:
            logger.error("Failed to analyze split lens video: %s", exc)
            self._video_info = None

    def _reject_container_in_lens_picker(self, path: str) -> bool:
        """Refuse a dual-fisheye container in the per-lens pickers (plan
        D5c): Dual Lens Video takes pre-split front/back files; containers
        belong to Fisheye → Video. Returns True when refused."""
        from ..core.input_detect import detect_input_type

        if detect_input_type(path)[0] != "dual_fisheye":
            return False
        self._error_message = (
            "This is a dual-fisheye container — use Video (not Dual Lens "
            "Video) for .osv/.insv files. Dual Lens Video takes pre-split "
            "front and back .mp4s."
        )
        if self._handle:
            self._handle.dirty_all()
        return True

    def _on_select_front_video(self, handle, event, args):
        del handle, event, args
        path = self._open_video_file_dialog_ext("Select Front Lens Video")
        if not path:
            return
        if self._reject_container_in_lens_picker(path):
            return
        self._error_message = ""
        self._front_video_path = path
        self._probe_split_lens_info()
        if self._handle:
            self._handle.dirty_all()

    def _on_select_back_video(self, handle, event, args):
        del handle, event, args
        path = self._open_video_file_dialog_ext("Select Back Lens Video")
        if not path:
            return
        if self._reject_container_in_lens_picker(path):
            return
        self._error_message = ""
        self._back_video_path = path
        self._probe_split_lens_info()
        if self._handle:
            self._handle.dirty_all()

    def _on_clear_front_video(self, handle, event, args):
        del handle, event, args
        self._front_video_path = ""
        self._probe_split_lens_info()
        if self._handle:
            self._handle.dirty_all()

    def _on_clear_back_video(self, handle, event, args):
        del handle, event, args
        self._back_video_path = ""
        self._probe_split_lens_info()
        if self._handle:
            self._handle.dirty_all()

    # ── Output path browsing ──────────────────────────────────

    def _on_browse_output(self, handle, event, args):
        del handle, event, args
        path = lf.ui.open_folder_dialog(
            title="Select Output Folder",
            start_dir=self._output_path or "",
        )
        if path:
            self._output_path = path
            if self._handle:
                self._handle.dirty_all()

    # ── Pipeline execution ────────────────────────────────────

    def _on_run_pipeline(self, handle, event, args):
        del handle, event, args
        self._import_after = True
        self._start_pipeline()

    def _on_run_pipeline_only(self, handle, event, args):
        del handle, event, args
        self._import_after = False
        self._start_pipeline()

    def _validate_image_folder_selection(self):
        """Return an error string if the image-folder selection is incomplete
        or a source overlaps the output dir (B6); else None."""
        from ..core.frame_source import assert_source_reads_safe
        if self._projection == "erp":
            dirs = {"Image folder": self._image_dir}
        elif self._fisheye_folder_mode == "one":
            dirs = {"Fisheye frames folder": self._fisheye_dir}
        else:
            dirs = {
                "Front lens folder": self._front_dir,
                "Back lens folder": self._back_dir,
            }
        for label, d in dirs.items():
            if not d:
                return f"{label} is not selected"
        if self._mask_source == "preexisting":
            if self._get_output_mode() != "pinhole":
                return ("Pre-existing masks are only available for "
                        "Equirectangular + Pinhole output")
            if not self._preexisting_mask_dir:
                return "Pre-existing masks folder is not selected"
        # Pre-run guard: frames are read in place, so reject only a source that
        # overlaps a folder the run writes to (colmap/, masks/, metadata/).
        if self._output_path:
            checks = list(dirs.values())
            if self._mask_source == "preexisting":
                checks.append(self._preexisting_mask_dir)
            for d in checks:
                try:
                    assert_source_reads_safe(d, self._output_path)
                except ValueError as exc:
                    return str(exc)
        return None

    def _start_pipeline(self):
        if self._is_processing:
            return
        # In Fisheye + split mode the user provides front + back files instead of
        # one container; the standard "video loaded" check doesn't apply.
        # In image-folder mode, no video is needed; only the output directory.
        is_fisheye_split = (
            self._output_mode_idx in _FISHEYE_MODES
            and self._source_mode_idx == 1
        )
        is_image_folder = self._source_mode_idx == 2
        if is_image_folder:
            # Image-folder mode (AR-5): validate the required folder(s) for the
            # current projection / folder-mode + B6 path-safety; no video needed.
            _img_err = self._validate_image_folder_selection()
            if _img_err:
                self._error_message = _img_err
                if self._handle:
                    self._handle.dirty_all()
                return
        elif is_fisheye_split:
            if not self._front_video_path or not self._back_video_path:
                self._error_message = (
                    "Split mode requires both front and back video files"
                )
                if self._handle:
                    self._handle.dirty_all()
                return
        elif not self._video_loaded or not self._video_path:
            self._error_message = "No video loaded"
            if self._handle:
                self._handle.dirty_all()
            return
        if not self._output_path:
            self._error_message = "No output path set"
            if self._handle:
                self._handle.dirty_all()
            return

        self._error_message = ""
        preset_name = self._get_selected_preset_name()
        output_mode = self._get_output_mode()
        if (
            output_mode == "fisheye"
            and self._source_mode_idx == 0
            and self._video_path
        ):
            from ..core.input_detect import detect_input_type

            if (
                not self._fisheye_single_lens
                and detect_input_type(self._video_path)[0] != "dual_fisheye"
            ):
                self._error_message = (
                    "This isn't a recognized dual-fisheye container — choose the "
                    "fisheye video again."
                )
                if self._handle:
                    self._handle.dirty_all()
                return
        colmap_matcher = COLMAP_MATCHERS[self._colmap_matcher_idx] if 0 <= self._colmap_matcher_idx < len(COLMAP_MATCHERS) else "sequential"
        match_budget_tier = MATCH_BUDGET_TIERS[self._match_budget_idx] if 0 <= self._match_budget_idx < len(MATCH_BUDGET_TIERS) else "custom"

        prompts = [p.strip() for p in self._mask_prompts_str.split(",") if p.strip()]
        sharpness_preset = EXTRACT_SHARPNESS_PRESETS[self._extract_sharpness_idx]
        sharpness_modes = ["none", "basic", "better", "best"]
        blur_metric = BLUR_METRICS[self._blur_metric_idx]["value"]

        # The UI now exposes only the SAM 3 masking path.
        masking_method = "sam3_cubemap"
        mask_backend = "sam3"

        # Image-folder mask source is authoritative (M1/M2): only "generate"
        # runs SAM 3; "preexisting"/"none" skip generation.
        if is_image_folder:
            enable_masking = (self._mask_source == "generate") and self._masking_available
        else:
            enable_masking = self._enable_masking and self._masking_available

        # Heavy import (cv2 / torch / pycolmap load here). Guarded so a
        # broken dependency surfaces its real error in the panel instead
        # of dying silently in the data-model event handler (#6/#8).
        try:
            from ..core.pipeline import PipelineConfig, PipelineJob
        except ImportError as exc:
            self._error_message = (
                f"A required dependency failed to load: {exc} — "
                "try reinstalling the plugin, or report this message on GitHub."
            )
            logger.error("Pipeline import failed: %s", exc, exc_info=True)
            if self._handle:
                self._handle.dirty_all()
            return

        config = PipelineConfig(
            video_path=self._video_path,
            output_dir=self._output_path,
            interval=1.0 / max(0.1, self._extract_fps),
            extraction_sharpness=sharpness_modes[self._extract_sharpness_idx],
            blur_metric=blur_metric,
            scene_threshold=sharpness_preset["scene_threshold"],
            blur_scale_width=sharpness_preset["scale_width"],
            all_frames=self._extract_all_frames,
            expected_frame_count=(
                self._video_info.frame_count if self._video_info else 0
            ),
            quality=self._jpeg_quality,
            enable_masking=enable_masking,
            masking_method=masking_method,
            enable_diagnostics=self._enable_diagnostics,
            mask_prompts=prompts if prompts else ["person"],
            mask_backend=mask_backend,
            preset_name=preset_name,
            output_size=self._output_size,
            jpeg_quality=self._jpeg_quality,
            colmap_matcher=colmap_matcher,
            colmap_match_budget_tier=match_budget_tier,
            colmap_max_num_matches=self._colmap_max_num_matches,
            sift_preset=SIFT_PRESETS[self._sift_preset_idx],
            sift_max_features=self._sift_max_features,
            sift_max_image_size=self._sift_max_image_size,
            # COLMAP 4.1 features
            colmap_feature_type=FEATURE_TYPES[self._feature_type_idx],
            colmap_matcher_type=MATCHER_TYPES[self._matcher_type_idx],
            colmap_mapper=MAPPERS[self._mapper_idx],
            colmap_ba_solver=BA_SOLVERS[self._ba_solver_idx],
            fisheye_circle_margin=self._fisheye_circle_margin,
            vocab_tree_path="",  # auto-resolved from feature type by ColmapRunner
            loop_detection=self._loop_closure_enabled,
            colmap_sequential_overlap=self._sequential_overlap,
            colmap_guided_matching=self._guided_matching_enabled,
            colmap_sift_affine_dsp=self._sift_affine_dsp_enabled,
            output_mode=output_mode,
            fisheye_training_output=(
                self._get_fisheye_training_output()
                if output_mode == "fisheye"
                else "native"
            ),
            # training_output applies to image-folder Native (both projections)
            # and to video ERP Native; Pinhole output is a single dataset.
            training_output=(
                self._get_training_output()
                if (output_mode == "erp_native"
                    or (self._source_mode_idx == 2
                        and self._processing_idx == NATIVE_PROCESSING_IDX))
                else "native"
            ),
            # Dual fisheye fields
            input_type=(
                "single_fisheye"
                if (output_mode == "fisheye" and self._fisheye_single_lens)
                else "dual_fisheye" if output_mode == "fisheye" else "erp"
            ),
            camera_family=(
                None
                if (output_mode == "fisheye" and self._fisheye_single_lens)
                else self._camera_family_detected
                if (output_mode == "fisheye" and self._source_mode_idx == 0)
                else (
                    CAMERA_FAMILIES[self._camera_family_idx]
                    if (output_mode == "fisheye"
                        and 0 <= self._camera_family_idx < len(CAMERA_FAMILIES))
                    else None
                )
            ),
            source_mode=(
                "container"
                if (output_mode == "fisheye" and self._fisheye_single_lens)
                else SOURCE_MODES[self._source_mode_idx]
                if (output_mode == "fisheye"
                    and 0 <= self._source_mode_idx < len(SOURCE_MODES))
                else "container"
            ),
            # Image-folder source (AR-C0-1): the pipeline gates on these dirs
            # being set, uniformly across projections — not on source_mode.
            image_source_dir=(
                self._image_dir
                if (is_image_folder and self._projection == "erp")
                else self._fisheye_dir
                if (is_image_folder and self._projection == "fisheye"
                    and self._fisheye_folder_mode == "one")
                else ""
            ),
            image_front_dir=(
                self._front_dir
                if (is_image_folder and self._projection == "fisheye"
                    and self._fisheye_folder_mode == "two")
                else ""
            ),
            image_back_dir=(
                self._back_dir
                if (is_image_folder and self._projection == "fisheye"
                    and self._fisheye_folder_mode == "two")
                else ""
            ),
            mask_source=self._mask_source if is_image_folder else "generate",
            preexisting_mask_dir=(
                self._preexisting_mask_dir if is_image_folder else ""
            ),
            front_video_path=self._front_video_path,
            back_video_path=self._back_video_path,
            keep_streams=self._keep_streams,
            keep_extracted_data=self._keep_extracted_data,
        )

        self._is_processing = True
        self._processing_stage = "Starting..."
        self._processing_status = ""
        self._processing_progress = 0.0
        self._processing_log_lines = []
        self._completion_summary = ""
        self._completion_report = ""
        if output_mode == "erp_native":
            self._append_processing_log("Preset: native EQUIRECTANGULAR (no reframe)")
        elif output_mode == "fisheye":
            family = (
                self._camera_family_detected
                or (CAMERA_FAMILIES[self._camera_family_idx]
                    if 0 <= self._camera_family_idx < len(CAMERA_FAMILIES)
                    else "unknown")
            )
            family_label = (
                CAMERA_FAMILY_LABELS[CAMERA_FAMILIES.index(family)]
                if family in CAMERA_FAMILIES else family
            )
            self._append_processing_log(
                "Preset: OPENCV_FISHEYE x PER_FOLDER (no rig)"
            )
            self._append_processing_log(
                f"Training output: {self._get_fisheye_training_output_label()}"
            )
            self._append_processing_log(f"Camera family: {family_label}")
            if self._source_mode_idx == 1:
                self._append_processing_log(
                    f"Source: split files (front + back)"
                )
        else:
            self._append_processing_log(f"Preset: {PRESET_LABELS[self._preset_idx]}")
        self._append_processing_log(
            f"Output mode: {OUTPUT_MODE_LABELS[self._output_mode_idx]}"
        )
        self._append_processing_log(f"Output: {self._output_path}")
        self._append_processing_log("Pipeline queued.")

        self._job = PipelineJob(
            config,
            on_progress=self._on_pipeline_progress,
            on_complete=self._on_pipeline_complete,
        )
        self._job.start()

        if self._handle:
            self._handle.dirty_all()

    def _on_cancel(self, handle, event, args):
        del handle, event, args
        if self._job:
            self._job.cancel()
            self._processing_status = "Cancelling..."
            self._append_processing_log("Cancellation requested.")
            if self._handle:
                self._handle.dirty_all()

    # ── Pipeline callbacks (called from background thread) ────

    # Stage display names with step numbers
    _STAGE_LABELS = {
        "extraction":  "Step 1/5 \u2014 Frame Extraction",
        "masking":     "Step 2/5 \u2014 Operator Masking",
        "reframe":     "Step 3/5 \u2014 Reframing",
        "rig_config":  "Step 4/5 \u2014 Rig Config",
        "colmap":      "Step 4/5 \u2014 COLMAP Alignment",
        "output":      "Step 5/5 \u2014 Writing Output",
        "complete":    "Complete",
    }

    # Image-folder runs never extract frames from a video -- they read/stage a
    # folder -- so "Frame Extraction" would be a lie. Covers EVERY stage key the
    # image-folder flows emit (ERP pinhole also emits reframe / overlap_masks /
    # rig_config); any key absent falls through to the raw stage name.
    _IMAGE_FOLDER_STAGE_LABELS = {
        "staging":       "Step 1/5 \u2014 Reading Image Folder",
        "masking":       "Step 2/5 \u2014 Operator Masking",
        "reframe":       "Step 3/5 \u2014 Reframing",
        "overlap_masks": "Step 3/5 \u2014 Overlap Masks",
        "rig_config":    "Step 4/5 \u2014 Rig Config",
        "colmap":        "Step 4/5 \u2014 COLMAP Alignment",
        "output":        "Step 5/5 \u2014 Writing Output",
        "complete":      "Complete",
    }

    # Item count patterns for throughput tracking
    _RE_REFRAME = re.compile(r"Reframing (\d+)/(\d+)")
    _RE_EXTRACT = re.compile(r"Extracting (\d+)/(\d+)")
    _RE_SCORING = re.compile(r"Scoring (\d+)/(\d+)")

    # COLMAP substage detection
    _COLMAP_SUBSTAGES = {
        "feature": "feature_extraction",
        "sift": "feature_extraction",
        "match": "matching",
        "map": "mapping",
        "incremental": "mapping",
    }

    def _on_pipeline_progress(self, stage: str, percent: float, message: str):
        now = time.time()

        with self._pending_lock:
            prev_stage = self._timing_current_stage
            _labels = (
                self._IMAGE_FOLDER_STAGE_LABELS
                if self._source_mode_idx == 2 else self._STAGE_LABELS
            )
            self._processing_stage = _labels.get(stage, stage)
            self._processing_progress = percent
            self._processing_status = message
            if stage != prev_stage:
                self._append_processing_log(self._processing_stage)
            self._append_processing_log(message)

            # -- Timing accumulation --

            # Initialize on first call
            if self._timing_t0 == 0:
                self._timing_t0 = now

            # Detect stage transition
            if stage != self._timing_current_stage:
                # Close previous stage
                if self._timing_current_stage and self._timing_current_stage in self._timing_stages:
                    prev = self._timing_stages[self._timing_current_stage]
                    prev["ended"] = now
                    # Close last COLMAP substage
                    if self._timing_current_stage == "colmap" and self._timing_colmap_substage:
                        prev["substeps"][self._timing_colmap_substage] = now - self._timing_colmap_substage_t0
                        self._timing_colmap_substage = ""

                # Open new stage
                self._timing_stages[stage] = {
                    "started": now,
                    "ended": 0.0,
                    "items": 0,
                    "substeps": {},
                }
                self._timing_current_stage = stage

            # Parse item counts
            if stage in self._timing_stages:
                rec = self._timing_stages[stage]
                if stage == "reframe":
                    m = self._RE_REFRAME.search(message)
                    if m:
                        rec["items"] = int(m.group(1))
                elif stage == "extraction":
                    m = self._RE_EXTRACT.search(message) or self._RE_SCORING.search(message)
                    if m:
                        rec["items"] = int(m.group(1))

            # Track COLMAP substeps
            if stage == "colmap":
                msg_lower = message.lower()
                for keyword, substage_name in self._COLMAP_SUBSTAGES.items():
                    if keyword in msg_lower:
                        if substage_name != self._timing_colmap_substage:
                            # Close previous substep
                            if self._timing_colmap_substage and stage in self._timing_stages:
                                self._timing_stages[stage]["substeps"][self._timing_colmap_substage] = (
                                    now - self._timing_colmap_substage_t0
                                )
                            self._timing_colmap_substage = substage_name
                            self._timing_colmap_substage_t0 = now
                        break

    def _on_pipeline_complete(self, result: PipelineResult):
        with self._pending_lock:
            # Close last open stage
            now = time.time()
            if self._timing_current_stage and self._timing_current_stage in self._timing_stages:
                s = self._timing_stages[self._timing_current_stage]
                if s["ended"] == 0:
                    s["ended"] = now
                if self._timing_current_stage == "colmap" and self._timing_colmap_substage:
                    s["substeps"][self._timing_colmap_substage] = now - self._timing_colmap_substage_t0

            self._pending_result = result

    # ── Main-thread result consumption ────────────────────────

    def _consume_pending_result(self) -> bool:
        with self._pending_lock:
            result = self._pending_result
            self._pending_result = None
        if result is None:
            return False
        try:
            self._apply_result(result)
        except Exception as exc:
            logger.exception("Failed to apply pipeline result")
            self._job = None
            self._is_processing = False
            self._processing_stage = ""
            self._processing_status = ""
            self._processing_progress = 0.0
            self._processing_log_lines = []
            self._error_message = result.error or f"Failed to render diagnostics: {exc}"
            summary, report = self._build_minimal_failure_report(
                result,
                result.elapsed_sec,
                report_error=f"{type(exc).__name__}: {exc}",
            )
            self._completion_summary = summary
            self._completion_report = report

            try:
                matcher, tier = self._get_matcher_and_tier()
                self._write_failure_timing_json(
                    result,
                    result.elapsed_sec,
                    {},
                    matcher,
                    tier,
                    report_error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception("Failed to write minimal failure timing.json")
        return True

    def _apply_result(self, result: PipelineResult):
        self._job = None
        self._is_processing = False
        self._processing_stage = ""
        self._processing_status = ""
        self._processing_progress = 0.0

        # -- Build timing summary --
        stage_names = {
            "extraction": "Extraction",
            "reframe": "Reframe",
            "rig_config": "Rig Config",
            "colmap": "COLMAP",
            "output": "Output",
            "masking": "Masking",
        }
        timing_dict = {}
        for key, data in self._timing_stages.items():
            if key == "complete":
                continue
            elapsed = data["ended"] - data["started"] if data["ended"] > 0 else 0

            entry: dict = {"elapsed_sec": round(elapsed, 3)}
            if data["items"] > 0:
                rate = data["items"] / elapsed if elapsed > 0 else 0
                entry["items"] = data["items"]
                entry["rate_per_sec"] = round(rate, 2)
            if data["substeps"]:
                entry["substeps"] = {k: round(v, 3) for k, v in data["substeps"].items()}
            timing_dict[key] = entry

        total = result.elapsed_sec
        matcher, tier = self._get_matcher_and_tier()

        if result.success:
            self._error_message = ""

            summary, report = self._build_completion_report(
                result, stage_names, timing_dict, total
            )
            self._completion_summary = summary
            self._completion_report = report

            # Print to Python console
            print(f"\n{'=' * 60}")
            print(f"360 Plugin \u2014 Timing Report")
            print(f"{'=' * 60}")
            print(self._completion_report)
            print(f"{'=' * 60}\n")

            # Write timing.json into metadata/
            try:
                metadata_dir = Path(self._output_path) / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                timing_path = metadata_dir / "timing.json"
                mask_diagnostics_summary = self._load_mask_diagnostics_summary(
                    result.mask_diagnostics_path
                )
                timing_output = {
                    "success": True,
                    "total_sec": round(total, 3),
                    "stages": timing_dict,
                    "result": {
                        "source_frames": result.num_source_frames,
                        "output_images": result.num_output_images,
                        "aligned_cameras": result.num_aligned_cameras,
                        "matcher": matcher,
                        "match_budget_tier": tier,
                        "max_num_matches": self._colmap_max_num_matches,
                        "preset_geometry": result.preset_signature,
                        "views_per_frame": result.views_per_frame,
                        "registered_frames": result.num_registered_frames,
                        "complete_rig_frames": result.num_complete_frames,
                        "partial_rig_frames": result.num_partial_frames,
                        "dropped_rig_frames": max(
                            result.num_source_frames - result.num_registered_frames, 0
                        ),
                        "registered_images_by_view": result.registered_images_by_view,
                        "expected_images_by_view": result.expected_images_by_view,
                        "partial_frame_examples": result.partial_frame_examples,
                        "dropped_frame_examples": result.dropped_frame_examples,
                        "mask_backend": result.mask_backend_name,
                        "mask_diagnostics_path": result.mask_diagnostics_path,
                        "mask_diagnostics_summary": mask_diagnostics_summary,
                        "video_backend": result.video_backend_name,
                        "used_fallback_video_backend": result.used_fallback_video_backend,
                        "video_backend_error": result.video_backend_error,
                        "gpu_extraction": result.gpu_extraction,
                        "masking_timers": result.masking_timers,
                    },
                }
                timing_path.write_text(json.dumps(timing_output, indent=2))
            except Exception as exc:
                logger.warning("Failed to write timing.json: %s", exc)

            # Log summary
            logger.info(
                "Pipeline complete: %d source frames -> %d output images, "
                "%d registered frames, %d aligned images in %.1fs",
                result.num_source_frames,
                result.num_output_images,
                result.num_registered_frames,
                result.num_aligned_cameras,
                total,
            )

            if self._import_after and result.dataset_path:
                try:
                    # ERP native passes the transforms.json FILE (BlenderLoader:
                    # EQUIRECTANGULAR cameras; its flat masks resolve fine).
                    # Fisheye native imports the dataset DIRECTORY instead, so
                    # ColmapLoader wins: its sparse/0 IS the OPENCV_FISHEYE
                    # reconstruction, and ColmapLoader resolves front/back masks
                    # by full relative path. BlenderLoader cannot — it matches
                    # masks by bare filename, and front/000001 + back/000001
                    # collide ("ambiguous across the dataset mask folders").
                    if result.output_mode == "fisheye":
                        dataset_base_path, import_output_path = self._resolve_transforms_directory_import_target(
                            result.dataset_path
                        )
                    elif result.output_mode == "erp_native":
                        dataset_base_path, import_output_path = self._resolve_erp_import_target(
                            result.dataset_path
                        )
                    else:
                        dataset_base_path, import_output_path = self._resolve_import_target(
                            result.dataset_path
                        )
                    lf.load_file(
                        dataset_base_path,
                        is_dataset=True,
                        output_path=import_output_path,
                    )
                    self._processing_status = (
                        f"Import requested: {Path(dataset_base_path).name}"
                    )
                    logger.info(
                        "Requested dataset import: base_path=%s output_path=%s",
                        dataset_base_path,
                        import_output_path,
                    )
                except Exception as exc:
                    logger.error("Failed to import dataset: %s", exc)
                    print(f"[360] IMPORT FAILED: {exc}")
                    import traceback; traceback.print_exc()
                    self._error_message = f"Import failed: {exc}"
        else:
            self._error_message = result.error or "Pipeline failed"
            report_error = ""
            try:
                summary, report = self._build_failure_report(
                    result, stage_names, timing_dict, total
                )
            except Exception as exc:
                report_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to build detailed failure report")
                summary, report = self._build_minimal_failure_report(
                    result,
                    total,
                    report_error=report_error,
                )
            self._completion_summary = summary
            self._completion_report = report
            logger.error("Pipeline failed: %s", self._error_message)

            print(f"\n{'=' * 60}")
            print("360 Plugin — Failure Report")
            print(f"{'=' * 60}")
            print(self._completion_report)
            print(f"{'=' * 60}\n")

            try:
                self._write_failure_timing_json(
                    result,
                    total,
                    timing_dict,
                    matcher,
                    tier,
                    report_error=report_error,
                )
            except Exception as exc:
                logger.warning("Failed to write failure timing.json: %s", exc)

        # Reset timing state for next run
        self._timing_stages = {}
        self._timing_current_stage = ""
        self._timing_t0 = 0.0
        self._timing_colmap_substage = ""
        self._timing_colmap_substage_t0 = 0.0
