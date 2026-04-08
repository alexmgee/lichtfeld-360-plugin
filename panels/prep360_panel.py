# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Plugin panel — full UI with data model, threading, and import."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

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
from ..core.pipeline import PipelineConfig, PipelineJob, PipelineResult
from ..core.presets import VIEW_PRESETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset ordering — indices match the <select> in the RML
# ---------------------------------------------------------------------------

PRESET_NAMES = [
    "default",
    "cubemap",
]

PRESET_LABELS = [
    "Default (16 views)",
    "Cubemap (6 views)",
]

CUBEMAP_PRESET_IDX = PRESET_NAMES.index("cubemap")

COLMAP_MATCHERS = ["sequential", "exhaustive"]
MATCH_BUDGET_TIERS = ["fast", "balanced", "default", "high", "custom"]

OUTPUT_SIZES = [960, 1280, 1536, 1920]

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
}

# Extraction sharpness tiers
# - None: no analysis, fixed-interval extraction
# - Basic: ~10 candidates per interval, OpenCV scoring
# - Best: score every frame with OpenCV
EXTRACT_SHARPNESS_PRESETS = [
    {"label": "None",  "scale_width": 0,   "scene_threshold": 0.0},
    {"label": "Basic", "scale_width": 640, "scene_threshold": 0.3},
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
    "default": "8+8 two-ring layout at +/-35°, no poles",
}

# ---------------------------------------------------------------------------
# Section management
# ---------------------------------------------------------------------------

SECTIONS = ["extraction", "masking", "reframe", "quality"]


class Plugin360Panel(lf.ui.Panel):
    id = "plugin360.main"
    label = "360 Plugin"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 10100
    template = str(Path(__file__).resolve().with_name("prep360_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
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
        self._extract_sharpness_idx: int = 2  # default: Best
        self._blur_metric_idx: int = 0        # default: Tenengrad

        # Masking
        self._setup_state: MaskingSetupState = check_masking_setup()
        self._sam3_setup_report: Sam3SetupReport = check_sam3_setup(
            setup_state=self._setup_state
        )
        self._masking_method_idx: int = 0  # 0=Default, 1=SAM 3
        self._masking_available: bool = self._setup_state.fullcircle_ready
        self._enable_masking: bool = False
        self._enable_diagnostics: bool = False
        self._mask_prompts_str: str = "person"
        self._hf_token_input: str = ""
        self._hf_verify_text: str = self._sam3_setup_report.message
        self._hf_verify_ok: bool = False
        self._install_busy: bool = False
        self._install_button_text: str = "Repair Default Masking"
        self._sam3_check_button_text: str = "Check Setup"
        self._sam3_edit_token: bool = False
        self._sam3_notice_text: str = ""
        self._sam3_install_in_progress: bool = False
        self._sam3_install_progress_phase: float = 0.0

        # Reframe
        self._preset_idx: int = 0
        self._output_size_idx: int = 3  # index into OUTPUT_SIZES, default 1920
        self._output_size: int = OUTPUT_SIZES[3]
        self._jpeg_quality: int = 97
        self._colmap_matcher_idx: int = 1  # default: exhaustive
        self._match_budget_idx: int = 2  # default
        self._colmap_max_num_matches: int = MATCH_BUDGETS["default"]

        # Output
        self._output_path: str = ""

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
        model.bind_func("est_frames_text", self._get_est_frames_text)

        # -- Masking --
        model.bind("masking_method_idx", lambda: str(self._masking_method_idx), self._set_masking_method)
        model.bind("enable_masking", lambda: self._enable_masking, self._set_enable_masking)
        model.bind("enable_diagnostics", lambda: self._enable_diagnostics, self._set_enable_diagnostics)
        model.bind("mask_prompts_str", lambda: self._mask_prompts_str, self._set_mask_prompts)
        model.bind("hf_token_input", lambda: self._hf_token_input, self._set_hf_token_input)
        model.bind_func("hf_verify_text", lambda: self._hf_verify_text)
        model.bind_func("masking_available", lambda: self._masking_available)
        model.bind_func("masking_backend_text", self._get_masking_backend_text)
        model.bind_func("fullcircle_status_text", self._get_fullcircle_status_text)
        model.bind_func("fullcircle_action_text", lambda: self._install_button_text)
        # FullCircle conditional states
        model.bind_func("show_masking_fullcircle", lambda: self._masking_method_idx == 0)
        model.bind_func("show_masking_install", lambda: not self._setup_state.fullcircle_ready)
        model.bind_func("show_masking_controls", lambda: self._setup_state.fullcircle_ready)
        # SAM 3 conditional states
        model.bind_func("show_masking_sam3_setup", lambda: self._masking_method_idx == 1 and not self._setup_state.sam3_ready)
        model.bind_func("show_masking_sam3_ready", lambda: self._masking_method_idx == 1 and self._setup_state.sam3_ready)
        model.bind_func("sam3_status_message", lambda: self._sam3_setup_report.message)
        model.bind_func("show_sam3_notice", lambda: bool(self._sam3_notice_text))
        model.bind_func("sam3_notice_text", lambda: self._sam3_notice_text)
        model.bind_func("sam3_next_action_text", lambda: self._sam3_setup_report.next_action)
        model.bind_func("sam3_token_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.token_status))
        model.bind_func("sam3_access_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.access_status))
        model.bind_func("sam3_runtime_status_text", lambda: self._format_sam3_status(self._sam3_setup_report.runtime_status))
        model.bind_func("sam3_weights_status_text", self._get_sam3_weights_status_text)
        model.bind_func("show_sam3_cached_weights_note", self._show_sam3_cached_weights_note)
        model.bind_func("sam3_cached_weights_note_text", self._get_sam3_cached_weights_note_text)
        model.bind_func("sam3_reassurance_text", self._get_sam3_reassurance_text)
        model.bind_func("sam3_check_button_text", lambda: self._sam3_check_button_text)
        model.bind_func("sam3_install_button_text", self._get_sam3_install_button_text)
        model.bind_func("sam3_install_disabled", self._get_sam3_install_disabled)
        model.bind_func("show_sam3_install_button", self._show_sam3_install_button)
        model.bind_func("show_sam3_install_progress", self._show_sam3_install_progress)
        model.bind_func("sam3_install_progress_value", lambda: f"{self._sam3_install_progress_phase:.4f}")
        model.bind_func("show_sam3_check_button", self._show_sam3_check_button)
        model.bind_func("show_sam3_external_actions", self._show_sam3_external_actions)
        model.bind_func("show_sam3_local_actions", self._show_sam3_local_actions)
        model.bind_func("show_sam3_saved_token_notice", self._show_sam3_saved_token_notice)
        model.bind_func("sam3_saved_token_text", self._get_sam3_saved_token_text)
        model.bind_func("show_sam3_account_step", self._show_sam3_account_step)
        model.bind_func("show_sam3_request_access_step", self._show_sam3_request_access_step)
        model.bind_func("show_sam3_token_editor", self._show_sam3_token_editor)
        model.bind_func("show_sam3_token_actions", self._show_sam3_token_actions)
        model.bind_func("sam3_change_token_button_text", self._get_sam3_change_token_button_text)
        model.bind_func("install_button_text", lambda: self._install_button_text)
        model.bind_func("install_busy", lambda: self._install_busy)

        # -- Reframe --
        model.bind("preset_idx", lambda: str(self._preset_idx), self._set_preset)
        model.bind_func("coverage_text", self._get_coverage_text)
        model.bind_func("total_output_text", self._get_total_output_text)
        model.bind("output_size_idx", lambda: str(self._output_size_idx), self._set_output_size_idx)
        model.bind("jpeg_quality_str", lambda: str(self._jpeg_quality), self._set_jpeg_quality)
        model.bind("colmap_matcher_idx", lambda: str(self._colmap_matcher_idx), self._set_colmap_matcher)
        model.bind("match_budget_idx", lambda: str(self._match_budget_idx), self._set_match_budget_idx)
        model.bind("colmap_max_matches_str", lambda: str(self._colmap_max_num_matches), self._set_colmap_max_matches)
        model.bind_func("match_budget_text", self._get_match_budget_text)

        # -- Output --
        model.bind_func("output_path_display", lambda: self._output_path or "(not set)")
        model.bind_func("dataset_summary_text", self._get_dataset_summary)

        # -- Processing --
        model.bind_func("show_processing", lambda: self._is_processing)
        model.bind_func("show_idle", lambda: not self._is_processing)
        model.bind_func("processing_stage_text", lambda: self._processing_stage)
        model.bind_func("processing_status_text", lambda: self._processing_status)
        model.bind_func("processing_progress_value", lambda: f"{self._processing_progress / 100:.4f}")
        model.bind_func("processing_progress_pct", lambda: f"{self._processing_progress:.1f}%")
        model.bind_func("processing_recent_text", self._get_processing_recent_text)
        model.bind_func("show_error", lambda: bool(self._error_message))
        model.bind_func("error_text", lambda: self._error_message)
        model.bind_func("completion_summary_text", self._get_completion_summary_text)
        model.bind_func("completion_report_text", self._get_completion_report_text)

        # -- Events --
        model.bind_event("select_video", self._on_select_video)
        model.bind_event("clear_video", self._on_clear_video)
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
        model.bind_func("show_video_tracking_install",
                         lambda: False)

        self._handle = model.get_handle()

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        self._doc = doc
        self._scrub_fields.mount(doc)

    def on_unmount(self, doc):
        self._scrub_fields.unmount()
        self._doc = None

    def on_update(self, doc):
        if not self._handle:
            return False

        dirty = self._consume_pending_result()
        if self._scrub_fields.sync_all():
            dirty = True

        if self._sam3_install_in_progress:
            self._sam3_install_progress_phase += 0.08
            if self._sam3_install_progress_phase > 1.0:
                self._sam3_install_progress_phase = 0.0
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
            self._extract_fps,
            self._extract_sharpness_idx,
            self._blur_metric_idx,
            self._masking_method_idx,
            self._masking_available,
            self._enable_masking,
            self._enable_diagnostics,
            self._mask_prompts_str,
            self._hf_token_input,
            self._hf_verify_text,
            self._install_busy,
            self._install_button_text,
            self._sam3_check_button_text,
            self._sam3_notice_text,
            self._sam3_install_in_progress,
            round(self._sam3_install_progress_phase, 3),
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
            self._output_size_idx,
            self._jpeg_quality,
            self._colmap_matcher_idx,
            self._match_budget_idx,
            self._colmap_max_num_matches,
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

    def _get_current_view_config(self):
        name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "default"
        return VIEW_PRESETS.get(name, VIEW_PRESETS["default"])

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

    def _get_fullcircle_status_text(self) -> str:
        if self._setup_state.fullcircle_ready:
            return "Default masking is ready on this install. No HuggingFace account required."
        return "Default masking runtime is missing or damaged. Repair the local masking runtime."

    def _get_sam3_reassurance_text(self) -> str:
        if self._setup_state.fullcircle_ready:
            return "Default masking remains available while SAM 3 setup is incomplete."
        return "Default masking runtime needs repair on this install."

    def _get_sam3_saved_token_text(self) -> str:
        if self._sam3_setup_report.access_status == "granted":
            return (
                "Using a saved HuggingFace token from this machine. "
                "Change or forget it below if you want to test first-run setup."
            )
        return "Using a saved HuggingFace token from this machine."

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
        if self._install_busy and self._masking_method_idx == 1:
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

    def _show_sam3_install_progress(self) -> bool:
        return self._sam3_install_in_progress and self._masking_method_idx == 1

    def _show_sam3_check_button(self) -> bool:
        return (
            self._setup_state.has_token
            or self._sam3_setup_report.access_status in {"pending", "granted", "network_error"}
            or self._sam3_setup_report.runtime_status in {"installed", "broken"}
        )

    def _show_sam3_external_actions(self) -> bool:
        return True

    def _show_sam3_local_actions(self) -> bool:
        return (
            self._show_sam3_token_actions()
            or self._show_sam3_token_editor()
            or self._show_sam3_check_button()
            or self._show_sam3_install_button()
        )

    def _show_sam3_saved_token_notice(self) -> bool:
        return self._setup_state.has_token and not self._sam3_edit_token

    def _show_sam3_account_step(self) -> bool:
        return not self._setup_state.has_token and self._show_sam3_token_editor()

    def _show_sam3_request_access_step(self) -> bool:
        return self._sam3_setup_report.access_status != "granted"

    def _show_sam3_token_editor(self) -> bool:
        return (
            self._sam3_edit_token
            or not self._setup_state.has_token
            or self._sam3_setup_report.token_status in {"missing", "invalid"}
        )

    def _show_sam3_token_actions(self) -> bool:
        return self._setup_state.has_token or self._sam3_edit_token

    def _get_sam3_change_token_button_text(self) -> str:
        if self._sam3_edit_token:
            return "Cancel Token Change"
        return "Change Token"

    def _get_est_frames_text(self) -> str:
        if not self._video_info:
            return "Select a video source"
        interval = 1.0 / max(0.1, self._extract_fps)
        base = VideoAnalyzer.estimate_frame_count(self._video_info, interval)
        preset = EXTRACT_SHARPNESS_PRESETS[self._extract_sharpness_idx]
        if preset["scene_threshold"] > 0:
            extra = int(base * 0.2)
            return f"Estimated frames   ~{base}\u2013{base + extra}"
        return f"Estimated frames   ~{base}"

    def _get_coverage_text(self) -> str:
        name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "default"
        return COVERAGE_DESCRIPTIONS.get(name, "")

    def _get_total_output_text(self) -> str:
        vc = self._get_current_view_config()
        views = vc.total_views()
        if not self._video_info:
            return f"{views} views per frame"
        interval = 1.0 / max(0.1, self._extract_fps)
        frames = VideoAnalyzer.estimate_frame_count(self._video_info, interval)
        total = views * frames
        return f"{views} views \u00d7 {frames} frames = {total:,} images"

    def _get_dataset_summary(self) -> str:
        if not self._video_loaded:
            return "No video loaded"
        vc = self._get_current_view_config()
        views = vc.total_views()
        interval = 1.0 / max(0.1, self._extract_fps)
        frames = VideoAnalyzer.estimate_frame_count(self._video_info, interval) if self._video_info else 0
        total = views * frames
        return f"Pinhole (COLMAP) | {total:,} images | {self._output_size}px"

    def _get_match_budget_text(self) -> str:
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
        tier_label = tier.title()
        return (
            f"{tier_label} match limit on {matcher} matching: "
            f"keeps up to {self._colmap_max_num_matches:,} matches per image pair. "
            "Higher limits preserve more correspondences but cost more time and memory."
        )

    def _get_completion_summary_text(self) -> str:
        return self._completion_summary or "No diagnostics yet."

    def _get_completion_report_text(self) -> str:
        return self._completion_report or "Run Process Only or Process & Import to capture timing and registration diagnostics."

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

        timing_path = Path(self._output_path) / "timing.json"
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
            lines.append(f"Frames extracted: {result.num_source_frames}")
            lines.append(f"Views per frame: {result.views_per_frame}")
            lines.append(f"Images written: {result.num_output_images}")
            lines.append(
                f"Registered frames: {result.num_registered_frames}/{result.num_source_frames}"
            )
            lines.append(f"Complete rig frames: {result.num_complete_frames}")
            lines.append(f"Dropped rig frames: {dropped_frames}")
            if result.dropped_frame_examples:
                lines.append(
                    f"Dropped examples: {', '.join(result.dropped_frame_examples)}"
                )
            if result.num_partial_frames > 0:
                lines.append(f"Partial rig frames: {result.num_partial_frames}")
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

    def _set_jpeg_quality(self, val):
        try:
            v = int(float(val))
            if 50 <= v <= 100:
                self._jpeg_quality = v
        except (ValueError, TypeError):
            pass

    def _set_enable_masking(self, val):
        self._enable_masking = bool(val) and self._masking_available

    def _set_enable_diagnostics(self, val):
        self._enable_diagnostics = bool(val)

    def _set_mask_prompts(self, val):
        self._mask_prompts_str = str(val)

    def _selected_masking_ready(self) -> bool:
        if self._masking_method_idx == 1:
            return self._setup_state.sam3_ready
        return self._setup_state.fullcircle_ready

    def _refresh_masking_availability(self, *, auto_enable: bool = False) -> None:
        ready = self._selected_masking_ready()
        self._masking_available = ready
        if auto_enable:
            self._enable_masking = ready
        else:
            self._enable_masking = self._enable_masking and ready

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

    def _set_masking_method(self, val):
        try:
            idx = int(val)
            if idx in (0, 1):
                self._masking_method_idx = idx
                if idx == 1 and self._preset_idx != CUBEMAP_PRESET_IDX:
                    self._preset_idx = CUBEMAP_PRESET_IDX
                    self._sam3_notice_text = (
                        "SAM 3 uses the Cubemap preset in this version, "
                        "so the preset was switched to Cubemap."
                    )
                self._refresh_masking_availability()
        except (ValueError, TypeError):
            pass

    def _get_masking_backend_text(self):
        if self._masking_method_idx == 1:
            return "SAM 3 (large model)"
        if self._setup_state.fullcircle_ready:
            return "Default (YOLO + SAM v1 + SAM v2)"
        if self._setup_state.default_tier_ready:
            return "Default runtime needs repair"
        return "Default runtime unavailable"

    def _set_hf_token_input(self, val):
        self._hf_token_input = str(val)

    # ── Masking setup event handlers ─────────────────────────

    def _on_open_hf_signup(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/")

    def _on_open_hf_model(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/facebook/sam3")

    def _on_open_hf_tokens(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/settings/tokens")

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
        self._sam3_install_in_progress = True
        self._sam3_install_progress_phase = 0.08
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
            self._sam3_install_in_progress = False
            self._sam3_install_progress_phase = 0.0
            state = check_masking_setup()
            self._setup_state = state
            if ok:
                self._sam3_setup_report = check_sam3_setup(setup_state=state)
                self._hf_verify_text = self._sam3_setup_report.message
                self._hf_verify_ok = self._sam3_setup_report.access_status == "granted"
                self._refresh_masking_availability(auto_enable=True)
                self._install_button_text = "SAM 3 installed"
            else:
                self._sam3_setup_report = make_sam3_install_failure_report(
                    last_progress or "SAM 3 install appears incomplete.",
                    setup_state=state,
                )
                self._hf_verify_text = self._sam3_setup_report.message
                self._hf_verify_ok = False
                self._refresh_masking_availability()
                self._install_button_text = "Retry Install"
                self._enable_masking = False
            self._sam3_check_button_text = "Re-check Setup"
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_install, daemon=True).start()

    def _set_preset(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(PRESET_NAMES):
                if self._masking_method_idx == 1 and idx != CUBEMAP_PRESET_IDX:
                    self._preset_idx = CUBEMAP_PRESET_IDX
                    self._sam3_notice_text = (
                        "SAM 3 uses the Cubemap preset in this version. "
                        "The preset stays on Cubemap while SAM 3 is selected."
                    )
                else:
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
        except (ValueError, TypeError):
            pass

    def _set_output_path(self, val):
        self._output_path = str(val)

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
        raise KeyError(prop)

    def _set_scrub_value(self, prop: str, value: float) -> None:
        if prop == "extract_fps_str":
            self._extract_fps = max(0.1, min(5.0, float(value)))
        elif prop == "jpeg_quality_str":
            self._jpeg_quality = max(50, min(100, int(value)))
        elif prop == "colmap_max_matches_str":
            self._set_colmap_max_matches(value)
        else:
            raise KeyError(prop)

    # ── Video selection ───────────────────────────────────────

    def _on_select_video(self, handle, event, args):
        del handle, event, args
        path = lf.ui.open_video_file_dialog()
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str):
        self._error_message = ""
        try:
            analyzer = VideoAnalyzer()
            info = analyzer.analyze(path)
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

            # Auto-set output path next to video
            if not self._output_path:
                video_dir = Path(path).parent
                stem = Path(path).stem
                self._output_path = str(video_dir / f"{stem}_LFS360")

        except Exception as exc:
            logger.error("Failed to analyze video: %s", exc)
            self._error_message = f"Video analysis failed: {exc}"

        if self._handle:
            self._handle.dirty_all()

    def _on_clear_video(self, handle, event, args):
        del handle, event, args
        self._video_loaded = False
        self._video_path = ""
        self._video_info = None
        self._video_info_text = ""
        self._error_message = ""
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

    def _start_pipeline(self):
        if self._is_processing:
            return
        if not self._video_loaded or not self._video_path:
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
        preset_name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "default"
        colmap_matcher = COLMAP_MATCHERS[self._colmap_matcher_idx] if 0 <= self._colmap_matcher_idx < len(COLMAP_MATCHERS) else "sequential"
        match_budget_tier = MATCH_BUDGET_TIERS[self._match_budget_idx] if 0 <= self._match_budget_idx < len(MATCH_BUDGET_TIERS) else "custom"

        prompts = [p.strip() for p in self._mask_prompts_str.split(",") if p.strip()]
        sharpness_preset = EXTRACT_SHARPNESS_PRESETS[self._extract_sharpness_idx]
        sharpness_modes = ["none", "basic", "best"]
        blur_metric = BLUR_METRICS[self._blur_metric_idx]["value"]

        # Pin masking_method and mask_backend explicitly by UI selection —
        # do not pass active_backend through, which can auto-promote to sam3
        masking_method = "sam3_cubemap" if self._masking_method_idx == 1 else "fullcircle"
        mask_backend = "sam3" if self._masking_method_idx == 1 else "yolo_sam1"

        config = PipelineConfig(
            video_path=self._video_path,
            output_dir=self._output_path,
            interval=1.0 / max(0.1, self._extract_fps),
            extraction_sharpness=sharpness_modes[self._extract_sharpness_idx],
            blur_metric=blur_metric,
            scene_threshold=sharpness_preset["scene_threshold"],
            blur_scale_width=sharpness_preset["scale_width"],
            quality=self._jpeg_quality,
            enable_masking=self._enable_masking and self._masking_available,
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
            output_mode="pinhole",
        )

        self._is_processing = True
        self._processing_stage = "Starting..."
        self._processing_status = ""
        self._processing_progress = 0.0
        self._processing_log_lines = []
        self._completion_summary = ""
        self._completion_report = ""
        self._append_processing_log(f"Preset: {PRESET_LABELS[self._preset_idx]}")
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
            self._processing_stage = self._STAGE_LABELS.get(stage, stage)
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

            # Write timing.json
            try:
                timing_path = Path(self._output_path) / "timing.json"
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
