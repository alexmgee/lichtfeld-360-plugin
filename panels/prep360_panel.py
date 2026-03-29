# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Camera preprocessing panel — full UI with data model, threading, and import."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import lichtfeld as lf

try:
    from lfs_plugins import ScrubFieldController, ScrubFieldSpec
except ImportError:
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec

from ..core.analyzer import VideoAnalyzer, VideoInfo
from ..core.masker import is_masking_available
# Masking setup UI — disabled pending python3.dll fix (see plugin_masking_blocker.md)
# from ..core.setup_checks import (
#     MaskingSetupState, check_masking_setup, verify_hf_token, download_model_weights,
#     install_torch_to_plugin_venv, install_sam3_to_plugin_venv,
# )
from ..core.pipeline import PipelineConfig, PipelineJob, PipelineResult
from ..core.presets import VIEW_PRESETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset ordering — indices match the <select> in the RML
# ---------------------------------------------------------------------------

PRESET_NAMES = [
    "cubemap",
    "balanced",
    "standard",
    "dense",
    "full",
]

PRESET_LABELS = [
    "Cubemap (6 views)",
    "Balanced (9 views)",
    "Standard (13 views)",
    "Dense (17 views)",
    "Full (26 views)",
]

COLMAP_PRESETS = ["sequential", "exhaustive"]
OUTPUT_MODES = ["pinhole", "erp"]

OUTPUT_SIZES = [960, 1280, 1536, 1920]

# Scrub field specifications — keys MUST match the data-value attribute in the RML
SCRUB_FIELD_SPECS = {
    "extract_fps_str": ScrubFieldSpec(
        min_value=0.1, max_value=5.0, step=0.1, fmt="%.1f", data_type=float,
    ),
    "jpeg_quality_str": ScrubFieldSpec(
        min_value=50.0, max_value=100.0, step=1.0, fmt="%d", data_type=int,
    ),
}

# Extraction quality presets: (label, use_blur, scale_width, scene_threshold)
# - Interval: no blur analysis, just extract at FPS rate
# - Sharp: fast blur check at very low res, no scene splitting
# - Sharpest: full blur analysis with scene-aware chunking
# - Sharpest+: thorough analysis at higher resolution
EXTRACT_QUALITY_PRESETS = [
    {"label": "None",    "use_blur": False, "scale_width": 0,    "scene_threshold": 0.0},
    {"label": "Fast",    "use_blur": True,  "scale_width": 480,  "scene_threshold": 0.0},
    {"label": "Normal",  "use_blur": True,  "scale_width": 640,  "scene_threshold": 0.5},
    {"label": "Maximum", "use_blur": True,  "scale_width": 1280, "scene_threshold": 0.3},
]

# Human-readable coverage descriptions per preset
COVERAGE_DESCRIPTIONS = {
    "cubemap": "4 horizon, 1 top, 1 bottom",
    "balanced": "6 horizon, 2 below, zenith",
    "standard": "8 horizon, 4 below, zenith",
    "dense": "8 horizon, 4 above, 4 below, zenith",
    "full": "8 horizon, 8 above, 8 below, zenith, nadir",
}

# ---------------------------------------------------------------------------
# Section management
# ---------------------------------------------------------------------------

SECTIONS = ["video", "extraction", "masking", "reframe", "output"]


class Prep360Panel(lf.ui.Panel):
    id = "lichtfeld_360_camera.prep360"
    label = "360 Camera"
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
        self._extract_quality_idx: int = 2  # default: Sharpest (standard)

        # Masking (coming soon — setup UI disabled pending python3.dll fix)
        self._enable_masking: bool = False
        self._masking_available: bool = is_masking_available()
        self._mask_prompts_str: str = "person"
        # self._setup_state: MaskingSetupState = check_masking_setup()
        # self._hf_token_input: str = ""
        # self._hf_verify_text: str = ""
        # self._hf_verify_ok: bool = False
        # self._install_busy: bool = False
        # self._install_button_text: str = "Install PyTorch + SAM 3.1"

        # Reframe
        self._preset_idx: int = 0
        self._output_size_idx: int = 3  # index into OUTPUT_SIZES, default 1920
        self._output_size: int = OUTPUT_SIZES[3]
        self._jpeg_quality: int = 95
        self._colmap_preset_idx: int = 0

        # Output
        self._output_mode_idx: int = 0
        self._output_path: str = ""

        # Processing state
        self._is_processing: bool = False
        self._processing_stage: str = ""
        self._processing_status: str = ""
        self._processing_progress: float = 0.0
        self._error_message: str = ""
        self._import_after: bool = False

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
        model = ctx.create_data_model("lichtfeld_360_camera")
        if model is None:
            return

        # -- Video state --
        model.bind_func("show_no_video", lambda: not self._video_loaded)
        model.bind_func("show_video_loaded", lambda: self._video_loaded)
        model.bind_func("video_info_text", lambda: self._video_info_text)
        model.bind_func("video_path_text", lambda: self._video_path or "No file selected")

        # -- Extraction (range sliders need two-way bindings for data-value) --
        model.bind("extract_fps_str", lambda: f"{self._extract_fps:.1f}", self._set_extract_fps)
        model.bind("extract_quality_idx", lambda: str(self._extract_quality_idx), self._set_extract_quality)
        model.bind_func("est_frames_text", self._get_est_frames_text)

        # -- Masking (coming soon — setup UI disabled) --
        model.bind("enable_masking", lambda: self._enable_masking, self._set_enable_masking)
        model.bind("mask_prompts_str", lambda: self._mask_prompts_str, self._set_mask_prompts)

        # -- Reframe --
        model.bind("preset_idx", lambda: str(self._preset_idx), self._set_preset)
        model.bind_func("coverage_text", self._get_coverage_text)
        model.bind_func("total_output_text", self._get_total_output_text)
        model.bind("output_size_idx", lambda: str(self._output_size_idx), self._set_output_size_idx)
        model.bind("jpeg_quality_str", lambda: str(self._jpeg_quality), self._set_jpeg_quality)
        model.bind("colmap_preset_idx", lambda: str(self._colmap_preset_idx), self._set_colmap_preset)

        # -- Output --
        model.bind("output_mode_idx", lambda: str(self._output_mode_idx), self._set_output_mode)
        model.bind_func("output_path_display", lambda: self._output_path or "(not set)")
        model.bind_func("dataset_summary_text", self._get_dataset_summary)

        # -- Processing --
        model.bind_func("show_processing", lambda: self._is_processing)
        model.bind_func("show_idle", lambda: not self._is_processing)
        model.bind_func("processing_stage_text", lambda: self._processing_stage)
        model.bind_func("processing_status_text", lambda: self._processing_status)
        model.bind_func("processing_progress_value", lambda: f"{self._processing_progress / 100:.4f}")
        model.bind_func("processing_progress_pct", lambda: f"{self._processing_progress:.1f}%")
        model.bind_func("show_error", lambda: bool(self._error_message))
        model.bind_func("error_text", lambda: self._error_message)

        # -- Events --
        model.bind_event("select_video", self._on_select_video)
        model.bind_event("clear_video", self._on_clear_video)
        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("run_pipeline", self._on_run_pipeline)
        model.bind_event("run_pipeline_only", self._on_run_pipeline_only)
        model.bind_event("cancel_pipeline", self._on_cancel)
        model.bind_event("toggle_section", self._on_toggle_section)
        # Masking setup events — disabled pending python3.dll fix
        # model.bind_event("open_hf_signup", self._on_open_hf_signup)
        # model.bind_event("open_hf_model", self._on_open_hf_model)
        # model.bind_event("open_hf_tokens", self._on_open_hf_tokens)
        # model.bind_event("verify_hf_token", self._on_verify_hf_token)
        # model.bind_event("install_masking_deps", self._on_install_deps)
        # model.bind_event("download_weights", self._on_download_weights)

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
            self._extract_quality_idx,
            self._enable_masking,
            self._mask_prompts_str,
            self._preset_idx,
            self._output_size_idx,
            self._jpeg_quality,
            self._colmap_preset_idx,
            self._output_mode_idx,
            self._output_path,
            self._is_processing,
            self._processing_stage,
            self._processing_progress,
            self._error_message,
        )

    # ── Computed text helpers ─────────────────────────────────

    def _get_current_view_config(self):
        name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "cubemap"
        return VIEW_PRESETS.get(name, VIEW_PRESETS["cubemap"])

    def _get_est_frames_text(self) -> str:
        if not self._video_info:
            return "Load a video first"
        interval = 1.0 / max(0.1, self._extract_fps)
        base = VideoAnalyzer.estimate_frame_count(self._video_info, interval)
        preset = EXTRACT_QUALITY_PRESETS[self._extract_quality_idx]
        if preset["scene_threshold"] > 0:
            extra = int(base * 0.2)
            return f"~{base}\u2013{base + extra} frames"
        return f"~{base} frames"

    def _get_coverage_text(self) -> str:
        name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "cubemap"
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
        mode = OUTPUT_MODES[self._output_mode_idx] if 0 <= self._output_mode_idx < len(OUTPUT_MODES) else "pinhole"
        vc = self._get_current_view_config()
        views = vc.total_views()
        interval = 1.0 / max(0.1, self._extract_fps)
        frames = VideoAnalyzer.estimate_frame_count(self._video_info, interval) if self._video_info else 0
        total = views * frames
        mode_label = "Pinhole (COLMAP)" if mode == "pinhole" else "ERP"
        return f"{mode_label} | {total:,} images | {self._output_size}px"

    # ── Setters (called by data model on user input) ──────────

    def _set_extract_fps(self, val):
        try:
            v = float(val)
            if 0.1 <= v <= 5.0:
                self._extract_fps = v
        except (ValueError, TypeError):
            pass

    def _set_extract_quality(self, val):
        try:
            v = int(float(val))
            if 0 <= v < len(EXTRACT_QUALITY_PRESETS):
                self._extract_quality_idx = v
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
        self._enable_masking = bool(val)

    def _set_mask_prompts(self, val):
        self._mask_prompts_str = str(val)

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
        webbrowser.open("https://huggingface.co/facebook/sam3.1")

    def _on_open_hf_tokens(self, handle, event, args):
        del handle, event, args
        import webbrowser
        webbrowser.open("https://huggingface.co/settings/tokens")

    def _on_verify_hf_token(self, handle, event, args):
        del handle, event, args
        token = self._hf_token_input.strip()
        if not token:
            self._hf_verify_text = "Please paste a token"
            self._hf_verify_ok = False
            if self._handle:
                self._handle.dirty_all()
            return
        self._hf_verify_text = "Verifying..."
        if self._handle:
            self._handle.dirty_all()

        def _verify():
            if verify_hf_token(token):
                self._hf_verify_text = "Access verified"
                self._hf_verify_ok = True
                self._setup_state = check_masking_setup()
            else:
                self._hf_verify_text = "Access denied or pending. Check your email and try again."
                self._hf_verify_ok = False
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_verify, daemon=True).start()

    def _on_install_deps(self, handle, event, args):
        del handle, event, args
        if self._install_busy:
            return
        self._install_busy = True
        self._install_button_text = "Installing PyTorch..."
        if self._handle:
            self._handle.dirty_all()

        def _install_all():
            ok = install_torch_to_plugin_venv()
            if ok:
                self._install_button_text = "Installing SAM 3.1..."
                if self._handle:
                    self._handle.dirty_all()
                ok = install_sam3_to_plugin_venv()
            self._install_busy = False
            if ok:
                self._install_button_text = "Installed"
            else:
                self._install_button_text = "Install failed \u2014 retry"
            self._setup_state = check_masking_setup()
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_install_all, daemon=True).start()

    def _on_download_weights(self, handle, event, args):
        del handle, event, args

        def _download():
            download_model_weights()
            self._setup_state = check_masking_setup()
            if self._handle:
                self._handle.dirty_all()

        threading.Thread(target=_download, daemon=True).start()

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

    def _set_colmap_preset(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(COLMAP_PRESETS):
                self._colmap_preset_idx = idx
        except (ValueError, TypeError):
            pass

    def _set_output_mode(self, val):
        try:
            idx = int(val)
            if 0 <= idx < len(OUTPUT_MODES):
                self._output_mode_idx = idx
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
        elif prop == "jpeg_quality_str":
            return float(self._jpeg_quality)
        raise KeyError(prop)

    def _set_scrub_value(self, prop: str, value: float) -> None:
        if prop == "extract_fps_str":
            self._extract_fps = max(0.1, min(5.0, float(value)))
        elif prop == "jpeg_quality_str":
            self._jpeg_quality = max(50, min(100, int(value)))
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
        preset_name = PRESET_NAMES[self._preset_idx] if 0 <= self._preset_idx < len(PRESET_NAMES) else "cubemap"
        colmap_preset = COLMAP_PRESETS[self._colmap_preset_idx] if 0 <= self._colmap_preset_idx < len(COLMAP_PRESETS) else "normal"
        output_mode = OUTPUT_MODES[self._output_mode_idx] if 0 <= self._output_mode_idx < len(OUTPUT_MODES) else "pinhole"

        prompts = [p.strip() for p in self._mask_prompts_str.split(",") if p.strip()]
        quality_preset = EXTRACT_QUALITY_PRESETS[self._extract_quality_idx]
        quality_names = ["none", "fast", "normal", "maximum"]

        config = PipelineConfig(
            video_path=self._video_path,
            output_dir=self._output_path,
            interval=1.0 / max(0.1, self._extract_fps),
            extraction_quality=quality_names[self._extract_quality_idx],
            scene_threshold=quality_preset["scene_threshold"],
            blur_scale_width=quality_preset["scale_width"],
            quality=self._jpeg_quality,
            enable_masking=self._enable_masking and self._masking_available,
            mask_prompts=prompts if prompts else ["person"],
            preset_name=preset_name,
            output_size=self._output_size,
            jpeg_quality=self._jpeg_quality,
            colmap_preset=colmap_preset,
            output_mode=output_mode,
        )

        self._is_processing = True
        self._processing_stage = "Starting..."
        self._processing_status = ""
        self._processing_progress = 0.0

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

    def _on_pipeline_progress(self, stage: str, percent: float, message: str):
        with self._pending_lock:
            self._processing_stage = self._STAGE_LABELS.get(stage, stage)
            self._processing_progress = percent
            self._processing_status = message

    def _on_pipeline_complete(self, result: PipelineResult):
        with self._pending_lock:
            self._pending_result = result

    # ── Main-thread result consumption ────────────────────────

    def _consume_pending_result(self) -> bool:
        with self._pending_lock:
            result = self._pending_result
            self._pending_result = None
        if result is None:
            return False
        self._apply_result(result)
        return True

    def _apply_result(self, result: PipelineResult):
        self._job = None
        self._is_processing = False
        self._processing_stage = ""
        self._processing_status = ""
        self._processing_progress = 0.0

        if result.success:
            elapsed_str = f"{result.elapsed_sec:.1f}s"
            logger.info(
                "Pipeline complete: %d source frames -> %d output images, "
                "%d aligned cameras in %s",
                result.num_source_frames,
                result.num_output_images,
                result.num_aligned_cameras,
                elapsed_str,
            )
            self._error_message = ""

            if self._import_after and result.dataset_path:
                try:
                    lf.load_file(
                        result.dataset_path,
                        is_dataset=True,
                        output_path=self._output_path,
                    )
                except Exception as exc:
                    logger.error("Failed to import dataset: %s", exc)
                    self._error_message = f"Import failed: {exc}"
        else:
            self._error_message = result.error or "Pipeline failed"
            logger.error("Pipeline failed: %s", self._error_message)
