# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
COLMAP SfM pipeline runner for rig-aware 360° reconstruction.

Pipeline steps:
  1. Extract features  (CameraMode.PER_FOLDER — one camera per virtual view)
  2. Apply rig config  (reassigns cameras per view, groups images into frames)
  3. Feature matching   (sequential or exhaustive, configurable)
  4. Incremental mapping

Based on the Lichtfeld-COLMAP-Plugin's proven patterns (daemon thread,
_try_set_attr resilience, os.fspath paths, GPU/CPU fallback).

Image layout (camera-first, produced by the reframer):
    images/{view}/{station}.jpg

COLMAP stores image names with the subfolder prefix, e.g.
``00_00/frame_001.jpg``. The rig config uses literal folder prefixes
such as ``00_00/`` so the remaining filename ``frame_001.jpg`` becomes
the shared rig-frame key across all virtual cameras.
"""

from __future__ import annotations

from collections import Counter
import gc
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Patterns for parsing COLMAP's stderr log lines
_RE_PROCESSED_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_RE_IMAGE_NAME = re.compile(r"Name:\s+(\S+)")
_RE_MATCHING_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")

_IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png"))
MATCH_BUDGETS = {
    "fast": 8192,
    "balanced": 16384,
    "default": 32768,
    "high": 65536,
}


def _try_set_attr(obj, attr: str, value) -> bool:
    """Best-effort set for pybind option objects (older pycolmap builds may lack some fields)."""
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def _count_images(directory: Path) -> int:
    """Count image files recursively under *directory*."""
    count = 0
    for f in directory.rglob("*"):
        if f.suffix.lower() in _IMAGE_EXTENSIONS:
            count += 1
    return count


def _trim_process_memory() -> None:
    """Best-effort memory trim after heavy COLMAP stages."""
    gc.collect()


def resolve_match_budget(
    tier: str = "default",
    override: Optional[int] = None,
) -> int:
    """Resolve a user-facing match budget tier into COLMAP's max_num_matches."""
    if override is not None:
        return max(1024, int(override))
    return MATCH_BUDGETS.get(tier, MATCH_BUDGETS["default"])


def infer_shared_pinhole_camera_params(
    view_fovs_deg: Sequence[float],
    image_size: int,
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Infer shared PINHOLE intrinsics when all reframed views use one FOV."""
    if image_size <= 0:
        return None, None, None

    unique_fovs = sorted({round(float(fov), 6) for fov in view_fovs_deg})
    if len(unique_fovs) != 1:
        return None, None, None

    fov_deg = unique_fovs[0]
    if not 0.0 < fov_deg < 179.0:
        return None, None, None

    focal = 0.5 * image_size / math.tan(math.radians(fov_deg) / 2.0)
    cx = image_size / 2.0
    cy = image_size / 2.0
    params = f"{focal:.6f},{focal:.6f},{cx:.6f},{cy:.6f}"
    return params, focal / float(image_size), fov_deg


def _split_image_name(image_name: str) -> tuple[str, str]:
    """Split ``view/frame.jpg`` into (view, frame.jpg)."""
    normalized = image_name.replace("\\", "/")
    if "/" not in normalized:
        return "", normalized
    return normalized.split("/", 1)


def _collect_staged_image_names(images_dir: Path) -> list[str]:
    """Collect staged image names as POSIX-like relative paths."""
    names: list[str] = []
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            names.append(path.relative_to(images_dir).as_posix())
    return sorted(names, key=os.path.normcase)


# ── COLMAP 4.1 dispatch maps ─────────────────────────────────

# Maps (feature_type, matcher_type) → FeatureMatcherType enum member name.
# The enum values are compound: each combines a detector and a matching algo.
_MATCHER_TYPE_MAP: dict[tuple[str, str], str] = {
    ("sift", "bruteforce"): "SIFT_BRUTEFORCE",
    ("sift", "lightglue"): "SIFT_LIGHTGLUE",
    ("aliked_n16rot", "bruteforce"): "ALIKED_BRUTEFORCE",
    ("aliked_n16rot", "lightglue"): "ALIKED_LIGHTGLUE",
    ("aliked_n32", "bruteforce"): "ALIKED_BRUTEFORCE",
    ("aliked_n32", "lightglue"): "ALIKED_LIGHTGLUE",
}


def _feature_extractor_type(feature_type: str):
    """Map config string to pycolmap.FeatureExtractorType enum (lazy import)."""
    import pycolmap
    _MAP = {
        "sift": pycolmap.FeatureExtractorType.SIFT,
        "aliked_n16rot": pycolmap.FeatureExtractorType.ALIKED_N16ROT,
        "aliked_n32": pycolmap.FeatureExtractorType.ALIKED_N32,
    }
    return _MAP.get(feature_type, pycolmap.FeatureExtractorType.SIFT)


def _ba_backend(solver: str, camera_model: str):
    """Resolve BA solver backend enum from config strings."""
    import pycolmap
    CASPAR = pycolmap.BundleAdjustmentBackend(1)  # not named in pybind11
    CERES = pycolmap.BundleAdjustmentBackend.CERES
    CASPAR_MODELS = {"PINHOLE", "SIMPLE_RADIAL"}

    if solver == "ceres":
        return CERES
    if solver == "caspar":
        if camera_model in CASPAR_MODELS:
            return CASPAR
        return CERES  # unsupported model → silent fallback
    # "auto": Caspar when model is supported
    return CASPAR if camera_model in CASPAR_MODELS else CERES


@dataclass
class ColmapConfig:
    preset: str = "normal"
    camera_model: str = "PINHOLE"
    camera_params: Optional[str] = None
    default_focal_length_factor: Optional[float] = None
    matcher: str = "sequential"  # "sequential", "exhaustive", or "vocab_tree"
    match_budget_tier: str = "default"
    max_num_matches_override: Optional[int] = None
    refine_focal_length: bool = True
    refine_principal_point: bool = False  # set True for OPENCV_FISHEYE w/ a prior
    refine_extra_params: bool = False     # set True for OPENCV_FISHEYE w/ a prior
    vocab_tree_path: Optional[str] = None  # Required when matcher == "vocab_tree"
    # Override knobs for the fisheye path (where 2048 features at 1600px is
    # too sparse for 3840-wide raw fisheye). When None, fall back to the
    # preset-based defaults below.
    sift_max_num_features_override: Optional[int] = None
    sift_max_image_size_override: Optional[int] = None
    # COLMAP 4.1 features
    feature_type: str = "sift"        # "sift", "aliked_n16rot", "aliked_n32"
    matcher_type: str = "bruteforce"  # "bruteforce", "lightglue"
    mapper: str = "incremental"       # "incremental", "global"
    ba_solver: str = "auto"           # "auto", "ceres", "caspar"

    @property
    def sift_max_image_size(self) -> int:
        if self.sift_max_image_size_override is not None:
            return self.sift_max_image_size_override
        return 1600 if self.preset == "normal" else 1200

    @property
    def sift_max_num_features(self) -> int:
        if self.sift_max_num_features_override is not None:
            return self.sift_max_num_features_override
        return 2048 if self.preset == "normal" else 1536

    @property
    def sift_max_num_matches(self) -> int:
        return resolve_match_budget(
            self.match_budget_tier,
            self.max_num_matches_override,
        )


@dataclass
class ColmapResult:
    success: bool
    reconstruction_path: str = ""
    num_registered_images: int = 0
    num_expected_frames: int = 0
    num_registered_frames: int = 0
    num_complete_frames: int = 0
    num_partial_frames: int = 0
    views_per_frame: int = 0
    expected_images_by_view: dict[str, int] = field(default_factory=dict)
    registered_images_by_view: dict[str, int] = field(default_factory=dict)
    partial_frame_examples: list[str] = field(default_factory=list)
    dropped_frame_examples: list[str] = field(default_factory=list)
    num_points3d: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


@dataclass
class RegistrationDiagnostics:
    expected_frames: int = 0
    registered_frames: int = 0
    complete_frames: int = 0
    partial_frames: int = 0
    views_per_frame: int = 0
    expected_images_by_view: dict[str, int] = field(default_factory=dict)
    registered_images_by_view: dict[str, int] = field(default_factory=dict)
    partial_frame_examples: list[str] = field(default_factory=list)
    dropped_frame_examples: list[str] = field(default_factory=list)


def _summarize_registration(
    expected_image_names: Iterable[str],
    registered_image_names: Iterable[str],
) -> RegistrationDiagnostics:
    """Summarize registration by rig frame and by virtual camera view."""
    expected_by_view: Counter[str] = Counter()
    expected_frame_names: set[str] = set()
    for image_name in expected_image_names:
        view_name, frame_name = _split_image_name(image_name)
        if not frame_name:
            continue
        expected_by_view[view_name] += 1
        expected_frame_names.add(frame_name)

    registered_by_view: Counter[str] = Counter()
    registered_frame_counts: Counter[str] = Counter()
    for image_name in registered_image_names:
        view_name, frame_name = _split_image_name(image_name)
        if not frame_name:
            continue
        registered_by_view[view_name] += 1
        registered_frame_counts[frame_name] += 1

    views_per_frame = len(expected_by_view)
    complete_frames = 0
    partial_examples: list[str] = []
    for frame_name in sorted(registered_frame_counts):
        count = registered_frame_counts[frame_name]
        if views_per_frame > 0 and count == views_per_frame:
            complete_frames += 1
        else:
            partial_examples.append(f"{frame_name} ({count}/{views_per_frame})")

    dropped_frame_examples = [
        frame_name
        for frame_name in sorted(expected_frame_names)
        if frame_name not in registered_frame_counts
    ]

    return RegistrationDiagnostics(
        expected_frames=len(expected_frame_names),
        registered_frames=len(registered_frame_counts),
        complete_frames=complete_frames,
        partial_frames=max(len(registered_frame_counts) - complete_frames, 0),
        views_per_frame=views_per_frame,
        expected_images_by_view=dict(sorted(expected_by_view.items())),
        registered_images_by_view=dict(sorted(registered_by_view.items())),
        partial_frame_examples=partial_examples[:8],
        dropped_frame_examples=dropped_frame_examples[:8],
    )


ProgressCallback = Callable[[str, float, str], None]
CancelCheck = Callable[[], bool]


class _StderrCapture:
    """Redirect stderr fd to a pipe and read lines from it in a thread.

    COLMAP (via glog) writes per-image progress to stderr. This captures
    those lines in real time by redirecting file descriptor 2 to a pipe
    and reading from the read end in a background thread.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._read_fd: int = -1
        self._write_fd: int = -1
        self._old_stderr_fd: int = -1
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._read_fd, self._write_fd = os.pipe()
        self._old_stderr_fd = os.dup(2)
        os.dup2(self._write_fd, 2)

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(self._read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    with self._lock:
                        self._lines.append(text)

    def get_new_lines(self) -> list[str]:
        with self._lock:
            lines = self._lines[:]
            self._lines.clear()
        return lines

    def stop(self) -> None:
        if self._old_stderr_fd >= 0:
            os.dup2(self._old_stderr_fd, 2)
            os.close(self._old_stderr_fd)
            self._old_stderr_fd = -1
        if self._write_fd >= 0:
            os.close(self._write_fd)
            self._write_fd = -1
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._read_fd >= 0:
            os.close(self._read_fd)
            self._read_fd = -1


class ColmapRunner:
    def __init__(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        rig_config_path: str | Path,
        mask_path: str | Path | None = None,
        config: Optional[ColmapConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> None:
        self._images_dir = Path(images_dir)
        self._output_dir = Path(output_dir)
        self._rig_config_path = Path(rig_config_path)
        self._mask_path = Path(mask_path) if mask_path else None
        self._config = config or ColmapConfig()
        self._on_progress = on_progress
        self._cancel_check = cancel_check

    def _progress(self, stage: str, percent: float, message: str) -> None:
        logger.info("[%s %.0f%%] %s", stage, percent * 100, message)
        if self._on_progress:
            self._on_progress(stage, percent, message)

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_check and self._cancel_check():
            raise RuntimeError("Cancelled by user")

    def run(self) -> ColmapResult:
        t0 = time.monotonic()
        try:
            import pycolmap
        except ImportError:
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error="pycolmap is not installed. Install it with: pip install pycolmap",
            )
        try:
            return self._run_pipeline(pycolmap, t0)
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            if "Cancelled" in str(exc):
                return ColmapResult(success=False, elapsed_sec=elapsed, error="Cancelled by user")
            logger.exception("COLMAP pipeline failed")
            return ColmapResult(success=False, elapsed_sec=elapsed, error=str(exc))
        except Exception as exc:
            logger.exception("COLMAP pipeline failed")
            return ColmapResult(success=False, elapsed_sec=time.monotonic() - t0, error=str(exc))

    def _run_colmap_with_progress(
        self,
        stage: str,
        target_fn: Callable[[], None],
        parse_fn: Callable[[str], Optional[tuple[int, int, str]]],
        _log: Callable[[str], None],
    ) -> Optional[Exception]:
        """Run a COLMAP function on a thread while capturing stderr for progress."""
        error: list[Optional[Exception]] = [None]

        def _worker():
            try:
                target_fn()
            except Exception as exc:
                error[0] = exc

        capture = _StderrCapture()
        capture.start()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while thread.is_alive():
                thread.join(timeout=0.5)
                if self._cancel_check and self._cancel_check():
                    break

                for line in capture.get_new_lines():
                    _log(f"  colmap: {line}")
                    parsed = parse_fn(line)
                    if parsed is not None:
                        cur, tot, detail = parsed
                        pct = cur / max(tot, 1)
                        self._progress(stage, pct, detail)
        finally:
            capture.stop()

        for line in capture.get_new_lines():
            _log(f"  colmap: {line}")

        return error[0]

    # ------------------------------------------------------------------
    # Matcher helpers (GPU/CPU fallback, API compat)
    # ------------------------------------------------------------------

    def _run_matcher(self, pycolmap, fn_name: str, match_kwargs: dict, _log) -> None:
        """Run a pycolmap.match_* function with TypeError fallback."""
        fn = getattr(pycolmap, fn_name, None)
        if fn is None:
            raise RuntimeError(f"pycolmap.{fn_name} is not available in this build")
        try:
            fn(**match_kwargs)
        except TypeError:
            # Drop optional kwargs one at a time until it works
            fallback = dict(match_kwargs)
            for key in ("matching_options", "sift_options"):
                if key in fallback:
                    fallback.pop(key)
                    try:
                        fn(**fallback)
                        return
                    except TypeError:
                        continue
            raise

    def _run_matcher_with_gpu_fallback(
        self,
        pycolmap,
        fn_name: str,
        match_kwargs: dict,
        sift_opts,
        gpu_requested: bool,
        _log,
    ) -> None:
        """Run matcher, falling back from GPU to CPU on failure."""
        try:
            self._run_matcher(pycolmap, fn_name, match_kwargs, _log)
        except Exception as exc:
            if gpu_requested and sift_opts is not None and _try_set_attr(sift_opts, "use_gpu", False):
                _log(f"  GPU matching failed ({exc}), retrying CPU")
                self._run_matcher(pycolmap, fn_name, match_kwargs, _log)
            else:
                raise

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, pycolmap, t0: float) -> ColmapResult:
        debug_log = self._output_dir / "colmap_debug.log"

        def _log(msg: str) -> None:
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(f"[{time.monotonic() - t0:.1f}s] {msg}\n")
                f.flush()

        # --- Setup ---
        self._output_dir.mkdir(parents=True, exist_ok=True)
        _log("Pipeline started")

        sparse_dir = self._output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        database_path = os.fspath(self._output_dir / "database.db")
        image_path = os.fspath(self._images_dir)
        sparse_path = os.fspath(sparse_dir)

        # Clean stale database + WAL/SHM files
        for db_file in [database_path, f"{database_path}-wal", f"{database_path}-shm"]:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

        num_threads = min(8, os.cpu_count() or 4)
        sift_max_image_size = self._config.sift_max_image_size
        sift_max_num_features = self._config.sift_max_num_features

        # Count images recursively under the view folders.
        total_images = _count_images(self._images_dir)
        _log(f"Total images: {total_images}")

        # Enable COLMAP's stderr logging
        pycolmap.logging.logtostderr = True
        pycolmap.logging.verbose_level = 1

        # ================================================
        # STEP 1: FEATURE EXTRACTION (CameraMode.PER_FOLDER)
        # ================================================
        self._ensure_not_cancelled()
        feat_type = self._config.feature_type
        _log(f"Step 1: Feature extraction (type={feat_type})")
        self._progress("features", 0.0, f"Extracting {feat_type.upper()} features (0/{total_images})...")

        from pycolmap import CameraMode
        camera_mode = CameraMode.PER_FOLDER

        reader_opts = None
        extraction_opts = None
        extraction_gpu_requested = False

        if hasattr(pycolmap, "ImageReaderOptions"):
            reader_opts = pycolmap.ImageReaderOptions()
            _try_set_attr(reader_opts, "camera_model", self._config.camera_model)
            if self._config.camera_params:
                _try_set_attr(reader_opts, "camera_params", self._config.camera_params)
                _log(f"Step 1: Using camera_params={self._config.camera_params}")
            elif self._config.default_focal_length_factor is not None:
                _try_set_attr(
                    reader_opts,
                    "default_focal_length_factor",
                    self._config.default_focal_length_factor,
                )
                _log(
                    "Step 1: Using default_focal_length_factor="
                    f"{self._config.default_focal_length_factor:.6f}"
                )
            if self._mask_path and self._mask_path.is_dir():
                _try_set_attr(reader_opts, "mask_path", os.fspath(self._mask_path))
                _log(f"Step 1: Using masks from {self._mask_path}")

        if hasattr(pycolmap, "FeatureExtractionOptions"):
            extraction_opts = pycolmap.FeatureExtractionOptions()
            extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
            _try_set_attr(extraction_opts, "num_threads", num_threads)
            _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
            # Set feature type enum (string assignment raises TypeError)
            extraction_opts.type = _feature_extractor_type(feat_type)
            # Route max_num_features to the correct sub-options object
            if feat_type == "sift":
                _try_set_attr(extraction_opts.sift, "max_num_features", sift_max_num_features)
            else:
                _try_set_attr(extraction_opts.aliked, "max_num_features", sift_max_num_features)
        elif hasattr(pycolmap, "SiftExtractionOptions"):
            extraction_opts = pycolmap.SiftExtractionOptions()
            extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
            _try_set_attr(extraction_opts, "num_threads", num_threads)
            _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
            _try_set_attr(extraction_opts, "max_num_features", sift_max_num_features)

        extract_kwargs = dict(
            database_path=database_path,
            image_path=image_path,
            camera_mode=camera_mode,
        )

        if reader_opts is not None:
            extract_kwargs["reader_options"] = reader_opts
        else:
            extract_kwargs["camera_model"] = self._config.camera_model

        if extraction_opts is not None:
            if hasattr(pycolmap, "FeatureExtractionOptions"):
                extract_kwargs["extraction_options"] = extraction_opts
            else:
                extract_kwargs["sift_options"] = extraction_opts

        def _call_extract() -> None:
            try:
                pycolmap.extract_features(**extract_kwargs)
            except TypeError:
                fallback_kwargs = dict(extract_kwargs)
                for k in ("extraction_options", "sift_options", "reader_options", "camera_model"):
                    if k in fallback_kwargs:
                        fallback_kwargs.pop(k)
                        try:
                            pycolmap.extract_features(**fallback_kwargs)
                            return
                        except TypeError:
                            continue
                raise

        def _extract_with_fallback():
            try:
                _call_extract()
            except Exception as exc:
                if extraction_gpu_requested and extraction_opts is not None and _try_set_attr(extraction_opts, "use_gpu", False):
                    _log(f"Step 1: GPU failed ({exc}), retrying CPU")
                    _call_extract()
                else:
                    raise

        last_image_name = [""]

        def _parse_extraction(line: str) -> Optional[tuple[int, int, str]]:
            m = _RE_PROCESSED_FILE.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                name = last_image_name[0]
                return (cur, tot, f"Extracting features: {cur}/{tot} ({name})")
            m = _RE_IMAGE_NAME.search(line)
            if m:
                last_image_name[0] = m.group(1)
            return None

        err = self._run_colmap_with_progress(
            "features", _extract_with_fallback, _parse_extraction, _log,
        )
        if err is not None:
            raise err

        _log("Step 1: Feature extraction complete")
        self._progress("features", 1.0, f"Feature extraction complete ({total_images} images)")

        del reader_opts, extraction_opts, extract_kwargs
        _trim_process_memory()

        # ================================================
        # STEP 2: APPLY RIG CONFIG
        # ================================================
        self._ensure_not_cancelled()
        rig_config_path = os.fspath(self._rig_config_path)
        if os.path.exists(rig_config_path):
            _log("Step 2: Applying rig config")
            self._progress("rig", 0.0, "Applying rig constraints...")

            rig_configs = pycolmap.read_rig_config(rig_config_path)
            db = pycolmap.Database.open(database_path)
            pycolmap.apply_rig_config(rig_configs, db)

            n_cams = db.num_cameras()
            n_rigs = db.num_rigs()
            n_frames = db.num_frames()
            db.close()

            _log(f"Step 2: Rig applied — cameras={n_cams}, rigs={n_rigs}, frames={n_frames}")
            self._progress("rig", 1.0, f"Rig applied: {n_cams} cameras, {n_frames} frames")
        else:
            _log("Step 2: No rig config file found, skipping")

        # ================================================
        # STEP 3: FEATURE MATCHING
        # ================================================
        self._ensure_not_cancelled()
        matcher_name = self._config.matcher
        _log(f"Step 3: Feature matching ({matcher_name})")
        self._progress("matching", 0.0, f"Matching features ({matcher_name})...")
        _log(
            f"Step 3: Match budget — tier={self._config.match_budget_tier}, "
            f"max_num_matches={self._config.sift_max_num_matches}"
        )

        matching_opts = pycolmap.FeatureMatchingOptions()
        matching_gpu_requested = _try_set_attr(matching_opts, "use_gpu", True)
        _try_set_attr(matching_opts, "max_num_matches", self._config.sift_max_num_matches)
        _try_set_attr(matching_opts, "rig_verification", True)
        _try_set_attr(matching_opts, "skip_image_pairs_in_same_frame", True)

        # Set compound matcher type (e.g. SIFT_LIGHTGLUE, ALIKED_BRUTEFORCE)
        _mt_key = (self._config.feature_type, self._config.matcher_type)
        _mt_name = _MATCHER_TYPE_MAP.get(_mt_key, "SIFT_BRUTEFORCE")
        if hasattr(pycolmap, "FeatureMatcherType"):
            _mt_enum = getattr(pycolmap.FeatureMatcherType, _mt_name, None)
            if _mt_enum is not None:
                matching_opts.type = _mt_enum
                _log(f"Step 3: Matcher type = {_mt_name}")
            else:
                _log(f"Step 3: FeatureMatcherType.{_mt_name} not found, using default")

        match_kwargs: dict = {
            "database_path": database_path,
            "matching_options": matching_opts,
        }

        if matcher_name == "sequential":
            if hasattr(pycolmap, "SequentialPairingOptions"):
                pairing_opts = pycolmap.SequentialPairingOptions()
                _try_set_attr(pairing_opts, "loop_detection", True)
                match_kwargs["pairing_options"] = pairing_opts
            fn_name = "match_sequential"
        elif matcher_name == "vocab_tree":
            tree_path = self._config.vocab_tree_path
            if not tree_path or not os.path.exists(tree_path):
                raise RuntimeError(
                    "Vocab tree matching requires ColmapConfig.vocab_tree_path "
                    "to point at an existing vocabulary tree file. "
                    f"Got: {tree_path!r}"
                )
            if hasattr(pycolmap, "VocabTreePairingOptions"):
                vocab_opts = pycolmap.VocabTreePairingOptions()
                _try_set_attr(vocab_opts, "vocab_tree_path", tree_path)
                _try_set_attr(vocab_opts, "num_images", 100)
                _try_set_attr(vocab_opts, "num_nearest_neighbors", 5)
                match_kwargs["pairing_options"] = vocab_opts
            fn_name = "match_vocabtree"
        else:  # exhaustive
            if hasattr(pycolmap, "ExhaustivePairingOptions"):
                exhaustive_opts = pycolmap.ExhaustivePairingOptions()
                _try_set_attr(exhaustive_opts, "block_size", 15)
                match_kwargs["pairing_options"] = exhaustive_opts
            fn_name = "match_exhaustive"

        def _parse_matching(line: str) -> Optional[tuple[int, int, str]]:
            m = _RE_MATCHING_BLOCK.search(line)
            if m:
                row, row_total, col, col_total = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                total_blocks = row_total * col_total
                current_block = (row - 1) * col_total + col
                return (current_block, total_blocks, f"Matching: block {current_block}/{total_blocks}")
            return None

        def _do_matching():
            self._run_matcher_with_gpu_fallback(
                pycolmap, fn_name, match_kwargs, matching_opts, matching_gpu_requested, _log,
            )

        err = self._run_colmap_with_progress(
            "matching", _do_matching, _parse_matching, _log,
        )
        if err is not None:
            raise err

        _log("Step 3: Matching complete")
        self._progress("matching", 1.0, "Feature matching complete")

        del matching_opts
        _trim_process_memory()

        # ================================================
        # STEP 4: RECONSTRUCTION (incremental or global)
        # ================================================
        self._ensure_not_cancelled()
        mapper_name = self._config.mapper
        ba_backend = _ba_backend(self._config.ba_solver, self._config.camera_model)
        _log(f"Step 4: Mapping (mapper={mapper_name}, ba_solver={self._config.ba_solver}, "
             f"ba_backend={ba_backend})")

        if mapper_name == "global":
            # ── GLOMAP (global SfM) ──
            self._progress("mapping", 0.0, f"Running GLOMAP ({total_images} images)...")
            global_opts = pycolmap.GlobalPipelineOptions()
            # BA refinement settings (nested in mapper.bundle_adjustment)
            ba = global_opts.mapper.bundle_adjustment
            ba.refine_focal_length = self._config.refine_focal_length
            ba.refine_principal_point = self._config.refine_principal_point
            ba.refine_extra_params = self._config.refine_extra_params
            ba.backend = ba_backend
            # Lock rig geometry in all sub-stages
            ba.refine_sensor_from_rig = False
            global_opts.mapper.refine_sensor_from_rig = False
            global_opts.mapper.global_positioning.refine_sensor_from_rig = False
            global_opts.mapper.rotation_averaging.refine_sensor_from_rig = False
            _log(
                "Step 4: GLOMAP BA — refine_sensor_from_rig=False (all stages), "
                f"refine_focal_length={self._config.refine_focal_length}, "
                f"refine_principal_point={self._config.refine_principal_point}, "
                f"refine_extra_params={self._config.refine_extra_params}"
            )

            _log("Step 4: Calling pycolmap.global_mapping()")
            # global_mapping has no next_image_callback — progress is start/finish only
            self._progress("mapping", 0.1, "GLOMAP running...")

            reconstructions = pycolmap.global_mapping(
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_path,
                options=global_opts,
            )
        else:
            # ── Incremental mapping (default) ──
            self._progress("mapping", 0.0, f"Running incremental mapper ({total_images} images)...")

            pipeline_opts = pycolmap.IncrementalPipelineOptions()
            if hasattr(pipeline_opts, "multiple_models"):
                _try_set_attr(pipeline_opts, "multiple_models", False)
            if hasattr(pipeline_opts, "max_num_models"):
                _try_set_attr(pipeline_opts, "max_num_models", 1)
            _try_set_attr(pipeline_opts, "ba_refine_sensor_from_rig", False)
            _try_set_attr(pipeline_opts, "ba_refine_focal_length", self._config.refine_focal_length)
            _try_set_attr(pipeline_opts, "ba_refine_principal_point", self._config.refine_principal_point)
            _try_set_attr(pipeline_opts, "ba_refine_extra_params", self._config.refine_extra_params)
            # Lock rig geometry: constant_rigs takes Set[int] of rig IDs.
            if os.path.exists(self._rig_config_path):
                pipeline_opts.constant_rigs = {0}
            # Set BA solver backend
            gba = pipeline_opts.get_global_bundle_adjustment()
            gba.backend = ba_backend
            _log(
                "Step 4: Incremental BA — refine_sensor_from_rig=False, "
                f"refine_focal_length={self._config.refine_focal_length}, "
                f"refine_principal_point={self._config.refine_principal_point}, "
                f"refine_extra_params={self._config.refine_extra_params}, "
                f"constant_rigs={pipeline_opts.constant_rigs}"
            )

            _log("Step 4: Calling pycolmap.incremental_mapping()")

            registered_count = [0]

            def _on_next_image():
                registered_count[0] += 1
                if total_images > 0:
                    pct = registered_count[0] / total_images
                    self._progress(
                        "mapping", pct,
                        f"Mapping: {registered_count[0]}/{total_images} images registered",
                    )

            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_path,
                options=pipeline_opts,
                next_image_callback=_on_next_image,
            )

        _log(f"Step 4: Mapping returned {len(reconstructions) if reconstructions else 0} reconstruction(s)")

        if not reconstructions:
            _log("Step 4: No reconstructions")
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error=f"{mapper_name} mapping produced no reconstructions",
            )

        reconstruction = next(iter(reconstructions.values()))
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)
        registered_image_names = [image.name for image in reconstruction.images.values()]
        registration = _summarize_registration(
            _collect_staged_image_names(self._images_dir),
            registered_image_names,
        )

        _log(
            f"Step 4: {num_images} images, {num_points} points, "
            f"{registration.registered_frames}/{registration.expected_frames} frames "
            f"({registration.complete_frames} complete, {registration.partial_frames} partial)"
        )
        if registration.registered_images_by_view:
            per_view = ", ".join(
                f"{view}={registration.registered_images_by_view.get(view, 0)}/{registration.expected_images_by_view.get(view, 0)}"
                for view in registration.expected_images_by_view
            )
            _log(f"Step 4: Per-view registration — {per_view}")
        if registration.partial_frame_examples:
            _log(
                f"Step 4: Partial frames — {', '.join(registration.partial_frame_examples)}"
            )
        if registration.dropped_frame_examples:
            _log(
                f"Step 4: Dropped frames — {', '.join(registration.dropped_frame_examples)}"
            )

        # Write sparse model
        if hasattr(reconstruction, "write_text"):
            reconstruction.write_text(sparse_path)
            _log(f"Step 4: Sparse model saved (text format)")
        else:
            reconstruction.write(sparse_path)
            _log(f"Step 4: Sparse model saved")

        self._progress(
            "mapping", 1.0,
            f"Mapping complete: {num_images} images, {num_points} points",
        )

        del reconstruction, reconstructions
        _trim_process_memory()

        _log("Done")
        elapsed = time.monotonic() - t0
        return ColmapResult(
            success=True,
            reconstruction_path=os.fspath(self._output_dir),
            num_registered_images=num_images,
            num_expected_frames=registration.expected_frames,
            num_registered_frames=registration.registered_frames,
            num_complete_frames=registration.complete_frames,
            num_partial_frames=registration.partial_frames,
            views_per_frame=registration.views_per_frame,
            expected_images_by_view=registration.expected_images_by_view,
            registered_images_by_view=registration.registered_images_by_view,
            partial_frame_examples=registration.partial_frame_examples,
            dropped_frame_examples=registration.dropped_frame_examples,
            num_points3d=num_points,
            elapsed_sec=elapsed,
        )
