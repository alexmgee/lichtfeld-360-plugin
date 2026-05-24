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
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Patterns for parsing COLMAP's stderr log lines
_RE_PROCESSED_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_RE_IMAGE_NAME = re.compile(r"Name:\s+(\S+)")
_RE_MATCHING_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")
_RE_MATCHING_IMAGE = re.compile(r"Processing image \[(\d+)/(\d+)\]")

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


def _assign_to_kill_on_close_job(proc) -> None:
    """Assign a subprocess to a Windows Job Object that kills it when the parent exits.

    When LFS closes (or crashes), the job handle is released and Windows
    terminates all processes in the job. No-op on non-Windows platforms or
    if the Win32 calls fail.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # CreateJobObjectW(lpJobAttributes, lpName)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )

        # AssignProcessToJobObject(hJob, hProcess)
        handle = int(proc._handle)  # subprocess.Popen stores the process handle on Windows
        kernel32.AssignProcessToJobObject(job, handle)

        # Store the job handle so it isn't garbage-collected (which would close it)
        proc._job_handle = job
    except Exception:
        pass  # best-effort; fall back to manual cleanup


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


def _configure_ba(solver: str, camera_model: str):
    """Configure local and global BA options for the hybrid solver strategy.

    Returns (local_ba_opts, global_ba_opts) — both are BundleAdjustmentOptions.

    Strategy by solver mode:
      "auto"     — Local: Ceres-GPU (size-adaptive). Global: Caspar if supported, else Ceres-GPU.
      "ceres"    — Both: Ceres CPU only (for debugging/comparison).
      "ceres_gpu"— Both: Ceres-GPU (cuDSS), no Caspar.
      "caspar"   — Both: Caspar if supported, else falls back to Ceres-GPU.
    """
    import pycolmap

    CASPAR = pycolmap.BundleAdjustmentBackend(1)  # not named in pybind11
    CERES = pycolmap.BundleAdjustmentBackend.CERES
    CASPAR_MODELS = {"PINHOLE", "SIMPLE_RADIAL"}
    SLA = type(pycolmap.IncrementalPipelineOptions()
               .get_local_bundle_adjustment()
               .ceres.solver_options.sparse_linear_algebra_library_type)

    local_ba = pycolmap.BundleAdjustmentOptions()
    global_ba = pycolmap.BundleAdjustmentOptions()

    def _enable_ceres_gpu(opts):
        """Enable GPU-accelerated Ceres with cuDSS sparse solver."""
        opts.backend = CERES
        opts.ceres.use_gpu = True
        opts.ceres.auto_select_solver_type = True
        opts.ceres.solver_options.sparse_linear_algebra_library_type = SLA.CUDA_SPARSE

    if solver == "ceres":
        # Pure CPU Ceres — for debugging or baseline comparison
        local_ba.backend = CERES
        local_ba.ceres.use_gpu = False
        global_ba.backend = CERES
        global_ba.ceres.use_gpu = False

    elif solver == "ceres_gpu":
        # Ceres-GPU on both — uses cuDSS, no Caspar
        _enable_ceres_gpu(local_ba)
        _enable_ceres_gpu(global_ba)

    elif solver == "caspar":
        # Caspar on both (original behavior), fall back to Ceres-GPU if unsupported
        if camera_model in CASPAR_MODELS:
            local_ba.backend = CASPAR
            global_ba.backend = CASPAR
        else:
            _enable_ceres_gpu(local_ba)
            _enable_ceres_gpu(global_ba)

    else:  # "auto" — the hybrid strategy
        # Local BA: always Ceres-GPU (size-adaptive threshold handles small problems)
        _enable_ceres_gpu(local_ba)
        local_ba.ceres.min_num_images_gpu_solver = 40

        # Global BA: Caspar for supported models, Ceres-GPU otherwise
        if camera_model in CASPAR_MODELS:
            global_ba.backend = CASPAR
        else:
            _enable_ceres_gpu(global_ba)

    return local_ba, global_ba


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
    loop_detection: bool = False  # Sequential matcher loop detection (opt-in via UI checkbox)
    sequential_overlap: int = 10  # pairs each frame with N sequential neighbors
    # Override knobs for the fisheye path (where 2048 features at 1600px is
    # too sparse for 3840-wide raw fisheye). When None, fall back to the
    # preset-based defaults below.
    sift_max_num_features_override: Optional[int] = None
    sift_max_image_size_override: Optional[int] = None
    # COLMAP 4.1 features
    feature_type: str = "sift"        # "sift", "aliked_n16rot", "aliked_n32"
    matcher_type: str = "bruteforce"  # "bruteforce", "lightglue"
    mapper: str = "incremental"       # "incremental", "global"
    ba_solver: str = "auto"           # "auto", "ceres", "ceres_gpu", "caspar"
    camera_mode: str = "PER_FOLDER"   # "PER_FOLDER", "AUTO", "SINGLE"

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


_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_PLUGIN_LIB_DIR = str(_PLUGIN_DIR / "lib")
_VENV_PYTHON = str(_PLUGIN_DIR / ".venv" / "Scripts" / "python.exe")

# Bundled faiss-format vocabulary trees (pycolmap 4.1+, COLMAP post-May 2025).
# One per feature type — COLMAP requires matching feature type.
_VOCAB_TREES = {
    "sift": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words256K.bin",
    "aliked_n16rot": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin",
    "aliked_n32": _PLUGIN_DIR / "lib" / "vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin",
}

# ONNX model files for feature extraction and matching (downloaded from COLMAP 3.13.0 release)
_ONNX_MODELS = {
    "aliked_n16rot": _PLUGIN_DIR / "lib" / "aliked-n16rot.onnx",
    "aliked_n32": _PLUGIN_DIR / "lib" / "aliked-n32.onnx",
    "aliked_lightglue": _PLUGIN_DIR / "lib" / "aliked-lightglue.onnx",
    "bruteforce_matcher": _PLUGIN_DIR / "lib" / "bruteforce-matcher.onnx",
    "sift_lightglue": _PLUGIN_DIR / "lib" / "sift-lightglue.onnx",
}


def _resolve_vocab_tree(
    feature_type: str = "sift",
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """Return a valid vocab tree path for the given feature type.

    Priority: explicit path > bundled tree for feature_type > any bundled tree > None.
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    # Match tree to feature type
    bundled = _VOCAB_TREES.get(feature_type)
    if bundled and bundled.exists():
        return str(bundled)
    # Fallback: any available tree (better than nothing for loop detection)
    for p in _VOCAB_TREES.values():
        if p.exists():
            return str(p)
    return None

# Inline script executed by the subprocess via `python -c`.
# Receives fn_name and kwargs as a JSON file, runs the pycolmap function,
# writes status to a result file. Stderr goes to a separate file for progress.
_WORKER_SCRIPT = r'''
import json, os, sys, traceback

lib_dir, kwargs_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

if os.name == "nt" and lib_dir:
    try:
        os.add_dll_directory(lib_dir)
    except OSError:
        pass

try:
    import pycolmap
    pycolmap.logging.logtostderr = True
    pycolmap.logging.verbose_level = 1

    with open(kwargs_path, "r") as f:
        spec = json.load(f)

    fn_name = spec["fn_name"]
    fn_kwargs = spec["fn_kwargs"]

    fn = getattr(pycolmap, fn_name, None)
    if fn is None:
        with open(result_path, "w") as f:
            json.dump({"status": "error", "detail": f"pycolmap.{fn_name} not found"}, f)
        sys.exit(1)

    fn(**fn_kwargs)
    with open(result_path, "w") as f:
        json.dump({"status": "ok"}, f)

except Exception as exc:
    sys.stderr.write(f"WORKER ERROR: {exc}\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    with open(result_path, "w") as f:
        json.dump({"status": "error", "detail": str(exc)}, f)
    sys.exit(1)
'''



class ColmapRunner:
    def __init__(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        rig_config_path: str | Path | None = None,
        mask_path: str | Path | None = None,
        config: Optional[ColmapConfig] = None,
        on_progress: Optional[ProgressCallback] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> None:
        self._images_dir = Path(images_dir)
        self._output_dir = Path(output_dir)
        self._rig_config_path = Path(rig_config_path) if rig_config_path is not None else None
        self._mask_path = Path(mask_path) if mask_path else None
        self._config = config or ColmapConfig()
        self._on_progress = on_progress
        self._cancel_check = cancel_check
        self._active_proc = None
        self._rig_ids: set = set()  # populated by apply_rig_config in step 2

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

    def _run_colmap_in_subprocess(
        self,
        stage: str,
        fn_name: str,
        fn_kwargs: dict,
        parse_fn: Callable[[str], Optional[tuple[int, int, str]]],
        _log: Callable[[str], None],
    ) -> None:
        """Run a pycolmap function in a subprocess with cancellation support.

        Uses subprocess.Popen with the venv Python to avoid LFS embedded
        Python issues. The child receives kwargs as a JSON file, writes
        status to a result file, and stderr goes to a log file for progress.

        Raises RuntimeError on failure or cancellation.
        """
        import json as _json
        import subprocess as _sp

        # Kill any leftover process from a previous cancelled run
        self._kill_active_subprocess()

        out_dir = str(self._output_dir)
        stderr_path = tempfile.mktemp(suffix=f"_colmap_{stage}.log", dir=out_dir)
        kwargs_path = tempfile.mktemp(suffix=f"_colmap_{stage}_kwargs.json", dir=out_dir)
        result_path = tempfile.mktemp(suffix=f"_colmap_{stage}_result.json", dir=out_dir)

        # Serialize kwargs to JSON. Most pycolmap option objects survive pickle,
        # but ImageReaderOptions does NOT — pickle corrupts internal C++ state
        # causing extract_features to silently skip all images (0 in database).
        # Fix: serialize ImageReaderOptions fields as plain JSON, reconstruct
        # in the subprocess via setattr. Everything else uses pickle.
        import pickle, base64
        serialized_kwargs = {}
        for k, v in fn_kwargs.items():
            try:
                _json.dumps(v)
                serialized_kwargs[k] = v
            except (TypeError, ValueError):
                if type(v).__name__ == "ImageReaderOptions":
                    # Serialize scalar fields only. CRITICAL: skip mask_path
                    # and camera_mask_path when they're the default "." —
                    # explicitly setting these triggers COLMAP's mask loader,
                    # which silently skips every image when no masks exist.
                    # Only serialize them if they point to a real directory.
                    fields = {}
                    for fk, fv in v.todict().items():
                        if hasattr(fv, "__fspath__"):
                            path_str = str(fv)
                            if path_str != "." and path_str != "":
                                fields[fk] = path_str
                        elif isinstance(fv, (str, int, float, bool, type(None))):
                            fields[fk] = fv
                    serialized_kwargs[k] = {"__reader_opts__": fields}
                else:
                    serialized_kwargs[k] = {
                        "__pickle__": base64.b64encode(pickle.dumps(v)).decode("ascii")
                    }

        spec = {"fn_name": fn_name, "fn_kwargs": serialized_kwargs}
        with open(kwargs_path, "w") as f:
            _json.dump(spec, f)

        # Build the inline script that handles pickle deserialization
        script = r'''
import json, os, sys, traceback, pickle, base64

lib_dir, kwargs_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

if os.name == "nt" and lib_dir:
    try:
        os.add_dll_directory(lib_dir)
    except OSError:
        pass

try:
    import pycolmap
    pycolmap.logging.logtostderr = True
    pycolmap.logging.verbose_level = 1

    with open(kwargs_path, "r") as f:
        spec = json.load(f)

    fn_name = spec["fn_name"]
    raw_kwargs = spec["fn_kwargs"]

    # Deserialize: ImageReaderOptions via setattr, everything else via pickle
    fn_kwargs = {}
    for k, v in raw_kwargs.items():
        if isinstance(v, dict) and "__reader_opts__" in v:
            obj = pycolmap.ImageReaderOptions()
            for fk, fv in v["__reader_opts__"].items():
                try:
                    setattr(obj, fk, fv)
                except Exception:
                    pass
            fn_kwargs[k] = obj
        elif isinstance(v, dict) and "__pickle__" in v:
            fn_kwargs[k] = pickle.loads(base64.b64decode(v["__pickle__"]))
        else:
            fn_kwargs[k] = v

    fn = getattr(pycolmap, fn_name, None)
    if fn is None:
        with open(result_path, "w") as f:
            json.dump({"status": "error", "detail": f"pycolmap.{fn_name} not found"}, f)
        sys.exit(1)

    fn(**fn_kwargs)
    with open(result_path, "w") as f:
        json.dump({"status": "ok"}, f)

except Exception as exc:
    sys.stderr.write(f"WORKER ERROR: {exc}\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    try:
        with open(result_path, "w") as f:
            json.dump({"status": "error", "detail": str(exc)}, f)
    except Exception:
        pass
    sys.exit(1)
'''

        stderr_fh = open(stderr_path, "w", encoding="utf-8")
        _sp_flags = {"creationflags": _sp.CREATE_NO_WINDOW} if os.name == "nt" else {}

        proc = _sp.Popen(
            [_VENV_PYTHON, "-c", script, _PLUGIN_LIB_DIR, kwargs_path, result_path],
            stderr=stderr_fh,
            stdout=_sp.DEVNULL,
            **_sp_flags,
        )
        self._active_proc = proc
        _assign_to_kill_on_close_job(proc)
        _log(f"  subprocess: spawned pid={proc.pid} for {fn_name}")

        read_pos = 0
        try:
            while proc.poll() is None:
                time.sleep(0.5)

                if self._cancel_check and self._cancel_check():
                    _log(f"  subprocess: cancelling {fn_name} (pid={proc.pid})")
                    self._kill_active_subprocess()
                    raise RuntimeError("Cancelled by user")

                read_pos = self._read_stderr_progress(
                    stderr_path, read_pos, parse_fn, stage, _log,
                )
        finally:
            stderr_fh.close()
            self._read_stderr_progress(
                stderr_path, read_pos, parse_fn, stage, _log,
            )
            self._active_proc = None

        # Read result
        result = None
        try:
            if os.path.exists(result_path):
                result = _json.loads(open(result_path, "r").read())
        except Exception:
            pass

        if result and result.get("status") == "error":
            raise RuntimeError(f"COLMAP {fn_name} failed: {result.get('detail', 'unknown')}")
        elif proc.returncode != 0:
            stderr_content = ""
            try:
                stderr_content = open(stderr_path, "r", encoding="utf-8", errors="replace").read()
            except OSError:
                pass
            detail = f"subprocess exited with code {proc.returncode}"
            if stderr_content.strip():
                last_lines = stderr_content.strip().splitlines()[-10:]
                for line in last_lines:
                    _log(f"  stderr: {line}")
                detail += f"\nLast stderr: {last_lines[-1]}"
            raise RuntimeError(f"COLMAP {fn_name} failed: {detail}")

        # Clean up temp files on success
        for p in (stderr_path, kwargs_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    def _read_stderr_progress(
        self,
        stderr_path: str,
        read_pos: int,
        parse_fn: Callable[[str], Optional[tuple[int, int, str]]],
        stage: str,
        _log: Callable[[str], None],
    ) -> int:
        """Read new lines from the stderr file starting at read_pos."""
        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(read_pos)
                new_data = f.read()
                new_pos = f.tell()
        except (FileNotFoundError, OSError):
            return read_pos

        if not new_data:
            return read_pos

        for line in new_data.splitlines():
            line = line.rstrip()
            if not line:
                continue
            _log(f"  colmap: {line}")
            parsed = parse_fn(line)
            if parsed is not None:
                cur, tot, detail = parsed
                pct = cur / max(tot, 1)
                self._progress(stage, pct, detail)

        return new_pos

    def _kill_active_subprocess(self) -> None:
        """Terminate any active COLMAP subprocess."""
        proc = getattr(self, "_active_proc", None)
        if proc is None:
            return
        # subprocess.Popen uses poll()/terminate()/kill()
        if proc.poll() is not None:
            return  # already exited
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        self._active_proc = None

    # ------------------------------------------------------------------
    # Matcher helpers (GPU/CPU fallback, API compat)
    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _validate_onnx_models(self, _log: Callable[[str], None]) -> None:
        """Check that required ONNX model files exist for the current config.

        Raises RuntimeError with a clear message if any are missing, so the
        user gets a diagnostic instead of an opaque COLMAP crash.
        """
        needed: list[str] = []
        ft = self._config.feature_type
        mt = self._config.matcher_type

        if ft.startswith("aliked"):
            needed.append(ft)  # aliked_n16rot or aliked_n32 (extraction model)
        if ft.startswith("aliked") and mt == "lightglue":
            needed.append("aliked_lightglue")
        elif ft.startswith("aliked") and mt == "bruteforce":
            needed.append("bruteforce_matcher")
        elif ft == "sift" and mt == "lightglue":
            needed.append("sift_lightglue")
        # sift + bruteforce uses native CUDA/CPU codepath — no ONNX model needed

        missing = []
        for key in needed:
            path = _ONNX_MODELS.get(key)
            if not path or not path.exists():
                missing.append(f"{key} ({path})" if path else key)

        if missing:
            raise RuntimeError(
                f"Missing ONNX model(s) for {ft}+{mt}: {', '.join(missing)}. "
                f"Download from COLMAP 3.13.0 release to lib/."
            )
        if needed:
            _log(f"Pre-flight: ONNX models verified for {ft}+{mt}: {needed}")

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

        # Pre-flight: verify required ONNX models exist for the current config
        self._validate_onnx_models(_log)

        # ================================================
        # STEP 1: FEATURE EXTRACTION (CameraMode.PER_FOLDER)
        # ================================================
        self._ensure_not_cancelled()
        feat_type = self._config.feature_type
        _log(f"Step 1: Feature extraction (type={feat_type})")
        self._progress("features", 0.0, f"Extracting {feat_type.upper()} features (0/{total_images})...")

        from pycolmap import CameraMode
        camera_mode = getattr(CameraMode, self._config.camera_mode, CameraMode.PER_FOLDER)

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
                # Set ONNX model paths for ALIKED extraction
                model_path = _ONNX_MODELS.get(feat_type)
                if model_path and model_path.exists():
                    attr = "n16rot_model_path" if "n16rot" in feat_type else "n32_model_path"
                    _try_set_attr(extraction_opts.aliked, attr, str(model_path))
                    _log(f"Step 1: ALIKED model: {model_path.name}")
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

        try:
            self._run_colmap_in_subprocess(
                "features", "extract_features", extract_kwargs,
                _parse_extraction, _log,
            )
        except RuntimeError as exc:
            if extraction_gpu_requested and extraction_opts is not None and "Cancelled" not in str(exc):
                _log(f"Step 1: GPU extraction failed ({exc}), retrying CPU")
                _try_set_attr(extraction_opts, "use_gpu", False)
                self._run_colmap_in_subprocess(
                    "features", "extract_features", extract_kwargs,
                    _parse_extraction, _log,
                )
            else:
                raise

        _log("Step 1: Feature extraction complete")
        self._progress("features", 1.0, f"Feature extraction complete ({total_images} images)")

        del reader_opts, extraction_opts, extract_kwargs
        _trim_process_memory()

        # ================================================
        # STEP 2: APPLY RIG CONFIG
        # ================================================
        self._ensure_not_cancelled()
        if self._rig_config_path is not None:
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
                # Capture actual rig IDs for constant_rigs in mapping
                self._rig_ids = {rig.rig_id for rig in db.read_all_rigs()}
                db.close()

                _log(f"Step 2: Rig applied — cameras={n_cams}, rigs={n_rigs}, "
                     f"frames={n_frames}, rig_ids={self._rig_ids}")
                self._progress("rig", 1.0, f"Rig applied: {n_cams} cameras, {n_frames} frames")
            else:
                _log("Step 2: No rig config file found, skipping")
        else:
            _log("Step 2: No rig config specified, skipping")

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

        # Set ONNX model paths for ALIKED matching (LightGlue or bruteforce)
        if self._config.feature_type.startswith("aliked"):
            lg_model = _ONNX_MODELS.get("aliked_lightglue")
            if self._config.matcher_type == "lightglue" and lg_model and lg_model.exists():
                _try_set_attr(matching_opts.aliked.lightglue, "model_path", str(lg_model))
                _log(f"Step 3: LightGlue model: {lg_model.name}")
            # ALIKED+Bruteforce uses ONNX inference (unlike SIFT+Bruteforce which
            # uses the native CUDA/CPU codepath and needs no model file).
            bf_model = _ONNX_MODELS.get("bruteforce_matcher")
            if self._config.matcher_type == "bruteforce" and bf_model and bf_model.exists():
                _try_set_attr(matching_opts.aliked.brute_force, "model_path", str(bf_model))
                _log(f"Step 3: Bruteforce ONNX model: {bf_model.name}")

        # Set ONNX model path for SIFT+LightGlue matching
        if self._config.feature_type == "sift" and self._config.matcher_type == "lightglue":
            slg_model = _ONNX_MODELS.get("sift_lightglue")
            if slg_model and slg_model.exists():
                _try_set_attr(matching_opts.sift.lightglue, "model_path", str(slg_model))
                _log(f"Step 3: SIFT LightGlue model: {slg_model.name}")

        match_kwargs: dict = {
            "database_path": database_path,
            "matching_options": matching_opts,
        }

        feat = self._config.feature_type

        if matcher_name == "sequential":
            if hasattr(pycolmap, "SequentialPairingOptions"):
                pairing_opts = pycolmap.SequentialPairingOptions()
                if self._config.loop_detection:
                    tree_path = _resolve_vocab_tree(feat, self._config.vocab_tree_path)
                    if tree_path:
                        _try_set_attr(pairing_opts, "loop_detection", True)
                        _try_set_attr(pairing_opts, "vocab_tree_path", tree_path)
                    else:
                        _try_set_attr(pairing_opts, "loop_detection", False)
                else:
                    _try_set_attr(pairing_opts, "loop_detection", False)
                _try_set_attr(pairing_opts, "overlap", self._config.sequential_overlap)
                match_kwargs["pairing_options"] = pairing_opts
            fn_name = "match_sequential"
        elif matcher_name == "vocab_tree":
            tree_path = _resolve_vocab_tree(feat, self._config.vocab_tree_path)
            if not tree_path:
                raise RuntimeError(
                    "Vocab tree matching requires a vocabulary tree file. "
                    "Expected bundled trees in lib/ or explicit ColmapConfig.vocab_tree_path. "
                    f"Got: {self._config.vocab_tree_path!r}"
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
            # Exhaustive matcher: "Processing block [R/RT, C/CT]"
            m = _RE_MATCHING_BLOCK.search(line)
            if m:
                row, row_total, col, col_total = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                total_blocks = row_total * col_total
                current_block = (row - 1) * col_total + col
                return (current_block, total_blocks, f"Matching: block {current_block}/{total_blocks}")
            # Sequential matcher: "Processing image [N/M]"
            m = _RE_MATCHING_IMAGE.search(line)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                return (current, total, f"Matching: image {current}/{total}")
            return None

        try:
            self._run_colmap_in_subprocess(
                "matching", fn_name, match_kwargs,
                _parse_matching, _log,
            )
        except RuntimeError as exc:
            if matching_gpu_requested and "Cancelled" not in str(exc):
                _log(f"Step 3: GPU matching failed ({exc}), retrying CPU")
                _try_set_attr(matching_opts, "use_gpu", False)
                self._run_colmap_in_subprocess(
                    "matching", fn_name, match_kwargs,
                    _parse_matching, _log,
                )
            else:
                raise

        _log("Step 3: Matching complete")
        self._progress("matching", 1.0, "Feature matching complete")

        del matching_opts
        _trim_process_memory()

        # ================================================
        # STEP 4: RECONSTRUCTION (incremental or global)
        # ================================================
        self._ensure_not_cancelled()
        mapper_name = self._config.mapper
        local_ba_opts, global_ba_opts = _configure_ba(self._config.ba_solver, self._config.camera_model)
        _log(f"Step 4: Mapping (mapper={mapper_name}, ba_solver={self._config.ba_solver}, "
             f"local_ba={local_ba_opts.backend}, global_ba={global_ba_opts.backend}, "
             f"local_gpu={local_ba_opts.ceres.use_gpu})")

        # Parse mapping progress from stderr (works for both incremental and global)
        _RE_REGISTERING = re.compile(r"Registering image #(\d+)")

        def _parse_mapping(line: str) -> Optional[tuple[int, int, str]]:
            m = _RE_REGISTERING.search(line)
            if m and total_images > 0:
                img_id = int(m.group(1))
                # img_id is the COLMAP image ID, not a sequential count.
                # Use it as a rough progress indicator.
                return (img_id, total_images, f"Mapping: registering image #{img_id}")
            return None

        if mapper_name == "global":
            # ── GLOMAP (global SfM) ──
            self._progress("mapping", 0.0, f"Running GLOMAP ({total_images} images)...")
            global_opts = pycolmap.GlobalPipelineOptions()
            ba = global_opts.mapper.bundle_adjustment
            ba.refine_focal_length = self._config.refine_focal_length
            ba.refine_principal_point = self._config.refine_principal_point
            ba.refine_extra_params = self._config.refine_extra_params
            ba.backend = global_ba_opts.backend
            if global_ba_opts.backend == pycolmap.BundleAdjustmentBackend.CERES:
                ba.ceres.use_gpu = global_ba_opts.ceres.use_gpu
                ba.ceres.auto_select_solver_type = global_ba_opts.ceres.auto_select_solver_type
                if global_ba_opts.ceres.use_gpu:
                    ba.ceres.solver_options.sparse_linear_algebra_library_type = (
                        global_ba_opts.ceres.solver_options.sparse_linear_algebra_library_type
                    )
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

            mapping_kwargs = dict(
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_path,
                options=global_opts,
            )
            self._run_colmap_in_subprocess(
                "mapping", "global_mapping", mapping_kwargs,
                _parse_mapping, _log,
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
            if hasattr(self, '_rig_ids') and self._rig_ids:
                pipeline_opts.constant_rigs = self._rig_ids

            # Apply hybrid BA configuration — local and global independently
            lba = pipeline_opts.get_local_bundle_adjustment()
            gba = pipeline_opts.get_global_bundle_adjustment()

            # Local BA
            lba.backend = local_ba_opts.backend
            lba.ceres.use_gpu = local_ba_opts.ceres.use_gpu
            lba.ceres.auto_select_solver_type = local_ba_opts.ceres.auto_select_solver_type
            lba.ceres.min_num_images_gpu_solver = local_ba_opts.ceres.min_num_images_gpu_solver
            if local_ba_opts.ceres.use_gpu:
                lba.ceres.solver_options.sparse_linear_algebra_library_type = (
                    local_ba_opts.ceres.solver_options.sparse_linear_algebra_library_type
                )

            # Global BA
            gba.backend = global_ba_opts.backend
            if global_ba_opts.backend == pycolmap.BundleAdjustmentBackend.CERES:
                gba.ceres.use_gpu = global_ba_opts.ceres.use_gpu
                gba.ceres.auto_select_solver_type = global_ba_opts.ceres.auto_select_solver_type
                if global_ba_opts.ceres.use_gpu:
                    gba.ceres.solver_options.sparse_linear_algebra_library_type = (
                        global_ba_opts.ceres.solver_options.sparse_linear_algebra_library_type
                    )

            _log(
                "Step 4: Incremental BA — refine_sensor_from_rig=False, "
                f"refine_focal_length={self._config.refine_focal_length}, "
                f"refine_principal_point={self._config.refine_principal_point}, "
                f"refine_extra_params={self._config.refine_extra_params}, "
                f"constant_rigs={pipeline_opts.constant_rigs}"
            )

            # next_image_callback can't cross process boundary — rely on
            # stderr parsing for progress instead.
            mapping_kwargs = dict(
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_path,
                options=pipeline_opts,
            )
            self._run_colmap_in_subprocess(
                "mapping", "incremental_mapping", mapping_kwargs,
                _parse_mapping, _log,
            )

        # Read reconstruction back from disk (subprocess wrote it)
        _log("Step 4: Reading reconstruction from disk")
        reconstruction = pycolmap.Reconstruction()
        sparse_0 = os.path.join(sparse_path, "0")
        if os.path.isdir(sparse_0):
            reconstruction.read(sparse_0)
        elif os.path.isdir(sparse_path):
            reconstruction.read(sparse_path)
        else:
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error=f"{mapper_name} mapping produced no reconstruction output",
            )

        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)

        if num_images == 0:
            _log("Step 4: No registered images in reconstruction")
            return ColmapResult(
                success=False,
                elapsed_sec=time.monotonic() - t0,
                error=f"{mapper_name} mapping produced empty reconstruction",
            )

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

        # Write text-format sparse model alongside binary
        if hasattr(reconstruction, "write_text"):
            reconstruction.write_text(sparse_path)
            _log("Step 4: Sparse model saved (text format)")

        self._progress(
            "mapping", 1.0,
            f"Mapping complete: {num_images} images, {num_points} points",
        )

        del reconstruction
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
