# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized masking setup state detection for two-tier backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAM3_MODEL_ID = "facebook/sam3"

_hf_access_cache: bool | None = None


@dataclass
class MaskingSetupState:
    """Current state of masking dependencies."""

    # Default tier (YOLO + SAM v1)
    has_torch: bool = False
    has_yolo: bool = False
    has_sam1: bool = False

    # Video tracking (SAM v2, optional)
    has_sam2: bool = False

    # Premium tier (SAM 3)
    has_token: bool = False
    has_access: bool = False
    has_sam3: bool = False
    has_weights: bool = False

    @property
    def default_tier_ready(self) -> bool:
        return self.has_torch and self.has_yolo and self.has_sam1

    @property
    def premium_tier_ready(self) -> bool:
        return self.has_torch and self.has_sam3 and self.has_weights

    @property
    def active_backend(self) -> str | None:
        """Return the best available backend.

        SAM 3 is only active if installed and weights are present.
        Falls back to YOLO+SAM v1 if available.
        """
        if self.premium_tier_ready:
            return "sam3"
        if self.default_tier_ready:
            return "yolo_sam1"
        return None

    @property
    def is_ready(self) -> bool:
        return self.active_backend is not None

    @property
    def first_incomplete_step(self) -> int:
        """For default tier setup: 0 = done, 1 = need deps."""
        if self.default_tier_ready:
            return 0
        return 1

    @property
    def video_tracking_ready(self) -> bool:
        """Pass 2 uses SAM v2 temporal tracking."""
        return self.default_tier_ready and self.has_sam2

    @property
    def capability_level(self) -> int:
        """Masking capability level for UI reporting.

        0 = nothing installed
        1 = YOLO+SAM v1 (Pass 1 + fallback Pass 2)
        2 = YOLO+SAM v1 + SAM v2 (full video tracking)
        3 = SAM 3 (premium)
        """
        if self.premium_tier_ready:
            return 3
        if self.video_tracking_ready:
            return 2
        if self.default_tier_ready:
            return 1
        return 0


def _check_torch_installed() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_yolo_installed() -> bool:
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except ImportError:
        return False


def _check_sam1_installed() -> bool:
    try:
        from segment_anything import SamPredictor  # noqa: F401
        return True
    except ImportError:
        return False


def _check_sam2_installed() -> bool:
    try:
        from sam2.build_sam import build_sam2_video_predictor_hf  # noqa: F401
        return True
    except ImportError:
        return False


def _check_sam3_installed() -> bool:
    try:
        from sam3.model_builder import build_sam3_image_model  # noqa: F401
        return True
    except ImportError:
        return False


def _check_hf_token() -> bool:
    try:
        from huggingface_hub import get_token
        token = get_token()
        return token is not None and len(token) > 0
    except Exception:
        return False


def _check_hf_access() -> bool:
    """Check if cached token has access to SAM 3 model.

    Result is cached in-memory after first successful check to avoid
    network calls on every panel load.
    """
    global _hf_access_cache
    if _hf_access_cache is not None:
        return _hf_access_cache
    try:
        from huggingface_hub import get_token, model_info
        token = get_token()
        if not token:
            return False
        info = model_info(SAM3_MODEL_ID, token=token)
        result = info is not None
        if result:
            _hf_access_cache = True
        return result
    except Exception:
        return False


def _check_weights_downloaded() -> bool:
    """Check if SAM 3 weights exist in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(SAM3_MODEL_ID, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def check_masking_setup() -> MaskingSetupState:
    """Check all masking setup conditions. Returns current state."""
    torch_ok = _check_torch_installed()
    return MaskingSetupState(
        has_torch=torch_ok,
        has_yolo=_check_yolo_installed(),
        has_sam1=_check_sam1_installed(),
        has_sam2=_check_sam2_installed(),
        has_token=_check_hf_token(),
        has_access=_check_hf_access(),
        has_sam3=_check_sam3_installed(),
        has_weights=_check_weights_downloaded(),
    )


def verify_hf_token(token: str) -> bool:
    """Verify a HuggingFace token has access to SAM 3.

    Saves the token via huggingface_hub.login() if valid.
    Returns True if token is valid and has model access.
    """
    global _hf_access_cache
    try:
        from huggingface_hub import login, model_info
        login(token=token)
        info = model_info(SAM3_MODEL_ID, token=token)
        if info is not None:
            _hf_access_cache = True
            return True
        return False
    except Exception as exc:
        logger.warning("HF token verification failed: %s", exc)
        return False


def download_model_weights() -> bool:
    """Download SAM 3 model weights from HuggingFace.

    Returns True on success. Weights download eagerly (not on first use).
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(SAM3_MODEL_ID)
        return True
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Plugin-venv package installation
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_VENV_DIR = _PLUGIN_DIR / ".venv"
def _find_uv() -> str:
    """Find uv binary — bundled with LFS or on PATH."""
    try:
        import lichtfeld as lf
        uv = lf.packages.uv_path()
        if uv:
            return uv
    except Exception:
        pass
    import shutil
    return shutil.which("uv") or ""


def _run_uv_command(args: list[str], on_output=None) -> bool:
    """Run a uv command in the plugin directory. Returns True on success."""
    uv = _find_uv()
    if not uv:
        logger.error("uv not found — cannot install packages")
        return False

    import subprocess, os
    cmd = [uv] + args
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    env = os.environ.copy()
    uv_cache_dir = _PLUGIN_DIR / "tmp" / "uv-cache-runtime"
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("UV_CACHE_DIR", str(uv_cache_dir))

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(_PLUGIN_DIR), env=env, **flags,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line and on_output:
            on_output(line)
    proc.wait()
    return proc.returncode == 0


def install_default_tier(on_output=None) -> bool:
    """Install YOLO + SAM v1 + torch into the plugin venv.

    Syncs the locked project environment without the dev group.
    PyTorch CUDA wheel selection is encoded in pyproject.toml/uv.lock.
    Downloads SAM v1 ViT-H weights eagerly after install.
    """
    ok = _run_uv_command([
        "sync",
        "--locked",
        "--no-dev",
    ], on_output=on_output)
    if not ok:
        return False

    # Eagerly download SAM v1 ViT-H weights
    try:
        import os
        import urllib.request
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
        if not os.path.exists(ckpt_path):
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            if on_output:
                on_output("Downloading SAM v1 ViT-H weights (~2.56 GB)...")
            urllib.request.urlretrieve(url, ckpt_path)
            if on_output:
                on_output("SAM v1 weights downloaded.")
    except Exception as exc:
        logger.warning("SAM v1 weight download failed (will retry on first use): %s", exc)

    return True


def _try_import_sam2_c() -> bool:
    """Try importing sam2._C, returning True on success.

    On Windows, torch must be imported first so that CUDA runtime DLLs
    are on the loader search path before the PE loader resolves _C.pyd's
    dependencies.
    """
    try:
        import torch  # noqa: F401  — seeds DLL search paths on Windows
        from sam2 import _C  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _normalize_acl(path: Path) -> None:
    """Best-effort ACL normalization on Windows.

    After copying _C.pyd, its ACL may inherit restrictive permissions
    from the source or the copy operation.  Match the ACL of a known-good
    file in the same directory (sam2/__init__.py) so the PE loader can
    open it in the LichtFeld runtime.
    """
    import os
    if os.name != "nt":
        return
    try:
        sam2_dir = path.parent
        reference = sam2_dir / "__init__.py"
        if not reference.exists():
            return
        # icacls /reset clears inherited ACEs and reapplies parent defaults
        import subprocess
        subprocess.run(
            ["icacls", str(path), "/reset"],
            capture_output=True, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as exc:
        logger.debug("ACL normalization failed for %s: %s", path, exc)


def _find_real_sam2_package_dir() -> Path | None:
    """Return the installed sam2 package dir only if build_sam is present."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("sam2.build_sam")
        if spec is None or spec.origin is None:
            return None
        return Path(spec.origin).resolve().parent
    except Exception:
        return None


def _quarantine_broken_sam2_namespace(on_output=None) -> Path | None:
    """Move aside a ghost sam2 directory that lacks real package modules.

    A stale ``site-packages/sam2`` folder containing only copied extension
    artifacts can survive package-manager changes and then import as a namespace
    package. If ``sam2.build_sam`` is missing but the directory still exists,
    move it out of the way before reinstalling the real package.
    """
    if _find_real_sam2_package_dir() is not None:
        return None

    try:
        import sysconfig
        import time

        purelib = Path(sysconfig.get_path("purelib"))
        broken = purelib / "sam2"
        if not broken.is_dir():
            return None

        backup = purelib / f"sam2.broken_{time.strftime('%Y%m%d_%H%M%S')}"
        broken.rename(backup)
        logger.warning("Quarantined broken sam2 namespace directory at %s", backup)
        if on_output:
            on_output(f"Moved broken SAM v2 package aside: {backup.name}")
        return backup
    except Exception as exc:
        logger.warning("Failed to quarantine broken sam2 namespace package: %s", exc)
        return None


def _install_sam2_c_extension(on_output=None) -> bool:
    """Copy the bundled _C.pyd into the installed sam2 package.

    The sam2 PyPI package ships without the compiled CUDA extension for
    connected-component mask hole-filling.  We pre-built it and bundle
    it in lib/_C.pyd.  This copies it into the sam2 package so that
    ``from sam2 import _C`` succeeds at runtime.

    After copying, normalizes Windows ACLs and verifies the import
    actually succeeds.  Returns True only if _C is importable after
    this call.
    """
    bundled = _PLUGIN_DIR / "lib" / "_C.pyd"
    if not bundled.exists():
        logger.debug("No bundled _C.pyd found in lib/")
        return False

    sam2_dir = _find_real_sam2_package_dir()
    if sam2_dir is None:
        logger.debug("real sam2 package not importable — skipping _C.pyd install")
        return False

    dest = sam2_dir / "_C.pyd"
    needs_copy = not dest.exists() or dest.stat().st_size != bundled.stat().st_size

    if needs_copy:
        import shutil
        shutil.copy2(str(bundled), str(dest))
        _normalize_acl(dest)
        logger.info("Installed _C.pyd into %s", sam2_dir)
        if on_output:
            on_output("Installed SAM v2 mask post-processing extension.")

    # Verify the import actually works — file presence alone is not enough
    if _try_import_sam2_c():
        return True

    # File is there but won't load — try ACL repair on existing file
    if not needs_copy:
        _normalize_acl(dest)
        if _try_import_sam2_c():
            logger.info("_C.pyd loaded after ACL repair")
            return True

    logger.warning(
        "sam2._C extension is installed at %s but cannot be loaded. "
        "SAM v2 will still work but mask hole-filling is disabled. "
        "See docs/2026-04-04-sam2-c-extension-bundling.md for details.",
        dest,
    )
    return False


def ensure_sam2_c_extension() -> None:
    """Runtime safety net: install _C.pyd if missing from sam2 package.

    Called from Sam2VideoBackend.initialize() before loading the model.
    Imports torch first (seeds Windows DLL search paths), then attempts
    to import sam2._C.  If the import fails, tries to copy and load the
    bundled extension.  Logs a clear status message on permanent failure.
    """
    if _try_import_sam2_c():
        return
    _install_sam2_c_extension()


def install_video_tracking(on_output=None) -> bool:
    """Install SAM v2 for video tracking on synthetic views.

    Requires torch already installed (from default tier).
    Syncs the locked optional ``video-tracking`` extra instead of
    mutating the environment ad hoc with ``uv add sam2``.
    """
    _quarantine_broken_sam2_namespace(on_output=on_output)

    ok = _run_uv_command([
        "sync",
        "--locked",
        "--no-dev",
        "--extra", "video-tracking",
    ], on_output=on_output)
    if not ok:
        return False

    if not _check_sam2_installed():
        logger.error("sam2 install completed but sam2.build_sam is still unavailable")
        if on_output:
            on_output("SAM v2 install appears incomplete: sam2.build_sam is missing.")
        return False

    # Install bundled _C.pyd (CUDA connected-components extension) into
    # the sam2 package.  Without this, SAM v2 skips mask hole-filling
    # post-processing and emits a warning on every propagation call.
    _install_sam2_c_extension(on_output=on_output)

    if on_output:
        on_output("SAM v2 installed. Model weights download on first use.")
    return True


def install_premium_tier(on_output=None) -> bool:
    """Install SAM 3 into the plugin venv.

    Requires torch already installed (from default tier).
    Downloads SAM 3 weights eagerly after install.
    """
    ok = _run_uv_command([
        "add", "sam3",
    ], on_output=on_output)
    if not ok:
        return False

    # Eagerly download SAM 3 weights
    try:
        if on_output:
            on_output("Downloading SAM 3 weights (~3.5 GB)...")
        download_model_weights()
        if on_output:
            on_output("SAM 3 weights downloaded.")
    except Exception as exc:
        logger.warning("SAM 3 weight download failed: %s", exc)

    return True
