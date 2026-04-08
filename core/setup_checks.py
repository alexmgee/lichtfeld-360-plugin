# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized masking setup state detection for operator masking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .sam3_compat import sam3_image_api_available, sam3_video_api_available

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

    # Video tracking (SAM v2, required for operator masking runs)
    has_sam2: bool = False

    # Premium tier (SAM 3)
    has_token: bool = False
    has_access: bool = False
    has_sam3: bool = False
    has_weights: bool = False

    # SAM 3.1 video tracking (future)
    has_sam3_video: bool = False

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
    def fullcircle_ready(self) -> bool:
        """FullCircle masking requires YOLO+SAM v1 image backend plus SAM v2 tracking."""
        return self.default_tier_ready and self.has_sam2

    @property
    def sam3_ready(self) -> bool:
        """SAM 3 cubemap masking requires only torch + sam3 + weights."""
        return self.has_torch and self.has_sam3 and self.has_weights

    @property
    def masking_ready(self) -> bool:
        """Backward-compatible alias for fullcircle_ready.

        Panel and pipeline code that predates method-specific gating
        may still reference this. It maps to the FullCircle contract only.
        """
        return self.fullcircle_ready

    @property
    def is_ready(self) -> bool:
        """True if any masking method is ready to run."""
        return self.fullcircle_ready or self.sam3_ready

    @property
    def first_incomplete_step(self) -> int:
        """For default tier setup: 0 = done, 1 = need deps."""
        if self.default_tier_ready:
            return 0
        return 1

    @property
    def video_tracking_ready(self) -> bool:
        """Operator masking runtime is ready for SAM v2 temporal tracking."""
        return self.masking_ready

    @property
    def capability_level(self) -> int:
        """Masking capability level for UI reporting.

        0 = nothing installed
        1 = image backend only (not enough for masking runs)
        2 = full masking stack (image backend + SAM v2)
        3 = SAM 3 image backend + SAM v2
        """
        if self.premium_tier_ready and self.has_sam2:
            return 3
        if self.masking_ready:
            return 2
        if self.active_backend is not None:
            return 1
        return 0


@dataclass
class Sam3SetupReport:
    """User-facing SAM 3 setup status for the panel onboarding flow."""

    token_status: str = "missing"       # missing | saved | verified | invalid | network_error
    access_status: str = "unknown"      # unknown | pending | granted | network_error
    runtime_status: str = "missing"     # missing | installed | broken
    weights_status: str = "missing"     # missing | present | failed
    overall_stage: str = "needs_token"  # needs_token | needs_access | ready_to_install | needs_weights | ready | error
    message: str = ""
    next_action: str = ""
    detail: str = ""


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
    return sam3_image_api_available()


def _check_sam3_video_installed() -> bool:
    return sam3_video_api_available()


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


def _classify_hf_access_exception(exc: Exception, *, token_verified: bool) -> tuple[str, str, str]:
    """Map HuggingFace verification failures to user-facing setup states."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(term in text for term in (
        "401", "unauthorized", "invalid token", "invalid user token",
        "authentication", "whoami",
    )):
        return "invalid", "unknown", "HuggingFace rejected this token."

    if any(term in text for term in (
        "403", "forbidden", "gated", "awaiting review", "access to model",
        "restricted", "not in the authorized list", "not have access",
    )):
        return ("verified" if token_verified else "saved"), "pending", (
            "SAM 3 model access is still pending HuggingFace approval."
        )

    if any(term in text for term in (
        "timed out", "connection", "network", "dns", "ssl", "temporarily unavailable",
        "name or service not known", "connection aborted", "connection reset",
    )):
        return ("verified" if token_verified else "saved"), "network_error", (
            "Could not reach HuggingFace to verify SAM 3 access."
        )

    return ("verified" if token_verified else "saved"), "unknown", (
        "Could not verify SAM 3 access."
    )


def _check_weights_downloaded() -> bool:
    """Check if SAM 3 weights exist in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(SAM3_MODEL_ID, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def _build_sam3_setup_report(
    state: MaskingSetupState,
    *,
    token_status: str | None = None,
    access_status: str | None = None,
    runtime_status: str | None = None,
    weights_status: str | None = None,
    detail: str = "",
) -> Sam3SetupReport:
    """Build a SAM 3 onboarding report from current setup state."""
    token_status = token_status or ("saved" if state.has_token else "missing")
    access_status = access_status or ("granted" if state.has_access else "unknown")
    runtime_status = runtime_status or ("installed" if state.has_sam3 else "missing")
    weights_status = weights_status or ("present" if state.has_weights else "missing")
    if access_status == "granted" and token_status == "saved":
        token_status = "verified"

    if runtime_status == "broken":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="error",
            message="SAM 3 install failed.",
            next_action="Retry the install, then click Re-check Setup.",
            detail=detail,
        )

    if token_status == "missing":
        return Sam3SetupReport(
            token_status=token_status,
            access_status="unknown",
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_token",
            message="SAM 3 setup needs a HuggingFace token.",
            next_action="Paste a HuggingFace token, then click Verify Access.",
            detail=detail,
        )

    if token_status == "invalid":
        return Sam3SetupReport(
            token_status=token_status,
            access_status="unknown",
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_token",
            message="HuggingFace rejected this token.",
            next_action="Create a new token and verify access again.",
            detail=detail,
        )

    if access_status == "network_error" or token_status == "network_error":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="Could not verify SAM 3 access right now.",
            next_action="Check your connection, then click Re-check Setup.",
            detail=detail,
        )

    if access_status == "pending":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="SAM 3 model access is still pending approval.",
            next_action="Wait for HuggingFace approval, then click Re-check Setup.",
            detail=detail,
        )

    if access_status != "granted":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="SAM 3 access has not been verified yet.",
            next_action="Click Check Setup or verify your HuggingFace token.",
            detail=detail,
        )

    if runtime_status != "installed":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="ready_to_install",
            message="SAM 3 access is approved and ready to install.",
            next_action="Click Install SAM 3 to add the runtime.",
            detail=detail,
        )

    if weights_status != "present":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_weights",
            message="SAM 3 runtime is installed but the model weights are missing.",
            next_action="Click Install SAM 3 to download the weights.",
            detail=detail,
        )

    return Sam3SetupReport(
        token_status=token_status,
        access_status=access_status,
        runtime_status=runtime_status,
        weights_status=weights_status,
        overall_stage="ready",
        message="SAM 3 is ready to use.",
        next_action="Enable masking to use SAM 3 Cubemap.",
        detail=detail,
    )


def check_sam3_setup(
    *,
    setup_state: MaskingSetupState | None = None,
    force_access_check: bool = False,
) -> Sam3SetupReport:
    """Return a user-facing SAM 3 setup report for onboarding and repair UX."""
    state = setup_state or check_masking_setup()
    token_status = "saved" if state.has_token else "missing"
    access_status = "granted" if state.has_access else "unknown"
    detail = ""

    if force_access_check and state.has_token:
        try:
            from huggingface_hub import get_token, model_info

            token = get_token()
            if token:
                info = model_info(SAM3_MODEL_ID, token=token)
                if info is not None:
                    access_status = "granted"
                    token_status = "verified"
                    state.has_access = True
                    global _hf_access_cache
                    _hf_access_cache = True
        except Exception as exc:
            token_status, access_status, detail = _classify_hf_access_exception(
                exc,
                token_verified=True,
            )

    return _build_sam3_setup_report(
        state,
        token_status=token_status,
        access_status=access_status,
        detail=detail,
    )


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
        has_sam3_video=_check_sam3_video_installed(),
    )


def is_operator_masking_ready() -> bool:
    """Fast runtime check for FullCircle masking readiness.

    FullCircle masking requires at least one image backend (YOLO+SAM v1
    or SAM 3) plus SAM v2 temporal tracking.
    """
    if not _check_torch_installed() or not _check_sam2_installed():
        return False
    if _check_yolo_installed() and _check_sam1_installed():
        return True
    return _check_sam3_installed() and _check_weights_downloaded()


def is_sam3_masking_ready() -> bool:
    """Fast runtime check for SAM 3 cubemap masking readiness.

    SAM 3 cubemap masking requires only torch + sam3 + weights.
    Does NOT require SAM v2, YOLO, or SAM v1.
    """
    return _check_torch_installed() and _check_sam3_installed() and _check_weights_downloaded()


def verify_hf_token(token: str) -> bool:
    """Verify a HuggingFace token has access to SAM 3.

    Saves the token via huggingface_hub.login() if valid.
    Returns True if token is valid and has model access.
    """
    report = verify_hf_token_detailed(token)
    return report.access_status == "granted"


def verify_hf_token_detailed(token: str) -> Sam3SetupReport:
    """Verify a HuggingFace token and return a detailed SAM 3 setup report."""
    global _hf_access_cache

    token = token.strip()
    if not token:
        return _build_sam3_setup_report(
            check_masking_setup(),
            token_status="missing",
        )

    try:
        from huggingface_hub import login, model_info

        login(token=token)
        info = model_info(SAM3_MODEL_ID, token=token)
        if info is not None:
            _hf_access_cache = True
            state = check_masking_setup()
            state.has_token = True
            state.has_access = True
            return _build_sam3_setup_report(
                state,
                token_status="verified",
                access_status="granted",
            )
    except Exception as exc:
        logger.warning("HF token verification failed: %s", exc)
        token_status, access_status, detail = _classify_hf_access_exception(
            exc,
            token_verified=True,
        )
        state = check_masking_setup()
        state.has_token = token_status not in {"missing", "invalid"}
        state.has_access = access_status == "granted"
        return _build_sam3_setup_report(
            state,
            token_status=token_status,
            access_status=access_status,
            detail=detail,
        )

    state = check_masking_setup()
    state.has_token = True
    return _build_sam3_setup_report(
        state,
        token_status="verified",
        access_status="unknown",
    )


def forget_hf_token() -> bool:
    """Remove the saved HuggingFace login for this machine."""
    global _hf_access_cache

    try:
        from huggingface_hub import logout

        logout()
        _hf_access_cache = None
        return True
    except Exception as exc:
        logger.warning("Could not forget saved HuggingFace token: %s", exc)
        return False


def make_sam3_install_failure_report(
    detail: str = "",
    *,
    setup_state: MaskingSetupState | None = None,
) -> Sam3SetupReport:
    """Create a SAM 3 report for local runtime/install failures."""
    state = setup_state or check_masking_setup()
    return _build_sam3_setup_report(
        state,
        access_status="granted" if state.has_access else "unknown",
        runtime_status="broken",
        weights_status="failed" if not state.has_weights else "present",
        detail=detail,
    )


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


def _download_sam1_weights(on_output=None) -> None:
    """Best-effort eager download of SAM v1 ViT-H weights."""
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


def _install_sam2_runtime(on_output=None) -> bool:
    """Repair or bootstrap the shipped SAM v2 runtime used by FullCircle."""
    _quarantine_broken_sam2_namespace(on_output=on_output)

    ok = _run_uv_command([
        "sync",
        "--locked",
        "--no-dev",
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
    return True


def install_default_tier(on_output=None) -> bool:
    """Repair or bootstrap the shipped FullCircle runtime in the plugin venv.

    FullCircle is expected to ship as part of the normal plugin runtime.
    This helper remains as a repair/bootstrap path for damaged or partial
    environments and still ensures the local SAM v1 weights are available.
    """
    if not _install_sam2_runtime(on_output=on_output):
        return False

    _download_sam1_weights(on_output=on_output)
    if on_output:
        on_output("FullCircle runtime ready. SAM v2 model weights download on first use.")
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
    """Legacy alias to repair the shipped SAM v2 runtime used by FullCircle."""
    if not _install_sam2_runtime(on_output=on_output):
        return False
    if on_output:
        on_output("SAM v2 runtime ready. Model weights download on first use.")
    return True


def install_premium_tier(on_output=None) -> bool:
    """Install SAM 3 into the plugin venv.

    Requires torch already installed (from default tier).
    Downloads SAM 3 weights eagerly after install.
    """
    ok = _run_uv_command([
        "sync",
        "--locked",
        "--no-dev",
        "--extra", "sam3-masking",
    ], on_output=on_output)
    if not ok:
        return False

    if not _check_sam3_installed():
        logger.error("sam3 install completed but image API is still unavailable")
        if on_output:
            on_output("SAM 3 install appears incomplete: image API import failed.")
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
