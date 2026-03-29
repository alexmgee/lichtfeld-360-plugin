# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized masking setup state detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAM3_MODEL_ID = "facebook/sam3.1"

_hf_access_cache: bool | None = None


@dataclass
class MaskingSetupState:
    """Current state of masking dependencies."""

    has_token: bool = False
    has_access: bool = False
    has_torch: bool = False
    has_sam3: bool = False
    has_weights: bool = False

    @property
    def is_ready(self) -> bool:
        # If deps + weights are present, masking works regardless of token
        # (token is only needed for initial download)
        if self.has_torch and self.has_sam3 and self.has_weights:
            return True
        return all([
            self.has_token, self.has_torch, self.has_sam3, self.has_weights,
        ])

    @property
    def first_incomplete_step(self) -> int:
        """1 = HF access, 2 = dependencies, 3 = weights, 0 = all done."""
        # If everything needed to run is present, skip all setup
        if self.has_torch and self.has_sam3 and self.has_weights:
            return 0
        if not self.has_token:
            return 1
        if not self.has_torch or not self.has_sam3:
            return 2
        if not self.has_weights:
            return 3
        return 0


def _check_hf_token() -> bool:
    try:
        from huggingface_hub import get_token

        token = get_token()
        return token is not None and len(token) > 0
    except Exception:
        return False


def _check_hf_access() -> bool:
    """Check if cached token has access to SAM 3.1 model.

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


def _check_torch_installed() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _check_sam3_installed() -> bool:
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: F401

        return True
    except ImportError:
        return False


def _check_weights_downloaded() -> bool:
    """Check if SAM 3.1 weights exist in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(SAM3_MODEL_ID, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def check_masking_setup() -> MaskingSetupState:
    """Check all masking setup conditions. Returns current state."""
    token = _check_hf_token()
    access = _check_hf_access()
    torch_ok = _check_torch_installed()
    sam3_ok = _check_sam3_installed()
    weights = _check_weights_downloaded()
    return MaskingSetupState(
        has_token=token,
        has_access=access,
        has_torch=torch_ok,
        has_sam3=sam3_ok,
        has_weights=weights,
    )


def verify_hf_token(token: str) -> bool:
    """Verify a HuggingFace token has access to SAM 3.1.

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
    """Download SAM 3.1 model weights from HuggingFace.

    Returns True on success.
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
_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"


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


def _venv_python() -> Path:
    """Get the plugin venv's Python interpreter."""
    import os
    if os.name == "nt":
        return _VENV_DIR / "Scripts" / "python.exe"
    return _VENV_DIR / "bin" / "python"


def install_torch_to_plugin_venv(on_output=None) -> bool:
    """Install torch + torchvision into the plugin's own .venv/.

    Uses the CUDA 12.8 PyTorch index (same as densification plugin).
    """
    uv = _find_uv()
    if not uv:
        logger.error("uv not found — cannot install torch")
        return False

    venv_py = _venv_python()
    if not venv_py.exists():
        logger.error("Plugin venv not found at %s", venv_py)
        return False

    import subprocess, os
    cmd = [
        uv, "pip", "install",
        "torch", "torchvision",
        "--extra-index-url", _PYTORCH_INDEX,
        "--python", str(venv_py),
    ]
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    logger.info("Installing torch: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **flags,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line and on_output:
            on_output(line)
    proc.wait()
    return proc.returncode == 0


def install_sam3_to_plugin_venv(on_output=None) -> bool:
    """Install sam3.1 + triton into the plugin's own .venv/.

    Installs from the GitHub repo (not PyPI) since sam3.1 requires
    the facebookresearch/sam3 source with HuggingFace model access.
    """
    uv = _find_uv()
    if not uv:
        logger.error("uv not found — cannot install sam3")
        return False

    venv_py = _venv_python()
    if not venv_py.exists():
        logger.error("Plugin venv not found at %s", venv_py)
        return False

    import subprocess, os
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    # Install triton first (sam3 dependency)
    cmd_triton = [
        uv, "pip", "install", "triton",
        "--python", str(venv_py),
    ]
    logger.info("Installing triton: %s", " ".join(cmd_triton))
    proc = subprocess.Popen(
        cmd_triton, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **flags,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line and on_output:
            on_output(line)
    proc.wait()
    if proc.returncode != 0:
        logger.error("triton installation failed")
        return False

    # Install sam3 from GitHub
    cmd_sam3 = [
        uv, "pip", "install",
        "sam3 @ git+https://github.com/facebookresearch/sam3.git",
        "--python", str(venv_py),
    ]
    logger.info("Installing sam3: %s", " ".join(cmd_sam3))
    proc = subprocess.Popen(
        cmd_sam3, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, **flags,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line and on_output:
            on_output(line)
    proc.wait()
    return proc.returncode == 0
