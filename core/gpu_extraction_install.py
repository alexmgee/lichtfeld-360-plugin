# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""State + staging + apply/rollback for opt-in GPU frame extraction.

Pure stdlib on purpose: this module is imported from ``__init__.py`` on
every plugin load (to apply a staged swap before anything imports cv2),
so it must never depend on cv2/torch/pycolmap. See the extractor probe
in ``core/sharpest_extractor.py`` for the runtime GPU capability check;
nothing here imports OpenCV.

The installed CUDA OpenCV build is version-camouflaged as the lock's CPU
pin so the host's ``uv sync`` leaves it in place. A build is identified
as CUDA by a ``CUDA_PATH`` default in its ``cv2/config.py`` (the CPU
wheel has none) — this is the single fingerprint used everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ".gpu_extraction_state.json"

_DEFAULT_STATE = {
    "state": "disabled",
    "detail": "",
    "staged_at": "",
    "wheel_sha256": "",
    "camouflage_version": "",
}


def get_state(root: Path = PLUGIN_ROOT) -> dict:
    """Return the state dict; the default (disabled) if no file exists.

    Reading never creates the file. A corrupt file degrades to default
    rather than raising — plugin load must never break on it.
    """
    path = Path(root) / STATE_FILE
    if not path.exists():
        return dict(_DEFAULT_STATE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)
    merged = dict(_DEFAULT_STATE)
    merged.update(data)
    return merged


def set_state(state: str, root: Path = PLUGIN_ROOT, **fields) -> None:
    """Write ``state`` plus any extra fields, preserving existing ones."""
    data = get_state(root=root)
    data["state"] = state
    data.update(fields)
    (Path(root) / STATE_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8")


def _site_packages(root: Path) -> Path:
    return Path(root) / ".venv" / "Lib" / "site-packages"


def _installed_build(root: Path) -> str:
    """Classify the installed cv2 as 'cuda', 'cpu', or 'missing'.

    'missing' covers both an absent ``cv2/config.py`` and an absent
    opencv dist-info — either is a crash remnant the host sync self-heals
    and, for an ``active`` state, means we were reverted.
    """
    sp = _site_packages(root)
    config = sp / "cv2" / "config.py"
    if not config.exists():
        return "missing"
    if not list(sp.glob("opencv_contrib_python-*.dist-info")):
        return "missing"
    text = config.read_text(encoding="utf-8", errors="replace")
    return "cuda" if "CUDA_PATH" in text else "cpu"


def detect_runtime_state(root: Path = PLUGIN_ROOT) -> str:
    """Reconcile the recorded state against the installed cv2.

    Only ``active`` is cross-checked: if the installed build is no longer
    the CUDA one (host sync or a manual action undid us, or a crash left
    no dist-info), transition to ``reverted`` and persist it. All other
    states pass through unchanged. Cheap and cv2-free; safe from the
    panel probe thread.
    """
    state = get_state(root=root)["state"]
    if state != "active":
        return state
    if _installed_build(root) == "cuda":
        return "active"
    set_state("reverted", root=root)
    return "reverted"
