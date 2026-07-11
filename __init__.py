# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Plugin for LichtFeld Studio."""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _lib_dir = Path(__file__).resolve().parent / "lib"
    if _lib_dir.is_dir():
        os.add_dll_directory(str(_lib_dir))
    try:
        # Apply a staged GPU-extraction enable/disable BEFORE anything can
        # import cv2 (a loaded cv2.pyd is locked and cannot be swapped).
        from .core.gpu_extraction_install import apply_pending

        apply_pending()
    except Exception:
        # GPU opt-in must never be able to break plugin load.
        pass
    _gpu_dir = _lib_dir / "gpu"
    if _gpu_dir.is_dir():
        os.add_dll_directory(str(_gpu_dir))

try:
    from .plugin import on_load, on_unload

    __all__ = ["on_load", "on_unload"]
except ImportError:
    # Allow importing sub-packages (e.g. core) outside the LichtFeld
    # plugin runtime — needed for standalone tests.
    pass
