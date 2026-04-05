# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""PanoSplat plugin for LichtFeld Studio."""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _lib_dir = Path(__file__).resolve().parent / "lib"
    if _lib_dir.is_dir():
        os.add_dll_directory(str(_lib_dir))

try:
    from .plugin import on_load, on_unload

    __all__ = ["on_load", "on_unload"]
except ImportError:
    # Allow importing sub-packages (e.g. core) outside the LichtFeld
    # plugin runtime — needed for standalone tests.
    pass
