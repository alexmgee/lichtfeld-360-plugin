# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pytest configuration — bootstraps Windows DLL lookup and plugin imports."""

import os
import subprocess
import sys
from pathlib import Path

_DLL_HANDLES = []


def _windows_dll_search_dirs() -> list[Path]:
    """Return candidate directories that may contain python3.dll on Windows."""
    candidates: list[Path] = []
    versioned_dll = f"python{sys.version_info.major}{sys.version_info.minor}.dll"

    for raw_path in {
        Path(sys.executable).resolve().parent,
        Path(sys.base_prefix).resolve(),
        Path(sys.base_exec_prefix).resolve(),
        Path(sys.exec_prefix).resolve(),
    }:
        candidates.append(raw_path)
        candidates.append(raw_path / "DLLs")

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    candidates.append(system_root / "System32")

    try:
        result = subprocess.run(
            ["where", "python3.dll"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        result = None

    if result is not None:
        for line in result.stdout.splitlines():
            dll_path = Path(line.strip())
            if dll_path.name.lower() == "python3.dll":
                candidates.append(dll_path.parent)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        unique_candidates.append(path)

    compatible_dirs = [
        path for path in unique_candidates
        if (path / "python3.dll").exists() and (path / versioned_dll).exists()
    ]

    return compatible_dirs + [
        path for path in unique_candidates
        if path.name.lower() == "system32"
    ]


def _configure_windows_dll_search() -> None:
    """Expose python3.dll for extension modules such as OpenCV during tests."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    for directory in _windows_dll_search_dirs():
        _DLL_HANDLES.append(os.add_dll_directory(str(directory)))


_configure_windows_dll_search()

# Ensure `core` package is importable without installing the plugin.
_plugin_root = str(Path(__file__).resolve().parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)
