# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-fast provenance check for the in-process pycolmap module.

LichtFeld Studio runs every plugin in one shared Python interpreter and
persistently prepends each plugin venv's site-packages to sys.path, so the
first plugin to import pycolmap wins process-wide. When a sibling plugin
ships a different pycolmap version, this plugin silently executes that
version's native code — broken GPU BA options and hard crashes follow
(see docs/pycolmap-collision-guard-and-upstream-plan.md).

This guard cannot fix the collision (sys.modules is poisoned before any
plugin code runs); it converts undefined behavior into an actionable error.
Full isolation of pycolmap behind the venv subprocess is the planned cure.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


def verify_pycolmap_provenance(pycolmap_module) -> Optional[str]:
    """Return an actionable error string if pycolmap was not loaded from
    this plugin's venv, else None."""
    module_file = getattr(pycolmap_module, "__file__", None)
    if not module_file:
        return None
    loaded = os.path.normcase(str(Path(module_file).resolve()))
    expected_root = os.path.normcase(str((_PLUGIN_DIR / ".venv").resolve()))
    if loaded.startswith(expected_root + os.sep):
        # Right venv — but a CPU pycolmap inside OUR OWN venv (e.g. an
        # accidental `pip install -U pycolmap` pulling PyPI's CPU-only
        # Windows wheel) breaks GPU BA. PyPI's wheel reports has_cuda=False;
        # the pinned CUDA wheel reports True.
        if getattr(pycolmap_module, "has_cuda", None) is False:
            version = getattr(pycolmap_module, "__version__", "unknown")
            logger.error(
                "pycolmap %s in the plugin venv has no CUDA "
                "(has_cuda=False) — the CUDA wheel was replaced, likely by "
                "PyPI's CPU-only build", version,
            )
            return (
                f"COLMAP blocked: the installed pycolmap {version} is a "
                f"CPU-only build (has_cuda=False). The plugin requires its "
                f"pinned CUDA wheel. Fix: run 'uv sync' in the plugin "
                f"directory (the lockfile pins the correct wheel), then "
                f"restart LichtFeld Studio. Never run 'pip install -U "
                f"pycolmap' — PyPI's Windows wheel has no CUDA."
            )
        return None
    version = getattr(pycolmap_module, "__version__", "unknown")
    culprit = _plugin_name_from_path(Path(module_file))
    logger.error(
        "pycolmap provenance mismatch: loaded %s (version %s), expected a "
        "module under %s",
        loaded, version, expected_root,
    )
    source = f"the '{culprit}' plugin" if culprit else f"outside this plugin ({module_file})"
    workaround = (
        f"disable or uninstall '{culprit}'" if culprit
        else "disable other plugins that bundle pycolmap"
    )
    return (
        f"COLMAP blocked: LichtFeld loaded pycolmap {version} from {source} "
        f"instead of this plugin's own copy. Plugins share one Python "
        f"process and the first import wins, which breaks GPU bundle "
        f"adjustment and can crash the app. Workaround: {workaround}, then "
        f"restart LichtFeld Studio. (A proper fix is in progress.)"
    )


def check_loaded_pycolmap() -> Optional[str]:
    """Provenance-check the pycolmap already in sys.modules, if any.

    Returns the error string, or None when pycolmap is absent (a later
    import is checked at ColmapRunner.run) or correctly ours.
    """
    module = sys.modules.get("pycolmap")
    if module is None:
        return None
    return verify_pycolmap_provenance(module)


def _plugin_name_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part.lower() == "plugins":
            return parts[i + 1]
    return None
