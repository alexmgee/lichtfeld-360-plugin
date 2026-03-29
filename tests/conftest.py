# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pytest configuration — adds plugin root to sys.path."""

import sys
from pathlib import Path

# Ensure `core` package is importable without installing the plugin.
_plugin_root = str(Path(__file__).resolve().parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)
