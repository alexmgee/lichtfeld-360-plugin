# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical mapping from (projection, processing) to the pipeline's internal
output-mode string.

Output mode is two orthogonal axes:
  - projection: "erp" | "fisheye"
  - processing: "native" | "pinhole"

The panel exposes these as separate controls and derives the single internal
``output_mode`` string the pipeline dispatches on. This module is the one
source of truth for that mapping, kept free of any panel/heavy imports so it
stays unit-testable on its own.
"""

# Internal mode strings, in a stable order. The obsolete "erp_scaffold" mode
# is intentionally absent (COLMAP handles equirectangular natively).
OUTPUT_MODES = ("erp_native", "pinhole", "fisheye", "fisheye_pinhole")

_MODE_BY_AXES = {
    ("erp", "native"): "erp_native",
    ("erp", "pinhole"): "pinhole",
    ("fisheye", "native"): "fisheye",
    ("fisheye", "pinhole"): "fisheye_pinhole",
}


def output_mode_string(projection: str, processing: str) -> str:
    """Return the internal output_mode string for a (projection, processing) pair.

    Raises ValueError for any unknown combination.
    """
    try:
        return _MODE_BY_AXES[(projection, processing)]
    except KeyError:
        raise ValueError(
            f"Invalid (projection, processing): ({projection!r}, {processing!r})"
        ) from None


def output_mode_index(projection: str, processing: str) -> int:
    """Return the index of the derived mode within OUTPUT_MODES."""
    return OUTPUT_MODES.index(output_mode_string(projection, processing))
