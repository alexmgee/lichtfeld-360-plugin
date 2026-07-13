# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP artifact cleanup helpers used by the native-ERP output path."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def cleanup_colmap_artifacts(output_dir: Path, log_fn: Callable[..., object] = logger.info) -> None:
    """Delete COLMAP reconstruction artifacts after native-ERP export.

    Called once rig poses have been extracted from the reconstruction. The
    ``sparse/`` model references pinhole crop camera geometry, but ``images/``
    now contains ERP frames, so loading ``sparse/`` would produce a broken
    dataset (the same ERP image assigned to every pinhole camera slot). These
    artifacts must be removed.
    """
    import shutil
    for item_name in ("sparse", "database.db", "rig_config.json",
                       "database.db-wal", "database.db-shm"):
        target = output_dir / item_name
        if target.is_dir():
            shutil.rmtree(str(target), ignore_errors=True)
        elif target.is_file():
            target.unlink(missing_ok=True)
    log_fn("Cleaned COLMAP reconstruction artifacts from %s", output_dir)
