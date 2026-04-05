# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify rig configs are accepted by COLMAP (pycolmap)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest

from core.presets import VIEW_PRESETS
from core.reframer import reframe_view
from core.rig_config import write_rig_config


@pytest.mark.parametrize("preset_name", ["cubemap", "low", "medium", "high"])
def test_colmap_accepts_rig_config(preset_name, tmp_path):
    """Feature extraction + rig_from_json must succeed for every preset."""
    config = VIEW_PRESETS[preset_name]
    images_dir = tmp_path / "images"

    # Generate a synthetic equirect and reframe into all views
    equirect = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    for yaw, pitch, fov, name, flip_v in config.get_all_views():
        view_dir = images_dir / name
        view_dir.mkdir(parents=True, exist_ok=True)
        face = reframe_view(equirect, fov_deg=fov, yaw_deg=yaw,
                            pitch_deg=pitch, out_size=256)
        if flip_v:
            face = np.flipud(face)
        cv2.imwrite(str(view_dir / "frame_001.jpg"), face)

    # Write rig config
    rig_path = str(tmp_path / "rig_config.json")
    write_rig_config(config, rig_path)

    # Validate structure
    with open(rig_path) as f:
        rig = json.load(f)
    cameras = rig[0]["cameras"]
    ref_count = sum(1 for c in cameras if c.get("ref_sensor"))
    assert ref_count == 1, f"Expected 1 ref_sensor, got {ref_count}"
    assert len(cameras) == config.total_views()

    # COLMAP feature extraction
    db_path = str(tmp_path / "database.db")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=str(images_dir),
    )

    # COLMAP rig config application — this is what failed before
    rig_configs = pycolmap.read_rig_config(rig_path)
    db = pycolmap.Database.open(db_path)
    pycolmap.apply_rig_config(rig_configs, db)
