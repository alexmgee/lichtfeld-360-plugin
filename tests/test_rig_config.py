"""Tests for dual fisheye rig config generation."""
import json
import tempfile
from pathlib import Path

import pytest

from core.rig_config import write_dual_fisheye_rig_config


def test_rig_config_structure():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig_config.json"
        write_dual_fisheye_rig_config(
            str(path),
            baseline_m=0.025,
            rotation_quat=[0, 0, 1, 0],
        )
        data = json.loads(path.read_text())

    assert len(data) == 1
    cameras = data[0]["cameras"]
    assert len(cameras) == 2

    # Front is ref sensor
    assert cameras[0]["image_prefix"] == "front/"
    assert cameras[0]["ref_sensor"] is True

    # Back has rotation and translation
    assert cameras[1]["image_prefix"] == "back/"
    assert cameras[1]["cam_from_rig_rotation"] == [0, 0, 1, 0]
    # Translation Z must be negative (back camera looks opposite to front)
    assert cameras[1]["cam_from_rig_translation"][2] < 0
    assert abs(cameras[1]["cam_from_rig_translation"][2] - (-0.025)) < 1e-9


def test_rig_config_rejects_wrong_sign():
    """Negative baseline_m produces positive Z translation → assertion fires."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig_config.json"
        with pytest.raises(AssertionError):
            write_dual_fisheye_rig_config(str(path), baseline_m=-0.025)


def test_rig_config_custom_baseline():
    """Custom baseline value propagates correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig_config.json"
        write_dual_fisheye_rig_config(str(path), baseline_m=0.030)
        data = json.loads(path.read_text())
    assert abs(data[0]["cameras"][1]["cam_from_rig_translation"][2] - (-0.030)) < 1e-9
