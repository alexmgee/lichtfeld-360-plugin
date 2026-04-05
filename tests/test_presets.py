# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ring/view preset configuration system."""

from core.presets import Ring, ViewConfig, VIEW_PRESETS, DEFAULT_PRESET


def test_ring_yaw_positions():
    ring = Ring(pitch=0, count=8, fov=65)
    yaws = ring.get_yaw_positions()
    assert len(yaws) == 8
    assert yaws[0] == 0.0
    assert abs(yaws[1] - 45.0) < 0.01


def test_ring_zero_count():
    ring = Ring(pitch=0, count=0, fov=65)
    assert ring.get_yaw_positions() == []


def test_ring_start_yaw_offset():
    ring = Ring(pitch=0, count=4, fov=65, start_yaw=10.0)
    yaws = ring.get_yaw_positions()
    assert yaws[0] == 10.0
    assert abs(yaws[1] - 100.0) < 0.01


def test_default_preset_is_cubemap():
    assert DEFAULT_PRESET == "cubemap"
    assert DEFAULT_PRESET in VIEW_PRESETS


def test_all_preset_counts():
    expected = {
        "cubemap": 6,
        "low": 9,
        "medium": 14,
        "high": 18,
    }
    for name, count in expected.items():
        assert (
            VIEW_PRESETS[name].total_views() == count
        ), f"{name} should have {count} views"


def test_view_config_serialization():
    config = VIEW_PRESETS["cubemap"]
    d = config.to_dict()
    restored = ViewConfig.from_dict(d)
    assert restored.total_views() == config.total_views()


def test_freeform_view_config_serialization():
    config = VIEW_PRESETS["low"]
    d = config.to_dict()
    restored = ViewConfig.from_dict(d)
    assert restored.total_views() == config.total_views()
    orig_views = config.get_all_views()
    rest_views = restored.get_all_views()
    for orig, rest in zip(orig_views, rest_views):
        assert orig[3] == rest[3], "Names should match"
        assert orig[0] == rest[0], "Yaw should match"
        assert orig[1] == rest[1], "Pitch should match"


def test_view_config_get_all_views_length():
    config = VIEW_PRESETS["medium"]
    views = config.get_all_views()
    assert len(views) == config.total_views()


def test_view_config_get_all_views_zenith_name():
    config = VIEW_PRESETS["medium"]
    views = config.get_all_views()
    names = [v[3] for v in views]
    assert "ZN_00" in names


def test_medium_preset_freeform_camera_positions():
    config = VIEW_PRESETS["medium"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert len(views) == 14
    assert views["00_00"][:2] == (159, 12)
    assert views["01_00"][:2] == (23, 30)
    assert views["02_00"][:2] == (68, -30)
    assert views["ZN_00"][:2] == (0, 90)


def test_low_preset_freeform_camera_positions():
    config = VIEW_PRESETS["low"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert len(views) == 9
    assert views["00_00"][:2] == (52, 16)
    assert views["00_05"][:2] == (-37, 69)
    assert views["01_00"][:2] == (30, -62)
    assert views["02_00"][:2] == (210, -44)
    assert views["02_01"][:2] == (-19, 0)


def test_high_preset_freeform_camera_positions():
    config = VIEW_PRESETS["high"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert len(views) == 18
    assert views["ZN_00"][:2] == (0, 90)
    assert views["ND_00"][:2] == (180, -90)
    assert views["00_00"][:2] == (3, 34)
    assert views["02_03"][:2] == (-180, -29)

def test_cubemap_no_zenith_nadir():
    config = VIEW_PRESETS["cubemap"]
    views = config.get_all_views()
    names = [v[3] for v in views]
    assert "ZN_00" not in names
    assert "ND_00" not in names


def test_cubemap_swaps_01_00_and_02_00_pole_faces():
    config = VIEW_PRESETS["cubemap"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert views["01_00"][:2] == (0.0, -90)
    assert views["02_00"][:2] == (0.0, 90)
