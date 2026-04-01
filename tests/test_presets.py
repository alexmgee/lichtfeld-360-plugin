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
        "balanced": 9,
        "standard": 13,
        "dense": 17,
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


def test_view_config_get_all_views_length():
    config = VIEW_PRESETS["standard"]
    views = config.get_all_views()
    assert len(views) == config.total_views()


def test_view_config_get_all_views_zenith_name():
    config = VIEW_PRESETS["standard"]
    views = config.get_all_views()
    names = [v[3] for v in views]
    assert "ZN_00" in names


def test_standard_preset_staggers_oblique_views_between_horizon_views():
    config = VIEW_PRESETS["standard"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert views["01_00"][:2] == (22.5, 25)
    assert views["01_01"][:2] == (202.5, 25)
    assert views["02_00"][:2] == (112.5, -25)
    assert views["02_01"][:2] == (292.5, -25)


def test_balanced_preset_uses_horizon_heavy_layout_with_two_oblique_views():
    config = VIEW_PRESETS["balanced"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert views["00_00"][:2] == (0.0, 0)
    assert views["00_01"][:2] == (60.0, 0)
    assert views["00_05"][:2] == (300.0, 0)
    assert views["01_00"][:2] == (30, 40)
    assert views["02_00"][:2] == (210, -40)


def test_dense_preset_staggers_upper_and_lower_rings_between_horizon_views():
    config = VIEW_PRESETS["dense"]
    views = {name: (yaw, pitch, fov) for yaw, pitch, fov, name, _flip in config.get_all_views()}

    assert views["01_00"][:2] == (22.5, 30)
    assert views["01_01"][:2] == (112.5, 30)
    assert views["02_00"][:2] == (67.5, -30)
    assert views["02_01"][:2] == (157.5, -30)

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
