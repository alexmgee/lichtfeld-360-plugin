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
        "full": 26,
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


def test_view_config_nadir_included():
    config = VIEW_PRESETS["full"]
    views = config.get_all_views()
    names = [v[3] for v in views]
    assert "ND_00" in names


def test_cubemap_no_zenith_nadir():
    config = VIEW_PRESETS["cubemap"]
    views = config.get_all_views()
    names = [v[3] for v in views]
    assert "ZN_00" not in names
    assert "ND_00" not in names
