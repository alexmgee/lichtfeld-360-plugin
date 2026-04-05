# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local-only diagnostics for evaluating 360 preset layouts.

Usage:
    .venv\Scripts\python.exe dev\preset_diagnostics.py
    .venv\Scripts\python.exe dev\preset_diagnostics.py --preset standard

This script intentionally loads ``core/presets.py`` directly so it can run
without importing the full plugin package.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Iterable


def _load_presets_module():
    repo_root = Path(__file__).resolve().parents[1]
    presets_path = repo_root / "core" / "presets.py"
    spec = importlib.util.spec_from_file_location("preset_diagnostics_presets", presets_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load presets module from {presets_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spherical_to_vector(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cos_pitch = math.cos(pitch)
    return (
        math.sin(yaw) * cos_pitch,
        math.sin(pitch),
        math.cos(yaw) * cos_pitch,
    )


def _angular_distance_deg(a: Iterable[float], b: Iterable[float]) -> float:
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def _pairwise_angles(vectors: list[tuple[str, tuple[float, float, float]]]) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for i, (name_a, vec_a) in enumerate(vectors):
        distances: list[tuple[str, float]] = []
        for j, (name_b, vec_b) in enumerate(vectors):
            if i == j:
                continue
            distances.append((name_b, _angular_distance_deg(vec_a, vec_b)))
        distances.sort(key=lambda item: item[1])
        out[name_a] = distances
    return out


def _format_fovs(views) -> str:
    unique = sorted({round(float(fov), 4) for _yaw, _pitch, fov, _name in views})
    if len(unique) == 1:
        return f"shared {unique[0]:g} deg"
    return "mixed " + ", ".join(f"{value:g}" for value in unique) + " deg"


def analyze_preset(name: str, config) -> str:
    views = config.get_all_views()
    vectors = [
        (view_name, _spherical_to_vector(yaw_deg, pitch_deg))
        for yaw_deg, pitch_deg, _fov_deg, view_name in views
    ]
    distances = _pairwise_angles(vectors)
    nearest = {name: neighbors[0] for name, neighbors in distances.items()}
    nearest_values = [angle for _neighbor, angle in nearest.values()]

    lines = [
        f"[{name}]",
        f"views: {len(views)}",
        f"fov: {_format_fovs(views)}",
        (
            "nearest-neighbor angle: "
            f"min {min(nearest_values):.2f} deg, "
            f"max {max(nearest_values):.2f} deg, "
            f"avg {sum(nearest_values) / len(nearest_values):.2f} deg"
        ),
        "view layout:",
    ]

    for yaw_deg, pitch_deg, fov_deg, view_name in views:
        neighbor_name, neighbor_angle = nearest[view_name]
        lines.append(
            "  "
            f"{view_name:>5}  yaw={yaw_deg:>6.1f}  pitch={pitch_deg:>6.1f}  "
            f"fov={fov_deg:>5.1f}  nearest={neighbor_name} ({neighbor_angle:.2f} deg)"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["cubemap", "balanced", "standard", "dense", "full"])
    args = parser.parse_args()

    module = _load_presets_module()
    preset_names = [args.preset] if args.preset else list(module.VIEW_PRESETS)

    for index, preset_name in enumerate(preset_names):
        if index:
            print()
        print(analyze_preset(preset_name, module.VIEW_PRESETS[preset_name]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
