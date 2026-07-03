# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dual-fisheye calibration resolution.

This module keeps camera-family decisions in one place so an Insta360 path
cannot accidentally inherit DJI Osmo 360 calibration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .fisheye_calibration import (
    DualFisheyeCalibration,
    FisheyeCalibration,
    default_osmo360_calibration,
)


SCHEMA_V1 = "lichtfeld.dual_fisheye_calibration.v1"
ENV_CALIBRATION_PATH = "LICHTFELD_DUAL_FISHEYE_CALIBRATION"
SOURCE_CONFIDENCE_VALUES = {
    "measured",
    "community_reference",
    "reverse_engineered",
    "sdk_reported",
    "unknown",
}


class CalibrationResolutionError(ValueError):
    """Raised when no safe dual-fisheye calibration can be resolved."""


@dataclass(frozen=True)
class CalibrationResolution:
    """Resolved calibration plus provenance for logs and manifests."""

    calibration: DualFisheyeCalibration
    source: str
    source_path: Optional[str] = None
    source_confidence: str = "unknown"
    warning: Optional[str] = None


def resolve_dual_fisheye_calibration(
    camera_family: Optional[str],
    *,
    override_path: str = "",
    output_mode: str = "fisheye_pinhole",
) -> CalibrationResolution:
    """Resolve calibration for dual-fisheye pinhole reframing.

    DJI Osmo 360 has a built-in empirical calibration. Insta360 falls back to
    an unverified estimated calibration so the plugin remains all-in-one while
    keeping provenance explicit.
    """

    family = (camera_family or "").strip().lower()
    explicit_override = _coalesce_override_path(override_path)
    if explicit_override:
        return load_dual_fisheye_calibration_override(explicit_override, family)

    if family == "dji_osmo360":
        return CalibrationResolution(
            calibration=default_osmo360_calibration(),
            source="builtin:dji_osmo360",
            source_confidence="measured",
        )

    if family == "insta360":
        return CalibrationResolution(
            calibration=estimated_insta360_calibration(),
            source="builtin:insta360_estimated_generic",
            source_confidence="unknown",
            warning=(
                "Using unverified generic Insta360 calibration. This keeps "
                "Fisheye (Pinhole) mode usable without user calibration, but "
                "geometry should be validated against real captures."
            ),
        )

    raise CalibrationResolutionError(
        "Fisheye (Pinhole) mode needs a known camera family or an explicit "
        "dual-fisheye calibration JSON. "
        f"camera_family={camera_family!r}, output_mode={output_mode!r}"
    )


def estimated_insta360_calibration(
    image_size: tuple[int, int] = (3840, 3840),
) -> DualFisheyeCalibration:
    """Return an unverified generic Insta360 dual-fisheye calibration.

    This is intentionally labeled low-confidence. The values model a centered
    approximately 190-degree fisheye and avoid reusing DJI-specific lens
    centers/distortion/baseline for Insta360 captures.
    """

    width, height = image_size
    f = min(width, height) / 3.65
    cx = width / 2.0
    cy = height / 2.0
    k = np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d = np.zeros((4, 1), dtype=np.float64)
    front = FisheyeCalibration(
        camera_matrix=k.copy(),
        dist_coeffs=d.copy(),
        image_size=image_size,
        rms_error=-1.0,
        num_images_used=0,
        fov_degrees=190.0,
    )
    back = FisheyeCalibration(
        camera_matrix=k.copy(),
        dist_coeffs=d.copy(),
        image_size=image_size,
        rms_error=-1.0,
        num_images_used=0,
        fov_degrees=190.0,
    )
    return DualFisheyeCalibration(
        front=front,
        back=back,
        front_rotation_deg=0.0,
        back_rotation_deg=180.0,
        camera_model="Insta360 generic estimated dual fisheye (unverified)",
        baseline_m=0.0,
        baseline_axis=(0.0, 0.0, 1.0),
    )


def load_dual_fisheye_calibration_override(
    path: str | Path,
    expected_family: str = "",
) -> CalibrationResolution:
    """Load a user-supplied dual-fisheye calibration JSON."""

    p = Path(path).expanduser()
    if not p.is_file():
        raise CalibrationResolutionError(f"Calibration file not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationResolutionError(
            f"Calibration file is not valid JSON: {p}: {exc}"
        ) from exc

    schema = data.get("schema")
    if schema == SCHEMA_V1:
        calibration = _load_schema_v1(data, p)
        family = str(data.get("camera_family", "")).strip().lower()
        confidence = str(data.get("source_confidence", "unknown")).strip()
        source = str(data.get("source", "user_calibration")).strip() or "user_calibration"
        if confidence not in SOURCE_CONFIDENCE_VALUES:
            raise CalibrationResolutionError(
                f"Invalid source_confidence {confidence!r} in {p}; expected one "
                f"of {sorted(SOURCE_CONFIDENCE_VALUES)}"
            )
    elif _looks_like_legacy_dual_calibration(data):
        calibration = DualFisheyeCalibration.load(str(p))
        family = str(data.get("camera_family", "")).strip().lower()
        confidence = "unknown"
        source = "legacy_dual_fisheye_calibration"
    else:
        raise CalibrationResolutionError(
            f"Unsupported calibration schema in {p}. Expected {SCHEMA_V1!r} "
            "or a legacy DualFisheyeCalibration JSON."
        )

    normalized_expected = (expected_family or "").strip().lower()
    if normalized_expected and family and family != normalized_expected:
        raise CalibrationResolutionError(
            f"Calibration family {family!r} does not match input camera family "
            f"{normalized_expected!r}: {p}"
        )

    warning = None
    if confidence != "measured":
        warning = (
            f"Calibration source confidence is {confidence!r}; validate "
            "reconstruction quality before relying on this preset."
        )

    return CalibrationResolution(
        calibration=calibration,
        source=source,
        source_path=str(p),
        source_confidence=confidence,
        warning=warning,
    )


def _coalesce_override_path(override_path: str) -> str:
    value = (override_path or "").strip()
    if value:
        return value
    return os.environ.get(ENV_CALIBRATION_PATH, "").strip()


def _load_schema_v1(data: dict[str, Any], path: Path) -> DualFisheyeCalibration:
    camera_family = str(data.get("camera_family", "")).strip()
    if not camera_family:
        raise CalibrationResolutionError(f"Missing camera_family in {path}")

    front = _load_lens("front", data.get("front"), data.get("image_size"), path)
    back = _load_lens("back", data.get("back"), data.get("image_size"), path)
    rig = data.get("rig") or {}
    baseline_axis = _tuple_of_floats(
        rig.get("baseline_axis", (0.0, 0.0, 1.0)),
        3,
        "rig.baseline_axis",
        path,
    )

    return DualFisheyeCalibration(
        front=front,
        back=back,
        front_rotation_deg=float(rig.get("front_rotation_deg", 0.0)),
        back_rotation_deg=float(rig.get("back_rotation_deg", 180.0)),
        camera_model=str(data.get("camera_model", "Unknown")),
        baseline_m=float(rig.get("baseline_m", 0.0)),
        baseline_axis=baseline_axis,
    )


def _load_lens(
    name: str,
    lens_data: Any,
    shared_image_size: Any,
    path: Path,
) -> FisheyeCalibration:
    if not isinstance(lens_data, dict):
        raise CalibrationResolutionError(f"Missing {name} lens calibration in {path}")

    camera_matrix = np.array(
        lens_data.get("camera_matrix"),
        dtype=np.float64,
    )
    if camera_matrix.shape != (3, 3):
        raise CalibrationResolutionError(
            f"{name}.camera_matrix must be 3x3 in {path}"
        )

    dist_coeffs = np.array(lens_data.get("dist_coeffs"), dtype=np.float64).reshape(-1)
    if dist_coeffs.shape != (4,):
        raise CalibrationResolutionError(
            f"{name}.dist_coeffs must contain four fisheye coefficients in {path}"
        )

    image_size_raw = lens_data.get("image_size", shared_image_size)
    image_size = _tuple_of_ints(image_size_raw, 2, f"{name}.image_size", path)

    validation = lens_data.get("validation") or {}
    return FisheyeCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(4, 1),
        image_size=image_size,
        rms_error=float(lens_data.get("rms_error", validation.get("rms_error", -1.0))),
        num_images_used=int(
            lens_data.get("num_images_used", validation.get("num_images_used", 0)) or 0
        ),
        fov_degrees=float(lens_data.get("fov_degrees", 190.0)),
    )


def _tuple_of_ints(value: Any, length: int, field: str, path: Path) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CalibrationResolutionError(
            f"{field} must be a {length}-element array in {path}"
        )
    return tuple(int(v) for v in value)


def _tuple_of_floats(value: Any, length: int, field: str, path: Path) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CalibrationResolutionError(
            f"{field} must be a {length}-element array in {path}"
        )
    return tuple(float(v) for v in value)


def _looks_like_legacy_dual_calibration(data: dict[str, Any]) -> bool:
    return (
        isinstance(data.get("front"), dict)
        and isinstance(data.get("back"), dict)
        and "camera_matrix" in data["front"]
        and "camera_matrix" in data["back"]
    )
