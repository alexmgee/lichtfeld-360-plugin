# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for loading and summarizing masking diagnostics documents."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _round_pct(value: float) -> float:
    return round(float(value), 3)


def _avg(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return _round_pct(sum(vals) / len(vals))


def build_mask_diagnostics_summary(frames: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate frame-level diagnostics into a compact summary."""
    face_values: dict[str, list[float]] = defaultdict(list)
    view_max_removed: dict[str, float] = defaultdict(float)
    flag_counts: Counter[str] = Counter()

    pre_coverages: list[float] = []
    post_coverages: list[float] = []
    views_with_removed_pixels: list[float] = []
    max_removed_view_pct = 0.0

    for frame in frames:
        pre_coverages.append(float(frame.get("erp_detection_coverage_pct_pre", 0.0)))
        post_coverages.append(float(frame.get("erp_detection_coverage_pct_post", 0.0)))
        views_with_removed_pixels.append(float(frame.get("views_with_removed_pixels", 0.0)))

        for flag in frame.get("flags", []) or []:
            flag_counts[str(flag)] += 1

        for face_name, pct in (frame.get("face_detection_pct") or {}).items():
            face_values[str(face_name)].append(float(pct))

        for view_name, pct in (frame.get("per_view_removed_pct") or {}).items():
            value = float(pct)
            if value > view_max_removed[str(view_name)]:
                view_max_removed[str(view_name)] = value
            if value > max_removed_view_pct:
                max_removed_view_pct = value

    flagged_frames = [
        frame for frame in frames
        if frame.get("flags")
    ]
    ranked_frames = sorted(
        flagged_frames,
        key=lambda frame: (
            -len(frame.get("flags", []) or []),
            -float(frame.get("erp_detection_coverage_pct_post", 0.0)),
            str(frame.get("frame", "")),
        ),
    )

    return {
        "flagged_frames": len(flagged_frames),
        "flag_counts": dict(sorted(flag_counts.items())),
        "avg_erp_detection_coverage_pct_pre": _avg(pre_coverages),
        "avg_erp_detection_coverage_pct_post": _avg(post_coverages),
        "max_erp_detection_coverage_pct_post": (
            _round_pct(max(post_coverages)) if post_coverages else 0.0
        ),
        "avg_views_with_removed_pixels": _avg(views_with_removed_pixels),
        "max_removed_view_pct": _round_pct(max_removed_view_pct),
        "avg_face_detection_pct": {
            face_name: _avg(values)
            for face_name, values in sorted(face_values.items())
        },
        "max_per_view_removed_pct": {
            view_name: _round_pct(value)
            for view_name, value in sorted(view_max_removed.items())
        },
        "top_flagged_frames": [
            str(frame.get("frame", ""))
            for frame in ranked_frames[:5]
            if frame.get("frame")
        ],
    }


def load_mask_diagnostics_document(path: str | Path) -> dict[str, Any] | None:
    """Load a masking diagnostics JSON document from disk."""
    if not path:
        return None

    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    if not isinstance(doc, dict):
        return None
    return doc


def get_mask_diagnostics_summary(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a normalized summary from a diagnostics document."""
    if not doc:
        return None

    summary = doc.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)

    frames = doc.get("frames")
    if isinstance(frames, list):
        return build_mask_diagnostics_summary(frames)

    return None


def format_mask_diagnostics_overview(doc: Mapping[str, Any] | None) -> list[str]:
    """Render a compact human-readable overview from a diagnostics document."""
    if not doc:
        return []

    summary = get_mask_diagnostics_summary(doc)
    if not summary:
        return []

    total_frames = int(doc.get("total_frames", 0))
    masked_frames = int(doc.get("masked_frames", 0))
    frames_with_face_detections = int(doc.get("frames_with_face_detections", 0))
    lines: list[str] = []

    if total_frames > 0:
        lines.append(
            "Mask summary: "
            f"{masked_frames}/{total_frames} masked, "
            f"{frames_with_face_detections}/{total_frames} with face detections"
        )

    lines.append(
        "ERP detection coverage: "
        f"avg {summary['avg_erp_detection_coverage_pct_post']:.1f}% post, "
        f"max {summary['max_erp_detection_coverage_pct_post']:.1f}%"
    )

    if int(summary.get("flagged_frames", 0)) > 0:
        flag_counts = summary.get("flag_counts", {})
        flag_bits = ", ".join(
            f"{name}={count}" for name, count in sorted(flag_counts.items())
        )
        lines.append(f"Flagged frames: {summary['flagged_frames']} ({flag_bits})")
        top_frames = summary.get("top_flagged_frames", [])
        if top_frames:
            lines.append(f"Top flagged frames: {', '.join(top_frames)}")

    face_summary = summary.get("avg_face_detection_pct", {})
    if face_summary:
        face_bits = ", ".join(
            f"{name} {float(pct):.1f}%"
            for name, pct in sorted(face_summary.items())
        )
        lines.append(f"Avg face detection: {face_bits}")

    view_summary = summary.get("max_per_view_removed_pct", {})
    if view_summary:
        top_views = sorted(
            view_summary.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )[:3]
        view_bits = ", ".join(
            f"{name} {float(pct):.1f}%"
            for name, pct in top_views
            if float(pct) > 0.0
        )
        if view_bits:
            lines.append(f"Worst view removal: {view_bits}")

    return lines
