# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for loading and summarizing masking diagnostics documents."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


def _round_pct(value: float) -> float:
    return round(float(value), 3)


def _avg(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return _round_pct(sum(vals) / len(vals))


def _median(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return _round_pct(median(vals))


def _mad(values: Iterable[float], med: float | None = None) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    center = float(med) if med is not None else float(median(vals))
    deviations = [abs(value - center) for value in vals]
    return _round_pct(median(deviations))


def _max_value(mapping: Mapping[str, Any] | None) -> float:
    if not mapping:
        return 0.0
    values = [float(value) for value in mapping.values()]
    return _round_pct(max(values)) if values else 0.0


def _high_outlier_score(
    value: float,
    center: float,
    spread: float,
    *,
    abs_floor: float,
    rel_floor: float,
) -> float:
    """Return a soft outlier score for values unusually high within a clip."""
    delta = float(value) - float(center)
    if delta <= abs_floor:
        return 0.0

    fallback = max(abs_floor, abs(float(center)) * rel_floor, 1e-6)
    scale = float(spread) if float(spread) > 1e-6 else fallback
    score = delta / scale
    return round(float(max(score, 0.0)), 3)


def _rank_unusual_frames(frames: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Identify frames that are unusual relative to the rest of the clip.

    These signals are intentionally descriptive rather than judgemental. A frame
    may be unusual and still be correct for a difficult scene.
    """
    if not frames:
        return [], Counter()

    post_coverages = [
        float(frame.get("erp_detection_coverage_pct_post", 0.0))
        for frame in frames
    ]
    views_removed = [
        float(frame.get("views_with_removed_pixels", 0.0))
        for frame in frames
    ]
    max_view_removed = [
        _max_value(frame.get("per_view_removed_pct") or {})
        for frame in frames
    ]
    down_face_share = [
        float(frame.get("down_face_share_pct", 0.0))
        for frame in frames
    ]
    component_counts = [
        float(frame.get("component_count", 0.0))
        for frame in frames
    ]
    secondary_component_share = [
        float(frame.get("secondary_component_share_pct", 0.0))
        for frame in frames
    ]

    post_median = _median(post_coverages)
    post_mad = _mad(post_coverages, post_median)
    views_median = _median(views_removed)
    views_mad = _mad(views_removed, views_median)
    max_view_median = _median(max_view_removed)
    max_view_mad = _mad(max_view_removed, max_view_median)
    down_median = _median(down_face_share)
    down_mad = _mad(down_face_share, down_median)
    component_median = _median(component_counts)
    component_mad = _mad(component_counts, component_median)
    secondary_median = _median(secondary_component_share)
    secondary_mad = _mad(secondary_component_share, secondary_median)

    coverage_jumps = [0.0]
    view_jumps = [0.0]
    for idx in range(1, len(frames)):
        coverage_jumps.append(abs(post_coverages[idx] - post_coverages[idx - 1]))
        view_jumps.append(abs(views_removed[idx] - views_removed[idx - 1]))

    coverage_jump_median = _median(coverage_jumps[1:])
    coverage_jump_mad = _mad(coverage_jumps[1:], coverage_jump_median)
    view_jump_median = _median(view_jumps[1:])
    view_jump_mad = _mad(view_jumps[1:], view_jump_median)

    unusual_frames: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for idx, frame in enumerate(frames):
        reasons: list[str] = []
        score = 0.0

        reason_specs = [
            (
                "coverage_spike_rel",
                _high_outlier_score(
                    post_coverages[idx],
                    post_median,
                    post_mad,
                    abs_floor=max(0.25, post_median * 0.15),
                    rel_floor=0.35,
                ),
            ),
            (
                "view_spread_spike",
                _high_outlier_score(
                    views_removed[idx],
                    views_median,
                    views_mad,
                    abs_floor=1.0,
                    rel_floor=0.25,
                ),
            ),
            (
                "heavy_view_spike_rel",
                _high_outlier_score(
                    max_view_removed[idx],
                    max_view_median,
                    max_view_mad,
                    abs_floor=1.0,
                    rel_floor=0.3,
                ),
            ),
            (
                "down_face_dominant_rel",
                _high_outlier_score(
                    down_face_share[idx],
                    down_median,
                    down_mad,
                    abs_floor=5.0,
                    rel_floor=0.2,
                ),
            ),
            (
                "fragmented_mask_rel",
                _high_outlier_score(
                    component_counts[idx],
                    component_median,
                    component_mad,
                    abs_floor=1.0,
                    rel_floor=0.5,
                ),
            ),
            (
                "secondary_blob_rel",
                _high_outlier_score(
                    secondary_component_share[idx],
                    secondary_median,
                    secondary_mad,
                    abs_floor=5.0,
                    rel_floor=0.35,
                ),
            ),
            (
                "coverage_jump",
                _high_outlier_score(
                    coverage_jumps[idx],
                    coverage_jump_median,
                    coverage_jump_mad,
                    abs_floor=max(0.35, coverage_jump_median * 1.5),
                    rel_floor=0.5,
                ),
            ),
            (
                "view_jump",
                _high_outlier_score(
                    view_jumps[idx],
                    view_jump_median,
                    view_jump_mad,
                    abs_floor=2.0,
                    rel_floor=0.5,
                ),
            ),
        ]

        for reason, reason_score in reason_specs:
            if reason_score >= 3.0:
                reasons.append(reason)
                score += reason_score

        if reasons and (len(reasons) >= 2 or score >= 4.0):
            for reason in reasons:
                reason_counts[reason] += 1
            unusual_frames.append(
                {
                    "frame": str(frame.get("frame", "")),
                    "score": round(score, 3),
                    "reasons": reasons,
                    "erp_detection_coverage_pct_post": _round_pct(post_coverages[idx]),
                    "views_with_removed_pixels": int(round(views_removed[idx])),
                    "max_view_removed_pct": _round_pct(max_view_removed[idx]),
                }
            )

    unusual_frames.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -len(item.get("reasons", [])),
            str(item.get("frame", "")),
        )
    )
    return unusual_frames, reason_counts


def build_mask_diagnostics_summary(frames: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate frame-level diagnostics into a compact summary."""
    face_values: dict[str, list[float]] = defaultdict(list)
    view_max_removed: dict[str, float] = defaultdict(float)
    flag_counts: Counter[str] = Counter()

    pre_coverages: list[float] = []
    post_coverages: list[float] = []
    views_with_removed_pixels: list[float] = []
    component_counts: list[float] = []
    secondary_component_share_pct: list[float] = []
    down_face_share_pct: list[float] = []
    max_removed_view_pct = 0.0

    for frame in frames:
        pre_coverages.append(float(frame.get("erp_detection_coverage_pct_pre", 0.0)))
        post_coverages.append(float(frame.get("erp_detection_coverage_pct_post", 0.0)))
        views_with_removed_pixels.append(float(frame.get("views_with_removed_pixels", 0.0)))
        component_counts.append(float(frame.get("component_count", 0.0)))
        secondary_component_share_pct.append(
            float(frame.get("secondary_component_share_pct", 0.0))
        )
        down_face_share_pct.append(float(frame.get("down_face_share_pct", 0.0)))

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

    unusual_frames, unusual_reason_counts = _rank_unusual_frames(frames)

    return {
        "flagged_frames": len(flagged_frames),
        "flag_counts": dict(sorted(flag_counts.items())),
        "unusual_frames": len(unusual_frames),
        "unusual_reason_counts": dict(sorted(unusual_reason_counts.items())),
        "avg_erp_detection_coverage_pct_pre": _avg(pre_coverages),
        "avg_erp_detection_coverage_pct_post": _avg(post_coverages),
        "median_erp_detection_coverage_pct_post": _median(post_coverages),
        "max_erp_detection_coverage_pct_post": (
            _round_pct(max(post_coverages)) if post_coverages else 0.0
        ),
        "avg_views_with_removed_pixels": _avg(views_with_removed_pixels),
        "median_views_with_removed_pixels": _median(views_with_removed_pixels),
        "max_removed_view_pct": _round_pct(max_removed_view_pct),
        "avg_component_count": _avg(component_counts),
        "median_component_count": _median(component_counts),
        "avg_secondary_component_share_pct": _avg(secondary_component_share_pct),
        "max_secondary_component_share_pct": (
            _round_pct(max(secondary_component_share_pct))
            if secondary_component_share_pct else 0.0
        ),
        "avg_down_face_share_pct": _avg(down_face_share_pct),
        "max_down_face_share_pct": (
            _round_pct(max(down_face_share_pct))
            if down_face_share_pct else 0.0
        ),
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
        "top_unusual_frames": unusual_frames[:5],
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
        f"median {summary.get('median_erp_detection_coverage_pct_post', 0.0):.1f}%, "
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

    if int(summary.get("unusual_frames", 0)) > 0:
        unusual_counts = summary.get("unusual_reason_counts", {})
        unusual_bits = ", ".join(
            f"{name}={count}" for name, count in sorted(unusual_counts.items())
        )
        lines.append(
            f"Unusual frames: {summary['unusual_frames']}"
            + (f" ({unusual_bits})" if unusual_bits else "")
        )
        top_unusual = summary.get("top_unusual_frames", [])
        if top_unusual:
            top_bits = []
            for item in top_unusual[:3]:
                frame_name = str(item.get("frame", ""))
                reasons = ", ".join(item.get("reasons", []) or [])
                if frame_name:
                    top_bits.append(f"{frame_name} [{reasons}]")
            if top_bits:
                lines.append(f"Top unusual frames: {', '.join(top_bits)}")

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

    if float(summary.get("avg_secondary_component_share_pct", 0.0)) > 0.0:
        lines.append(
            "Secondary mask area: "
            f"avg {float(summary['avg_secondary_component_share_pct']):.1f}% of detection, "
            f"max {float(summary.get('max_secondary_component_share_pct', 0.0)):.1f}%"
        )

    if float(summary.get("avg_down_face_share_pct", 0.0)) > 0.0:
        lines.append(
            "Down-face share: "
            f"avg {float(summary['avg_down_face_share_pct']):.1f}%, "
            f"max {float(summary.get('max_down_face_share_pct', 0.0)):.1f}%"
        )

    return lines
