# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for mask diagnostics summary helpers."""

import json
from pathlib import Path
import shutil
import uuid

from core.mask_diagnostics import (
    build_mask_diagnostics_summary,
    format_mask_diagnostics_overview,
    get_mask_diagnostics_summary,
    load_mask_diagnostics_document,
)


def _make_temp_workspace() -> Path:
    root = Path.cwd() / "tmp" / "pytest"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"maskdiag_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_mask_diagnostics_summary_aggregates_frame_stats():
    frames = [
        {
            "frame": "frame_00001",
            "erp_detection_coverage_pct_pre": 10.0,
            "erp_detection_coverage_pct_post": 12.5,
            "views_with_removed_pixels": 2,
            "face_detection_pct": {"front": 20.0, "left": 5.0},
            "per_view_removed_pct": {"00_00": 15.0, "00_01": 4.0},
            "flags": ["heavy_removed_view"],
        },
        {
            "frame": "frame_00002",
            "erp_detection_coverage_pct_pre": 0.0,
            "erp_detection_coverage_pct_post": 0.0,
            "views_with_removed_pixels": 0,
            "face_detection_pct": {"front": 10.0, "right": 7.0},
            "per_view_removed_pct": {"00_00": 3.0, "00_02": 9.0},
            "flags": ["no_face_detections"],
        },
    ]

    summary = build_mask_diagnostics_summary(frames)

    assert summary["flagged_frames"] == 2
    assert summary["flag_counts"] == {
        "heavy_removed_view": 1,
        "no_face_detections": 1,
    }
    assert summary["avg_erp_detection_coverage_pct_pre"] == 5.0
    assert summary["avg_erp_detection_coverage_pct_post"] == 6.25
    assert summary["max_erp_detection_coverage_pct_post"] == 12.5
    assert summary["avg_face_detection_pct"]["front"] == 15.0
    assert summary["max_per_view_removed_pct"]["00_00"] == 15.0
    assert summary["max_removed_view_pct"] == 15.0
    assert summary["top_flagged_frames"] == ["frame_00001", "frame_00002"]


def test_load_and_format_mask_diagnostics_document_handles_legacy_docs():
    tmp_path = _make_temp_workspace()
    try:
        diag_path = tmp_path / "masking_diagnostics.json"
        doc = {
            "mode": "sam3_cubemap",
            "backend": "Sam3Backend",
            "total_frames": 2,
            "masked_frames": 1,
            "frames_with_face_detections": 1,
            "frames": [
                {
                    "frame": "frame_00001",
                    "erp_detection_coverage_pct_pre": 2.0,
                    "erp_detection_coverage_pct_post": 3.0,
                    "views_with_removed_pixels": 1,
                    "face_detection_pct": {"front": 4.0},
                    "per_view_removed_pct": {"00_00": 12.0},
                    "flags": ["heavy_removed_view"],
                },
                {
                    "frame": "frame_00002",
                    "erp_detection_coverage_pct_pre": 0.0,
                    "erp_detection_coverage_pct_post": 0.0,
                    "views_with_removed_pixels": 0,
                    "face_detection_pct": {},
                    "per_view_removed_pct": {},
                    "flags": [],
                },
            ],
        }
        diag_path.write_text(json.dumps(doc), encoding="utf-8")

        loaded = load_mask_diagnostics_document(diag_path)
        summary = get_mask_diagnostics_summary(loaded)
        lines = format_mask_diagnostics_overview(loaded)

        assert loaded is not None
        assert summary is not None
        assert summary["avg_erp_detection_coverage_pct_post"] == 1.5
        assert any("Mask summary:" in line for line in lines)
        assert any("Flagged frames:" in line for line in lines)
        assert any("Worst view removal:" in line for line in lines)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
