# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Phase 1 smoke test for the dual fisheye pipeline.

Runs the new fisheye-native path end-to-end on a real capture, bypassing
the LichtFeld plugin UI entirely. Verifies:
    - .osv / .insv input is recognized
    - Paired sharpest extraction produces front/back frames in sync
    - COLMAP runs PER_FOLDER + OPENCV_FISHEYE and registers images
    - The pipeline completes without exceptions

Phase 1 deliberately skips: masking, fisheye_transforms.json output, the
experimental rig path. Those land in subsequent phases.

Run from the plugin venv:
    .venv/Scripts/python.exe test_phase1.py [osv|insv]

Default = osv. Pass `insv` to test the Insta360 file-pair path instead.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

# Configure logging BEFORE importing core modules (so their logger.info calls go to stdout).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Test inputs (per the user's plan).
TEST_INPUTS = {
    "osv": {
        "video_path": r"D:\Capture\deskTest\CAM_20260323172324_0023_D.OSV",
        "output_dir": r"D:\Capture\deskTest\plugin_phase1_osv",
        "camera_family": "dji_osmo360",
    },
    "insv": {
        "video_path": r"D:\Capture\meadowlark_insv\VID_20230813_194602_10_045.insv",
        "output_dir": r"D:\Capture\meadowlark_insv\plugin_phase1_insv",
        "camera_family": "insta360",
    },
}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "osv"
    if which not in TEST_INPUTS:
        print(f"Unknown input '{which}'. Expected: {list(TEST_INPUTS)}", file=sys.stderr)
        return 2
    case = TEST_INPUTS[which]

    video_path = case["video_path"]
    output_dir = case["output_dir"]

    if not Path(video_path).exists():
        print(f"Test input not found on disk: {video_path}", file=sys.stderr)
        return 2

    print(f"=== Phase 1 dual fisheye test: {which} ===")
    print(f"  input:  {video_path}")
    print(f"  output: {output_dir}")
    print()

    # Import after logging setup so module-level loggers inherit basicConfig.
    from core.pipeline import PipelineConfig, PipelineJob, detect_input_type

    # Verify auto-detect agrees with the manual config.
    detected_type, detected_family = detect_input_type(video_path)
    print(f"  detect_input_type() → ({detected_type!r}, {detected_family!r})")
    if detected_type != "dual_fisheye":
        print(f"  ERROR: expected 'dual_fisheye', got {detected_type!r}", file=sys.stderr)
        return 1
    if detected_family != case["camera_family"]:
        print(
            f"  WARN: detected family {detected_family!r} != expected "
            f"{case['camera_family']!r} — proceeding with expected",
            file=sys.stderr,
        )
    print()

    cfg = PipelineConfig(
        video_path=video_path,
        output_dir=output_dir,
        # Extraction settings — keep this fast for first smoke test
        interval=2.0,                 # 1 pair per 2 seconds
        extraction_sharpness="basic",  # sharpest selection, no scene detection
        blur_metric="tenengrad",
        blur_scale_width=1920,
        quality=95,
        # Phase 1 deferred features
        enable_masking=False,
        # COLMAP
        colmap_preset="normal",
        colmap_matcher="sequential",
        colmap_match_budget_tier="default",
        # Output
        output_mode="fisheye",
        # Dual fisheye dispatch
        input_type="dual_fisheye",
        camera_family=case["camera_family"],
        keep_streams=False,
    )

    last_msg = {"v": ""}

    def on_progress(stage: str, progress_pct: float, status_msg: str) -> None:
        # Throttle — only print when message changes
        line = f"[{stage:>10s} {progress_pct:6.1f}%] {status_msg}"
        if line != last_msg["v"]:
            print(line)
            last_msg["v"] = line

    completion_event = threading.Event()
    final_result = {"r": None}

    def on_complete(result) -> None:
        final_result["r"] = result
        completion_event.set()

    job = PipelineJob(cfg, on_progress=on_progress, on_complete=on_complete)
    job.start()

    # Wait for the daemon thread to finish (no timeout — large captures
    # can take many minutes).
    completion_event.wait()

    result = final_result["r"]
    print()
    print("=== Result ===")
    print(f"  success:                   {result.success}")
    print(f"  output_mode:               {result.output_mode}")
    print(f"  dataset_path:              {result.dataset_path}")
    print(f"  num_source_frames (pairs): {result.num_source_frames}")
    print(f"  num_output_images:         {result.num_output_images}")
    print(f"  num_aligned_cameras:       {result.num_aligned_cameras}")
    print(f"  num_registered_frames:     {result.num_registered_frames}")
    print(f"  num_complete_frames:       {result.num_complete_frames}")
    print(f"  num_partial_frames:        {result.num_partial_frames}")
    print(f"  views_per_frame:           {result.views_per_frame}")
    print(f"  expected_images_by_view:   {result.expected_images_by_view}")
    print(f"  registered_images_by_view: {result.registered_images_by_view}")
    print(f"  preset_signature:          {result.preset_signature}")
    print(f"  elapsed_sec:               {result.elapsed_sec:.1f}s")
    if result.error:
        print(f"  error:                     {result.error}")
    if result.partial_frame_examples:
        print(f"  partial_frame_examples:    {result.partial_frame_examples[:5]}")
    if result.dropped_frame_examples:
        print(f"  dropped_frame_examples:    {result.dropped_frame_examples[:5]}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
