# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pipeline Benchmark Tool
=======================

Standalone timing harness for the PanoSplat pipeline. Hooks into the
pipeline's progress callbacks to capture per-stage timing, substep
breakdowns, and live throughput rates — without modifying any shipped code.

Usage:
    python dev/benchmark.py <video_path> [--output-dir <dir>] [--preset cubemap]
                                          [--fps 1.0] [--quality normal]

Outputs:
    - Live terminal dashboard with per-stage rates
    - timing.json in the output directory
    - Summary table at the end

Examples:
    python dev/benchmark.py D:/Capture/test.mp4
    python dev/benchmark.py D:/Capture/test.mp4 --preset balanced --fps 2.0
    python dev/benchmark.py D:/Capture/test.mp4 --quality none --output-dir D:/tmp/bench
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add plugin root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import PipelineConfig, PipelineJob, PipelineResult


# ---------------------------------------------------------------------------
# Stage timing tracker
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """Accumulated timing for one pipeline stage."""

    name: str
    started: float = 0.0
    ended: float = 0.0
    items: int = 0
    item_label: str = "items"
    last_message: str = ""
    substeps: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        if self.ended > 0:
            return self.ended - self.started
        if self.started > 0:
            return time.time() - self.started
        return 0.0

    @property
    def rate(self) -> float:
        e = self.elapsed
        return self.items / e if e > 0 and self.items > 0 else 0.0

    def to_dict(self) -> dict:
        d: dict = {"elapsed_sec": round(self.elapsed, 3)}
        if self.items > 0:
            d["items"] = self.items
            d["item_label"] = self.item_label
            d["rate_per_sec"] = round(self.rate, 2)
        if self.substeps:
            d["substeps"] = {k: round(v, 3) for k, v in self.substeps.items()}
        return d


# Patterns for extracting counts from progress messages
_REFRAME_PAT = re.compile(r"Reframing (\d+)/(\d+)")
_EXTRACT_PAT = re.compile(r"Extracting (\d+)/(\d+)")
_SCORING_PAT = re.compile(r"Scoring (\d+)/(\d+)")
_ANALYZING_PAT = re.compile(r"Analyzing.*?(\d+)%")
_MAPPING_PAT = re.compile(r"(\d+) images.*?(\d+) points")

# Map stage names to human labels and item units
_STAGE_META = {
    "extraction": ("Frame Extraction", "frames"),
    "masking": ("Operator Masking", "frames"),
    "reframe": ("Reframing", "images"),
    "rig_config": ("Rig Config", ""),
    "colmap": ("COLMAP Alignment", "images"),
    "output": ("Write Output", ""),
    "complete": ("Complete", ""),
}


class BenchmarkTracker:
    """Captures timing data from pipeline progress callbacks."""

    def __init__(self) -> None:
        self.stages: dict[str, StageRecord] = {}
        self._current_stage: str = ""
        self._t0 = time.time()
        self._colmap_substep_start: float = 0.0
        self._colmap_last_substage: str = ""

    def on_progress(self, stage: str, percent: float, message: str) -> None:
        """Progress callback — pass this to PipelineJob."""
        now = time.time()

        # Detect stage transition
        if stage != self._current_stage:
            # Close previous stage
            if self._current_stage and self._current_stage in self.stages:
                self.stages[self._current_stage].ended = now
                # Close last COLMAP substep
                if self._current_stage == "colmap" and self._colmap_last_substage:
                    self.stages["colmap"].substeps[self._colmap_last_substage] = (
                        now - self._colmap_substep_start
                    )

            # Open new stage
            label, item_label = _STAGE_META.get(stage, (stage, "items"))
            self.stages[stage] = StageRecord(
                name=label, started=now, item_label=item_label
            )
            self._current_stage = stage

        record = self.stages[stage]
        record.last_message = message

        # Extract item counts from messages
        self._parse_items(stage, message, record)

        # Track COLMAP substeps (features → matching → mapping)
        if stage == "colmap":
            self._track_colmap_substep(message, now)

        # Live terminal output
        self._print_live(stage, percent, message, record)

    def _parse_items(self, stage: str, message: str, record: StageRecord) -> None:
        if stage == "reframe":
            m = _REFRAME_PAT.search(message)
            if m:
                record.items = int(m.group(1))
        elif stage == "extraction":
            m = _EXTRACT_PAT.search(message)
            if m:
                record.items = int(m.group(1))
            else:
                m = _SCORING_PAT.search(message)
                if m:
                    record.items = int(m.group(1))
        elif stage == "colmap":
            m = _MAPPING_PAT.search(message)
            if m:
                record.items = int(m.group(1))

    def _track_colmap_substep(self, message: str, now: float) -> None:
        msg_lower = message.lower()
        if "sift" in msg_lower or "feature" in msg_lower:
            substage = "feature_extraction"
        elif "match" in msg_lower:
            substage = "matching"
        elif "map" in msg_lower or "incremental" in msg_lower:
            substage = "mapping"
        else:
            return

        if substage != self._colmap_last_substage:
            # Close previous substep
            if self._colmap_last_substage and "colmap" in self.stages:
                self.stages["colmap"].substeps[self._colmap_last_substage] = (
                    now - self._colmap_substep_start
                )
            self._colmap_substep_start = now
            self._colmap_last_substage = substage

    def _print_live(
        self, stage: str, percent: float, message: str, record: StageRecord
    ) -> None:
        elapsed = time.time() - self._t0
        rate_str = ""
        if record.rate > 0:
            rate_str = f" | {record.rate:.1f} {record.item_label}/sec"

        line = (
            f"\r[{elapsed:6.1f}s] {record.name:<20s} "
            f"{percent:5.1f}%{rate_str}  {message[:60]:<60s}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def finalize(self) -> None:
        """Close the last open stage."""
        now = time.time()
        if self._current_stage and self._current_stage in self.stages:
            s = self.stages[self._current_stage]
            if s.ended == 0:
                s.ended = now
            if self._current_stage == "colmap" and self._colmap_last_substage:
                s.substeps[self._colmap_last_substage] = now - self._colmap_substep_start

    def to_dict(self) -> dict:
        total = sum(s.elapsed for s in self.stages.values() if s.name != "Complete")
        return {
            "total_sec": round(total, 3),
            "stages": {k: v.to_dict() for k, v in self.stages.items() if k != "complete"},
        }

    def print_summary(self) -> None:
        """Print a formatted summary table."""
        total = sum(s.elapsed for s in self.stages.values() if s.name != "Complete")
        print("\n")
        print("=" * 72)
        print("PIPELINE BENCHMARK RESULTS")
        print("=" * 72)
        print(f"{'Stage':<24s} {'Time':>8s} {'%':>6s} {'Items':>8s} {'Rate':>14s}")
        print("-" * 72)

        for key, s in self.stages.items():
            if key == "complete":
                continue
            pct = (s.elapsed / total * 100) if total > 0 else 0
            items_str = str(s.items) if s.items > 0 else ""
            rate_str = f"{s.rate:.1f} {s.item_label}/s" if s.rate > 0 else ""
            print(f"  {s.name:<22s} {s.elapsed:7.1f}s {pct:5.0f}%  {items_str:>8s} {rate_str:>14s}")

            for sub_name, sub_elapsed in s.substeps.items():
                sub_pct = (sub_elapsed / s.elapsed * 100) if s.elapsed > 0 else 0
                print(f"    {sub_name:<20s} {sub_elapsed:7.1f}s {sub_pct:5.0f}%")

        print("-" * 72)
        print(f"  {'TOTAL':<22s} {total:7.1f}s")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark the PanoSplat pipeline")
    parser.add_argument("video", help="Path to 360° equirectangular video")
    parser.add_argument("--output-dir", help="Output directory (default: next to video)")
    parser.add_argument("--preset", default="cubemap", help="Reframing preset")
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    parser.add_argument("--quality", default="normal",
                        choices=["none", "fast", "normal", "maximum"],
                        help="Extraction quality")
    parser.add_argument("--crop-size", type=int, default=1920, help="Output crop size")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--colmap-preset", default="exhaustive",
                        choices=["sequential", "exhaustive"],
                        help="COLMAP matching strategy")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Error: video not found: {video}")
        sys.exit(1)

    output_dir = args.output_dir or str(video.parent / f"{video.stem}_benchmark")

    # Check pycolmap GPU status
    try:
        import pycolmap
        gpu_status = f"CUDA: {pycolmap.has_cuda}"
    except ImportError:
        gpu_status = "pycolmap not installed"

    print(f"Video:    {video}")
    print(f"Output:   {output_dir}")
    print(f"Preset:   {args.preset} | FPS: {args.fps} | Quality: {args.quality}")
    print(f"COLMAP:   {args.colmap_preset} | pycolmap {gpu_status}")
    print(f"Crop:     {args.crop_size}px | JPEG: {args.jpeg_quality}")
    print()

    config = PipelineConfig(
        video_path=str(video),
        output_dir=output_dir,
        interval=1.0 / max(0.1, args.fps),
        extraction_quality=args.quality,
        preset_name=args.preset,
        output_size=args.crop_size,
        jpeg_quality=args.jpeg_quality,
        colmap_preset=args.colmap_preset,
    )

    tracker = BenchmarkTracker()
    result_holder: list[PipelineResult] = []

    def on_complete(result: PipelineResult) -> None:
        result_holder.append(result)

    job = PipelineJob(config, on_progress=tracker.on_progress, on_complete=on_complete)
    job.start()

    # Wait for completion
    while job.is_running:
        time.sleep(0.1)

    tracker.finalize()

    if not result_holder:
        print("\nPipeline returned no result")
        sys.exit(1)

    result = result_holder[0]

    # Print summary
    tracker.print_summary()

    if result.success:
        print(f"\nSource frames:    {result.num_source_frames}")
        print(f"Output images:    {result.num_output_images}")
        print(f"Aligned cameras:  {result.num_aligned_cameras}")
        print(f"Dataset:          {result.dataset_path}")
    else:
        print(f"\nPipeline FAILED: {result.error}")

    # Write timing.json
    timing_path = Path(output_dir) / "timing.json"
    timing_data = tracker.to_dict()
    timing_data["config"] = {
        "video": str(video),
        "preset": args.preset,
        "fps": args.fps,
        "quality": args.quality,
        "crop_size": args.crop_size,
        "colmap_preset": args.colmap_preset,
        "pycolmap_cuda": gpu_status,
    }
    if result.success:
        timing_data["result"] = {
            "source_frames": result.num_source_frames,
            "output_images": result.num_output_images,
            "aligned_cameras": result.num_aligned_cameras,
        }
    timing_path.write_text(json.dumps(timing_data, indent=2))
    print(f"\nTiming saved to:  {timing_path}")


if __name__ == "__main__":
    main()
