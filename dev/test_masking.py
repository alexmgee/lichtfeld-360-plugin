#!/usr/bin/env python
"""Quick masking-only test runner.

Runs just the masking stage on already-extracted ERP frames, skipping
extraction, reframing, and COLMAP.  Useful for iterating on mask quality
without waiting for the full pipeline.

Usage:
    .venv/Scripts/python.exe dev/test_masking.py <frames_dir> [output_dir]

If output_dir is omitted, defaults to D:/Capture/deskTest/default_test/mask_testing/
"""
import sys
import os
import time
from pathlib import Path

# Bootstrap: add plugin root to sys.path
_plugin_root = str(Path(__file__).resolve().parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

# Windows DLL search paths
if sys.platform == "win32":
    _lib_dir = Path(_plugin_root) / "lib"
    if _lib_dir.is_dir():
        os.add_dll_directory(str(_lib_dir))

from core.masker import Masker, MaskConfig
from core.presets import VIEW_PRESETS


def main():
    default_frames = Path(r"D:\Capture\deskTest\default_test\extracted\frames")
    default_output = Path(r"D:\Capture\deskTest\default_test\mask_testing\2")

    frames_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_frames
    if not frames_dir.is_dir():
        print(f"Error: {frames_dir} is not a directory")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = default_output

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default preset views (needed for MaskConfig but not used
    # by the synthetic pipeline — detection uses DETECTION_LAYOUT)
    preset = VIEW_PRESETS["default"]
    views = preset.get_all_views()

    cfg = MaskConfig(
        targets=["person"],
        output_size=1920,
        views=views,
        enable_synthetic=True,
    )

    print(f"Frames:  {frames_dir}")
    print(f"Output:  {output_dir}")
    print(f"Views:   {len(views)}")
    print()

    masker = Masker(cfg)

    t0 = time.time()
    print("Initializing backends...")
    masker.initialize()

    print("Running masking pipeline...")
    result = masker.process_frames(
        str(frames_dir),
        str(output_dir),
    )
    masker.cleanup()
    elapsed = time.time() - t0

    print()
    print(f"{'=' * 50}")
    print(f"Result:  {'OK' if result.success else 'FAILED'}")
    print(f"Frames:  {result.masked_frames}/{result.total_frames}")
    print(f"Masks:   {result.masks_dir}")
    print(f"Time:    {elapsed:.1f}s")
    if result.error:
        print(f"Error:   {result.error}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
