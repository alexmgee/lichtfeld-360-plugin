# SAM2 Runtime Repair Report

**Date:** 2026-04-05  
**Status:** Repair completed and verified  
**Related report:** [2026-04-05-sam2-broken-runtime-report.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-05-sam2-broken-runtime-report.md)

---

## Summary

The broken `sam2` runtime was repaired successfully.

Before repair:

- `sam2` imported only as a namespace package
- `sam2.build_sam` was missing
- there was no `sam2` distribution metadata
- `site-packages/sam2` contained only `_C.pyd` artifacts

After repair:

- `sam2` is a real installed package again
- `sam2.__file__` points to `sam2/__init__.py`
- `sam2` distribution metadata exists
- `sam2.build_sam` is importable
- `sam2._C` is importable

This restores the environment needed for the plugin to use the real SAM2 video
tracking path again.

---

## Repair Steps Performed

### 1. Preserve the broken package directory

The stale namespace-style directory was moved out of the way to:

- `.venv/Lib/site-packages/sam2.broken_20260405_012728`

This preserved the broken state for inspection instead of deleting it outright.

### 2. Reinstall locked video-tracking dependencies

Executed:

```powershell
uv sync --locked --extra video-tracking --reinstall-package sam2
```

This restored the real `sam2==1.1.0` package and its supporting runtime stack.

### 3. Reinstall the bundled `_C.pyd`

After the clean `sam2` package was back, the bundled extension was copied into:

- `.venv/Lib/site-packages/sam2/_C.pyd`

This restored the plugin's connected-components extension on top of the real
package install.

---

## Verification

The repaired environment now verifies successfully with:

```python
import torch
from sam2.build_sam import build_sam2_video_predictor_hf
from sam2 import _C
import importlib.metadata as m
```

Observed result:

- `sam2 version 1.1.0`
- `build_sam ok True`
- `_C ok True`
- `torch 2.11.0+cu128`
- `cuda_available True`

Additional import-spec checks:

- `sam2.build_sam spec` resolves to `sam2/build_sam.py`
- `sam2._C spec` resolves to `sam2/_C.pyd`

On-disk package state now includes:

- `sam2/`
- `sam2-1.1.0.dist-info/`
- preserved forensic backup: `sam2.broken_20260405_012728/`

---

## Practical Conclusion

The environment is no longer in the ghost-package state Claude identified.

The live plugin `.venv` once again contains:

- a real `sam2` Python package
- real `sam2` package metadata
- the plugin's `_C.pyd` extension

That means the environment is again capable of the plugin's real SAM2 video
tracking path rather than being limited to the fallback image-only path.

---

## Remaining Caution

This repair fixes the package/runtime state.

It does **not** by itself prove every earlier Default-preset benchmark was
performed with real SAM2 active.

So the right posture is:

- treat the current repaired environment as the trustworthy baseline
- avoid over-trusting any earlier conclusion that was drawn while the broken
  ghost-package state existed

---

## Follow-up Proof Pass

After repair, the Default preset was re-run in LichtFeld Studio on:

- `D:\Capture\deskTest\default_test2`

The end-of-run diagnostics confirmed the real backend path:

- `Mask backend: YoloSamBackend`
- `Video backend: Sam2VideoBackend`

And the run completed cleanly:

- `Registered frames: 11/11`
- `Complete rig frames: 11`
- `Registered images: 176`
- all 16 views registered `11/11`

User review of the outputs:

- ERP masks looked great
- pinhole masks looked good overall
- only one minor false-positive/noise artifact stood out in:
  - `masks/00_04/deskTest_trim_00007.png`
  - corresponding image `images/00_04/deskTest_trim_00007.jpg`

This follow-up proof pass closes the loop on the repair:

- the runtime is repaired
- the real SAM2 video backend is confirmed active in LFS
- the Default preset is back in a trustworthy working state
