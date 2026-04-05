# SAM v2 _C Extension Build Report

**Date:** 2026-04-04
**Goal:** Compile the `sam2._C` CUDA extension to eliminate the `fill_holes_in_mask_scores` warning during video propagation.

---

## The Warning

During SAM v2 video propagation, this warning appears every run:

```
UserWarning: cannot import name '_C' from 'sam2'

Skipping the post-processing step due to the error above. You can still use SAM 2
and it's OK to ignore the error above, although some post-processing functionality
may be limited (which doesn't affect the results in most cases; see
https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
```

## What _C Is

`sam2._C` is a compiled CUDA extension that provides a single function: `get_connected_componnets()` (note the typo — it's in Meta's source). It's used by `fill_holes_in_mask_scores()` in `sam2/utils/misc.py:312-338` to fill small holes in predicted masks during video propagation.

The source is a single CUDA file: `.venv/Lib/site-packages/sam2/csrc/connected_components.cu`

The call chain:
1. `SAM2VideoPredictor.propagate_in_video()` calls `fill_holes_in_mask_scores()` at line 786
2. `fill_holes_in_mask_scores()` calls `get_connected_components()` at line 322
3. `get_connected_components()` does `from sam2 import _C` at line 61
4. `_C` doesn't exist → ImportError → warning → post-processing skipped

## Why It's Not Built

The `sam2` package on PyPI (v1.1.0) is distributed as a source tarball, not a wheel. When installed via `uv add sam2`, the Python package installs but the CUDA extension does not get compiled. The extension requires:

1. MSVC C++ compiler (`cl.exe`) on PATH
2. CUDA toolkit (nvcc)
3. PyTorch's `torch.utils.cpp_extension` JIT compilation

## Build Attempt

Tried JIT compilation via:

```python
from torch.utils.cpp_extension import load
_C = load(name='_C', sources=['.../sam2/csrc/connected_components.cu'])
```

Failed with:
```
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
```

`cl.exe` (MSVC compiler) is not on the PATH in the plugin's venv environment.

## Environment

- Plugin venv Python: 3.12 (LichtFeld Studio embedded)
- torch: 2.11.0+cu128
- CUDA toolkit: 12.9 (at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`)
- sam2: 1.1.0 (installed via `uv add sam2`)
- OS: Windows 11
- Visual Studio: unknown status — needs to be checked

## What Needs To Happen

### Option A: JIT compile with MSVC on PATH

1. Find or install Visual Studio with C++ Desktop workload
2. Run the build from a "Developer Command Prompt" or set up vcvars64.bat to put `cl.exe` on PATH
3. JIT compile with `torch.utils.cpp_extension.load()`
4. The compiled `.pyd` file gets cached in `~/.cache/torch_extensions/` and reused on subsequent imports

### Option B: Pre-build and bundle the .pyd

1. Build the extension once in a proper build environment
2. Copy the resulting `_C.cp312-win_amd64.pyd` into `.venv/Lib/site-packages/sam2/`
3. The import `from sam2 import _C` would then find it directly

### Option C: Install sam2 from source with build

1. Clone the sam2 repo
2. Run `pip install -e .` from a Visual Studio developer environment
3. The setup.py would build the extension during install

### Option D: Find a pre-built wheel

1. Check if a Windows wheel exists for sam2 that includes the compiled extension
2. As of the preflight, only a source tarball was found on PyPI

## Impact If Not Fixed

The warning is non-blocking — SAM v2 tracking works without the extension. The only effect is that small holes in predicted masks are not filled during post-processing. In practice this means mask edges may have a few more tiny gaps than they would with the extension. The preflight and all testing runs confirmed tracking works correctly without it.

## Related Context: YOLO-Only Pass 1

During testing, we made a significant optimization to the masking pipeline that is relevant here:

### What Changed

Pass 1 (primary detection) no longer runs SAM v1 segmentation. It runs YOLO bounding box detection only. The rationale:

- Pass 1's only job is to estimate the person's direction on the sphere (for aiming the synthetic camera)
- YOLO bounding box centers give the same direction as SAM v1 mask centroids
- Pass 2 (SAM v2 video tracking on synthetic fisheye views) is now authoritative — it produces the final mask shape
- Pass 1's mask was being thrown away by the authoritative replace anyway

### Performance Impact

- Before: 657s masking (75% of pipeline) — 16 views × 11 frames × YOLO+SAM v1 = 176 full inference calls
- After: 83s masking (27% of pipeline) — 16 views × 11 frames × YOLO-only = 176 YOLO calls, no SAM v1 encoder
- 8x speedup on masking, total pipeline 878s → 303s

### Quality Impact

- Mask quality identical — confirmed by visual comparison of pinhole masks
- Direction estimation identical — YOLO box centers point the same direction as SAM v1 mask centroids
- SAM v2 tracking still 11/11 frames

### Code Location

The change is in `Masker._primary_detection()` in `core/masker.py`. Instead of calling `self._backend.detect_and_segment()` (which runs YOLO → SAM v1), it accesses `self._backend._yolo` directly and computes direction from bounding box centers.

Note: SAM v1 is still loaded by `YoloSamBackend.initialize()` because it's needed by `FallbackVideoBackend` if SAM v2 fails at runtime. A future optimization could defer SAM v1 loading until actually needed.

## Other Changes Made During Testing

### Dedicated 16-Camera Detection Layout

Pass 1 now uses a hardcoded FullCircle-style 16-camera layout (`DETECTION_LAYOUT` constant in `core/masker.py`) instead of the user's reconstruction preset. This decouples detection quality from preset choice. The layout is: 8 yaw × 2 pitch at ±35°, 90° FOV, upper row offset 22.5°.

### Authoritative Pass 2

The merge strategy changed from OR-merge (`np.maximum(pass1, pass2)`) to authoritative replace. When Pass 2 (SAM v2 tracking) produces a non-empty mask for a frame, it replaces Pass 1's mask entirely. Pass 1 only survives on frames where Pass 2 failed. This eliminated false positives that OR-merge was preserving.

### Removed ERP Morph-Close

The ERP-level morphological close + flood fill was removed from Phase 3 of the masker. It was bridging false positives across the sphere. Per-view erosion in the reframer (9×9 elliptical kernel) replaced it, matching FullCircle's approach.

### 5% Minimum Direction Coverage

Detections covering less than 5% of a view are excluded from the person direction computation (but still contribute to the ERP mask if Pass 1 mask is used). This filters out false positive YOLO detections that would pull the synthetic camera off-target.
