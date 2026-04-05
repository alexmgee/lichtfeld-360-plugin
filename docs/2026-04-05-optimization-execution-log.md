# Default Preset Optimization — Execution Log

**Date:** 2026-04-05
**Checklist:** `docs/2026-04-05-default-preset-optimization-execution-checklist.md`

---

## Phase 0 — Baseline Verification

**Status:** Passed

Environment confirmed:
- torch 2.11.0+cu128
- torch_cuda 12.8
- cuda_available True
- torchvision 0.26.0+cu128
- sam2.build_sam importable
- sam2._C importable

Worktree clean (only new docs uncommitted).

Baseline reference from proof pass:
- Clip: deskTest, 11 frames, Default preset
- TOTAL 404.7s, Masking 156.8s, Reframe 30.9s, COLMAP 94.0s
- Registration: 11/11, 176/176 images
- Known artifact: one small false positive in `masks/00_04/deskTest_trim_00007.png`

---

## Phase 1 — Detection Remap Cache Wiring

**Status:** Implemented, awaiting validation

**Change:** In `_primary_detection()` ([core/masker.py](core/masker.py)), replaced per-call `_reframe_to_detection()` with cached remap tables:

- On first call (or when ERP/detection dimensions change), builds 16 `(map_x, map_y)` pairs via `_build_detection_remap()` and stores them in `self._detection_remap_cache`
- Cache key: `(detection_size, erp_w, erp_h)`
- On all subsequent calls, reuses cached tables via `_apply_detection_remap()`
- Eliminates 16 × (N-1) redundant meshgrid + trig + rotation matrix computations

**What was NOT changed:**
- YOLO call structure (still per-view, not batched)
- Union-box direction logic
- Detection thresholds (conf=0.35, iou=0.6, 5% coverage)
- Pass 2 behavior
- Any other file

**Validation result:**
- Clip: deskTest, 11 frames, Default preset
- Masks: OK, match baseline quality
- Registration: 11/11, 176/176 — unchanged
- Backend: YoloSamBackend + Sam2VideoBackend — unchanged
- Masking runtime: 156.8s → 124.7s (**-32.1s, -20%**)
- Total runtime: 404.7s → 339.0s (**-65.7s, -16%**)

**Verdict:** Keep. Clean speed win, no quality regression.

---

## Phase 2 — Batched YOLO for Pass 1

**Status:** Implemented, awaiting validation

**Change:** In `_primary_detection()` ([core/masker.py](core/masker.py)), split the per-view loop into three stages:

1. Build all 16 detection images from cached remap tables
2. Single batched YOLO call on all 16 images
3. Parse results per view into the existing union-box direction logic

Reduces 16 YOLO GPU kernel launches per frame to 1.

**What was NOT changed:**
- conf/iou/classes parameters (identical)
- Union-box direction logic (identical)
- Coverage threshold (identical)
- Pass 2 behavior
- Any other file

**Validation result:**
- Clip: deskTest, 11 frames, Default preset
- Output: D:\Capture\deskTest\default_validation2
- Masks: OK, ERP and pinhole masks match baseline quality
- Registration: 11/11, 176/176 — unchanged
- Backend: YoloSamBackend + Sam2VideoBackend — unchanged
- Masking runtime: 124.7s → 97.9s (**-26.8s, -21%**)
- Total runtime: 339.0s → 312.5s (**-26.5s, -8%**)
- Cumulative from baseline: masking 156.8s → 97.9s (**-37.5%**), total 404.7s → 312.5s (**-22.8%**)

**Verdict:** Keep. Clean speed win, no quality regression. Batched YOLO did not affect mask output.

---

## Phase 3 — Backprojection Validation Harness

**Status:** Complete

**Created:** `dev/backprojection_harness.py`

Harness compares the production `_backproject_fisheye_mask_to_erp` (pycolmap path) against a candidate numpy equidistant fisheye implementation across 6 test cases:
- Centered blob + front direction
- Off-center blob + front direction
- Centered blob + side direction
- Person shape + front direction
- Person shape + side direction
- Person shape + front direction at full 7680×3840 ERP resolution

**Results:**
- IoU: 1.000000 across all 6 tests — zero changed pixels
- The numpy candidate is mathematically equivalent to pycolmap for the zero-distortion fisheye
- No speedup from the math replacement alone (0.9x) — the bottleneck is the ERP grid construction and rotation, not the projection function itself
- Full-res backprojection: ~5.9s per frame (reference), ~6.6s (candidate)

**Conclusion:** The numpy replacement is proven equivalent and safe to swap in. However, it alone does not solve the performance problem. The next optimization should target the grid/rotation work or reduce the number of points processed (e.g., downsampled backprojection).

---

## Phase 4 — Downsampled Backprojection

**Status:** Implemented, awaiting LFS validation

### Step 1-2: Harness evaluation

Added `_backproject_downsampled` candidate to the harness. Tested at 0.5x and 0.25x scales.

**0.5x results:**
- IoU: 0.991–0.996 across all test cases (all above 0.99 threshold)
- Speedup: 3.4–4.4x (avg 3.8x)
- Full-res (7680×3840): 6.0s → 1.5s per frame
- Area drift: max 0.10%
- Changed pixels: boundary-only, <0.02% of image
- **Verdict: Passes acceptance criteria**

**0.25x results:**
- IoU: 0.984–0.993 (some below 0.99 threshold)
- Speedup: 9.8–14.5x (avg 12.5x)
- **Verdict: Too aggressive — rejected**

### Step 3: Production implementation

**Change:** Added `BACKPROJECT_SCALE = 0.5` constant to `core/masker.py`.

Both backprojection paths updated consistently:
- **Direct path** (`_backproject_fisheye_mask_to_erp`): delegates to `_backproject_fisheye_mask_to_erp_full` at reduced resolution, upscales with `INTER_NEAREST`
- **Shared-map path** (`_build_backproject_map` + `_BackprojectMap.apply`): map built at reduced grid, `apply()` upscales result

**What was NOT changed:**
- Projection math (still uses production pycolmap path)
- Pass 1 detection
- Direction estimation
- SAM2 tracking
- Reframer

**Validation result:**
- Clip: deskTest, 11 frames, Default preset
- Output: D:\Capture\deskTest\default_validation3
- Masks: OK, ERP and pinhole masks match baseline quality
- Registration: 11/11, 176/176 — unchanged
- Backend: YoloSamBackend + Sam2VideoBackend — unchanged
- Masking runtime: 97.9s → 58.8s (**-39.1s, -40%**)
- Total runtime: 312.5s → 291.5s (**-21.0s, -7%**)
- Cumulative from baseline: masking 156.8s → 58.8s (**-62.5%**), total 404.7s → 291.5s (**-28.0%**)

**Verdict:** Keep. Significant masking speedup, no quality regression.

---

## Cumulative Results Summary

| Phase | Masking | Total | Key change |
|-------|---------|-------|------------|
| Baseline | 156.8s | 404.7s | — |
| Phase 1 (remap cache) | 124.7s (-20%) | 339.0s (-16%) | Detection remap reuse |
| Phase 2 (batched YOLO) | 97.9s (-21%) | 312.5s (-8%) | 16→1 YOLO calls/frame |
| Phase 4 (downsampled BP) | 58.8s (-40%) | 291.5s (-7%) | 0.5x ERP grid for backprojection |
| **Cumulative** | **-62.5%** | **-28.0%** | |

Registration: 11/11, 176/176 throughout. No quality regression at any phase.
