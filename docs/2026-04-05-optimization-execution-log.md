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
