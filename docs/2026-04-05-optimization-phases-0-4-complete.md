# Default Preset Optimization — Phases 0-4 Complete

**Date:** 2026-04-05
**Audience:** Codex / future sessions
**Status:** Phases 0-4 validated and committed. Phase 5 parked.

---

## Summary

The Default preset optimization pass is complete through Phase 4. Masking runtime dropped 62.5% from baseline with zero quality regression. The pipeline is committed at `85d6380` and the environment is lock-backed.

## Results

| Phase | Masking | Total | Change | Key optimization |
|-------|---------|-------|--------|-----------------|
| Baseline | 156.8s | 404.7s | — | Proof pass on stabilized env |
| Phase 1 | 124.7s | 339.0s | masking -20% | Detection remap cache |
| Phase 2 | 97.9s | 312.5s | masking -21% | Batched YOLO (16→1 call/frame) |
| Phase 4 | 58.8s | 291.5s | masking -40% | 0.5x downsampled backprojection |
| **Cumulative** | **58.8s** | **291.5s** | **masking -62.5%, total -28.0%** | |

Registration: 11/11 frames, 176/176 images throughout. No quality regression at any phase.

Test clip: `deskTest`, 11 frames, Default preset (16 views), exhaustive matching.

## What Was Done

### Phase 0 — Baseline verification
Confirmed environment (torch 2.11.0+cu128, sam2.build_sam importable, Sam2VideoBackend active) and recorded baseline timing.

### Phase 1 — Detection remap cache (`da31c07`)
Wired the existing `_build_detection_remap` / `_apply_detection_remap` scaffolding into the real `_primary_detection` hot path. The 16 detection layout remap tables are now built once per ERP size and reused across all frames. Eliminated 16×(N-1) redundant meshgrid+trig computations.

### Phase 2 — Batched YOLO (`da31c07`)
Split `_primary_detection` into three stages: build 16 detection images, one batched YOLO call, parse results per view. Reduced GPU kernel launch overhead from 16 per frame to 1.

### Phase 3 — Backprojection validation harness (`85d6380`)
Built `dev/backprojection_harness.py` to compare candidate backprojection implementations against production. Tests synthetic masks at multiple directions and ERP resolutions. Measures IoU, changed pixels, area drift, and runtime. Proved numpy math replacement is bit-identical to pycolmap (IoU=1.0) but offers no speedup alone.

### Phase 4 — Downsampled backprojection (`85d6380`)
Implemented `BACKPROJECT_SCALE = 0.5` in both the direct backprojection path and the shared-map path. Computes at half ERP resolution (7.4M vs 29.5M points), upscales binary result with `INTER_NEAREST`. Harness validated IoU > 0.99 across all test cases. Area drift < 0.11%, changed pixels boundary-localized only.

## What Was NOT Done

### Phase 5 — Quality-sensitive experiments (parked)
Two candidates remain for future work:

- **5A: Highest-confidence box direction** — replace union-box direction estimation with the single highest-confidence YOLO detection. May reduce false-positive drag on direction. Previously caused a regression, but that was during environment drift — needs retesting on the stable baseline.

- **5B: Smaller detection resolution** — test `min(512, erp_w // 4)` instead of `min(1024, erp_w // 4)`. May provide additional Pass 1 speedup but must be validated as a quality/performance tradeoff.

Both require one-at-a-time testing with the same validation protocol (masks, registration, backend confirmation). Neither should be bundled with other changes.

## Current Code State

### Live optimizations in `core/masker.py`
- Detection remap cache (`_detection_remap_cache`, `_detection_remap_key`)
- Batched YOLO in `_primary_detection` (builds 16 images → one YOLO call → parse results)
- `BACKPROJECT_SCALE = 0.5` in `_backproject_fisheye_mask_to_erp` and `_build_backproject_map`
- Prompt frame selection by `detection_counts` (not empty mask area)
- Union-box direction estimation (restored baseline)
- Detection size `min(1024, erp_w // 4)` (restored baseline)

### Live optimizations in `core/reframer.py`
- Reframe remap cache (`_build_reframe_remap` / `_apply_reframe_remap`)
- Shared remap tables between image and mask reprojection
- `cv2.remap(..., INTER_NEAREST)` for mask path

### Environment
- torch 2.11.0+cu128, torchvision 0.26.0+cu128
- sam2 1.1.0 with `_C.pyd` installed
- `video-tracking` optional dependency lock-backed
- `uv sync --locked --extra video-tracking` preserves working state

### Git state
- Latest commit: `85d6380` (Phase 3-4)
- Previous: `da31c07` (Phase 1-2)
- Baseline checkpoints: `ec158a1`, `93452b3`

## Remaining Performance Targets

After the optimization pass, the pipeline cost breakdown is:

| Stage | Time | % |
|-------|------|---|
| Extraction | 107.2s | 37% |
| Masking | 58.8s | 20% |
| COLMAP | 78.4s | 27% |
| Reframe | 29.3s | 10% |
| Overlap masks | 17.8s | 6% |

Masking is no longer the dominant cost. Extraction (FFmpeg + sharpness scoring) and COLMAP matching are now the largest stages. Further masking optimization has diminishing returns unless the quality experiments in Phase 5 reveal a cheaper direction path.

## Validation Protocol For Future Work

For any future Default-preset change, capture:
- Clip name, frame count, preset
- Total, masking, reframe, COLMAP runtime
- Registration result (frames, images, per-view)
- Backend lines (must show Sam2VideoBackend)
- ERP mask visual quality
- Pinhole mask visual quality
- Known-sensitive frame: `00_04/deskTest_trim_00007`
