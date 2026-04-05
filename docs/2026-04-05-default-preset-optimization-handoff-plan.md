# Default Preset Optimization & Quality Re-Application — Handoff Plan

**Date:** 2026-04-05  
**Audience:** Claude / future maintainers  
**Status:** Planning document for the next Default-preset work phase  
**Scope:** Default preset only  
**Purpose:** Provide a comprehensive, execution-ready plan for improving speed and quality on the stabilized Default preset baseline without reintroducing the recent environment or masking regressions.

**Companion checklist:** [2026-04-05-default-preset-optimization-execution-checklist.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-05-default-preset-optimization-execution-checklist.md)

---

## Executive Summary

The Default preset is back in a good state.

That baseline is now:

- masking quality restored
- plugin end-to-end behavior restored
- CUDA + SAM2 runtime restored
- `video-tracking` path lock-backed
- checkpointed in git

This means the next Default-preset work should no longer be recovery.

It should be:

- careful optimization
- careful quality improvement
- one contained change at a time
- measured from the current stabilized baseline

The two most important constraints for this next phase are:

1. **Do not destabilize the environment again.**
2. **Do not mix multiple quality-sensitive changes into one patch.**

The safest execution order is:

1. freeze the current baseline and benchmark it cleanly
2. finish the safe Pass 1 speed work that is partially present but not fully wired
3. attack Pass 2 backprojection with explicit equivalence validation
4. only then revisit any quality-sensitive detection/direction changes

---

## Baseline Preconditions

This plan assumes all of the following are already true.

### Runtime baseline

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`
- `sam2.build_sam` importable

### Project / installer baseline

- `video-tracking` is modeled as an optional dependency in [pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml)
- [core/setup_checks.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/setup_checks.py) installs video tracking via locked sync
- `uv sync --locked --extra video-tracking` preserves the working runtime

### Functional baseline

- masking-only rerun returned to prior good quality
- full plugin rerun returned good ERP masks and good pinhole masks
- Default masking regression was primarily traced to environment drift, not a still-broken Default algorithm baseline

### Checkpoints already in git

- `ec158a1` `Checkpoint masking stabilization and environment lock-in`
- `93452b3` `Document verified masking baseline and lock-backed env`

These are the recovery checkpoints that the next optimization phase should build on, not overwrite conceptually.

---

## Current Live Code State

This section is important because some optimization work exists in the repo already, some exists only as scaffolding, and some ideas were investigated but should not be treated as current behavior.

## Already live and should be preserved

### 1. Stage 3 reframe remap caching

[core/reframer.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/reframer.py) already has the main remap-cache split:

- `_build_reframe_remap(...)`
- `_apply_reframe_remap(...)`

The batch reframer already:

- builds remap tables once per ERP size / output geometry
- reuses them across frames
- reuses the same geometry for both RGB images and masks
- uses `cv2.remap(..., INTER_NEAREST)` for masks

This should be treated as a successful phase-1 optimization and kept intact.

### 2. Prompt-frame selection by `detection_counts`

[core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py) now selects the SAM2 prompt frame using `detection_counts`, not empty ERP mask area.

This fix should stay.

### 3. Pass 1 detection size restored to 1024 max

The live Default path currently uses:

```python
detection_size = min(1024, erp_w // 4)
```

This is the restored quality-safe baseline and should be treated as the starting point.

### 4. Shared-map backprojection path exists

[core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py) currently includes:

- `_BackprojectMap`
- `_build_backproject_map(...)`
- `_direction_angular_spread(...)`
- threshold-gated shared-map reuse in `_synthetic_pass(...)`

This should be preserved while further backprojection work is evaluated.

---

## Present in code, but not fully realized in the Default path

These are the most important items for Claude to understand.

### 1. Detection remap helper split exists, but the Default path still recomputes per view

[core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py) already contains:

- `_build_detection_remap(...)`
- `_apply_detection_remap(...)`
- cache fields on `Masker`:
  - `_detection_remap_cache`
  - `_detection_remap_key`

But the live Default `_primary_detection(...)` path still calls `_reframe_to_detection(...)` inside the per-view loop, which recomputes remap tables every call.

So the helper split exists, but the optimization is not fully wired into the hot path yet.

### 2. Batched YOLO is not yet actually live in `_primary_detection(...)`

The live `_primary_detection(...)` path still:

- builds one face image
- runs YOLO immediately on that one face
- repeats this 16 times per frame

So the idea is documented, but the Default path still needs the actual “build all 16 detection images first, then run one batched YOLO call” refactor.

---

## Live quality-sensitive behavior that should be treated carefully

### 1. Direction estimation currently uses union boxes

The live Default `_primary_detection(...)` path still computes direction from the union of all detected person boxes in a view.

That is the restored baseline, not necessarily the final ideal behavior.

It should not be changed casually in the same patch as speed work.

### 2. Backprojection is still the main remaining hotspot

The existing reports consistently point to Pass 2 backprojection as the main remaining cost center after the earlier wins.

That is where the next real performance work likely belongs, but it needs explicit correctness validation.

---

## What This Plan Is Not

This plan is **not**:

- another environment-repair plan
- a cubemap plan
- a prompt-frame-debug plan
- a “change everything and see what happens” pass

This plan is specifically for:

- re-applying or finishing the Default-preset optimization work from a stable baseline
- preserving current mask quality while reducing runtime

---

## Recommended Strategy

Treat the next Default phase as three layers:

### Layer A: Finish the safe speed work that is already partly prepared

This includes:

- wiring the detection remap cache into the real Pass 1 hot path
- converting Pass 1 YOLO to one batched call per frame

These are the lowest-risk, highest-confidence remaining Default optimizations.

### Layer B: Attack the real hotspot with an explicit validation harness

This includes:

- backprojection optimization
- but only after there is a small harness that can compare candidate outputs to the current implementation

### Layer C: Revisit quality-sensitive logic only after A and B

This includes:

- highest-confidence-box direction
- smaller detection resolution
- other direction-estimation refinements

These changes may still be good, but they should not be mixed into the safe speed work.

---

## Exact Execution Plan

## Phase 0 — Freeze the current Default baseline

Before any new Default optimization work, Claude should:

1. confirm the env is still the lock-backed CUDA + SAM2 baseline
2. confirm no new environment mutation has happened
3. run one reference Default-preset validation clip in LFS
4. record:
   - total runtime
   - masking runtime
   - reframe runtime
   - registration result
   - representative mask screenshots / frame references

Purpose:

- establish a clean “before” state for the next phase

Rule:

- do not start optimization patches until this reference run exists

---

## Phase 1 — Finish Pass 1 remap caching

### Goal

Stop recomputing detection-view remap tables inside `_primary_detection(...)`.

### Files

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

### Change

Refactor `_primary_detection(...)` so it:

1. builds or refreshes the detection remap cache once per `(detection_size, erp_w, erp_h)`
2. reuses cached `(map_x, map_y)` pairs for all 16 views
3. calls `_apply_detection_remap(...)` inside the loop instead of `_reframe_to_detection(...)`

### Do not change in this patch

- YOLO behavior
- union-box behavior
- detection thresholds
- coverage threshold
- Pass 2 behavior

### Acceptance

- identical visible mask quality on the benchmark clip
- lower Pass 1 wall-clock time
- no change in registered output quality

### Why this should come first

The helper split and cache scaffolding already exist, so this is mostly finishing an intended optimization rather than inventing a new one.

---

## Phase 2 — Convert Pass 1 YOLO to batched inference

### Goal

Reduce 16 YOLO launches per frame to 1 batched call.

### Files

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

### Change

Refactor `_primary_detection(...)` into three explicit stages:

1. build the 16 detection face images
2. run one batched YOLO call on the 16 images
3. parse results per view into the current direction-estimation logic

### Keep the behavior the same in this patch

- same `conf`
- same `iou`
- same `classes`
- same union-box direction logic
- same coverage threshold

### Acceptance

- no visible quality regression
- measurable Pass 1 speed improvement
- same or better registration outcome

### Why this patch should stay narrow

If quality changes at the same time as batching, it becomes impossible to tell whether the regression came from batching or from a simultaneous detection-logic change.

---

## Phase 3 — Add a backprojection validation harness

### Goal

Create a tiny, explicit mechanism to compare candidate backprojection implementations against the current one.

### Files

Suggested:

- new helper under [dev/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev)
- or a focused test under [tests/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/tests)

### Harness requirements

The harness should compare:

1. current backprojection
2. candidate backprojection

on at least:

- one stable synthetic case
- one off-center synthetic case
- one real tracked fisheye mask from a handheld clip

### Metrics

- runtime
- IoU vs current result
- changed-pixel count
- visual sanity output for a few representative masks

### Why this phase matters

Backprojection is the strongest next performance target, but it is also the easiest place to accidentally trade correctness for speed without noticing.

---

## Phase 4 — Prototype the next real hotspot optimization

### Preferred first candidate: same-math replacement

Prototype a direct NumPy equidistant fisheye projection path that aims to match `pycolmap.img_from_cam(...)` for the synthetic camera model actually in use.

The goal here is:

- replace a generic camera-model hot call with equivalent vectorized math

Why this is preferred:

- it attacks the dominant cost directly
- it works for moving-camera sequences
- it is conceptually a same-math replacement, not a sampling-model rewrite

### Fallback candidate: downsampled backprojection

If the NumPy replacement is too risky or too finicky, the next fallback is:

- compute backprojection on a reduced ERP grid
- upscale with nearest-neighbor

This is lower-risk than forward projection and easier to validate than a more radical mapping rewrite.

### Do not jump to yet

- forward projection / splatting
- heavy morphology to cover projection artifacts
- simultaneous direction-estimation changes

---

## Phase 5 — Revisit quality-sensitive Default refinements

Only after Phases 1-4 should Claude revisit the quality-sensitive ideas from the earlier notes.

### Candidate A: highest-confidence box for direction

This idea is still interesting:

- it may reduce false-positive drag on direction estimation
- it may improve synthetic-view centering

But it should now be tested as a pure quality change, not bundled into speed work.

### Candidate B: smaller Pass 1 detection size

This can still be tested later:

- `1024` baseline vs `512`

But it should be treated as a real quality/performance tradeoff experiment, not as a presumed free win.

### Rule

Only one of these should be tested at a time, on top of the already-stabilized optimized baseline.

---

## Validation Rules For Every Phase

For each patch or experiment, Claude should capture:

1. exact clip name
2. exact frame count
3. preset
4. total runtime
5. masking runtime
6. reframe runtime
7. COLMAP / registration result
8. a small fixed set of known-sensitive mask examples

And each patch should answer:

- did it get faster?
- did masks stay good?
- did registration stay good?

If any answer is “not sure,” the patch should not be treated as a success yet.

---

## Stop / Keep Criteria

## Keep a change if

- it gives a measurable speedup
- it preserves current Default mask quality
- it preserves or improves registration quality
- it does not require environment mutation to “make it work”

## Stop and revert or split the change if

- masking quality becomes ambiguous
- registration gets worse
- the patch mixes multiple behavioral changes
- the benchmark evidence is not clean enough to attribute the result

---

## Recommended File Touch Order

The safest sequence for Claude is:

1. [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)
   - finish detection remap cache wiring
2. [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)
   - add batched YOLO for Pass 1
3. [dev/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev) or [tests/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/tests)
   - add backprojection validation harness
4. [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)
   - prototype validated backprojection optimization
5. only then consider quality-sensitive detection/direction changes

Files that should generally stay untouched in the first next round:

- [core/reframer.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/reframer.py)
  - already carries the successful remap-cache work
- [core/setup_checks.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/setup_checks.py)
  - environment lock-in is already working
- cubemap-specific files/paths
  - should not be mixed into this Default pass

---

## Practical Guidance For Claude

The most important instruction is:

> do not treat the old optimization documents as if every listed change is currently live in code.

Some are live.
Some were reverted.
Some are partially scaffolded but not fully wired.

The first job is to preserve the stabilized Default baseline and only make changes that can be benchmarked cleanly against it.

In practice, that means:

- no broad refactors
- no simultaneous quality + speed + environment changes
- no assumption that “faster” means “safe”
- no trying to solve cubemap and Default in the same pass

---

## Recommended Outcome

If executed well, the next Default-preset phase should achieve this:

1. Pass 1 becomes genuinely cheaper through:
   - real detection remap reuse
   - real batched YOLO
2. Pass 2 backprojection is attacked in an evidence-based way
3. Quality-sensitive ideas are tested later, not mixed into the core speed work
4. The plugin remains push-ready throughout the process

That is the right way to improve speed and quality while keeping the recovered Default baseline intact.

---

## Related Documents

- [2026-04-04-default-masking-stabilization-report-and-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-default-masking-stabilization-report-and-plan.md)
- [2026-04-04-performance-optimization-results.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-performance-optimization-results.md)
- [2026-04-04-performance-optimization-results-response.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-performance-optimization-results-response.md)
- [2026-04-04-masking-performance-quality-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-performance-quality-plan.md)
- [2026-04-04-masking-stabilization-handoff-report.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-stabilization-handoff-report.md)
