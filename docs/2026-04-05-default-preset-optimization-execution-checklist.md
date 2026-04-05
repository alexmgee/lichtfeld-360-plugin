# Default Preset Optimization — Execution Checklist

**Date:** 2026-04-05  
**Audience:** Claude / future implementation pass  
**Status:** Strict execution checklist  
**Scope:** Default preset only  
**Use with:** [2026-04-05-default-preset-optimization-handoff-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-05-default-preset-optimization-handoff-plan.md)

---

## Purpose

This document turns the broader handoff plan into a strict implementation order.

It is meant to answer:

- what to change first
- what files each step is allowed to touch
- how to validate each step
- when to keep a change
- when to stop, split, or revert a change

The main goal is:

> improve Default-preset speed and quality without destabilizing the now-repaired
> SAM2-backed baseline.

---

## Locked Baseline

Before any new Default optimization work, treat this as the current trusted
baseline:

- environment repaired from ghost-package `sam2` state
- real `sam2==1.1.0` installed
- `sam2.build_sam` importable
- `sam2._C` importable
- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.cuda.is_available() == True`
- `video-tracking` path lock-backed via `uv sync --locked --extra video-tracking`

Default preset proof pass in LFS:

- clip/output: `D:\Capture\deskTest\default_test2`
- `Mask backend: YoloSamBackend`
- `Video backend: Sam2VideoBackend`
- `Frames extracted: 11`
- `Views per frame: 16`
- `Images written: 176`
- `Registered frames: 11/11`
- `Complete rig frames: 11`
- `Registered images: 176`
- all views `11/11`

Timing reference from that proof pass:

- `TOTAL 404.7s`
- `Extraction 101.9s`
- `Masking 156.8s`
- `Reframe 30.9s`
- `overlap_masks 21.1s`
- `COLMAP 94.0s`

Known minor visual note from that proof pass:

- one small pinhole false-positive/noise artifact in:
  - `masks/00_04/deskTest_trim_00007.png`
  - source image `images/00_04/deskTest_trim_00007.jpg`

This is the baseline every new phase must compare against.

---

## Global Rules

## Rule 1

Do not mutate the environment during optimization work unless the task is
explicitly environment-related.

## Rule 2

Do not combine speed work and quality-sensitive logic changes in one patch.

## Rule 3

Do not combine Default work with cubemap work in one patch.

## Rule 4

Each accepted phase should end with:

- validation evidence
- a short notes update
- ideally a checkpoint commit

## Rule 5

If a run is ambiguous, the phase is not done.

---

## Fixed Validation Set

Every phase should be checked against the same small reference set.

## Required run metadata

Capture all of:

- clip name
- frame count
- preset
- total runtime
- masking runtime
- reframe runtime
- COLMAP runtime
- registration result
- backend lines from diagnostics

## Required visual checks

At minimum inspect:

- ERP masks overall quality
- pinhole masks overall quality
- `00_04/deskTest_trim_00007` because it is already known-sensitive

## Required backend confirmation

The validation run must still report:

- `Mask backend: YoloSamBackend`
- `Video backend: Sam2VideoBackend`

If the run reports fallback, stop. Do not treat optimization results as valid.

---

## Phase 0 — Reconfirm Baseline Before Touching Code

## Goal

Make sure the baseline still exists before beginning the next patch.

## Allowed files

No code edits required.

## Required actions

1. confirm the env is still the lock-backed CUDA + SAM2 env
2. confirm `sam2.build_sam` and `sam2._C` are still importable
3. confirm the worktree is clean before starting the patch
4. if there is any doubt, rerun the reference Default clip first

## Keep criteria

- env still healthy
- Default proof path still valid

## Stop criteria

- any sign of fallback backend
- any sign of env drift

---

## Phase 1 — Finish Detection Remap Cache Wiring

## Goal

Remove repeated detection-view remap construction from the real Pass 1 hot path.

## Primary file

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

## Expected code change

Refactor `_primary_detection(...)` so it:

1. builds or refreshes the detection remap cache once per
   `(detection_size, erp_w, erp_h)`
2. reuses cached `(map_x, map_y)` for all 16 detection views
3. calls `_apply_detection_remap(...)` in the loop instead of recomputing via
   `_reframe_to_detection(...)`

## Not allowed in this patch

- no YOLO batching yet
- no threshold changes
- no union-box changes
- no direction changes
- no Pass 2 changes

## Validation

Run the reference Default clip and compare against baseline:

- masks still visually match baseline quality
- registration still `11/11`
- backend lines unchanged
- Pass 1 / masking runtime improves or at least moves in the right direction

## Keep criteria

- visible quality unchanged
- registration unchanged
- measurable speed win

## Revert or split if

- masks shift unexpectedly
- registration drops
- runtime win is not measurable

---

## Phase 2 — Add Real Batched YOLO For Pass 1

## Goal

Replace 16 independent YOLO calls per frame with one batched call.

## Primary file

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

## Expected code shape

Split `_primary_detection(...)` into clear substeps:

1. build all 16 detection images
2. run one batched YOLO call
3. parse results back per view
4. feed existing union-box direction logic exactly as before

## Not allowed in this patch

- no detection-size change
- no confidence change
- no IoU change
- no highest-confidence-box direction change
- no Pass 2 work

## Validation

Run the same Default reference clip and compare:

- masks visually unchanged
- registration unchanged
- backend lines unchanged
- masking runtime lower than Phase 1 baseline

## Keep criteria

- no visible mask regression
- same or better registration
- measurable runtime improvement

## Revert or split if

- quality moves at the same time as batching in a way that cannot be explained
- runtime does not improve enough to justify the patch

---

## Phase 3 — Build Backprojection Validation Harness

## Goal

Create a safe way to test Pass 2 backprojection changes without guessing.

## Allowed files

- new helper under [dev/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev)
- or focused tests under [tests/](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/tests)

## Expected harness behavior

Compare:

1. current backprojection
2. candidate backprojection

Across at least:

- one stable synthetic case
- one off-center synthetic case
- one real tracked fisheye case

## Required outputs

- runtime comparison
- IoU
- changed-pixel count
- a few saved comparison masks or images

## Not allowed in this phase

- no replacement of the production path yet

## Keep criteria

- harness is easy to rerun
- comparisons are clear enough to trust later optimization results

## Rework if

- the harness only measures speed and not correctness
- the harness depends on too much manual interpretation

---

## Phase 4 — Prototype One Backprojection Optimization

## Goal

Attack the main remaining Default hotspot with evidence, not guesswork.

## Primary file

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

## Preferred first candidate

Implement a same-math vectorized NumPy replacement for the current synthetic
camera projection path if it can be shown equivalent enough via the harness.

## Fallback candidate

If the direct replacement is too risky, test reduced-grid backprojection plus
nearest-neighbor upscale.

## Not allowed in this patch

- no simultaneous direction-estimation changes
- no morphology hacks to hide projection artifacts
- no quality-threshold tweaks at the same time

## Validation

1. pass the harness first
2. then run the full Default reference clip in LFS
3. compare:
   - ERP masks
   - pinhole masks
   - registration
   - masking runtime

## Keep criteria

- harness says candidate is close enough
- real run stays visually good
- registration stays good
- runtime improves meaningfully

## Revert if

- harness shows too much divergence
- real LFS run regresses quality
- runtime win is too small for the added risk

---

## Phase 5 — Quality-Sensitive Experiments

Only enter this phase after Phases 1-4 are stable.

## Candidate experiments

- highest-confidence-box direction
- smaller Pass 1 detection size
- other direction-estimation refinements

## Rule

One experiment per patch. No bundling.

## Required framing for each experiment

Each must be treated as a true quality/performance tradeoff trial, not an
assumed improvement.

## Validation

Use the same Default reference clip and the same inspection points.

## Keep criteria

- quality is same or better
- registration is same or better
- runtime is same or better, or any slowdown is explicitly judged worth it

## Revert if

- quality becomes arguable
- registration worsens
- the tradeoff is not obviously worthwhile

---

## Commit Strategy

After each successful phase:

1. record a short result note
2. checkpoint in git
3. start the next phase from a clean worktree

Suggested cadence:

- one commit per accepted phase
- no giant multi-phase optimization commit

---

## Practical “Do Next” Order

If Claude is picking this up immediately, the next steps should be:

1. reconfirm Phase 0 baseline
2. implement Phase 1 only
3. validate and checkpoint
4. implement Phase 2 only
5. validate and checkpoint
6. build Phase 3 harness
7. only then touch Phase 4

That is the safest path to real Default-preset improvement without falling back
into the earlier instability spiral.
