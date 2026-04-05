# Cubemap Quality-First v1 File-by-File Plan

**Date:** 2026-04-04  
**Status:** Implemented for the current cubemap baseline  
**Parent docs:**  
- [2026-04-04-cubemap-quality-v1-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-plan.md)  
- [2026-04-04-cubemap-quality-v1-implementation-checklist.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-implementation-checklist.md)  
- [2026-04-04-cubemap-test-review.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-test-review.md)

---

## Goal

Implement the cubemap quality-reset with the smallest safe code change set:

- remove ERP Pass 1 from cubemap
- remove cubemap face gating
- segment all 6 cubemap faces directly
- keep Default unchanged

---

## Primary Files To Change

### 1. [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py)

This file owns the high-level cubemap branch and is the place where cubemap still depends on ERP Pass 1.

#### Current cubemap behavior

The cubemap branch currently:

1. reframes ERP images into cubemap faces
2. creates a `Masker` with `enable_synthetic=False`
3. still calls `masker.estimate_person_directions(...)`
4. passes `person_directions` into `masker.process_reframed_views(...)`

Relevant lines at the time of writing:

- cubemap branch entry: [`core/pipeline.py#L314`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py#L314)
- reframe images only: [`core/pipeline.py#L325`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py#L325)
- cubemap masker config: [`core/pipeline.py#L346`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py#L346)
- ERP direction estimation: [`core/pipeline.py#L362`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py#L362)
- direct cubemap masking call: [`core/pipeline.py#L372`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py#L372)

#### Planned change

Change the cubemap branch so it becomes:

1. reframe ERP images into cubemap faces
2. create cubemap `Masker` with `enable_synthetic=False`
3. call direct cubemap masking immediately
4. continue to overlap masks / COLMAP unchanged

#### Exact edit intent

- delete the cubemap-only call to `estimate_person_directions()`
- stop constructing `person_directions`
- change the cubemap `process_reframed_views(...)` call to no longer pass directions
- keep the rest of the cubemap branch structure intact

#### Acceptance

- cubemap path no longer depends on ERP Pass 1
- Default branch is untouched
- overlap-mask generation still receives the same final `out/masks` structure

---

### 2. [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

This file owns the current gated cubemap logic and will carry most of the actual behavior change.

#### Current cubemap-specific pieces

- ERP Pass 1 helper:
  - [`core/masker.py#L785`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L785) `estimate_person_directions(...)`
- gating helper:
  - [`core/masker.py#L846`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L846) `_cubemap_face_visible(...)`
- direct cubemap masking:
  - [`core/masker.py#L875`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L875) `process_reframed_views(...)`

Inside `process_reframed_views(...)`, current behavior includes:

- `person_directions` parameter
- `gated_in_count` / `gated_out_count`
- face visibility check
- gated-out early write of all-white keep masks
- gated-in segmentation path

Relevant lines:

- function signature: [`core/masker.py#L875`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L875)
- directions parameter: [`core/masker.py#L879`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L879)
- counters: [`core/masker.py#L907`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L907)
- gate decision: [`core/masker.py#L934`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L934)
- gated-out write: [`core/masker.py#L946`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L946)
- segmentation path: [`core/masker.py#L957`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L957)
- per-view erosion: [`core/masker.py#L928`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L928) and [`core/masker.py#L975`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L975)

#### Planned change

Refactor `process_reframed_views(...)` into a true all-6-face direct segmentation helper.

#### Exact edit intent

1. Remove `person_directions` from the function signature.
2. Remove all face gating logic from the function.
3. Remove `_cubemap_face_visible(...)` from the active cubemap v1 path.
4. Remove `gated_in_count` and `gated_out_count`.
5. Keep the function focused on:
   - discover images
   - load image
   - run `self._backend.detect_and_segment(...)`
   - invert to COLMAP polarity
   - erode keep mask once
   - write final PNG
6. Preserve the current `9x9` erosion kernel initially.
7. Preserve the current all-white fallback only when segmentation returns no foreground, not when routing skips a face.
8. Update logs/progress text to reflect direct segmentation rather than gating.

#### Open decision

What to do with `_cubemap_face_visible(...)` and `estimate_person_directions(...)` after the change:

- safest cleanup now:
  - leave both in place but unused by the cubemap path
- cleaner follow-up later:
  - remove or deprecate them once the new cubemap path is proven

Recommendation:

- leave them in place for now
- do not delete them in the same change set as the behavior rewrite

#### Acceptance

- every cubemap face image is truly segmented
- adjacent-face misses are reduced relative to the current gated path
- no cubemap face remains all-white merely because it was skipped by routing

---

## Validation / Utility File To Consider

### 3. [dev/test_masking.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/test_masking.py)

This script is useful for the Default ERP masking path, but it is not the right harness for the cubemap direct-masking workflow.

Current behavior:

- uses the Default preset
- constructs `Masker` for ERP masking
- calls `process_frames(...)`

Relevant lines:

- imports: [`dev/test_masking.py#L29`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/test_masking.py#L29)
- Default preset assumption: [`dev/test_masking.py#L49`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/test_masking.py#L49)
- direct ERP masking entry: [`dev/test_masking.py#L73`](\/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/test_masking.py#L73)

#### Recommendation

Do not overload this file for cubemap.

Instead, if an isolated cubemap harness is needed, add a separate helper later, such as:

- `dev/test_cubemap_masking.py`

That helper should:

1. accept ERP frames input
2. reframe to cubemap images
3. run the cubemap direct masking helper
4. optionally skip COLMAP

For the initial implementation, this is optional because full pipeline cubemap runs already give usable evidence.

#### Acceptance

- no unnecessary churn in existing Default masking test tooling

---

## Files Expected To Stay Unchanged

### [core/reframer.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/reframer.py)

No behavior change needed for v1.

Reason:

- the cubemap quality reset keeps the same image geometry
- the reframer is already successfully producing the 6 cubemap face images

### [core/overlap_mask.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/overlap_mask.py)

No behavior change needed for v1.

Reason:

- overlap masks should consume the final `masks/` tree exactly as they do today

### [core/presets.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/presets.py)

No geometry change needed for v1.

Reason:

- this plan is about cubemap masking behavior, not cubemap camera layout

### UI files

- [panels/prep360_panel.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/panels/prep360_panel.py)
- [panels/prep360_panel.rml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/panels/prep360_panel.rml)

No v1 UI change needed.

Reason:

- the preset offering remains the same
- this is an implementation-quality change, not a user-workflow change

---

## Recommended Change Sequence

### Patch 1

Update [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py) so cubemap no longer calls ERP Pass 1 and no longer passes `person_directions`.

Validation:

- cubemap run still completes
- image output still looks structurally normal

### Patch 2

Update [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py) so `process_reframed_views(...)` segments all 6 faces directly and no longer uses gating.

Validation:

- `cubemap_test` rerun
- inspect the previously problematic adjacent-face example
- check that every face still has one mask per image

### Patch 3

Optional small logging cleanup in [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py) to make progress output match the new behavior.

Validation:

- logs are easier to interpret
- no functional change

### Patch 4

Optional documentation update after the behavior is confirmed good.

Validation:

- docs describe the new cubemap path accurately

---

## Core Success Case

The most important concrete success case for this change is:

- on the current `cubemap_test` clip, a face like `00_02` that still contains a visible operator fragment should no longer remain all-white simply because the previous routing logic skipped or effectively bypassed it

If the new all-6-face direct segmentation fixes that kind of miss while preserving the already-good main-face masks and keeping cubemap faster than Default, then the cubemap quality reset is working.

---

## Recommendation

Make this a **two-file functional change** first:

1. [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py)
2. [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

Everything else should stay unchanged unless the first validation rerun proves we need a follow-up harness or documentation refresh.
