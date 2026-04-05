# Cubemap Quality-First v1 Implementation Checklist

**Date:** 2026-04-04  
**Status:** Baseline implemented; short-sequence reconstruction follow-up remains  
**Parent plan:** [2026-04-04-cubemap-quality-v1-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-plan.md)

---

## Objective

Implement the quality-first cubemap masking path with the smallest practical change set:

- remove ERP Pass 1 from cubemap
- remove face gating from cubemap
- segment all 6 cubemap faces directly
- keep the Default preset unchanged

Current state:

- this baseline is now in the code
- the all-6-face reset improved both speed and mask recall relative to the
  original gated cubemap path
- the remaining instability now looks more like short-sequence reconstruction
  brittleness than a failure of the cubemap mask reset itself

---

## Preconditions

- Default preset masking baseline is restored and verified
- SAM2/video-tracking env is lock-backed and stable
- cubemap is currently functional enough to produce output, but masking reliability is still suspect

### Current quality signal

The current gated cubemap path is already good enough to complete reconstruction on at least some clips, but it still shows the exact fragility this checklist is meant to fix:

- adjacent-face operator fragments can survive as all-white keep masks
- the clearest current example is the `cubemap_test` run:
  - main-face masking can look good
  - an adjacent face can still contain visible operator pixels while remaining fully white

So this checklist is not trying to make cubemap "start working."

It is trying to make cubemap **reliably high-recall**.

---

## Implementation Checklist

### 1. Freeze cubemap scope

- Do not change the Default preset path
- Do not change synthetic Pass 2
- Do not change ERP masking behavior
- Do not change overlap-mask logic
- Do not change COLMAP wiring

Acceptance:

- only the cubemap masking branch in the pipeline is modified

### 1.5 Implementation order

Apply the cubemap quality reset in this order:

1. remove ERP Pass 1 from cubemap
2. remove face gating from cubemap
3. keep direct all-6-face segmentation as the new baseline
4. validate quality before considering any further speed recovery

Rule:

- do not reintroduce any routing/gating optimization until the all-6-face baseline has been tested and accepted

### 2. Simplify the cubemap branch in `pipeline.py`

- In the cubemap branch, remove the ERP `estimate_person_directions()` call
- Stop constructing cubemap masking around `person_directions`
- Keep the existing order:
  1. reframe images
  2. direct cubemap masking
  3. overlap masks
  4. rig/COLMAP/output

Acceptance:

- cubemap path no longer depends on ERP Pass 1 outputs
- cubemap path still produces the same image directory layout as before

File focus:

- [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py)

### 3. Narrow the `MaskConfig` usage for cubemap

- Keep `enable_synthetic=False` for cubemap
- Keep direct image masking backend only
- Do not initialize or require the video backend for cubemap

Acceptance:

- cubemap masking runs without SAM2/video tracking involvement
- cubemap still uses the same image masking backend selection as before

File focus:

- [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

### 4. Refactor `process_reframed_views()` in `masker.py`

- Remove the `person_directions` dependency from the cubemap direct masking helper
- Remove all face visibility gating logic from this v1 path
- Remove all code that writes all-white keep masks because of gating decisions
- Keep the helper focused on:
  1. iterate all view images
  2. load image
  3. run `detect_and_segment()`
  4. convert to COLMAP polarity
  5. apply per-view erosion
  6. write final mask

Also remove or bypass:

- `_cubemap_face_visible()` from the cubemap v1 path
- `gated_in_count`
- `gated_out_count`
- gating-specific progress/log wording

Acceptance:

- every cubemap face image is actually segmented
- no face is skipped because of routing
- the helper no longer takes `person_directions`

File focus:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)

### 5. Preserve the current mask polarity and output layout

- Keep final cubemap masks as:
  - white `255` = keep
  - black `0` = remove
- Keep output structure:
  - `images/<view_name>/<frame>.jpg`
  - `masks/<view_name>/<frame>.png`

Acceptance:

- existing downstream overlap-mask and COLMAP stages still work unchanged

Regression guard:

- no changes to overlap-mask directory ownership or naming
- no changes to `effective_mask_path` logic downstream

### 6. Keep conservative per-face postprocess only

- Start with the existing direct cubemap postprocess pattern:
  - invert detection result into keep-mask polarity
  - apply one `9x9` elliptical erosion to the keep region
- Do not add new morphology unless testing shows a repeated issue

Acceptance:

- v1 stays simple and easy to interpret
- if a face has no detection, it still gets a valid all-white keep mask because the detector returned no foreground, not because routing skipped the face

Important distinction:

- all-white because **no person was found after segmentation** is acceptable
- all-white because **segmentation never ran** is not acceptable in this v1

### 6.5 First cleanup pass: single-primary-box arbitration

This cleanup pass is now the current cubemap refinement in code.

The rule stays strictly conservative:

- keep the all-6-face cubemap path
- do not restore ERP Pass 1
- do not restore face gating
- do not change Default preset detection behavior
- treat direct cubemap masking as a single-operator workflow

Reason:

- the recurring false-positive cases were usually caused by a second weaker
  YOLO person box dragging SAM onto furniture or edge regions
- the useful recall wins were usually still present in the strongest box
- global threshold tuning helped one case but immediately regressed another,
  so it was too brittle to keep as the general rule

Implemented approach:

- the image backend now accepts a cubemap-only `single_primary_box` mode
- in that mode, only the strongest detected person box is passed to SAM
- the legacy all-box behavior remains for Default and other callers

What was tried and rejected before this:

- cubemap-only threshold tuning
- a single shared cubemap threshold was not stable enough to solve both the
  recovered true-positive case and the reintroduced false-positive case

Current practical conclusion:

- box arbitration generalized better than more threshold fiddling
- further cubemap tuning should not default back to face-specific or
  dataset-specific threshold hacks

Acceptance:

- previously recovered adjacent-face operator fragments remain masked
- obvious secondary junk blobs are reduced
- if the top-confidence-only rule drops legitimate split-person cases too often,
  revisit box arbitration logic instead of restoring routing

### 7. Add cubemap-specific debug clarity

- Update cubemap progress/log messages so they no longer mention gating counts
- Replace with simple direct-segmentation progress:
  - segmented views
  - views with detections
  - total frames processed

Acceptance:

- logs accurately reflect the new all-6-face behavior

Suggested replacement counters:

- `segmented_images`
- `images_with_detections`
- `images_without_detections`

### 8. Validate output integrity

- Confirm there are 6 image folders and 6 mask folders
- Confirm image count and mask count match for every cubemap face
- Confirm mask dimensions match image dimensions for every face

Acceptance:

- one final mask exists for every final cubemap image

Minimum spot-check:

- verify every face folder under `images/` has a matching face folder under `masks/`

### 9. Validate mask quality

- Use a fixed cubemap test clip set
- Inspect:
  - main operator face
  - adjacent face boundary spill
  - top face
  - bottom face
  - frame-to-frame consistency
- Explicitly look for:
  - previously skipped adjacent-face operator fragments now being masked
  - no obvious regression on main operator coverage

Acceptance:

- quality is visibly more reliable than the current gated cubemap path

Required first validation clip:

- `D:\Capture\deskTest\cubemap_test`

Required comparison target:

- the current gated cubemap output already produced for that clip

Required visual check from the current evidence:

- compare the face where the main operator is obvious and already masked well
- compare the adjacent face where a visible operator slice survived as all-white keep

Success condition for that case:

- the adjacent-face slice is no longer left unmasked merely because the face was previously not segmented

Secondary quality check:

- confirm the bottom face still masks the operator aggressively where appropriate
- confirm top-face behavior does not become noisy without benefit

### 10. Validate runtime

- Capture:
  - cubemap masking elapsed time
  - total cubemap pipeline time
  - registered frames
  - registered images
- Compare against:
  - current gated cubemap path
  - Default preset baseline

Acceptance:

- cubemap remains materially faster than Default
- runtime increase versus current gated cubemap is acceptable for the reliability gain

Priority rule:

- accept a moderate cubemap slowdown if it clearly removes routing-related misses
- do not reject the quality-first version just because it is slower than the gated experiment

---

## Stop / Keep Criteria

### Keep the change if

- operator coverage is clearly more reliable
- adjacent-face misses are reduced
- COLMAP registration is not worse
- runtime is still attractive relative to Default
- the resulting cubemap behavior is easier to explain to users and easier to debug internally

### Stop and revisit if

- all-6-face segmentation is unexpectedly close to Default runtime
- masks become noisy on many faces without improving recall
- registration quality gets materially worse
- the all-6-face version reveals a systematic detector weakness that routing had only been hiding

---

## Follow-On Optimization Only After v1

Once the all-6-face cubemap path is validated, consider later phases in this order:

1. Always segment 4 horizon faces, gate only top/bottom
2. Add cubemap masking overscan and crop-back for face-boundary quality
3. Add lightweight temporal stabilization if flicker remains
4. Only then consider cheaper routing heuristics

Do not skip straight to Phase 4.

If all-6-face segmentation quality is still not good enough, first ask whether the issue is:

- detector quality on cropped cubemap views
- face-boundary geometry
- per-face postprocess

before adding more routing logic

---

## Expected Outcome

If successful, this v1 should produce a cubemap preset that is:

- more trustworthy than the current gated cubemap path
- still meaningfully simpler than Default
- still meaningfully faster than Default
- much easier to reason about and improve incrementally

The most important v1 outcome is not "maximum speed."

It is:

- a cubemap preset users can plausibly trust as the faster option
