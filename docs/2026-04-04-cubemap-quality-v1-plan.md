# Cubemap Quality-First v1 Plan

**Date:** 2026-04-04  
**Status:** Baseline implemented and validated; follow-up stability investigation remains  
**Scope:** Cubemap preset only  
**Default preset:** Unchanged and out of scope

---

## Goal

Make the `cubemap` preset a trustworthy faster alternative to the Default preset by prioritizing **masking reliability first** and **speed second**.

The core idea is:

- mask the actual final cubemap output images directly
- remove direction-based face skipping
- avoid the FullCircle-style synthetic/video-tracking path entirely for cubemap
- keep the implementation simple enough to validate cleanly

---

## Current Cubemap Path

The current cubemap implementation already has a dedicated branch, but it still depends on ERP-side direction estimation and uses that direction to decide whether a cubemap face gets segmented at all.

Current behavior:

1. Extract ERP frames
2. Reframe ERP frames into 6 cubemap face image folders
3. Run ERP Pass 1 direction estimation using the fixed 16-view detection layout
4. For each cubemap face image:
   - if the face is considered visible from the estimated person direction, run `detect_and_segment()`
   - otherwise skip segmentation and write an all-white keep mask
5. Write per-face masks
6. Generate overlap masks for COLMAP
7. Run COLMAP

This is faster than the Default preset because cubemap does **not** use:

- synthetic fisheye rendering
- SAM2 video tracking
- fisheye-to-ERP backprojection
- ERP mask reprojection into final views

But it is still fragile because:

- cubemap quality still depends on ERP Pass 1 being directionally correct
- a wrong direction estimate can gate out a face that should have been segmented
- direct per-face misses have no ERP OR-merge safety net
- there is no temporal tracking to stabilize frame-to-frame misses

---

## Main Problem

The current cubemap path is trying to be fast by skipping segmentation on some faces.

That optimization introduces a high-risk failure mode:

- wrong or slightly imperfect direction estimate
- face gets gated out
- segmentation never runs on that face
- the final mask incorrectly keeps the operator in that face

For cubemap to be attractive to users, it first needs to become **predictable** and **high-recall**.

---

## Proposed Quality-First Cubemap Workflow

### v1 workflow

1. Extract ERP frames normally.
2. Reframe each ERP frame into the 6 final cubemap faces.
3. Segment **all 6 cubemap faces directly**.
4. Do **not** run the 16-view ERP Pass 1 first.
5. Do **not** gate faces in or out based on estimated person direction.
6. Convert each face segmentation result into final COLMAP polarity:
   - detected operator region = black `0`
   - keep/background = white `255`
7. Apply conservative per-face postprocess:
   - one small erosion of the keep region
   - optionally tiny-island cleanup if needed later
8. Write masks directly to the final output structure:
   - `images/<view>/<frame>.jpg`
   - `masks/<view>/<frame>.png`
9. Compute overlap masks for COLMAP exactly as today.
10. Run COLMAP exactly as today.

### What this removes

- ERP Pass 1 direction estimation for cubemap
- direction-based face gating
- synthetic Pass 2
- SAM2 video tracking for cubemap
- fisheye rendering and backprojection
- ERP mask generation for cubemap

### What this keeps

- the same cubemap image geometry
- the same final mask polarity
- the same overlap-mask stage
- the same rig / COLMAP / output flow

---

## Why This Is The Right v1

### 1. It masks the actual final images

The direct target of reconstruction is the cubemap face images, so masking those exact images is the most straightforward and least ambiguous path.

### 2. It removes the worst current cubemap failure mode

The current path can silently skip segmentation on a face because of a routing error. The proposed path removes that class of failure entirely.

### 3. It is still structurally simpler than Default

Even when all 6 faces are segmented, cubemap still avoids:

- temporal video tracking
- synthetic view generation
- ERP backprojection
- ERP mask reprojection

So this should still remain meaningfully simpler, and likely faster, than Default.

### 4. It gives a clean quality baseline

Once cubemap quality is proven with all-face direct segmentation, later speedups can be evaluated against a stable known-good cubemap behavior.

---

## Expected Tradeoff

This v1 is intentionally **not** the fastest possible cubemap path.

It trades some potential speed for:

- higher recall
- less routing fragility
- easier debugging
- easier user trust

Expected outcome:

- slower than the current gated cubemap experiment
- faster than Default
- much more reliable than the current cubemap masking path

---

## Detailed Implementation Shape

### Pipeline behavior

The cubemap branch should become:

1. Reframe ERP images into cubemap images
2. Run direct all-6-face masking on those images
3. Compute overlap masks
4. Continue to COLMAP

The cubemap branch should no longer call ERP Pass 1 direction estimation.

### Masker behavior

Add or revise the cubemap direct masking helper so that it:

1. walks every `images/<view_name>/*.jpg`
2. loads each image
3. runs `self._backend.detect_and_segment()`
4. converts the detection mask to COLMAP polarity
5. applies conservative per-face erosion
6. writes `masks/<view_name>/<frame>.png`

The direct cubemap helper should not take `person_directions`.

### Output ownership

For cubemap:

- `Reframer` owns `images/`
- direct cubemap masking owns `masks/`
- overlap-mask generation reads `masks/` and writes temporary COLMAP masks as it does today

---

## Per-Face Postprocess v1

Use the simplest postprocess that is already consistent with the current code style:

1. detection result is `0/1` or equivalent binary operator-region mask
2. invert into keep-mask polarity
3. erode the keep region once with a small elliptical kernel

Recommended v1 posture:

- keep the existing `9x9` elliptical erosion unless testing shows it is too aggressive
- do not add heavy morphology up front
- only add extra cleanup if specific artifacts are observed repeatedly

---

## Non-Goals For v1

This plan does **not** try to:

- make cubemap as robust as the Default preset in every scenario
- add temporal tracking to cubemap
- merge masks back through ERP
- introduce cubemap-specific routing heuristics
- optimize top/bottom face skipping
- optimize Pass 1 because Pass 1 is removed from cubemap in this plan

---

## Acceptance Criteria

The quality-first cubemap v1 is successful only if all of these are true:

1. Cubemap masking no longer depends on ERP direction estimation.
2. Every frame produces one mask per final cubemap face image.
3. No cubemap face is skipped because of direction gating.
4. Output mask dimensions always match the corresponding cubemap image dimensions.
5. The final masks have correct COLMAP polarity:
   - operator = black `0`
   - keep = white `255`
6. Cubemap masking quality is visibly more reliable than the current gated version on known troublesome clips.
7. Runtime remains materially lower than the Default preset on the same clip and output size.
8. Registration quality is not materially worse than the current cubemap path.

---

## Validation Plan

### Quality validation

Use a fixed set of test clips and inspect:

1. operator coverage on horizon faces
2. operator coverage near face boundaries
3. top/bottom face behavior when the operator is low or high in frame
4. frame-to-frame consistency
5. obvious misses where a face should clearly have been segmented

### Runtime validation

Capture at minimum:

- cubemap masking wall-clock time
- total cubemap pipeline wall-clock time
- number of segmented face images
- number of final output images

### Reconstruction validation

Compare against the current cubemap path:

- registered images
- registered frames
- complete rig frames
- obvious COLMAP degradation caused by worse masks

---

## Future Optimization Path

Only after v1 quality is proven should speed optimizations be considered.

Recommended order:

### Phase 2: Conservative optimization

1. Always segment the 4 horizon faces
2. Only gate top and bottom faces

This preserves most recall while reducing wasted work on pole faces.

### Phase 3: Overscan for boundary quality

Render temporary overscanned faces for masking, then crop back to the final 90° faces.

This is a quality improvement more than a speed optimization and may be worth doing even before further routing tricks.

### Phase 4: Temporal stabilization

Add a lightweight face-local stabilization pass if frame-to-frame flicker remains a problem.

### Phase 5: Faster routing only if needed

If cubemap is still too slow, consider a cheaper cubemap-specific routing heuristic later. But this should happen only after the all-6-face baseline is validated.

---

## Current Status Note

This quality-reset baseline has now been implemented:

- cubemap no longer runs ERP Pass 1
- cubemap no longer gates faces
- cubemap now segments all 6 final faces directly
- a conservative single-primary-box rule is now used for cubemap direct masking

The strongest current open issue is no longer basic cubemap mask recall.

It is:

- short-sequence reconstruction brittleness on some near-identical runs

Longer cubemap runs are currently looking healthier than very short runs, so the
next investigation should focus more on short-sequence COLMAP robustness than on
adding more mask-routing complexity back into cubemap.

---

## Recommendation

Treat the next cubemap iteration as a **quality-reset** rather than a speed optimization.

The cleanest next cubemap design is:

- no ERP Pass 1
- no face gating
- direct segmentation on all 6 final cubemap faces
- direct final mask writing

If that version is visibly reliable and still comfortably faster than Default, cubemap becomes a credible user-facing “fast preset.”
