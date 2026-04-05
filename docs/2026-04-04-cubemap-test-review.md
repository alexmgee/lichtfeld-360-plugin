# Cubemap Test Review

**Date:** 2026-04-04  
**Run reviewed:** `D:\Capture\deskTest\cubemap_test`  
**Related plan:** [2026-04-04-cubemap-quality-v1-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-plan.md)

---

## Summary

The current cubemap path is **functional**, but it still shows the directional-routing fragility that the quality-first cubemap plan is meant to remove.

This run does **not** look like a total failure:

- the run completed successfully
- all expected cubemap images were written
- all expected cubemap masks were written
- COLMAP registration was complete

But the mask distribution and sample face review strongly suggest that the current gated cubemap approach is still too fragile to present as a dependable fast preset.

---

## Run Facts

From `timing.json`:

- total time: `99.874s`
- extraction: `10.732s`
- reframe: `5.947s`
- masking: `44.256s`
- COLMAP: `38.938s`
- source frames: `11`
- output images: `66`
- views per frame: `6`
- registered frames: `11`
- complete rig frames: `11`
- dropped rig frames: `0`

Interpretation:

- cubemap is operational on this clip
- the masking stage is still the largest pre-COLMAP stage
- the run is good enough to evaluate quality, not just pipeline health

---

## Output Structure Check

The run produced:

- 6 image folders under `images/`
- 6 mask folders under `masks/`
- 11 images per face
- 11 masks per face

So the output structure itself is healthy.

---

## Mask Distribution Pattern

Quick mask statistics by face:

- `00_00`: all 11 masks all-white
- `00_01`: 10/11 masks all-white
- `00_02`: 8/11 masks all-white
- `00_03`: all 11 masks all-white
- `01_00`: 0/11 masks all-white, heavy masking throughout
- `02_00`: all 11 masks all-white

Interpretation:

- the operator is being masked primarily in the bottom face and a small number of horizon faces
- many faces are being left untouched throughout the run
- this could be partly correct for the clip geometry, but it also means the current path is highly selective about where segmentation effort is spent

That selectivity is the exact area of concern.

---

## Representative Visual Findings

### Positive example

Face `00_01`, frame `deskTest_trim_00007`:

- the operator is clearly present
- the mask captures the visible operator region reasonably well

This shows the current cubemap path can work on the main face.

### Important weakness

Face `00_02`, frame `deskTest_trim_00007`:

- the image still contains a visible slice of the operator at the edge
- the mask is fully white

Interpretation:

- the current path is allowing adjacent-face operator visibility to survive as keep-region
- whether that happened because of gating or because direct segmentation missed the small fragment, the result is the same:
  - visible operator content was not removed from that face

This is exactly the kind of miss that makes cubemap masking feel unreliable to a user.

### Bottom-face behavior

Face `01_00`, frame `deskTest_trim_00007`:

- the operator is large and obvious
- the mask is correspondingly aggressive

So the current cubemap path is not failing everywhere. It is strongest on the face where the operator is dominant, and weakest at cross-face edge coverage.

---

## What This Says About The Current Cubemap Path

This run suggests:

1. The cubemap branch is viable as a reconstruction path.
2. The current masking behavior is good enough to produce a complete registered result on at least some clips.
3. The current cubemap routing is still not trustworthy enough for a user-facing “fast preset” because visible edge fragments can survive in adjacent faces.

That means the problem is no longer “can cubemap work at all?”

The real question is now:

> can cubemap become predictably good enough that users will choose it for speed without feeling like they are gambling on masking quality?

This run suggests the answer is:

- not with the current gated path
- probably yes with a quality-first all-6-face direct segmentation path

---

## Does This Support The Proposed Direction?

Yes.

This run supports the quality-first cubemap plan for three reasons:

### 1. The main failure is recall, not total collapse

The issue is not that cubemap masks are universally terrible. The issue is that they are selectively incomplete.

That is the kind of problem best addressed by:

- removing routing fragility
- increasing segmentation recall
- masking all final faces directly

### 2. Adjacent-face misses matter

The `00_02` example shows that even a small visible operator fragment can be left unmasked.

That is exactly the type of failure that an all-6-face direct masking strategy is intended to reduce.

### 3. The run was otherwise healthy

Because the run completed and registered successfully, we can evaluate cubemap quality as a real design problem rather than an infrastructure problem.

That makes this a good moment to simplify cubemap behavior instead of adding more routing complexity.

---

## Recommendation

Treat this run as evidence that the current cubemap path is:

- promising
- fast enough to be interesting
- still too fragile to trust fully

So the right next cubemap move is:

1. remove ERP Pass 1 from cubemap
2. remove face gating from cubemap
3. segment all 6 cubemap faces directly
4. validate whether adjacent-face misses are reduced

Only after that quality-reset baseline is proven should further cubemap speed optimizations be considered.
