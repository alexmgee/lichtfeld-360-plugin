# Cubemap Preset Status Report

**Date:** 2026-04-05  
**Audience:** Claude / future maintainers  
**Purpose:** Record the current cubemap preset implementation, the run history that led to it, what improved, and what still appears unresolved.

---

## Executive Summary

Cubemap is in a much better place than it was at the start of this pass.

The current cubemap masking baseline is now:

- direct masking on all 6 final cubemap faces
- no ERP Pass 1 direction estimation
- no face gating
- cubemap-only single-primary-box arbitration before SAM

That reset improved the cubemap masking path in the ways we were actually
targeting:

- better adjacent-face recall
- materially lower masking time
- simpler behavior that is easier to reason about than the earlier gated path

The main remaining concern no longer looks like core cubemap mask quality.

It looks more like:

- short-sequence reconstruction brittleness on some near-identical cubemap runs

Longer cubemap runs currently look much healthier from a registration
standpoint.

So the current recommendation is:

- keep the new cubemap masking baseline
- stop fine-tuning mask logic aggressively for now
- treat short-run registration robustness as the next real follow-up question

---

## Current Cubemap Implementation

The current cubemap preset now works like this:

1. Extract ERP frames.
2. Reframe them into the 6 final cubemap faces.
3. Segment all 6 cubemap faces directly.
4. Convert each result into final keep-mask polarity.
5. Write masks directly to `masks/<view>/<frame>.png`.
6. Run overlap-mask generation and COLMAP as before.

Important implementation points:

- cubemap no longer runs ERP Pass 1 direction estimation
- cubemap no longer gates faces in or out
- cubemap no longer depends on `person_directions`
- cubemap still avoids the Default preset's synthetic/SAM2 video-tracking path
- cubemap currently uses single-primary-box arbitration before SAM to reduce
  disconnected junk caused by secondary weaker YOLO person boxes

Files carrying the current cubemap behavior:

- [core/pipeline.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/pipeline.py)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py)
- [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py)

Related planning docs:

- [2026-04-04-cubemap-quality-v1-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-plan.md)
- [2026-04-04-cubemap-quality-v1-implementation-checklist.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-implementation-checklist.md)
- [2026-04-04-cubemap-quality-v1-file-by-file-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-quality-v1-file-by-file-plan.md)

---

## Why We Changed Cubemap

The older cubemap path was fast, but too fragile:

- it still paid for ERP-side Pass 1 direction estimation using the 16-view layout
- it used that direction estimate to decide whether a cubemap face got segmented
- when that routing was wrong, a face could stay fully white even though visible
  operator pixels were present

That made cubemap feel risky as a user-facing fast preset.

The clearest early sign was:

- one face could mask the operator correctly
- an adjacent face could still contain a visible operator slice and remain fully
  white

So the goal of this pass was to make cubemap more dependable first, then worry
about speed second.

---

## Run History

## 1. Original gated cubemap baseline

Run:

- `D:\Capture\deskTest\cubemap_test`

Observed:

- total: `99.874s`
- masking: `44.256s`
- registered frames: `11/11`

Takeaway:

- cubemap was functional
- but adjacent-face recall was too fragile
- this run motivated the all-6-face quality reset

Reference:

- [2026-04-04-cubemap-test-review.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-cubemap-test-review.md)

---

## 2. All-6-face direct cubemap reset

Run:

- `D:\Capture\deskTest\cubemap_test2`

Observed:

- total: `73.909s`
- masking: `22.946s`
- registered frames: `11/11`

What changed:

- ERP Pass 1 removed from cubemap
- face gating removed
- all 6 faces segmented directly

Takeaway:

- this was the right overall direction
- it improved both speed and recall
- it also showed that the old ERP prepass was a bigger speed cost than expected

Remaining issue:

- some false-positive masks still appeared, especially on the bottom face

---

## 3. Threshold tuning experiments

Runs:

- `D:\Capture\deskTest\cubemap_test4`
- `D:\Capture\deskTest\cubemap_test5`

Why they happened:

- one bottom-face case improved when the cubemap-only detection threshold was
  raised
- another real operator case disappeared when the threshold stayed too high

What was learned:

- `0.55` was too strict
- `0.45` was too loose
- one global cubemap threshold could not cleanly separate the true-positive and
  false-positive cases

Takeaway:

- threshold tuning was informative
- threshold tuning was not durable enough to keep as the main cubemap strategy

---

## 4. Single-primary-box arbitration

Run:

- `D:\Capture\deskTest\cubemap_test6`

Why this change was tried:

- bad cubemap false positives were often caused by a second weaker YOLO person
  box pulling SAM onto furniture or edge regions
- the stronger box usually tracked the actual operator more cleanly

What changed:

- cubemap now keeps only the strongest YOLO person box before SAM
- Default and other paths keep legacy all-box behavior

Takeaway:

- this is a better generalization strategy than face-specific or dataset-specific
  threshold tuning
- the resulting mask changes looked reasonable

Important caveat:

- short-run registration still collapsed on some runs even when images and masks
  were nearly identical to earlier successful runs

That strongly suggests the remaining brittleness is not simply "cubemap masking
still bad."

---

## 5. Longer cubemap run

Run:

- `D:\Capture\deskTest\cubemap_test6_longer`

Observed:

- total: `325.357s`
- masking: `55.745s`
- COLMAP: `225.439s`
- frames extracted: `26`
- images written: `156`
- registered frames: `26/26`
- registered images: `156/156`

Takeaway:

- longer cubemap runs are looking healthy from a reconstruction standpoint
- this strengthens the idea that the main remaining brittleness is about short
  sequence robustness, not a basic cubemap mask failure

---

## Current Assessment

### What looks clearly improved

- cubemap masking is simpler than before
- cubemap no longer has the "wrong route, no segmentation" failure mode
- adjacent-face recall is better than the original gated version
- masking time is much better than the original gated cubemap baseline
- cubemap still remains meaningfully simpler than Default

### What no longer looks like the main problem

- basic cubemap mask recall
- the old ERP Pass 1 routing dependency
- face gating logic
- more threshold micro-tuning

### What now looks like the main remaining issue

- short-sequence reconstruction brittleness
- run-to-run registration instability on very short cubemap clips

The key evidence is:

- multiple short cubemap runs had identical reframed images and nearly identical
  masks but very different registration outcomes
- a longer cubemap run registered fully

That is much more consistent with downstream reconstruction instability than
with a meaningful mask-quality collapse.

---

## Recommendation

For now, keep the current cubemap masking baseline:

- all 6 faces segmented directly
- no ERP Pass 1
- no face gating
- single-primary-box arbitration

Do not add more cubemap mask-routing complexity back in right now.

Do not start face-specific tuning.

Do not institute a hard minimum clip-length rule yet.

Instead, treat the next follow-up question as:

> why are short cubemap reconstructions brittle between near-identical runs?

That is the investigation most likely to move cubemap forward now.

---

## Practical Product Position

Current realistic framing:

- `Default` remains the safest, highest-confidence preset
- `Cubemap` is now a legitimate faster preset
- `Cubemap` appears more dependable on medium/longer captures than on tiny short
  clips

That is a usable place to pause.

The cubemap preset no longer feels like a broken experiment.

It now feels like:

- a promising fast mode with a smaller remaining risk envelope

---

## Next Follow-Up, If We Revisit This

If cubemap work resumes soon, the next investigation should focus on:

1. short-sequence COLMAP robustness
2. run-to-run nondeterminism on near-identical short cubemap inputs
3. whether any rig, matcher, or mapper assumptions are more fragile on small
   cubemap clips than on longer ones

It should not start with more threshold tuning or more face-routing heuristics.
