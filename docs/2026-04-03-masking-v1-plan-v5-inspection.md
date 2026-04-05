# Masking Plan v5 Inspection

**Date:** 2026-04-03
**Subject:** Review of `docs/specs/2026-04-03-masking-v1-plan-v5.md`
**Context:** Follow-up inspection after the v4 review in `docs/2026-04-03-masking-v1-plan-v4-inspection.md`

---

## Executive Summary

The v5 plan is the strongest version so far.

It resolves the two remaining issues from the v4 inspection:

- the SAM v2 prompt coordinates are now explicitly defined in resized-frame space
- the bidirectional propagation result-ordering problem is now explicitly handled by storing masks by `out_frame_idx`

At this point, the plan is very close to implementation-ready.

I found one remaining issue worth fixing before implementation starts:

1. the SAM v2 backend flow still does not explicitly define what `window` value should be passed to `propagate_in_whole_video()` for full-sequence coverage

This is now an implementation-detail issue rather than an architecture issue, but it still matters because the helper logic depends on `window` and uses it to limit propagation range.

My current assessment:

- `Track A`: ready
- `Track B`: almost ready
- overall plan quality: high

---

## What v5 Fixed Well

### 1. Prompt coordinates are now tied to the resized SAM v2 frames

This was one of the two carryover issues from the v4 inspection.

The v5 plan now clearly says:

- click coordinates must be computed in the resized SAM v2 temp-frame coordinate space
- they must not be interpreted in original `SYNTHETIC_SIZE` coordinates
- `cx` and `cy` should be computed from the actual resized JPEG dimensions written to the temp directory

That is the right clarification. It makes the intended implementation much harder to misread.

Relevant reference:

- `docs/specs/2026-04-03-masking-v1-plan-v5.md:133`

### 2. Bidirectional propagation reassembly is now specified correctly

This was the other major carryover issue from the v4 inspection.

The v5 plan now explicitly says:

- preallocate `results: list[np.ndarray | None]`
- store each mask by `out_frame_idx`
- do not rely on append order
- bidirectional propagation returns results in processing order rather than frame order
- the prompt frame may appear twice

That is exactly the missing implementation note the v4 plan needed.

Relevant reference:

- `docs/specs/2026-04-03-masking-v1-plan-v5.md:370`

### 3. The plan still preserves the earlier structural improvements

The good changes from v4 remain intact:

- resize/back-projection geometry is explicit
- Track A preserves existing SAM 3 reporting
- the SAM v2 validation spike is framed as actual validation rather than assumption
- propagation is explicitly bidirectional

That means v5 is additive rather than regressive.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v5.md:135`
- `docs/specs/2026-04-03-masking-v1-plan-v5.md:311`
- `docs/specs/2026-04-03-masking-v1-plan-v5.md:336`
- `docs/specs/2026-04-03-masking-v1-plan-v5.md:369`

---

## Remaining Finding

### 1. The plan should explicitly define the propagation window used for SAM v2

This is now the main remaining issue.

The v5 plan correctly says that the backend should:

1. initialize the SAM v2 state
2. prompt on `best_frame_idx`
3. propagate forward and backward
4. collect masks by `out_frame_idx`

However, it still does not explicitly say what `window` value should be passed into `propagate_in_whole_video()`.

That matters because the inspected helper uses `window` directly to determine propagation bounds:

- `reverse_end_frame_idx = max(start_frame_idx - window, 0)`
- `forward_end_frame_idx = min(start_frame_idx + window, total_frames - 1)`

So `window` is not just a small tuning detail. It is part of the control flow that determines how much of the sequence is actually covered.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v5.md:369`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:88`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:94`

---

## Why This Matters

If the plan leaves `window` unspecified, an implementation can go wrong in more than one way:

### Case 1: `window` is omitted entirely

The helper signature allows `window=None`, but the inspected implementation uses `window` in arithmetic when computing forward and reverse bounds.

That means a naive caller that treats `window` as optional may hit a runtime failure.

### Case 2: `window` is too small

If the code passes a hardcoded or guessed value that is smaller than the actual sequence length:

- tracking will stop before reaching the whole clip
- late frames or early frames may remain untracked
- the failure may look like a model-quality issue rather than a control-flow bug

This is especially easy to miss when testing on short sequences.

### Case 3: behavior differs across clips

Even if a fixed value works on short clips, it may fail on longer ones.

That would create a plan that appears correct in spot testing but behaves inconsistently across real captures.

---

## Recommended Fix

The plan should explicitly define the intended whole-sequence behavior.

The simplest and clearest rule is:

- when invoking `propagate_in_whole_video()`, pass a `window` value that guarantees full coverage of the available frames

In practice, the cleanest wording would be something like:

> Pass `window=total_frames` (or an equivalent full-sequence span) so propagation can cover the entire clip in both directions from `best_frame_idx`.

That instruction is better than leaving the value implicit because it tells the implementer the real invariant:

- the propagation span must cover the whole sequence

It also leaves room for a mathematically equivalent expression if the final code prefers something like:

- `window = max(total_frames, 1)`
- `window = len(frames)`
- another equivalent full-coverage value

---

## Suggested Plan Edit

I would update the B2 flow block to read like this:

```text
Flow:
1. Write synthetic fisheye frames to tempdir as `{i:04d}.jpg` (resized to 512px min dimension, matching FullCircle)
2. `predictor.init_state(str(tempdir))`
3. Find the frame with strongest primary detection -> compute click coords in resized-frame space
4. `predictor.add_new_points_or_box(state, best_frame_idx, 0, [[cx, cy]], [1])`
5. Propagate forward AND backward from `best_frame_idx`, passing a `window` value that covers the entire sequence (for example `window=total_frames`)
6. Collect masks by `out_frame_idx` -> preallocate `results: list[np.ndarray | None]` for all frames, write each callback's mask into `results[out_frame_idx]`
7. Threshold at 0.5, resize each mask to `SYNTHETIC_SIZE` before returning
8. Clean up tempdir
```

That would close the last meaningful ambiguity I see in Track B.

---

## Instructional Notes For Implementation

If someone starts coding directly from this plan, these are the key behavioral rules they should carry into the implementation:

### For Track A

- keep the current image backend behavior intact for the fallback path
- isolate reframer convention math from synthetic-camera convention math
- treat the synthetic pass as an additive second pass, not a replacement for the primary ERP mask generation
- keep Track A shippable without any `sam2` dependency

### For Track B

- do not assume the SAM v2 install method before the validation spike proves it
- compute click coordinates from resized temp-frame dimensions, not original synthetic-camera dimensions
- propagate in both directions from the chosen prompt frame
- store results by `out_frame_idx`, not by callback order
- make the propagation window explicitly cover the full sequence
- resize masks back to the original synthetic camera resolution before ERP back-projection

These points are now almost all present in the plan already. The only one I would still make explicit in the plan text is the last missing window rule.

---

## Readiness Assessment

### Track A

`Track A` looks ready to implement.

Why:

- it avoids new dependency risk
- it is grounded in geometry and pipeline work that is already well understood
- the minimum product-facing updates are now appropriately scoped
- it delivers user-visible value even if SAM v2 never lands

### Track B

`Track B` is nearly ready.

Why:

- the packaging risk is now isolated in B1 where it belongs
- the sequence-ordering issue is fixed
- the prompt-coordinate ambiguity is fixed
- only one remaining propagation-control detail still needs to be spelled out

### Overall

This is now a high-quality implementation plan.

Compared with the earlier drafts:

- the major geometry risks have been addressed
- the UI/setup mismatches have been accounted for
- the SAM v2 flow now reflects the actual inspected helper behavior much more closely

The remaining work is now mostly about careful execution rather than discovering missing architecture.

---

## Final Recommendation

I would make one final wording pass to define the SAM v2 propagation window explicitly, then treat the plan as ready for implementation.

If that edit is added, my recommendation would be:

- begin `Track A`
- keep `Track B` gated behind the B1 validation spike
- use the updated B2 flow as the authoritative implementation guide for the SAM v2 backend

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v5.md`
- `docs/2026-04-03-masking-v1-plan-v4-inspection.md`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`
