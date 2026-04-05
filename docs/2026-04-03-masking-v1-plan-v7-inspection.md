# Masking Plan v7 Inspection

**Date:** 2026-04-03
**Subject:** Review of `docs/specs/2026-04-03-masking-v1-plan-v7.md`
**Context:** Follow-up inspection after the v6 review in `docs/2026-04-03-masking-v1-plan-v6-inspection.md`

---

## Executive Summary

The v7 plan is in excellent shape.

It resolves the remaining ownership issue from the v6 inspection by explicitly assigning the `best_frame_idx` selection to `_synthetic_pass()` and passing it into the video backend as `initial_frame_idx`.

That was the last meaningful contract ambiguity in the plan.

At this point, there is only one small remaining issue I would still clean up:

1. the Track B flow mixes `initial_frame_idx` and `best_frame_idx` terminology in adjacent steps even though the ownership model is now clear

This is a wording issue rather than an architectural one.

My current assessment:

- `Track A`: ready
- `Track B`: effectively ready
- overall plan quality: very high

---

## What v7 Fixed Well

### 1. Ownership of the prompt frame is now assigned clearly

This was the last substantive issue from the v6 inspection.

The v7 plan now explicitly says that `_synthetic_pass()`:

- owns the `best_frame_idx` selection
- chooses it from the primary-detection results
- passes it to the video backend as `initial_frame_idx`

That is the right division of responsibility.

It keeps:

- sequence-level decision making in the sequence-level orchestration layer
- backend logic focused on sequence execution

This is a good protocol fit and should reduce the chance of duplicate or divergent prompt-frame selection logic.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v7.md:278`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:367`

### 2. The earlier control-flow fixes remain intact

The improvements from the previous rounds are still preserved:

- click coordinates are computed in resized-frame space
- propagation uses full-sequence coverage
- masks are reassembled by `out_frame_idx`
- SAM v2 remains optional and validation-gated

This means the plan now reflects the real FullCircle-inspired execution model much more closely than the earlier drafts did.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v7.md:133`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:369`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:370`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:336`

### 3. The plan is now mostly in execution territory

At this point, the major issues are no longer about architecture or pipeline shape.

The plan now has:

- a credible Track A that ships value without `sam2`
- a well-scoped Track B gated behind validation
- a clearer caller/backend contract
- explicit notes on projection math, sequence ordering, and UI/setup implications

That means the remaining risk is mainly:

- actual SAM v2 packaging/runtime behavior in the plugin environment

rather than missing design work in the document itself.

---

## Remaining Finding

### 1. The backend flow should use one frame-index name consistently

This is now a small terminology issue.

The plan has already clarified the ownership model:

- `_synthetic_pass()` chooses `best_frame_idx`
- the backend receives that value as `initial_frame_idx`

However, the Track B flow still mixes the two names:

- step 3 refers to the caller-provided `initial_frame_idx`
- step 4 passes `initial_frame_idx` into `add_new_points_or_box(...)`
- step 5 then says to propagate from `best_frame_idx`

That is not functionally wrong if the two names refer to the same frame, but it reintroduces mild friction right after the plan finally clarified who owns the decision.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v7.md:367`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:368`
- `docs/specs/2026-04-03-masking-v1-plan-v7.md:369`

---

## Why This Matters

This is not a deep design problem, but it is still worth cleaning up because terminology drift at an interface boundary tends to create unnecessary confusion during implementation.

### Case 1: A reader wonders whether two different concepts still exist

Even if the intent is obvious after reading carefully, mixed naming can make it seem like:

- `best_frame_idx` is one thing
- `initial_frame_idx` is another thing

That is exactly the ambiguity the earlier revision was trying to eliminate.

### Case 2: Code comments and tests can drift in different directions

One person may write tests and comments using `best_frame_idx`, while another uses `initial_frame_idx`.

That kind of inconsistency is minor, but it makes the implementation harder to scan and reason about.

### Case 3: The final code contract is less obvious than it could be

The cleanest contract is:

- orchestration layer thinks in terms of "best frame"
- backend API thinks in terms of `initial_frame_idx`

If that is the model, the backend flow block should express it consistently once the handoff has happened.

---

## Recommended Fix

The simplest cleanup is:

- keep `best_frame_idx` language in `_synthetic_pass()` where the selection happens
- use `initial_frame_idx` consistently inside the backend flow once the value has been handed off

That means step 5 in the B2 flow should be rewritten from:

- "Propagate forward AND backward from `best_frame_idx` ..."

to something like:

- "Propagate forward AND backward from `initial_frame_idx` ..."

That would make the protocol boundary fully explicit.

---

## Suggested Plan Edit

I would rewrite the relevant B2 flow section like this:

```text
3. Use the caller-provided `initial_frame_idx` (selected by `_synthetic_pass()` from the strongest primary detection) and compute click coords from the resized temp-frame dimensions for that frame
4. `predictor.add_new_points_or_box(state, initial_frame_idx, 0, [[cx, cy]], [1])`
5. Propagate forward AND backward from `initial_frame_idx`, passing `window=total_frames` (or equivalent full-sequence span) so propagation covers the entire clip in both directions
```

That would align the flow description with the now-correct responsibility split.

---

## Instructional Notes For Implementation

If someone starts implementing from this version of the plan, the intended layering now appears to be:

### `_synthetic_pass()`

- select the strongest prompt frame
- call that value `best_frame_idx`
- pass it to the backend as `initial_frame_idx`

### `Sam2VideoBackend`

- treat `initial_frame_idx` as an input contract, not something to rediscover
- compute the prompt click for that frame in resized-frame coordinates
- start SAM v2 propagation from that provided frame
- store results by `out_frame_idx`
- resize results back to `SYNTHETIC_SIZE`

That is a clean and testable split.

The only thing I would still change in the document is making the wording reflect that split consistently all the way through.

---

## Readiness Assessment

### Track A

`Track A` looks ready.

Why:

- it is independent of `sam2` packaging success
- the geometry and pipeline refactor are now well specified
- it preserves honest product-facing messaging during the interim state
- it delivers value even on its own

### Track B

`Track B` now looks effectively ready.

Why:

- the protocol responsibility issue is resolved in substance
- the propagation window is fixed
- the result-ordering issue is fixed
- the prompt-coordinate issue is fixed
- the only remaining issue is one terminology cleanup in the flow text

### Overall

This is now a very high-quality plan.

The main remaining risk is no longer in the document design. It is in the actual validation and runtime behavior of `sam2` inside the plugin/LichtFeld environment.

---

## Final Recommendation

I would make one tiny wording cleanup so the backend flow uses `initial_frame_idx` consistently after the handoff from `_synthetic_pass()`. After that, I would consider the plan ready to execute.

If that small cleanup is made, I would treat this as the implementation-ready version of the plan.

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v7.md`
- `docs/2026-04-03-masking-v1-plan-v6-inspection.md`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`
