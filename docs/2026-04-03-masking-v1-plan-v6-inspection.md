# Masking Plan v6 Inspection

**Date:** 2026-04-03
**Subject:** Review of `docs/specs/2026-04-03-masking-v1-plan-v6.md`
**Context:** Follow-up inspection after the v5 review in `docs/2026-04-03-masking-v1-plan-v5-inspection.md`

---

## Executive Summary

The v6 plan is in very strong shape.

It fixes the last main issue from the v5 inspection:

- the SAM v2 propagation window is now explicitly defined so the full sequence is covered

It also preserves the major improvements from the previous rounds:

- prompt coordinates are explicitly defined in resized SAM v2 frame space
- bidirectional propagation results are explicitly reassembled by `out_frame_idx`
- the resize/back-projection geometry is explicit
- Track A still preserves existing SAM 3 reporting
- the SAM v2 installation path remains correctly gated behind the validation spike

At this point, there is only one remaining issue I would still tighten before implementation:

1. the plan should explicitly assign responsibility for choosing `best_frame_idx`

This is a much smaller issue than the earlier geometry or propagation problems. The plan is now very close to implementation-ready.

My current assessment:

- `Track A`: ready
- `Track B`: almost ready
- overall plan quality: high

---

## What v6 Fixed Well

### 1. The propagation window is now explicit

This was the main remaining issue from the v5 inspection.

The v6 plan now correctly says that the SAM v2 backend should:

- propagate forward and backward from `best_frame_idx`
- pass `window=total_frames` or an equivalent full-sequence span
- treat a too-small window as a correctness issue because it silently truncates coverage

That is the right clarification.

It now matches the inspected helper behavior more closely by acknowledging that `window` is not just a tuning parameter; it is part of the control flow that determines how much of the clip is actually processed.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v6.md:369`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:88`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:94`

### 2. The earlier v5 fixes remain intact

The v6 plan keeps the important clarifications that were added in v5:

- prompt coordinates are computed in resized-frame space
- results are stored by `out_frame_idx`
- duplicate prompt-frame callbacks are handled safely by index-based storage

Those were important implementation details, and it is good that they remain explicit here.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v6.md:133`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:370`

### 3. The broader structure still holds together well

The larger planning decisions still look solid:

- `Track A` remains useful and shippable without `sam2`
- `Track B` remains appropriately gated behind packaging/runtime validation
- the spec/UI/setup reconciliation is still accounted for
- the plan still reflects the real FullCircle-inspired workflow rather than a speculative port

At this point, the remaining questions are mostly about execution detail rather than missing architecture.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v6.md:317`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:330`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:381`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:471`

---

## Remaining Finding

### 1. The plan should explicitly say who chooses `best_frame_idx`

This is now the main remaining ambiguity.

The `VideoTrackingBackend` protocol currently exposes:

```python
def track_sequence(
    self,
    frames: list[np.ndarray],
    initial_mask: np.ndarray | None = None,
    initial_frame_idx: int = 0,
) -> list[np.ndarray]
```

That interface strongly suggests the caller is responsible for determining the prompt frame and passing it in as `initial_frame_idx`.

That matches the surrounding architecture, because `_synthetic_pass()` is the place that has access to:

- the synthetic frames
- the primary ERP masks
- the per-frame direction data
- the broader sequence context

However, the Task B2 flow still says:

- "Find the frame with strongest primary detection"

inside the backend implementation section itself.

That wording makes it sound as if `Sam2VideoBackend` is the layer responsible for choosing the strongest frame.

The problem is that the backend protocol, as written, does not actually receive the full sequence-level primary-detection information it would need to make that decision independently.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v6.md:232-237`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:278`
- `docs/specs/2026-04-03-masking-v1-plan-v6.md:367`

---

## Why This Matters

If the plan leaves ownership of `best_frame_idx` ambiguous, implementation can drift in a few unhelpful ways.

### Case 1: The caller and backend both try to own the decision

One implementation path might compute the strongest frame in `_synthetic_pass()`, while another might try to recompute it again inside `Sam2VideoBackend`.

That creates duplicate logic and increases the chance that the two paths diverge.

### Case 2: The backend lacks the data needed to make the decision well

If the backend only receives:

- `frames`
- `initial_mask`
- `initial_frame_idx`

then it does not naturally have the full primary-detection sequence summary that the planning text implies.

That could lead to:

- heuristic reimplementation inside the backend
- weaker prompt-frame selection
- or quiet fallback to frame `0` when the selection logic is unclear

### Case 3: The protocol no longer reflects the actual design intent

A good protocol should make ownership obvious.

Right now, the protocol suggests:

- caller chooses `initial_frame_idx`

while the B2 flow text partially suggests:

- backend chooses `best_frame_idx`

That is not a major flaw, but it is worth resolving before people start coding against the wrong mental model.

---

## Recommended Fix

The cleanest resolution is:

- `_synthetic_pass()` chooses the strongest frame from `primary_masks`
- `_synthetic_pass()` passes that frame index into `track_sequence(..., initial_frame_idx=best_frame_idx)`
- `Sam2VideoBackend` treats `initial_frame_idx` as the already-selected prompt frame

That keeps the responsibility in the layer that already has the full sequence-level context and avoids overloading the backend with policy that belongs one level up.

It also fits the existing protocol naturally without needing a larger interface redesign.

---

## Suggested Plan Edit

I would tighten the relevant sections with wording like this:

```text
In _synthetic_pass():
- determine best_frame_idx from the primary-detection sequence (e.g. strongest primary mask / strongest primary detection)
- pass that frame index into the video backend as initial_frame_idx

In Sam2VideoBackend:
- use initial_frame_idx as the prompt frame
- compute click coordinates for that frame in resized-frame space
- call add_new_points_or_box(... frame_idx=initial_frame_idx, ...)
```

And in the B2 flow, I would rephrase step 3 slightly:

```text
3. Use the caller-provided prompt frame (`initial_frame_idx`, selected from the strongest primary detection) and compute click coords in resized-frame space
```

That would make the contract much clearer without changing the architecture.

---

## Instructional Notes For Implementation

If someone starts implementing directly from this plan, the key control-flow rules should be:

### For `_synthetic_pass()`

- own the sequence-level reasoning
- choose the prompt frame based on the primary-detection results
- prepare the synthetic fisheye frames
- call the video backend with the chosen `initial_frame_idx`

### For `Sam2VideoBackend`

- do not rediscover the strongest frame on your own
- trust `initial_frame_idx` as the chosen prompt frame
- compute click coordinates from the resized temp-frame dimensions
- propagate in both directions with full-sequence coverage
- store results by `out_frame_idx`
- resize masks back to `SYNTHETIC_SIZE` before returning

### For the protocol itself

- keep `track_sequence()` focused on sequence execution, not prompt-frame policy
- let the caller decide which frame to prompt from

This separation keeps the backend reusable and the overall flow easier to test.

---

## Readiness Assessment

### Track A

`Track A` still looks ready to implement.

Why:

- it is decoupled from `sam2` packaging risk
- the geometry and pipeline work is now well specified
- the product-facing interim updates are scoped clearly enough
- it should deliver value even if Track B is delayed

### Track B

`Track B` is now extremely close.

Why:

- the propagation window is fixed
- the prompt-coordinate issue is fixed
- the output ordering issue is fixed
- only one responsibility/ownership detail still needs to be stated more clearly

### Overall

This is now a high-quality implementation plan.

The remaining ambiguity is not structural. It is the kind of contract clarification that is easiest to fix before coding starts and mildly annoying to fix afterward if two layers have already started to assume different responsibilities.

---

## Final Recommendation

I would make one final edit that explicitly assigns `best_frame_idx` selection to `_synthetic_pass()` and treats `initial_frame_idx` as the backend input contract. After that, I would consider the plan ready for implementation.

If that edit is made, my recommendation would be:

- start `Track A`
- keep `Track B` gated behind B1 validation
- use the clarified caller/backend division of responsibility as the implementation guide for the SAM v2 path

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v6.md`
- `docs/2026-04-03-masking-v1-plan-v5-inspection.md`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`
