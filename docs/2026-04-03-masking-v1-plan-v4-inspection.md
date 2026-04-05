# Masking Plan v4 Inspection

**Date:** 2026-04-03
**Subject:** Review of `docs/specs/2026-04-03-masking-v1-plan-v4.md`
**Context:** Follow-up inspection after the v3 reviews in `docs/2026-04-03-masking-v1-plan-v3-inspection.md` and `docs/2026-04-03-masking-v1-plan-v3-inspection-round2.md`

---

## Executive Summary

The v4 plan is in strong shape overall.

It successfully addresses the main concerns from the prior inspection round:

- the SAM v2 resize and back-projection geometry is now explicit
- the interim Track A product-facing updates now preserve existing SAM 3 reporting
- the SAM v2 install spike is now framed as validation of the working method, not a hardcoded dependency path
- the SAM v2 tracking flow now explicitly covers both forward and backward propagation from the selected prompt frame

At this point, `Track A` looks ready to implement and `Track B` is very close.

There are only two remaining issues I would still fix before implementation:

1. the SAM v2 backend flow should explicitly store masks by `out_frame_idx`, not by append order
2. the prompt-coordinate wording should be tightened so nobody mixes original synthetic-camera coordinates with resized SAM v2 frame coordinates

These are smaller than the earlier structural issues. The plan now reads as implementation-ready with a short final polish pass.

---

## What Improved Since the Last Inspection

### 1. The resize / back-projection path is now clearly specified

The previous review identified a serious geometry risk:

- the synthetic camera is defined at `SYNTHETIC_SIZE`
- SAM v2 runs on resized frames
- the plan had not clearly said how reduced-resolution masks get mapped back to the original synthetic camera geometry before ERP back-projection

That is now fixed.

The v4 plan explicitly states that:

- SAM v2 works on resized frames
- output masks are at reduced resolution
- masks must be resized back to `SYNTHETIC_SIZE × SYNTHETIC_SIZE`
- nearest-neighbor interpolation is used
- the back-projection math always uses the original synthetic camera model

This resolves the biggest remaining geometry ambiguity from the earlier drafts.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:135`

### 2. Track A now preserves existing SAM 3 reporting

The prior v3 inspection noted that the temporary capability-level plumbing in Track A could accidentally regress the plugin's current SAM 3 reporting path.

That concern is now addressed directly.

Task A7 now explicitly says:

- Track A only adds the minimum capability-level plumbing needed for the interim state
- existing SAM 3 reporting must remain intact
- the panel must still show `"Using SAM 3"` when `premium_tier_ready` is true
- only the non-SAM 3 path should be generalized to `"Masking ready"`

That is the right safeguard for an incremental rollout.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:311`
- `docs/specs/2026-04-03-masking-v1-plan-v4.md:312`

### 3. The SAM v2 install spike is now worded correctly

The earlier review noted that the plan still implied a preferred install path before the validation spike had actually proven anything.

That wording is now better.

B1.1 now says to start with `uv add sam2`, but explicitly document whether another method is required, including options such as:

- `uv pip install --no-deps`
- installing from Git
- other dependency-resolution workarounds

That properly frames B1 as discovery and validation rather than simple confirmation.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:336`

### 4. Reverse propagation is now part of the planned SAM v2 flow

The previous inspection's most important Track B issue was that a forward-only propagation flow would fail to track frames before the chosen prompt frame if the best detection occurred in the middle of the sequence.

That is now addressed.

The plan now explicitly states that SAM v2 should propagate forward and backward from `best_frame_idx`, which matches the inspected FullCircle helper's support for reverse traversal.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:369`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:63-105`

### 5. The verification section now fits the real Windows workflow better

The earlier Unix-only cleanup command problem is now cleaned up by listing both:

- a Git Bash `rm -rf` version
- a PowerShell `Remove-Item` version

That is a small but useful execution detail fix.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:523`

---

## Remaining Findings

### 1. The SAM v2 backend flow should explicitly store masks by frame index

This is now the main remaining technical issue.

The plan's `VideoTrackingBackend` protocol says:

```python
def track_sequence(...) -> list[np.ndarray]
```

That strongly implies the returned list is ordered by frame index.

However, the inspected FullCircle helper does not yield results in simple chronological order when reverse propagation is enabled. Its processing order is:

1. forward from `start_frame_idx` to the end window
2. then reverse from `start_frame_idx` back toward earlier frames

That means:

- yielded results are in processing order, not frame order
- the prompt frame can appear twice
- a naive implementation that appends masks as they arrive can misalign output masks with source frames

The v4 plan now correctly requires forward and backward propagation, but it does not yet explicitly say how the backend should reassemble those results.

### Why this matters

If `Sam2VideoBackend` returns masks in append order instead of storing by `out_frame_idx`:

- masks can be associated with the wrong ERP frames
- earlier frames may shift in the returned sequence
- the prompt frame may be duplicated in the output list

That would create incorrect results even if the underlying SAM v2 tracking itself worked correctly.

### Recommended fix

The plan should add one explicit implementation note in Task B2:

- preallocate `results: list[np.ndarray | None]` for all frames
- on each SAM callback, write the current mask into `results[out_frame_idx]`
- if both forward and reverse traversal produce the prompt frame, let the later write overwrite the earlier identical slot
- return the final list in strict frame-index order

This is a small addition, but it removes a real source of implementation error.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:232-237`
- `docs/specs/2026-04-03-masking-v1-plan-v4.md:369-370`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py:97-105`

### 2. The prompt-coordinate wording should be tied to the resized SAM v2 frames

This is a low-severity issue, but worth tightening.

The prompt strategy section says the click should be at:

- `(SYNTHETIC_SIZE/2, SYNTHETIC_SIZE/2)`

That is conceptually correct in the original synthetic camera space, because the person is supposed to be centered in the synthetic fisheye view.

But the same section also makes clear that SAM v2 operates on resized temp-frame JPEGs rather than the original `SYNTHETIC_SIZE × SYNTHETIC_SIZE` images.

That leaves a mild wording ambiguity:

- are the click coordinates defined in original synthetic-camera coordinates
- or in the resized SAM v2 temp-frame coordinates

The actual implementation intent appears to be "click the center of the resized temp frame," not "click literal 2048-space coordinates against a 512px-ish inference image."

### Why this matters

An implementer reading only the prompt sentence could incorrectly pass original-size center coordinates into the resized-frame pipeline.

That would be easy to avoid with one clearer sentence.

### Recommended fix

Rewrite the prompt wording to say something like:

- "Click the center of the resized SAM v2 temp frame. Because the synthetic camera is aimed at the person, the person should remain centered after resizing."

Or, if you want to stay more explicit:

- compute `cx` and `cy` from the resized frame dimensions actually written to the temp directory

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md:133`
- `docs/specs/2026-04-03-masking-v1-plan-v4.md:135`
- `docs/specs/2026-04-03-masking-v1-plan-v4.md:365-368`

---

## Readiness Assessment

### Track A

`Track A` looks ready to implement.

Why:

- it is built on geometry and pipeline work that is already well understood
- it avoids introducing `sam2` as a hard dependency
- the two-pass architecture is now described clearly enough to support implementation
- the interim setup/UI changes are now careful about preserving current product behavior

This is the strongest and lowest-risk part of the plan.

### Track B

`Track B` is nearly ready.

Why:

- the packaging spike is now framed correctly
- the propagation direction issue is fixed
- the remaining gaps are now implementation-detail clarifications rather than architecture problems

The one important remaining clarification is mask reassembly by `out_frame_idx`.

### Overall

The plan is now in good shape.

Compared with the previous inspection round:

- the large structural uncertainties have been resolved
- the remaining issues are small and correctable
- the plan now has a practical path to ship value even if `sam2` is delayed or fails validation

My current assessment:

- `Track A`: ready
- `Track B`: almost ready, with one meaningful sequencing/indexing clarification still recommended
- overall plan quality: high

---

## Recommended Final Edits

If I were making one last pass on the v4 plan before implementation, I would make these two edits:

1. add an explicit note in Task B2 that masks must be stored by `out_frame_idx`, not append order
2. rewrite the click-prompt wording so it clearly refers to the resized SAM v2 temp frames rather than only the original synthetic-camera resolution

Those would close out the last real ambiguities I see.

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v4.md`
- `docs/2026-04-03-masking-v1-plan-v3-inspection-round2.md`
- `core/setup_checks.py`
- `panels/prep360_panel.py`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`
