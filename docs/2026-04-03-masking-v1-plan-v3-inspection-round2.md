# Masking Plan v3 Inspection — Round 2

**Date:** 2026-04-03
**Subject:** Follow-up review of `docs/specs/2026-04-03-masking-v1-plan-v3.md`
**Context:** Second inspection after the first v3 review in `docs/2026-04-03-masking-v1-plan-v3-inspection.md`

---

## Executive Summary

The updated v3 plan is now in good shape overall.

It successfully addressed the main issues from the prior inspection:

- the SAM v2 resize/back-projection geometry is now explicit
- the installation path for SAM v2 is now correctly framed as something validated by the spike
- Track A now includes minimum product-facing updates if it ships before Track B
- the Windows verification command has been cleaned up

At this point, the plan looks ready for Track A implementation.

There are only a few remaining concerns:

1. the SAM v2 tracking flow still appears to propagate forward only from the chosen prompt frame
2. the temporary Track A capability-level work could accidentally regress the plugin's existing SAM 3 reporting if implemented too narrowly
3. one validation task still phrases the install test a little more narrowly than the later installation strategy

These are much smaller issues than before. The plan is now close to implementation-ready.

---

## What Improved Since the Last Inspection

## 1. The geometry ambiguity is resolved

The previous review identified a major risk around the SAM v2 frame-resize path:

- synthetic fisheye views are defined at `SYNTHETIC_SIZE`
- SAM v2 works on resized frames
- the plan had not yet explained how masks were mapped back to the original camera geometry before ERP back-projection

That issue is now explicitly addressed.

The updated plan now states that:

- SAM v2 masks are produced at reduced resolution
- those masks are resized back to the original synthetic camera resolution
- nearest-neighbor interpolation is used
- the back-projection math always uses the original synthetic camera model

That is the correct resolution of the geometry issue and makes the intended implementation much clearer.

## 2. The install strategy is better aligned with the validation spike

The earlier concern was that Track B still effectively assumed a single install method before B1 had validated it.

The updated plan now improves that by saying:

- `install_video_tracking()` will use the validated method from B1
- that method might be `uv add sam2`
- or it might be a different method if that turns out to be what works in the plugin environment

That is a much healthier framing for a dependency that still needs real-world validation.

## 3. Track A shipping now has at least minimal product-facing follow-through

The previous review pointed out that Track A could not really be considered shippable if:

- the UI still described the old world
- setup checks could not express the new fallback synthetic-pass capability
- the spec remained fully out of sync

The new Task A7 addresses that in a practical way:

- it adds minimum setup capability plumbing
- it updates panel text at a basic level
- it adds an interim spec note without forcing the full rewrite too early

That is the right compromise.

## 4. The verification section is now better suited to the real environment

The earlier Unix-only cleanup command was a small but real execution detail problem.

The updated plan now provides:

- a Git Bash variant
- a PowerShell variant

That is a simple but useful improvement.

---

## Remaining Findings

## 1. SAM v2 propagation still appears to cover only the forward direction

This is now the main remaining technical concern.

The plan says:

1. choose the frame with the strongest primary detection
2. add a prompt on that frame
3. propagate forward from that frame

That is not sufficient if the strongest primary detection is not at frame 0.

### Why this matters

If the prompt frame is in the middle of the sequence:

- frames after it get tracked masks
- frames before it do not

That would create an asymmetric result across the sequence and undercut the main benefit of video tracking.

### Why this is especially notable

The inspected `sam-ui` helper already supports reverse traversal behavior. That means the tooling appears capable of handling this better than the current plan describes.

### Recommended fix

The plan should explicitly choose one of these options:

- propagate both forward and backward from `best_frame_idx`
- or always prompt on frame 0 if a forward-only implementation is intended

The first option is clearly better if the API behavior remains as currently inspected.

---

## 2. Track A's temporary capability model should not regress existing SAM 3 reporting

The new Task A7 is a good addition, but its wording is still slightly risky.

It currently proposes:

- adding `capability_level` in Track A for levels 0 and 1 only
- updating panel text to say "Masking ready" when `capability_level >= 1`

The issue is that the current plugin already has an existing premium SAM 3 concept and reporting path.

### Why this matters

If Track A introduces a temporary capability model that only thinks in terms of levels 0 and 1, then a setup that already qualifies for SAM 3 could end up with less accurate reporting during the intermediate state.

That would be a regression, even if temporary.

### Recommended fix

Task A7 should explicitly preserve current SAM 3 behavior while adding the minimum Track A fallback-synthetic-pass language.

A good rule would be:

- do not collapse or hide any already-existing SAM 3 state in Track A
- only generalize the baseline status text for the non-SAM 3 path

That keeps the interim plumbing honest without rewriting all of Track B early.

---

## 3. B1.1 is still slightly narrower than the rest of the install strategy

This is a minor issue now, not a major one.

The later installation design correctly says the actual install method should come from the validation spike. But B1.1 still specifically says to run `uv add sam2`.

### Why this matters

That wording subtly frames the spike as:

- "verify that `uv add sam2` works"

instead of:

- "determine which installation method works"

Those are not quite the same question.

### Recommended fix

Widen B1.1 slightly. For example:

- "Test SAM v2 installation in the plugin venv, starting with `uv add sam2`, and document whether another method is required."

That wording preserves the current likely path without prejudging the answer.

---

## Readiness Assessment

## Track A

Track A now looks ready to implement.

Why:

- it stays inside proven plugin architecture
- it does not depend on SAM v2 packaging success
- the geometry plan is now explicit enough
- the product-facing minimum updates are now accounted for

This is the strongest part of the plan.

## Track B

Track B is close, but not fully locked down yet.

Why:

- the install strategy is now much better framed
- the geometry mismatch is fixed
- but the sequence-wide propagation behavior should be clarified before implementation begins

That means Track B is in good planning shape, but still benefits from one more small refinement.

---

## Recommended Final Edits

If I were making a final polish pass on this plan before implementation, I would make these edits:

### Edit 1

Update the SAM v2 flow to explicitly propagate both forward and backward from the chosen prompt frame.

### Edit 2

Clarify in Task A7 that interim capability changes must preserve existing SAM 3 reporting behavior.

### Edit 3

Widen B1.1 slightly so it validates the working install method rather than implicitly assuming only one.

---

## Final Assessment

This revision is strong.

Compared with the previous review:

- the major unresolved issues have been addressed
- the remaining issues are now tactical rather than structural
- the plan has a clear path to delivering value even if SAM v2 slips

My overall judgment:

- **Track A:** ready
- **Track B:** nearly ready, with one important propagation clarification
- **Overall plan quality:** high

This now reads like a practical implementation plan rather than a speculative architecture sketch.

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v3.md`
- `docs/2026-04-03-masking-v1-plan-v3-inspection.md`
- `core/setup_checks.py`
- `panels/prep360_panel.py`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`

