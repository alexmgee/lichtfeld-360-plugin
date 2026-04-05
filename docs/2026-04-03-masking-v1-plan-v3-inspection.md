# Masking Plan v3 Inspection

**Date:** 2026-04-03
**Subject:** Review of `docs/specs/2026-04-03-masking-v1-plan-v3.md`
**Context:** Follow-up inspection after the original review in `docs/2026-04-03-fullcircle-plugin-inspection.md`

---

## Executive Summary

Plan v3 is a substantial improvement over the previous plan.

It fixes the biggest structural problems identified in the earlier inspection:

- it separates the geometry/pipeline refactor from SAM v2 integration
- it no longer treats SAM v2 as a required hard dependency for the basic masking upgrade
- it explicitly adds setup/UI capability modeling
- it correctly reframes Track A as independently valuable and potentially shippable

The plan now feels much closer to an executable engineering plan than a speculative port.

That said, it still has a few important issues:

1. the SAM v2 frame-resize flow does not yet reconcile mask resolution with synthetic-camera back-projection geometry
2. the installation path in Track B is still too prescriptive before the validation spike proves the working method
3. the plan calls Track A shippable, but delays UI/spec reconciliation too long for that to be fully true
4. a couple of stale execution details remain from the prior version

Overall assessment:

- **Architecture:** strong
- **Geometry plan:** strong
- **Packaging/install plan:** improved, but still needs one more tightening pass
- **Readiness to implement Track A:** high
- **Readiness to implement Track B:** medium, pending clarification of the SAM v2 sizing/install details

---

## What Plan v3 Improved

## 1. It correctly splits proven work from risky work

This is the biggest improvement.

Track A isolates:

- synthetic fisheye projection math
- direction estimation
- two-pass masker refactor
- fallback second-pass detection using existing image backends

Track B isolates:

- SAM v2 environment validation
- SAM v2 backend implementation
- setup/install updates for video tracking
- UI changes for capability reporting

That separation is exactly what the previous inspection recommended. It reduces project risk and creates a clean stopping point if SAM v2 packaging or runtime integration proves difficult.

## 2. SAM v2 is no longer treated as a baseline requirement

Plan v3 now makes SAM v2 optional in the default tier:

- YOLO + SAM v1 remain the baseline requirement
- SAM v2 is framed as an optional enhancement for video tracking
- fallback image-based second-pass detection remains valid if SAM v2 is absent

This is a much more realistic product stance for the plugin.

It also aligns better with the fact that:

- the plugin already has usable image backends
- the geometry work has value even without temporal propagation
- SAM v2 installation/runtime behavior still needs proof in the real environment

## 3. The capability-level model is much better than the old two-tier language

The new capability table is a strong improvement:

- Level 0: nothing installed
- Level 1: masking ready with fallback synthetic pass
- Level 2: masking ready with SAM v2 video tracking
- Level 3: SAM 3

That matches the likely real-world user experience much better than a binary "default tier vs premium tier" framing.

This is especially important because it lets the UI communicate the difference between:

- basic masking support
- improved synthetic second-pass support
- full temporal propagation support

## 4. The SAM v2 API section is much more grounded

Plan v3 improves credibility by anchoring the SAM v2 design to inspected code in the checked-out `sam-ui` submodule rather than treating FullCircle as a black box.

That matters because it changes the conversation from:

- "maybe there is a usable API"

to:

- "the API exists and we have specific constraints to design around"

The key verified constraint is especially useful:

- `init_state()` consumes a directory of numbered frame images on disk

That is exactly the kind of practical implementation detail that should drive backend design.

## 5. Track A is now a real product milestone

This is a strong strategic improvement.

By the end of Track A, the plugin can already deliver:

- a two-pass masking structure
- synthetic camera targeting
- better framing of the detected person in pass 2
- improved masks without any new dependency risk

That is a real shipping story, not just a stepping stone.

---

## Remaining Issues

## 1. Synthetic-camera geometry and SAM v2 resize flow are not fully reconciled

This is the most important remaining plan issue.

The plan defines the synthetic camera at `2048x2048` with equidistant fisheye geometry. That geometry is then used for:

- rendering synthetic views
- projecting synthetic masks back into ERP

But the SAM v2 flow says synthetic frames are resized to 512px minimum dimension and written as numbered JPEGs before tracking.

That creates a geometry question the plan does not yet answer clearly:

- are masks produced by SAM v2 in resized-image coordinates?
- if yes, how are they mapped back to the original synthetic fisheye geometry?
- does back-projection use:
  - the original 2048 camera,
  - a resized temporary camera,
  - or an explicit resample step back into `synthetic_size`?

Without an explicit answer, there is a real risk that the second-pass masks will be geometrically misregistered during ERP back-projection.

### Why this matters

The synthetic camera is not just an image size choice. It is part of the projection model.

If you resize frames for SAM v2 but then back-project masks as if they were still on the original camera grid, the reprojection will be wrong.

### Recommended fix

Add one explicit rule to the plan:

- either always resize predicted masks back to `synthetic_size` before ERP back-projection

or

- define a resized synthetic camera model that matches the SAM v2 working resolution and use that camera consistently for both projection and inverse projection in the tracking path

The plan should not leave this implicit.

---

## 2. The install step in Track B still hardcodes a solution before validation fully proves it

Track B is much better than before, but it still says:

- validate installation in B1
- then install via `uv add sam2` in B3

That is still a bit too rigid.

The purpose of B1 is to determine the correct installation method in the plugin environment. If B1 finds that the working solution is not plain `uv add sam2`, the later task wording should not have to be corrected after the fact.

### Recommended fix

Change the wording from:

- "install_video_tracking() installs `sam2` via `uv add sam2`"

to:

- "install_video_tracking() uses the validated SAM v2 installation method determined in B1"

Then, if the answer is `uv add sam2`, great. If not, the plan remains accurate.

---

## 3. Track A is described as shippable, but the plan delays some product-facing reconciliation too long

Plan v3 correctly says Track A is shippable without SAM v2.

But if Track A truly ships, then the following also need to be true at or near the Track A milestone:

- the UI must accurately describe what the user now has
- setup checks must distinguish the fallback synthetic pass from true video tracking
- the spec should no longer describe a substantially different masking architecture

Right now, the plan places:

- setup/UI capability work in Track B
- spec rewrite after both tracks

That makes sense if Track A is purely internal.
It makes less sense if Track A is genuinely intended to ship.

### Recommended fix

Move the minimum required product-facing updates closer to Track A:

- Track A should update internal capability plumbing enough that the plugin can truthfully report "masking ready" with fallback synthetic pass
- the full polished UI progression can still remain in Track B
- the spec rewrite can stay later, but a short interim note or status addendum should exist if Track A ships first

---

## 4. A couple of stale details remain from the previous revision

The plan still contains a few details that should be cleaned up:

### Stale test count language

The plan still references "116+ existing tests must still pass."

That wording appears to be inherited from the earlier plan state and no longer matches the repo state inspected in this environment.

Even if the exact count is not critical, stale numbers reduce confidence in the document.

### Unix cleanup command in a Windows-oriented workflow

The verification section still uses `rm -rf` for pycache cleanup.

This project is clearly being worked on in a Windows environment with PowerShell, embedded Python, and LichtFeld Studio. The command should either:

- be written in a cross-platform-neutral way
- or include the PowerShell equivalent

This is minor, but easy to fix.

---

## 5. The SAM 3 video backend remains lightly specified compared to SAM v2

This is acceptable for now, because it is correctly positioned after the SAM v2 path. But it is still worth noting:

- SAM v2 now has a concrete inspected API reference
- SAM 3 video support is still largely aspirational in the plan

That is not a flaw as long as it stays clearly dependent on the SAM v2 track and is not treated as near-term guaranteed work.

Plan v3 mostly handles this well.

---

## What Looks Ready to Implement

## Ready now

These parts look ready for implementation planning or direct execution:

- Track A fisheye projection functions
- Track A direction helper functions
- Track A `VideoTrackingBackend` protocol and fallback backend
- Track A two-pass `Masker` refactor
- Track A integration tests
- `core/__init__.py` export updates

These are grounded in:

- existing plugin structure
- existing FullCircle math references
- no new dependency requirements

## Ready after one wording pass

These are almost ready, but should be clarified first:

- Track B SAM v2 install method
- Track B frame resize / mask reprojection geometry
- Track A shipping implications for setup/UI/spec language

## Not yet implementation-ready

These still belong in research-first territory:

- exact SAM 3 video predictor integration details
- any assumptions about long-term dependency stability for `sam2` in the embedded plugin runtime

---

## Recommended Edits Before Work Starts

If this plan were being finalized today, I would make these edits before implementation:

### Edit 1

Add a short subsection under Track B explaining how SAM v2 mask resolution maps back to synthetic-camera geometry.

### Edit 2

Replace any hardcoded SAM v2 installation method in later tasks with:

- "the validated install method from B1"

### Edit 3

If Track A is genuinely shippable, add one small Track A product task:

- update setup status text or capability plumbing so the plugin can accurately report fallback synthetic-pass support even before SAM v2 exists

### Edit 4

Remove or generalize stale test-count wording.

### Edit 5

Replace the Unix-only cleanup command in verification with a Windows-safe version or a neutral instruction.

---

## Final Assessment

Plan v3 looks good.

It addresses the most important problems from the previous review and now has a clear engineering shape:

- safe work first
- risky dependency work later
- useful product value even if the risky work slips

That is the right structure for this plugin.

My overall judgment is:

- **much better than v2**
- **good enough to begin Track A after a small cleanup pass**
- **Track B should wait for two clarifications: install method and resize/back-projection geometry**

If those remaining details are tightened up, this becomes a strong implementation plan.

---

## Inputs

This inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v3.md`
- `docs/2026-04-03-fullcircle-plugin-inspection.md`
- `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py`
- current plugin environment checks for Python and torch

