# FullCircle and Plugin Masking Inspection

**Date:** 2026-04-03
**Project:** PanoSplat / LichtFeld 360 Plugin
**Scope:** Inspection of the current plugin, `docs/specs/2026-04-03-masking-v1-updated-plan.md`, and the local `D:/Data/fullcircle` repository

---

## Executive Summary

The updated masking plan is directionally strong and captures the main missing capability in the plugin: a second masking pass that aims a synthetic camera at the detected person and uses temporal tracking to improve mask coverage and consistency.

However, the plan currently blends together three different things:

1. The plugin's current implementation
2. FullCircle's actual checked-out repository state
3. A desired future architecture for the plugin

Those three are not fully aligned yet.

The biggest conclusion is:

- The plugin currently implements the first-pass view sweep and ERP OR-merge correctly enough to serve as a baseline.
- The synthetic fisheye direction/geometry work in the updated plan is well motivated and maps cleanly to FullCircle's math.
- The SAM v2 integration section of the updated plan is not a direct port of the local FullCircle repo. It is a proposed re-architecture that still needs validation on Windows, in the plugin environment, and inside LichtFeld Studio.

In other words, the geometry portion of the plan looks mature. The backend packaging and product integration portion still needs one more design pass before implementation begins.

---

## What Exists Today in the Plugin

The plugin already has a working single-pass masking pipeline:

- `core/masker.py`
  - Reframes ERP images into preset pinhole views
  - Runs per-view detection
  - Back-projects each view mask into ERP
  - OR-merges all detections
  - Postprocesses the merged ERP mask
  - Inverts to COLMAP polarity and writes `extracted/masks/*.png`

- `core/pipeline.py`
  - Calls the masker in Stage 2
  - Passes ERP masks into the reframer
  - Reprojects those ERP masks into per-view masks
  - Optionally computes overlap masks before COLMAP

- `core/backends.py`
  - Supports two image backends only:
    - YOLO + SAM v1
    - SAM 3
  - Does not yet define any video-tracking backend interface

- `core/setup_checks.py`
  - Models a two-tier world:
    - Default tier: YOLO + SAM v1
    - Premium tier: SAM 3
  - Has no SAM v2 readiness concept yet

- `panels/prep360_panel.py`
  - Exposes install/status UI for the existing two-tier setup
  - Reports only:
    - `Using YOLO + SAM v1`
    - `Using SAM 3`
    - `Not installed`

This means the plugin is not missing "masking" in general. It is missing the second-pass synthetic-camera stage and the system-level plumbing around it.

---

## What the Updated Plan Gets Right

The updated plan correctly identifies the gap between the current plugin and FullCircle's stronger masking pipeline:

- Pass 1 in the plugin is already close to FullCircle's initial detection stage.
- The missing piece is the synthetic camera stage:
  - determine where the person is on the sphere
  - aim a synthetic fisheye camera there
  - run a stronger second pass
  - project that result back into ERP
  - merge it with the first-pass mask

The updated plan is especially strong in these areas:

### 1. Separation of the two passes

The plan's breakdown into:

- primary detection
- synthetic rendering
- temporal tracking or fallback detection
- back-projection and merge

is the right mental model for implementation in this plugin.

### 2. Camera convention awareness

The plan explicitly recognizes that the plugin's existing reframer math and FullCircle's synthetic fisheye math use different conventions:

- Plugin reframer:
  - world-to-camera
  - camera forward is `-Z`

- FullCircle synthetic fisheye:
  - world-from-camera
  - camera forward is `+Z`

That is a real integration risk, and the plan is right to isolate the bridge in helper functions such as:

- `_pixel_com_to_3d_direction()`
- `_look_at_rotation()`
- `_direction_to_yaw_pitch()`

### 3. Use of pycolmap fisheye projection primitives

Using `pycolmap.Camera` with `OPENCV_FISHEYE` and relying on:

- `cam_from_img()`
- `img_from_cam()`

is a sensible and maintainable approach. It is closer to FullCircle's actual synthetic camera behavior than trying to approximate the same thing with custom pinhole math.

### 4. FullCircle-inspired weighting

The plan's idea to use mask area as the weight for direction averaging matches the logic in FullCircle's direction aggregation. That is likely to outperform a naive average of detected centers.

---

## Where the Updated Plan Does Not Match the Local FullCircle Repo

This is the most important finding.

The updated plan reads partly like a FullCircle port, but the local `D:/Data/fullcircle` repo does not contain a ready-made in-process SAM v2 backend that can simply be transplanted into the plugin.

### 1. FullCircle's masking pipeline is script-orchestrated

FullCircle's masking flow is driven by `scripts/run_masking.sh`:

1. omni to perspectives
2. mask perspectives with YOLO + SAM v1
3. perspective masks back to omni + directions
4. omni to synthetic fisheyes
5. synthetic fisheye masks via a SAM2 tool
6. synthetic masks back to omni
7. omni masks to raw fisheye masks
8. fisheye mask dilation

That is materially different from the plugin's in-process architecture.

### 2. FullCircle uses an external SAM2 tool path

The critical masking step in FullCircle is:

- `python thirdparty/sam-ui/scripts/tracking_gui.py --frames-path ... --output-path ... --headless`

That means FullCircle's current checked-out pipeline relies on an external tool workflow, not a neat internal backend abstraction.

### 3. The `sam-ui` submodule is not present locally in usable form

In the local checkout:

- `thirdparty/sam-ui` exists as a directory
- but it appears empty in this environment
- `tracking_gui.py` is not present in the checked-out files I could read

So even FullCircle's concrete step-5 implementation is not fully inspectable from the local repo state.

### 4. FullCircle installs SAM2 in a special way

FullCircle does not treat SAM2 like a normal pinned Python dependency. Its install script does:

- `pip install --no-deps "git+https://github.com/facebookresearch/sam2.git"`

with a comment explaining that this avoids upgrading torch.

That is a major warning sign for the plugin plan:

- `uv add sam2`
- adding `sam2` to `pyproject.toml`

may not be safe or portable in the plugin's Windows + embedded-Python environment without validation.

---

## Specific Mismatches Between Plan, Plugin, and Spec

### 1. The updated plan changes the meaning of the default tier

The updated plan says the default tier is:

- YOLO + SAM v1 + SAM v2

But the current design spec still says:

- default tier = YOLO + SAM v1
- premium tier = SAM 3
- temporal propagation is a non-goal for v1

This means the project currently has two conflicting definitions of "v1 masking."

### 2. The plugin UI and setup flow still implement the old tier model

The panel and setup checks still assume:

- default tier readiness means YOLO + SAM v1 are installed
- premium tier means SAM 3
- backend reporting only mentions those two choices

If the updated plan were implemented without changing the UI and setup model, users would see misleading status:

- the app could say masking is "installed"
- but not communicate whether synthetic tracking is available
- and not distinguish first-pass-only masking from the full intended pipeline

### 3. The updated plan says "all 116 tests pass"

In this repo, `rg -n "^def test_" tests` found 92 test functions.

A local `pytest -q` run in this environment reported:

- `103 passed`
- `13 errors`

The errors were environment-level permission problems around temp/cache directories, not obvious masking failures, but the headline in the plan is still stale or incorrect for this checkout.

### 4. The design spec is now behind the updated plan

The main design spec still emphasizes:

- cubemap decomposition
- overlap masks
- default YOLO + SAM v1
- optional SAM 3
- no temporal propagation in v1

The updated plan supersedes that in practice, but the spec has not yet been rewritten to match.

---

## Assessment of FullCircle Math Ports

The FullCircle math references in the updated plan are mostly good choices.

### `look_at_camZ()`

This looks like an appropriate port target.

Why it matters:

- It cleanly defines a `world_from_cam` rotation
- It aligns camera `+Z` with the target direction
- It handles near-pole instability by switching the up vector

That makes it a good fit for the synthetic fisheye pass.

### ERP to synthetic fisheye

The intended approach matches FullCircle's structure:

- generate fisheye rays
- rotate to world
- convert to ERP coordinates
- sample with `cv2.remap()`

This is a natural fit for the plugin and should be testable in isolation.

### Synthetic fisheye back to ERP

The intended reverse mapping also matches FullCircle conceptually:

- ERP pixel to world ray
- world ray into synthetic camera frame
- filter forward-facing rays
- project with `img_from_cam()`
- sample the mask

This is the correct inverse relationship to test.

### Direction aggregation

The weighted-direction idea is sound, but this is the helper most likely to fail due to convention mixups:

- horizontal flip in the plugin detection views
- optional vertical flip
- plugin `-Z` forward convention
- synthetic `+Z` forward convention

This is the single most important math test area before any model work begins.

---

## Assessment of the Current Plugin Architecture for This Work

The plugin architecture is actually a decent host for the planned refactor.

### What is already in good shape

- `Masker` already owns the right stage in the pipeline
- `process_frames()` already works frame-by-frame with ERP inputs
- the pipeline already has a clean masking stage boundary
- masks are already written in the right place for downstream reframing and COLMAP

### What will need refactoring

- `Masker` currently assumes a single backend and a single pass
- `core/backends.py` currently assumes image segmentation only
- `core/setup_checks.py` and the panel need a richer capability model

### What should not be changed casually

- Existing reframer math
- Existing ERP mask polarity and output conventions
- Existing overlap-mask flow

The updated plan is correct to keep the existing reframer conventions intact and confine FullCircle-style conventions to the new synthetic fisheye path.

---

## Risks

## 1. SAM2 packaging and API risk

This is the largest product risk.

Open questions still needing proof:

- Does the `sam2` package install reliably in the plugin's Windows environment?
- Does it work with the plugin's existing torch version?
- Does it work under embedded Python inside LichtFeld Studio?
- Does it expose a stable in-process video predictor API suitable for direct backend integration?
- Does it require filesystem-based sequence input, or can it run on in-memory frames?

Until those are verified, `Sam2VideoBackend` should be treated as a prototype target, not a guaranteed implementation detail.

## 2. VRAM pressure

The updated plan correctly identifies this.

Potential issue:

- YOLO + SAM v1 + SAM2 + SAM3 assumptions can easily exceed available VRAM on common creator GPUs

The cleanup/init ordering in the plan is likely necessary, not optional.

## 3. Convention mismatch in direction math

If `_pixel_com_to_3d_direction()` is even slightly wrong:

- the synthetic camera will aim off-target
- the second pass will appear to "work" but quietly degrade quality
- debugging will be painful because masks may still look plausible

This helper needs strong unit tests before sequence code is added.

## 4. Plugin setup/UI drift

If backend readiness and UI language are not updated together:

- users will get ambiguous install states
- bug reports will become harder to interpret
- support burden will increase

## 5. Overcommitting v1 scope

The updated plan is now significantly bigger than the original v1:

- synthetic camera math
- direction estimation
- temporal fallback
- video backend abstraction
- SAM2 integration
- SAM3 video support
- setup and UI changes
- spec rewrite

That is no longer a "small masking enhancement." It is a substantial feature set.

---

## Recommended Plan Adjustments

The updated plan should be revised before implementation begins.

### Recommendation 1: split geometry from backend integration

Treat these as separate tracks:

Track A: Geometry and pipeline refactor

- synthetic camera helpers
- direction helpers
- two-pass `Masker` structure
- fallback second pass using the existing image backend
- tests for rendering/back-projection/direction correctness

Track B: SAM2 integration

- validate packaging
- validate predictor API
- validate Windows/LFS runtime behavior
- only then add optional SAM2 backend support

This reduces the risk of blocking the whole masking upgrade on one uncertain dependency.

### Recommendation 2: do not add `sam2` to hard dependencies yet

Do not move `sam2` straight into `pyproject.toml` as a required dependency until:

- installation is proven stable
- the versioning story is understood
- compatibility with existing torch is confirmed

Prefer:

- optional install path first
- setup checks for presence
- backend activation only when available

### Recommendation 3: update UI and setup tasks explicitly in the plan

The plan currently focuses on core code, but product behavior needs matching updates:

- setup checks should distinguish:
  - pass-1 ready
  - synthetic fallback ready
  - full SAM2 video-ready
- panel text should reflect actual active capability
- install UX should not imply that the full FullCircle pipeline exists when only pass 1 is installed

### Recommendation 4: revise the spec before or alongside implementation

The spec and updated plan now disagree on key product behavior.

At minimum, the spec should be revised to clarify:

- whether temporal tracking is in v1 or not
- what "default tier" means
- whether SAM2 is optional enhancement or part of baseline masking

### Recommendation 5: define a minimal successful milestone

A strong minimal milestone would be:

- two-pass masker structure implemented
- synthetic fisheye camera math implemented and tested
- pass 2 works with fallback per-frame image detection
- no SAM2 dependency required

That would already deliver meaningful quality improvement while de-risking the future SAM2 work.

---

## Suggested Revised Milestone Sequence

### Milestone 1: Math and refactor

- Implement synthetic fisheye projection helpers
- Implement direction helpers
- Refactor `Masker` into pass 1 / pass 2 / save stages
- Use fallback image backend for pass 2
- Add unit and integration tests

### Milestone 2: Capability modeling

- Extend setup checks with richer masking capability states
- Update panel text and install status
- Keep SAM2 optional

### Milestone 3: SAM2 research spike

- Verify install method
- Verify predictor API
- Verify checkpoint handling
- Verify runtime in plugin and LFS

### Milestone 4: Optional SAM2 backend

- Add real video backend
- Add cleanup/VRAM handling
- Add mocked tests and limited real-world validation

### Milestone 5: Premium video path

- Investigate SAM 3 video support only after the SAM2 path is stable

---

## Practical Conclusions

If the goal is to improve masking quality soon, the best immediate path is:

- implement the synthetic camera and second-pass structure now
- use the current image backend as the pass-2 fallback
- postpone true video-tracking integration until SAM2 is validated in this environment

That path preserves momentum and keeps the plan grounded in what is already proven locally.

If the goal is to faithfully mirror FullCircle, then the plan should explicitly say that the plugin is adapting FullCircle's geometry and staging concepts, not directly porting a self-contained SAM2 backend from the repo. The checked-out FullCircle repo does not provide that backend in a portable, in-process form.

---

## Inspection Inputs

This document was based on inspection of:

- `docs/specs/2026-04-03-masking-v1-updated-plan.md`
- `docs/specs/2026-04-02-masking-layer-v1-design.md`
- `core/masker.py`
- `core/backends.py`
- `core/pipeline.py`
- `core/setup_checks.py`
- `core/overlap_mask.py`
- `panels/prep360_panel.py`
- `panels/prep360_panel.rml`
- `pyproject.toml`
- `tests/`
- `D:/Data/fullcircle/masking/`
- `D:/Data/fullcircle/scripts/run_masking.sh`
- `D:/Data/fullcircle/install_env.sh`
- `D:/Data/fullcircle/CLAUDE.md`

---

## Final Recommendation

Proceed with the updated plan only after revising it in these ways:

- frame SAM2 as an unproven optional backend, not a guaranteed implementation detail
- add explicit setup/UI/spec updates
- split math refactor from model-integration work
- define a fallback-only milestone that delivers value even if SAM2 slips

That will make the plan more accurate, lower-risk, and much easier to execute cleanly inside the plugin codebase.
