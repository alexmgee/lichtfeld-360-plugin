# Masking Plan v8 Final Inspection

**Date:** 2026-04-03
**Subject:** Final review of `docs/specs/2026-04-03-masking-v1-plan-v8.md`
**Context:** Final inspection after the iterative plan review sequence from the original FullCircle/plugin inspection through the v7 inspection

---

## Executive Summary

The v8 plan is implementation-ready from a planning/documentation standpoint.

I do not see any remaining blocking architectural gaps in the document itself.

The plan now:

- matches the plugin's current masking stage shape much more closely
- adapts the relevant FullCircle masking steps into the plugin's ERP-first pipeline
- correctly treats SAM v2 as validation-gated rather than assumed
- keeps a shippable fallback path if SAM v2 fails in the actual plugin/LichtFeld environment

At this point, the main remaining risks are execution risks, not plan-design risks.

That said, there are still a few areas that are not blockers but are worth calling out as under-explained or easy to overlook during implementation.

---

## Final Status

### Plan Status

`docs/specs/2026-04-03-masking-v1-plan-v8.md` looks ready to execute.

### Track Status

- `Track A`: ready to start
- `Track B`: plan-ready, but still correctly gated by the B1 SAM v2 validation spike

### Main Remaining Risk

The main remaining uncertainty is not whether the plan makes sense.

It is whether `sam2`:

- installs cleanly in the plugin venv
- imports cleanly in the embedded LichtFeld runtime
- downloads the required model successfully
- runs reliably enough under the plugin's lifecycle and VRAM constraints

That is exactly the right kind of risk for B1 to own.

---

## What Looks Complete

### 1. The core FullCircle-inspired masking shape is present

The plan now covers the main ideas that matter from FullCircle's masking workflow:

- coarse first-pass person localization from ERP-derived views
- conversion of that localization into a synthetic re-centered fisheye view
- second-pass segmentation/tracking on the re-centered synthetic view
- back-projection of the synthetic mask back into ERP space
- merge with the primary mask result

That is the key structural adaptation needed for the plugin.

### 2. The plan fits the plugin's actual architecture

The plugin already:

- operates on ERP frames first
- writes ERP masks before reframing
- later reprojects those ERP masks into the pinhole outputs used for COLMAP

That means the plan is not trying to force the FullCircle data layout onto the plugin unchanged.

It is adapting the FullCircle masking logic into the plugin's existing stage order, which is the right approach.

Relevant code references:

- `core/pipeline.py:307-320`
- `core/pipeline.py:348-358`

### 3. Track A is genuinely viable inside the current plugin

Track A is no longer speculative:

- `pycolmap` is already a declared plugin dependency
- the current pipeline already has a masking stage and progress range where the two-pass refactor can live
- the plan keeps the existing image backend as the fallback path

Relevant code references:

- `pyproject.toml:9-17`
- `pyproject.toml:24-27`
- `core/pipeline.py:307-325`

### 4. The plan no longer depends on pretending FullCircle can be ported 1:1

This is one of the biggest improvements over the earlier drafts.

The plan now correctly treats these as separate concerns:

- the FullCircle masking logic that can be adapted directly
- the FullCircle-specific data layout and orchestration details that do not carry over unchanged
- the SAM v2 packaging/runtime problem that still has to be proven in the plugin environment

That is the right mental model.

---

## Intentional Differences From FullCircle That Are Not Actually Missing

These are worth stating explicitly because they can look like omissions if someone only compares step counts.

### 1. The plugin does not need FullCircle's final dual-fisheye conversion steps

FullCircle's shell pipeline ends with:

- synthetic masks back to omnidirectional masks
- omnidirectional masks back to raw fisheye masks
- dilation on raw fisheye masks

That is appropriate for FullCircle's data layout because it ultimately trains from fisheye inputs.

The plugin does not use that same final representation.

The plugin currently:

- writes ERP masks in the masking stage
- then reprojects those masks to pinhole outputs during reframing

So the absence of FullCircle's `omni2fisheye.py` and `dilate.py` steps from the plugin plan is not a missing feature. It is an intentional architecture difference.

Relevant references:

- `D:/Data/fullcircle/scripts/run_masking.sh`
- `core/pipeline.py:307-320`
- `core/pipeline.py:348-358`

### 2. The plugin does not need to shell out to `tracking_gui.py`

FullCircle uses `thirdparty/sam-ui/scripts/tracking_gui.py --headless` as the orchestration step.

The plugin plan instead imports the underlying SAM2 functionality into a backend.

That is a reasonable adaptation for the plugin and better matches the current architecture, as long as B1 proves the packaging/runtime side works.

This is an intentional integration improvement, not a missing step.

---

## Non-Blocking Watch Items

These are the main areas I would still call under-explained or easy to overlook.

None of them are severe enough to block implementation, but they are worth making explicit before or during coding.

### 1. The primary masking view layout is coupled to the reconstruction preset

This is the most important non-blocking watch item.

FullCircle's primary pass uses a dedicated masking view layout:

- 16 virtual cameras
- 8 yaw positions
- 2 pitch bands
- 90° FOV

The current plugin, by contrast, derives masking views from the active runtime preset used for reconstruction:

- `cubemap`: 6 views
- `low`: 9 views
- `medium`: 14 views
- `high`: 18 views

Those presets were designed around reconstruction/reframing needs, not specifically around coarse person localization.

This does not mean the plan is wrong.

It means the plan is making an implicit product choice:

- masking quality in Pass 1 will depend on the active preset geometry

That may be acceptable, especially for a first shipping version.

But it is under-explained, and it may be one of the first places where the plugin diverges in behavior from FullCircle's results.

Relevant references:

- `D:/Data/fullcircle/masking/omni2perspective.py:47-63`
- `core/pipeline.py:107-120`
- `core/pipeline.py:311-318`
- `core/presets.py:193-260`

### Recommendation

Add one sentence to the plan or implementation notes that makes the choice explicit:

- either the plugin intentionally uses the active reconstruction preset for primary masking in v1
- or a dedicated masking-only view layout may be introduced later if localization quality is not strong enough

If you want the highest likelihood of matching FullCircle's coarse localization behavior, a future hidden `masking_primary` preset would be the most direct follow-up.

### 2. The all-empty / no-direction sequence behavior should be stated explicitly

The plan already covers temporal fallback for gaps between valid detections.

That part is good.

What is still a little under-explained is the fully empty case:

- no usable primary detections anywhere in the clip
- no valid direction to aim the synthetic camera at

FullCircle's `omni2synthetic.py` makes this behavior explicit:

- if a frame has no center, it looks for a nearby frame
- if no usable center exists anywhere, it skips that frame

The plugin plan implies a similar behavior, but it never quite states the final rule in plain language.

Relevant references:

- `D:/Data/fullcircle/masking/omni2synthetic.py:62-89`
- `docs/specs/2026-04-03-masking-v1-plan-v8.md:38`
- `docs/specs/2026-04-03-masking-v1-plan-v8.md:211`
- `docs/specs/2026-04-03-masking-v1-plan-v8.md:278`

### Recommendation

State this explicitly in the plan or implementation notes:

- if some frames have directions and others do not, borrow the nearest valid direction
- if the entire clip has no valid direction, skip the synthetic pass entirely and preserve the primary ERP masks as the final result

That is probably already the intended behavior, but it is worth spelling out because it defines failure handling on difficult clips.

### 3. Tempdir and cleanup behavior should be exception-safe, not just success-path clean

The plan correctly says the SAM v2 backend writes numbered JPEGs to a temp directory and cleans that directory up afterward.

That is good.

What is slightly under-explained is that this cleanup should happen under failure as well as success.

The current plugin already follows a strong cleanup pattern in the pipeline:

- initialize backend
- run stage
- always cleanup in `finally`

That same discipline should carry into the new video backend and synthetic-pass logic.

Relevant references:

- `docs/specs/2026-04-03-masking-v1-plan-v8.md:131`
- `docs/specs/2026-04-03-masking-v1-plan-v8.md:365-372`
- `core/pipeline.py:322-341`
- `core/masker.py:239-250`

### Recommendation

Use:

- `tempfile.TemporaryDirectory()` or an equivalent `try/finally` pattern
- backend cleanup in `finally`
- explicit VRAM cleanup in the same code path used for early errors

This is not a missing concept in the plan, but it is easy to underspecify and then regret later.

### 4. The role of `initial_mask` in the video backend protocol is still a little under-explained

The protocol currently includes:

```python
initial_mask: np.ndarray | None = None
```

But the described SAM v2 flow uses:

- the selected prompt frame
- a center-point click
- no initial mask

That is not a problem, but it leaves one mild question open:

- is `initial_mask` intentionally reserved for future backends or future prompt styles
- or is it expected to be used by some fallback or SAM 3 flow later

Relevant reference:

- `docs/specs/2026-04-03-masking-v1-plan-v8.md:232-237`

### Recommendation

Clarify one of these:

- "`initial_mask` is reserved for future backends and may be ignored by Sam2VideoBackend"
- or remove it from the protocol until it is actually needed

This is a low-severity documentation clarity point, not a design flaw.

---

## Practical Execution Notes

If implementation starts from v8, these are the main practical points I would want the implementer to hold onto:

- Track A is not a speculative bridge. It should be implemented as a real, shippable fallback path.
- The plugin's ERP-first architecture means FullCircle's final fisheye-output steps are intentionally not part of the integration.
- The strongest likely quality delta versus FullCircle is the primary-pass view layout, not the synthetic fisheye math.
- The strongest likely runtime delta versus the written plan is still B1 packaging/runtime behavior.
- Difficult clips with no usable detections should degrade cleanly to "primary masks only," not fail the whole pipeline if that can be avoided.

---

## Final Assessment

This is the first version of the plan that I would call both:

- structurally complete
- well adapted to the plugin as it currently exists

I do not see any additional missing major FullCircle masking steps that must be ported for the plugin architecture to work.

The most important remaining work is now:

1. implement Track A carefully
2. run B1 in the real plugin/LichtFeld environment
3. measure whether the preset-coupled primary masking pass is good enough in practice

If that primary-pass quality is acceptable and B1 succeeds, the integration path described by v8 should be viable inside the plugin as it exists today.

---

## Inputs

This final inspection was based on:

- `docs/specs/2026-04-03-masking-v1-plan-v8.md`
- `docs/2026-04-03-masking-v1-plan-v7-inspection.md`
- `D:/Data/fullcircle/scripts/run_masking.sh`
- `D:/Data/fullcircle/CLAUDE.md`
- `D:/Data/fullcircle/docs/index.html`
- `D:/Data/fullcircle/masking/omni2perspective.py`
- `D:/Data/fullcircle/masking/omni2synthetic.py`
- `core/pipeline.py`
- `core/presets.py`
- `core/masker.py`
- `pyproject.toml`
