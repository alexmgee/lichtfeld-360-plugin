# Masking Testing Report Response And FullCircle Gap Analysis

> Date: 2026-04-04
> Inputs reviewed:
> - `docs/2026-04-04-masking-testing-report.md`
> - latest user-supplied `medium_test` console output
> - current plugin implementation in `core/masker.py`, `core/backends.py`, and `core/presets.py`
> - FullCircle masking pipeline in `D:/Data/fullcircle/masking/*` and `D:/Data/fullcircle/scripts/run_masking.sh`

## Executive Summary

The testing report identified a real problem, but the latest 16-camera run changes the diagnosis in an important way.

The earlier report said the synthetic pass was failing because direction was unstable and SAM v2 only produced masks on 2-3 frames. That was true for the earlier low-preset runs.

The newest run, however, shows this:

- Pass 1 found a person direction on `11/11` frames
- Pass 2 rendered `11` synthetic fisheye frames
- `Sam2VideoBackend` returned masks on `11/11` frames
- the pipeline completed successfully

That means the biggest active gap is no longer "SAM v2 cannot track." The bigger gap is now:

- how reliable and clean Pass 1 localization is
- how Pass 2 is merged back into the final mask
- whether the plugin is still preserving Pass 1 mistakes even after Pass 2 succeeds

The short answer to "what are we doing wrong that FullCircle does right?" is:

1. FullCircle uses a dedicated masking camera layout for localization.
2. FullCircle treats the synthetic/SAM2 pass as the authoritative final mask geometry.
3. FullCircle records direction evidence in a slightly more robust way than the plugin currently does.
4. FullCircle's postprocessing happens after the synthetic pass, in the raw-camera domain, not as an ERP rescue for noisy primary masks.

The most important practical conclusion is:

- the testing report is directionally correct that Pass 1 layout is the main quality problem
- but now that SAM v2 is tracking `11/11` frames, the plugin's unconditional Pass 1 + Pass 2 OR-merge is probably the next major thing holding quality back

## What The Latest Run Changes

The latest `medium_test` run materially changes the state of the investigation.

Compared to the report's earlier low-preset runs:

- direction coverage is now complete (`11/11` frames with person direction)
- SAM v2 coverage is now complete (`11/11` frames tracked)
- the synthetic pass is no longer merely a theoretical improvement path

This means some conclusions in `docs/2026-04-04-masking-testing-report.md` should now be read as historical findings from an earlier code/preset state rather than as the current full diagnosis.

In particular:

- "SAM v2 tracking yield is very low" is no longer true in the latest medium run
- "the synthetic pass adds almost no value" is no longer safe to assert generally

What still appears true:

- Pass 1 localization quality matters a lot
- view layout strongly affects the quality of direction estimation
- false positives in Pass 1 can still contaminate the final result if the merge policy preserves them

## Response To The Testing Report

## 1. What the report gets right

The report is strong on the following points:

### A. ERP-level morph close was the wrong cleanup stage

This was a real issue. The current plugin code now reflects that lesson:

- ERP morph-close has been removed from the save path
- the code comments explicitly note that ERP-level closing bridges false positives across the sphere

So this part of the report is correct and already acted on.

### B. Preset-coupled Pass 1 views are a real weakness

This is the most important structural point in the report, and it aligns with both:

- the final plan's "intentional v1 choice" language
- the FullCircle repo's actual masking layout

The report is right that the plugin is still asking a reconstruction-oriented view layout to do detector-localizer work.

### C. False positives in Pass 1 can poison direction estimation

This is also correct.

Even if tracking works later, the synthetic camera is only as good as the direction fed into it. If Pass 1 combines too much weak evidence or the wrong evidence, the synthetic pass starts from a compromised pose estimate.

## 2. What the report now overstates or leaves under-explained

### A. "SAM v2 tracking yield is very low" is now outdated

That statement fit runs 3 and 4, but it is contradicted by the newest medium run, where `Sam2VideoBackend` returned masks on all `11/11` frames.

That changes the priority stack:

- earlier priority: "make SAM2 work at all"
- current priority: "make the masks cleaner and stop preserving Pass 1 mistakes"

### B. "More views doesn't help" is too broad

The report's run 5 conclusion says more views do not help. That is too broad as written.

What the evidence really supports is:

- more reconstruction-optimized views do not automatically solve detector-localization problems

That is different from saying:

- more masking-optimized views do not help

Those are not the same claim.

In fact, FullCircle's fixed 16-camera masking layout suggests the opposite:

- the right 16 views help a lot
- the wrong 16 views mostly just give you more chances to accumulate false positives

### C. The "replace instead of OR-merge" fix needs gating

The report proposes replacing Pass 1 inside the synthetic hemisphere with Pass 2 instead of OR-merging.

This idea is directionally good, but it is unsafe if implemented unconditionally.

Why:

- if Pass 2 misses a frame
- or tracks the wrong object
- or produces a small empty-ish mask

then replacement can erase real Pass 1 detections.

Earlier, this would have been too risky because Pass 2 was weak.

Now, with `11/11` tracked frames in the latest run, a replacement-style policy is much more plausible, but it should still be conditional, not unconditional.

Recommended interpretation:

- when Pass 2 succeeds on a frame, it should be authoritative inside the synthetic support region
- when Pass 2 is empty or obviously bad, preserve Pass 1 for that frame

### D. The SAM v1 embedding-cache suggestion is not realistic in the current design

The report suggests caching SAM v1 image encoding across views that share the same ERP source.

That is not a good fit for how SAM v1 is used here.

In the current plugin:

- each reframed pinhole view is a different image
- `SamPredictor.set_image(...)` encodes that actual view image

So there is no shared per-ERP embedding that can be reused across all views in the current architecture.

The stronger performance ideas are the other ones already named in the report:

- YOLO-only direction estimation
- lower detection resolution for direction-only work
- cheaper or split-model primary pass

## What FullCircle Does Right That The Plugin Does Not Yet Do

## 1. FullCircle uses a dedicated masking layout, not the reconstruction layout

This is the clearest, highest-signal difference.

FullCircle's masking entry point is:

- `run_masking.sh`
- step 1: `omnidirectionals -> 16 perspectives`

Those 16 perspectives come from `masking/omni2perspective.py`, where `get_virtual_rotations(...)` builds:

- `8` yaw steps
- `2` pitch bands
- pitches at `-35°` and `+35°`
- `90°` FOV

That means FullCircle spends its entire Pass 1 camera budget on a masking layout designed to localize a standing human.

The plugin's `medium` preset, by contrast, spreads 16 cameras continuously from:

- `+90°`
- down through mid-latitudes
- all the way to `-90°`

That is useful for reconstruction coverage, but it is not the same optimization target.

### Why this matters for your scene

Your scene puts the person near:

- `pitch ≈ -70°`

That is exactly the kind of case where layout quality matters more than raw camera count.

I ran a quick visibility probe using the current plugin `low` / `medium` presets versus FullCircle's fixed 16-camera layout.

For a target at `pitch=-70°`:

- plugin `low`: between `1` and `3` cameras see the target depending on yaw, average `2.25`
- plugin `medium`: between `2` and `4` cameras see the target depending on yaw, average `2.83`
- FullCircle 16-camera layout: exactly `3` cameras see the target at every yaw sample tested, average `3.0`

That is the key difference:

- FullCircle gives consistent lower-band yaw redundancy
- the plugin gives variable coverage depending on how the spiral preset happens to line up with the subject

So the report's "primary cause" is basically right, but the sharper wording is:

- FullCircle does not merely use more views
- FullCircle uses a more uniform and task-specific lower-band layout

## 2. FullCircle's final mask geometry comes from the synthetic pass, not from a union with the primary pass

This is the second major difference, and it may now be the most important active one.

The FullCircle pipeline is:

1. `omni2perspective.py`
2. `mask_perspectives.py`
3. `perspective2omni.py` -> primary omni masks + centers
4. `omni2synthetic.py`
5. SAM2 tracking on synthetic fisheyes
6. `synthetic2omni.py` -> final omni masks

The important thing is what does **not** happen:

- `synthetic2omni.py` does not load and OR-merge the primary omni mask
- it backprojects the synthetic-tracking masks into a new final omni mask

So FullCircle uses the primary pass mainly to:

- localize the person
- derive the synthetic camera orientation

Then it lets the synthetic/SAM2 result define the final mask geometry.

The plugin currently does this instead:

- build primary ERP mask in Pass 1
- build synthetic ERP mask in Pass 2
- `np.maximum(...)` them together unconditionally

That means any false positive from Pass 1 remains alive forever, even if Pass 2 correctly focuses on the actual subject.

This is likely the biggest remaining quality gap now that your latest run shows successful `11/11` SAM2 tracking.

In other words:

- FullCircle uses Pass 1 to aim
- the plugin uses Pass 1 to aim and also to permanently contribute shape

That is a big behavioral difference.

## 3. FullCircle stores direction evidence in a slightly more robust way

The two systems are similar, but not identical.

The plugin currently:

- computes center of mass on the full per-view detection mask
- uses total mask area as the direction weight

FullCircle does this in two stages:

- `mask_perspectives.py` stores the per-view center of mass
- `perspective2omni.py` computes the connected-component area at that center point and uses that component area as the weight

This difference matters when a per-view mask contains:

- multiple disconnected detections
- one real person plus several small false positives
- a noisy union mask whose centroid is not representative of the intended object

FullCircle's approach is still not perfect, but it is slightly more object-aware than "moments of the full union mask + total union area."

That means the plugin may currently be over-trusting noisy multi-component view masks when computing direction.

## 4. FullCircle's postprocessing is downstream of the synthetic pass

FullCircle does not try to rescue noisy primary masks with ERP morph-close.

Its final cleanup stage is:

- convert final omni masks back to raw fisheye masks
- dilate those raw fisheye masks

The plugin has already corrected the worst mismatch by removing ERP-level morph-close, but there is still a conceptual difference:

- FullCircle's cleanup happens after synthetic tracking has already produced the final object geometry
- the plugin still spends more of its decision-making budget on noisy primary masks

This is probably not the number-one issue, but it reinforces the same overall theme:

- FullCircle lets the synthetic pass be authoritative
- the plugin still preserves too much upstream uncertainty

## 5. FullCircle can assume SAM2 exists in the masking pipeline

This is less glamorous, but important.

FullCircle's masking pipeline is not designed as a shipping fallback architecture.

It assumes:

- synthetic stage exists
- tracking stage exists
- final masks come from that stage

The plugin, by contrast, was designed to ship in a much safer way:

- Track A had to work even if SAM2 was absent or failed
- the plan intentionally kept OR-merge and fallback semantics conservative

That was the right design choice at the planning stage.

But now that your current medium run shows:

- stable directions on all frames
- successful SAM2 tracking on all frames

the plugin may be ready to move closer to FullCircle's more authoritative synthetic-pass behavior.

## What The Latest Run Suggests The Real Current Problem Is

The latest run strongly suggests this revised diagnosis:

### Old diagnosis

- Pass 1 is noisy
- direction is unstable
- synthetic pass fails
- therefore FullCircle's main advantage is that tracking works

### Revised diagnosis

- Pass 1 is still probably noisy
- but direction is now good enough to drive synthetic views on all frames
- synthetic tracking is now working on all frames
- therefore the remaining quality gap is likely in:
  - Pass 1 localization layout
  - direction robustness
  - final merge semantics

That is a much better problem to have.

It means the plugin is no longer "missing the FullCircle pipeline."

It means the plugin now likely has the pipeline, but not yet the same decision policy around:

- how to localize
- what evidence to trust
- what to throw away once better evidence exists

## Response To The Proposed Fixes

## Fix 1: Dedicated detection layout for Pass 1

Assessment: strongly agree.

This is the most justified and highest-priority change in the report.

I would frame it not as a speculative optimization, but as the most direct way to align the plugin with FullCircle's successful masking structure.

Recommended form:

- add a dedicated Pass 1 masking layout
- keep reconstruction presets unchanged
- do not reuse the active reconstruction preset for primary masking

Suggested naming:

- `masking_primary_fullcircle16`

Suggested initial geometry:

- `8` yaw steps
- `2` pitch bands
- `±35°`
- `90°` FOV

This should be treated as the next quality-first experiment.

## Fix 2: Replace strategy for Pass 2

Assessment: agree in principle, but not as an unconditional rule.

Best version:

- Pass 2 should be authoritative where it succeeds
- Pass 1 should survive only where Pass 2 has no trustworthy answer

Recommended merge policy:

1. Define the synthetic-support region in ERP for each frame.
2. If the Pass 2 mask for that frame is non-empty and plausibly centered, replace Pass 1 inside that region.
3. Keep Pass 1 outside that region.
4. If Pass 2 is empty or obviously failed, fall back to Pass 1 for that frame.

That would move the plugin much closer to FullCircle without sacrificing the plugin's safer failure behavior.

## Fix 3: SAM v2 tracking with correct direction

Assessment: partly already validated.

The latest run shows:

- prompt frame selected successfully
- SAM2 propagation covering all frames

So this fix is no longer just a hypothesis.

What remains is not "can SAM2 track?"

It is:

- "are we feeding it the best possible synthetic views?"
- "are we trusting its output enough once it succeeds?"

## Fix 4: Performance optimization

Assessment: mixed.

### Good performance ideas

- reduce detection resolution for direction-only work
- use YOLO-only direction estimation where possible
- possibly split direction estimation from final shape estimation

### Weak performance idea

- cache SAM v1 image encoding across views from the same ERP

That is not a good fit for the current backend because every reframed view is a different image embedding problem.

If performance becomes urgent after quality is stabilized, the best next experiment is probably:

- YOLO boxes only for Pass 1 direction estimation
- reserve full SAM segmentation for:
  - Pass 2 synthetic fallback
  - or a smaller subset of views

## Recommendations In Priority Order

## 1. Add a dedicated Pass 1 masking layout

This is the highest-value next step.

Do not wait on broader preset redesign.

Add one dedicated masking layout and test it against the same `deskTest` scene.

Success criterion:

- more stable direction statistics
- fewer false-positive direction contributions
- cleaner synthetic views
- equal or better final COLMAP behavior

## 2. Make Pass 2 conditionally authoritative

Given the latest `11/11` tracking result, the plugin is probably leaving quality on the table by preserving all Pass 1 geometry through OR-merge.

Recommended next experiment:

- replace-inside-support-region when Pass 2 succeeds
- preserve Pass 1 only when Pass 2 fails

This is the single most important post-tracking-quality change to test.

## 3. Make direction weighting component-aware

The plugin should stop treating the full union mask in a view as one clean direction cue.

A better short-term version would be one of:

- use the connected component containing the CoM
- use the largest connected component
- use the strongest YOLO box rather than union-mask moments

This would bring the direction stage closer to FullCircle's practical behavior.

## 4. Separate "direction estimation" from "mask geometry"

Right now Pass 1 is doing two jobs:

- estimating person direction
- contributing final mask geometry

Those jobs want different behavior.

Direction estimation wants:

- stable, large, centered detections
- maybe just boxes
- maybe aggressive rejection of tiny regions

Final mask geometry wants:

- conservative false-positive control
- stronger trust in Pass 2 once available

Those should be treated as separate design problems, not one combined heuristic.

## 5. Only then spend time on speed

The masking stage is clearly expensive, but the latest run suggests the quality path is finally becoming tractable.

So I would optimize performance only after:

- dedicated masking layout
- conditional replace strategy
- direction robustness improvements

Otherwise there is a real risk of making the wrong pipeline faster.

## Suggested Reframe Of The Core Question

Instead of asking:

- "why does FullCircle work and we do not?"

the better current question is:

- "now that our synthetic/SAM2 path works, why are we still preserving too much low-quality Pass 1 evidence?"

That is a more precise framing of the current state.

And the answer appears to be:

- FullCircle uses Pass 1 to localize
- FullCircle uses Pass 2 to define the final subject geometry
- the plugin still uses Pass 1 for both

## Bottom Line

The testing report was valuable because it identified the right structural weak point:

- Pass 1 view layout

But the newest medium run shows the project has moved forward:

- SAM2 is now tracking all frames

So the center of gravity has shifted.

The most important things FullCircle does right are now:

1. use a dedicated masking layout for primary localization
2. let the synthetic/SAM2 pass become the final mask geometry
3. treat direction cues more robustly than a raw union-mask average

If I had to rank the next changes by likely payoff:

1. dedicated FullCircle-style Pass 1 layout
2. conditional replacement of Pass 1 by Pass 2 inside the synthetic support region
3. component-aware direction weighting
4. only then performance work

That is the shortest path from "the pipeline now runs end to end" to "the masks behave like FullCircle-quality masks."
