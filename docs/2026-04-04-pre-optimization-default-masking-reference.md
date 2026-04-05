# Pre-Optimization Default Masking Reference

**Date:** 2026-04-04  
**Status:** Reference / reconstruction note  
**Scope:** Default preset masking baseline before the April 4 performance/quality changes  
**Purpose:** Preserve a clear picture of how the masking system was intended to work before the optimization pass, how it compared to FullCircle, and which later changes were most likely to have made masking worse.

---

## Why This Document Exists

The masking system has gone through:

- the original masking design/spec work
- the FullCircle comparison and adaptation phase
- the two-pass synthetic-camera implementation
- the later performance/quality optimization pass

At this point it is easy to mix up:

1. the original intended masking architecture
2. the pre-optimization working baseline
3. the later optimization-era changes
4. the bugs that may have existed before but only became visible later

This document is meant to keep those separate.

---

## Short Version

Before the optimization/quality changes, the Default preset masking system was supposed to work like this:

1. Do a coarse first-pass person localization from ERP-derived views.
2. Convert that localization into a single 3D direction per frame.
3. Aim a synthetic 180-degree fisheye camera at that direction.
4. Run a stronger second pass on those synthetic views.
5. Backproject the synthetic masks into ERP.
6. Merge them with the primary ERP masks.
7. Reframe ERP masks into final pinhole masks for COLMAP/LichtFeld.

That structure was already strongly inspired by FullCircle.

The most likely things that later made masking worse were:

- lowering Pass 1 detection resolution from `1024` to `512`
- replacing union-box direction estimation with highest-confidence single-box direction

The strongest live bug discovered afterward, however, is different:

- SAM2 prompt-frame selection still uses empty Pass 1 masks, so it effectively prompts on frame `0` every time

That bug may have existed before the optimization pass, but the optimization pass likely made it easier to trigger.

---

## The Pre-Optimization Mental Model

The cleanest pre-optimization picture is the one described across:

- [2026-04-02-masking-layer-v1-design.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/specs/2026-04-02-masking-layer-v1-design.md)
- [2026-04-03-masking-v1-plan-v8.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/specs/2026-04-03-masking-v1-plan-v8.md)
- [2026-04-03-fullcircle-plugin-inspection.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-03-fullcircle-plugin-inspection.md)
- [2026-04-03-masking-v1-plan-v8-final-inspection.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-03-masking-v1-plan-v8-final-inspection.md)

The key idea was:

- keep the plugin's **ERP-first pipeline**
- adapt the **important FullCircle masking logic**
- do **not** try to port FullCircle's shell-driven workflow 1:1

So the intended plugin masking system was not "copy FullCircle exactly."
It was:

> keep the plugin's dataset flow, but adopt FullCircle's coarse localization +
> synthetic re-centering + stronger second-pass masking strategy.

---

## What The Pre-Optimization Default Path Was Supposed To Be

### Stage order

For the Default preset, the intended order was:

1. Extract sharp ERP frames
2. Run Stage 2 masking on ERP frames
3. Write ERP masks
4. Reframe ERP images and ERP masks into pinhole views
5. Optionally compute overlap masks
6. Run COLMAP with the per-view masks

This is still the basic plugin shape in `core/pipeline.py`.

### Pass 1: Coarse localization

Pass 1 was the "find the person well enough to aim the synthetic camera" stage.

Its job was not to produce the final authoritative mask shape.

Its responsibilities were:

- generate detection views from the ERP
- run image-mode person detection/segmentation
- accumulate detections across views
- estimate a 3D person direction for the frame
- create a coarse ERP mask baseline

The important pre-optimization assumptions were:

- detection views were relatively dense and FullCircle-like
- detection resolution was conservative (`min(1024, erp_w // 4)`)
- direction estimation favored spatial stability over aggressiveness

In practice, that meant the baseline was tuned more toward:

- stable synthetic framing

than toward:

- squeezing the last bit of Pass 1 runtime out of the system

### Pass 2: Synthetic fisheye pass

Pass 2 was the core FullCircle-inspired improvement.

Its responsibilities were:

- resolve a usable direction for each frame
- render a synthetic fisheye view aimed at the person
- run a stronger second-pass segmentation/tracking backend
- backproject the resulting synthetic mask back into ERP
- merge with Pass 1

In the pre-optimization design, this was the quality-critical stage.

The whole purpose was:

- if Pass 1 only coarsely localizes the operator,
- Pass 2 should re-center them and make segmentation much easier

### Final ERP save and later pinhole reprojection

After Pass 2, the system would:

- save final ERP masks
- then let the reframer project those ERP masks into the pinhole outputs

That is an intentional difference from FullCircle's final data layout.

The plugin trains from pinhole images and wants ERP masks as the intermediate source of truth.
FullCircle continues further into its own fisheye/raw-camera-specific outputs.

---

## The Pre-Optimization Quality Assumptions

Before the optimization pass, the Default preset implicitly relied on a few quality assumptions:

### 1. Direction accuracy mattered more than small Pass 1 speed wins

The synthetic camera only works if the look-at direction is good.

If direction is slightly wrong:

- the person drifts off-center in the synthetic fisheye
- the center-click prompt becomes less trustworthy
- SAM2 or fallback segmentation becomes less stable
- backprojection quality gets worse

So the pre-optimization baseline leaned toward conservative direction estimation.

### 2. Union-style spatial averaging was safer than single-box confidence

In a single-operator clip, multiple person detections on a view often represent:

- different parts of the same person
- or slightly noisy boxes around the same subject

The union-box approach smoothed that noise.

The highest-confidence single-box idea can be attractive in theory, but only if false positives are the dominant problem.

That was not the safer default assumption for the `deskTest` style clips.

### 3. Higher detection resolution reduced direction quantization

At `1024`, box centers used for direction estimation had more positional precision than at `512`.

That matters more than it first seems because those centers feed:

- 3D direction estimation
- synthetic camera pointing
- person centering in the synthetic fisheye
- downstream prompt quality

The pre-optimization baseline effectively chose:

- more stable geometry

over:

- slightly cheaper Pass 1 input generation

### 4. Backprojection robustness was not fully solved

Even in the pre-optimization state, fisheye-to-ERP backprojection already had a latent weakness:

- off-center synthetic masks backproject sparsely
- hollow/stippled shapes can appear

That weakness did not necessarily dominate while synthetic framing stayed good.

This is important:

> The pre-optimization baseline was not perfect. It was just more forgiving because
> the operator was more likely to remain centered enough for the latent weakness to stay hidden.

---

## How This Compared To FullCircle

### What matched FullCircle closely

The pre-optimization plugin setup was already heavily aligned with FullCircle in the areas that mattered most:

#### 1. Two-pass structure

Both systems shared the same strategic idea:

- first pass for coarse person localization
- second pass on a synthetic view aimed at the person

That is the core FullCircle masking idea.

#### 2. Synthetic camera geometry

The plugin plan intentionally adopted FullCircle-style synthetic fisheye logic:

- synthetic camera aimed at the operator direction
- 180-degree fisheye view
- FullCircle-style `look_at_camZ()` / look-at rotation concept
- pycolmap-based fisheye projection primitives

#### 3. Center-click prompting strategy

The plugin also inherited FullCircle's prompt philosophy:

- if the synthetic camera is aimed correctly,
- the person should be at the center,
- so a center click is enough to start video tracking

That is elegant when centering is good, but fragile when centering is not.

#### 4. Temporal fallback for missing directions

The plugin plan also followed FullCircle's idea that if some frames lack a good direction:

- borrow from nearby valid frames

That is a reasonable adaptation of FullCircle's handling of missing center data.

### What intentionally differed from FullCircle

The plugin was never supposed to be a literal port of FullCircle.

These were the important intentional differences:

#### 1. ERP-first plugin architecture

The plugin keeps:

- ERP frames as the primary intermediate
- ERP masks as the masking-stage output
- later pinhole reprojection through the reframer

FullCircle's shell workflow continues through its own later camera-space conversions.

That difference is intentional, not a missing feature.

#### 2. In-process backend model

The plugin wraps masking in backends and pipeline stages.

FullCircle, by contrast, is much more script-orchestrated.

The plugin wanted:

- in-process lifecycle management
- UI/setup integration
- staged fallback behavior

#### 3. Fallback image backend path

The plugin deliberately kept a non-video fallback path:

- if SAM2 is unavailable or unstable,
- synthetic views can still be segmented frame-by-frame using the image backend

That is a productization choice, not a FullCircle requirement.

#### 4. Plugin-facing setup and dependency tiers

The plugin had to care about:

- installability inside the embedded environment
- Windows runtime behavior
- user-facing readiness states

FullCircle's repo does not have to solve those same product constraints in the same way.

---

## What Was Probably Working Better Before The Optimization Pass

If the pre-optimization system felt better in practice, it was most likely because of the combination below:

### 1. Pass 1 direction was more stable

The pre-optimization setup retained:

- higher detection resolution
- union-box smoothing

Those two together likely made synthetic look-at directions more stable.

### 2. Better synthetic centering hid prompt fragility

The center-click prompt strategy is inherently fragile.

But if the person is very close to the synthetic fisheye center, that fragility remains hidden.

So the earlier state may have "worked" less because its logic was fundamentally stronger and more because:

- it kept the person centered often enough that the fragile prompt strategy did not break visibly

### 3. Better centering hid backprojection weakness

The backprojection artifact is worst when the person is far from the fisheye center.

If the earlier state centered the operator better, then the aliasing weakness stayed relatively contained.

So the earlier system may have looked substantially healthier even if the underlying backprojection math was unchanged.

---

## What Later Changed That Most Likely Made Masking Worse

This is the most important section for revert/reference purposes.

### Highest-likelihood regression changes

#### 1. Lowering Pass 1 detection resolution to `512`

This is one of the strongest candidates.

Why it likely hurt:

- lower coordinate precision for detection centers
- noisier direction estimates
- weaker synthetic centering
- more prompt fragility
- more visible backprojection artifacts

This change is documented in:

- [2026-04-04-direction-estimation-regression.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-direction-estimation-regression.md)

#### 2. Replacing union-box direction estimation with highest-confidence single-box direction

This is the other strongest candidate.

Why it likely hurt:

- removes spatial averaging
- makes direction more sensitive to individual box noise
- offers little benefit in single-operator scenes without meaningful false positives

This again pushes the system toward worse synthetic centering.

### Medium-likelihood contributor

#### 3. General optimization-era focus shift from stability toward efficiency

The optimization pass did several things at once.

Even if only two changes were directly harmful, the overall mindset changed from:

- "protect synthetic quality"

toward:

- "reduce hot-path cost"

That makes it easier for a geometry-sensitive pipeline to drift away from its most stable behavior.

### Lower-likelihood direct culprits

These changes are much less likely to be the direct cause of worse masking quality:

#### Batched YOLO

This should be geometry-preserving if results are parsed correctly.

By itself it is not the kind of change that should turn good masks into bad masks.

#### Remap caching

Caching remap tables should not change geometry if the cached maps are correct.

It should change cost, not output.

#### Shared backprojection map

For the handheld `deskTest` style clip, this path appears not to have activated due to angular spread.

So it is not the best explanation for the observed failures in that clip family.

---

## What Was Probably Already Wrong Before The Optimization Pass

This is the nuance that matters most when reconstructing the baseline.

Not every current bug was newly introduced.

### 1. SAM2 prompt-frame selection was likely already flawed

The later prompt-frame bug report suggests:

- prompt-frame choice depended on Pass 1 mask area
- but Pass 1 masks were already empty in the new two-pass model
- so the system was effectively prompting on frame `0`

If that logic was already present before the optimization pass, then the bug itself may have been latent rather than newly introduced.

That means the optimization pass may have:

- exposed the bug

rather than:

- created it from scratch

This matters because a full revert may restore better-looking masks without actually fixing the underlying prompt-selection bug.

### 2. Backprojection sampling weakness was already latent

The fisheye-to-ERP point-sampling issue also appears to have existed independently of the optimization pass.

Again, the optimization pass may simply have made it easier to see.

So the pre-optimization state should not be remembered as:

- fully correct

but rather as:

- more stable under the clips tested so far

---

## Best Current Interpretation

The most disciplined reading is:

### Before optimization

The Default masking pipeline was:

- closer to the intended stable geometry
- closer to the quality priorities of the original FullCircle-inspired design
- less likely to expose latent prompt/backprojection weaknesses

### During optimization

The system likely became more fragile because:

- direction estimation got noisier
- synthetic framing got less stable

### After optimization

The resulting failures were amplified by:

- a likely pre-existing prompt-frame-selection bug
- a likely pre-existing backprojection sampling weakness

So the optimization pass is best seen as:

- the likely trigger/exposer

not necessarily:

- the sole root cause of everything now visible

---

## What To Preserve In A Reverted Baseline

If the goal is to get back to the safer pre-optimization behavior, these are the behaviors worth preserving:

1. Conservative Pass 1 detection resolution (`1024`-class behavior, not `512`)
2. Union-box-style direction smoothing
3. Quality-first synthetic centering
4. ERP-first plugin architecture
5. Two-pass structure:
   - coarse localization first
   - synthetic re-centered second pass
6. Reframer-based final pinhole mask generation

These are the parts of the system that were most aligned with the original design and least likely to be accidental regressions.

---

## What To Watch Even After Revert

A revert alone may not fully solve the masking issue if these are still present:

1. prompt-frame selection still effectively choosing frame `0`
2. center-click prompting still landing on non-person content in weak synthetic frames
3. fisheye-to-ERP point-sampling still producing fragmented masks when the person is off-center

So if the revert improves quality but not completely, that would still make sense.

It would mean:

- the revert restored a more stable baseline
- but some latent structural issues remain

---

## Bottom Line

The pre-optimization Default preset masking setup was a quality-first adaptation of FullCircle's two-pass masking strategy into the plugin's ERP-first architecture.

Its strengths were:

- conservative Pass 1 geometry
- synthetic re-centering
- compatibility with the plugin's existing reframe/COLMAP flow

Compared with FullCircle, it intentionally differed in orchestration and final data layout, but it was trying to preserve the same important masking logic.

The most likely changes that later made masking worse were:

1. lowering Pass 1 detection resolution
2. replacing union-box direction estimation with single-box selection

The most important caveat is that those changes probably exposed at least one bug that may already have been present:

- broken SAM2 prompt-frame selection

So the best future reference point is:

> the pre-optimization baseline was probably better because it was more stable and more forgiving, not because every deeper bug had already been solved.
