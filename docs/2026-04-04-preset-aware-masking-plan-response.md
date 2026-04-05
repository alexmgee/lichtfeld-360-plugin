# Preset-Aware Masking Plan Response

**Date:** 2026-04-04  
**Plan Under Review:** `docs/2026-04-04-preset-aware-masking-plan.md`

---

## Executive Summary

The plan is directionally correct: cubemap should not be forced through the same synthetic-camera masking path that works for the Default preset.

The current pipeline is optimized around the FullCircle-style 16-view layout:

1. Pass 1 uses a fixed 16-camera detection layout at `pitch=±35°`
2. Pass 2 renders a synthetic fisheye view aimed at the detected person
3. SAM2 tracking produces a mask in synthetic space
4. That mask is backprojected to ERP
5. Stage 3 reprojects the ERP mask into the preset's final pinhole views

That architecture makes sense for the Default preset because the final reconstruction views closely match the detection and synthetic coverage pattern.

For cubemap, it is a poor fit:

- the final views are `4x horizon + nadir + zenith`
- the operator is often well below the horizon ring
- the ERP mask becomes a weak intermediate
- backprojection is expensive
- the final cubemap masks inherit sparse/stippled artifacts that are not present in the underlying pinhole views

So the main conclusion holds: **masking should become preset-aware**.

---

## What The Plan Gets Right

### 1. The diagnosis is correct

The problem is not just "cubemap is slower." The bigger issue is that cubemap is being asked to consume a mask representation that was designed around a different viewing geometry.

### 2. The split should be preset-driven, not user-driven

This should not become another toggle. The pipeline already knows the active preset and its view geometry. The masking path should follow from that.

### 3. The Default preset should remain on the synthetic path

The current synthetic pipeline is still the best fit for the Default preset:

- the reconstruction views line up with the detection/synthetic geometry
- SAM2 tracking provides temporal consistency where it matters
- the expensive synthetic machinery is justified by the output layout

### 4. Cubemap should move toward direct per-view masking

For cubemap, pixel-aligned per-view masks are more valuable than preserving the synthetic ERP intermediate.

---

## Main Suggestions And Improvements

### 1. Change the cubemap implementation shape

The plan's biggest improvement area is **where** the direct path should live.

The current draft proposes that `Masker` itself:

1. reads each ERP frame
2. reframes it into preset views
3. runs per-view masking
4. writes final masks

That will work, but it duplicates the Stage 3 geometry and cache logic that already exists in the reframer.

### Recommended shape instead

For cubemap:

1. Reframe images first using the normal Stage 3 image path
2. Run direct masking on the already-reframed cubemap images
3. Write final masks directly to `out/masks/{view_name}/{frame_stem}.png`
4. Skip ERP mask generation and skip Stage 3 mask reprojection entirely

This is a better fit because it:

- uses the exact final pixels the user and COLMAP will see
- avoids duplicating ERP→pinhole geometry in `Masker`
- avoids building a second remap/cache path for the same preset views
- removes the ERP intermediate that is causing the cubemap artifacts in the first place

### Practical first version

The first implementation does not need deep refactoring:

- Stage 3 writes `images/{view_name}/*.jpg` as usual
- a new direct cubemap masking step reads those images back
- it runs `detect_and_segment` per image
- it writes `masks/{view_name}/*.png`

That adds some extra disk I/O, but it is a clean and low-risk first cut. Later, the process can be fused to avoid re-reading images.

---

### 2. Keep the strategy decision in the pipeline/preset layer

The current plan suggests adding `masking_strategy` to `MaskConfig`.

I would avoid that for the first implementation.

`MaskConfig` is mainly about model/backend behavior:

- targets
- device
- output size
- backend preference
- synthetic camera options

The choice between:

- synthetic ERP masking
- direct preset-view masking

is really an **output-geometry / pipeline-routing decision**, not a low-level mask backend setting.

### Recommendation

Make the decision in `pipeline.py`, based on the active preset:

- `default` → synthetic ERP path
- `cubemap` → direct per-view path

If the project later grows more preset families, then a formal preset-level strategy field can be added in a deliberate way.

---

### 3. Narrow the initial scope to explicit cubemap support

The draft already leans this way, and that is the right call.

Do not start with heuristics like:

- "if pitch exceeds 60°"
- "if overlap is low"
- "if views are pole-adjacent"

Those are reasonable future ideas, but they are unnecessary risk for v1.

### Recommendation

Start with:

- `default` = synthetic
- `cubemap` = direct

Only generalize after both paths are stable and benchmarked.

---

### 4. Reframe the performance section

The plan correctly points out that cubemap should avoid the synthetic backprojection bottleneck. But the direct-path cost estimate is probably too pessimistic in one important way.

The direct path is not automatically:

- `6 x full YOLO+SAM cost` per frame in the worst possible sense

because the current backend only runs SAM after YOLO finds a person. On cubemap, many faces will usually be:

- empty
- partial misses
- fast YOLO-only negatives

So the practical cost is more like:

- 6 view inspections per frame
- but only 1-2 faces are likely to trigger meaningful SAM work in many clips

### Recommendation

Rewrite the performance section to say:

- the direct path trades one expensive synthetic tracking/backprojection pipeline for many smaller per-view detection decisions
- actual cost depends heavily on how many cubemap faces contain the operator
- a cubemap benchmark is required before making strong timing claims

---

### 5. Preserve existing per-view mask finishing behavior

The direct path should not invent new edge-treatment rules unless there is a strong reason.

The current reframer already applies a per-view erosion of the keep region after reprojecting ERP masks. That is the current "final mask shaping" behavior for pinhole outputs.

### Recommendation

The direct cubemap path should:

- produce the same polarity as today: white=`keep`, black=`remove`
- apply the same per-view finishing behavior after segmentation
- write masks in the same final directory layout

That keeps downstream behavior consistent and makes A/B evaluation easier.

---

### 6. Add one small architectural note for later

If the direct cubemap path works well, there is a natural second-step optimization:

- let the reframer expose per-view images in memory during Stage 3
- let the direct masking step consume those images immediately
- avoid re-reading the just-written JPEGs from disk

That should **not** be required for the first version. It is just the obvious follow-on if the direct path becomes permanent.

---

## Recommended Implementation Shape

### Path A: Default preset

Keep the current path unchanged:

1. Pass 1 detection layout
2. Synthetic fisheye render
3. SAM2 tracking
4. ERP backprojection
5. Reframer image + mask reprojection

### Path B: Cubemap preset

Recommended path:

1. Extract ERP frames
2. Reframe ERP images into cubemap output views
3. Run direct per-view masking on `images/{view_name}/*.jpg`
4. Write final masks to `masks/{view_name}/*.png`
5. Skip ERP mask generation
6. Skip Stage 3 mask reprojection
7. Run overlap-mask and COLMAP stages as usual

This keeps cubemap-specific logic simple and avoids copying Stage 3 math into the masking module.

---

## Suggested File-Level Direction

| File | Suggested role |
|------|----------------|
| `core/pipeline.py` | Choose masking route by preset and wire the cubemap branch |
| `core/masker.py` | Add a direct per-view masking entry point that works on final preset views rather than ERP |
| `core/reframer.py` | Leave geometry logic alone for v1; optionally expose a future in-memory handoff later |
| `core/presets.py` | No required change for v1 beyond explicit preset-name routing |

One naming pattern that would stay readable:

- `process_frames()` = current synthetic ERP path
- `process_reframed_views()` = direct per-view path for already-generated output views

That keeps responsibilities clear.

---

## Validation Criteria I Would Add

Before treating the cubemap path as done, the plan should define success in both quality and speed terms.

### Quality

- cubemap masks are visually cleaner than the current ERP-derived masks
- no stippled ERP-backprojection artifacts in final pole/horizon views
- mask polarity matches current COLMAP expectations
- overlap masks still work correctly
- registration quality does not regress materially

### Performance

- cubemap total masking time is measured separately from Default
- direct path timing is broken down into:
  - image load / walk
  - per-view detection
  - mask write
- compare total cubemap wall-clock against the current synthetic cubemap path

### Practical run metadata

For each cubemap benchmark run, record:

- clip name
- frame count
- preset
- output size
- active masking backend
- whether `sam2._C` is active
- total masking stage time
- COLMAP registration outcome

---

## Final Recommendation

The document is right to make masking preset-aware.

The main change I would make is this:

> For cubemap, do not build a second ERP→view reprojection path inside `Masker`. Reframe the images first, then mask the final cubemap views directly.

That gives you the behavior you want with less duplicated geometry, less coupling, and a cleaner mental model:

- Default keeps the synthetic tracking workflow it was designed for
- Cubemap gets masks that are aligned to the actual output views

That is the implementation shape I would use for v1.
