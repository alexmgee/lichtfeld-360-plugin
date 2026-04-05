# Masking Performance & Quality Plan — Response Review

**Date:** 2026-04-04  
**Context:** Review of `docs/2026-04-04-masking-performance-quality-plan.md` against the current plugin implementation, with a broader goal of improving end-to-end pipeline speed wherever practical.

---

## Executive Assessment

The plan is directionally good, and several of its proposed optimizations are worth doing. The strongest near-term wins are:

1. Precompute and reuse Pass 1 detection remap tables
2. Batch YOLO inference for the 16 fixed detection views
3. Switch direction estimation from union-box to highest-confidence box

However, the report is too narrow for the stated goal of speeding up the plugin "everywhere possible." It treats masking as the whole problem, but the current pipeline still spends substantial time in Stage 3 reframing, and that code has the same repeated remap-table math that the report correctly identifies in Pass 1.

The result is:

- The report is a good **masking-stage optimization note**
- It is **not yet a complete pipeline performance plan**

---

## What The Report Gets Right

### 1. Pass 1 remap-table caching is a real win

This is the clearest optimization in the document.

Current Pass 1 detection in `core/masker.py` calls `_reframe_to_detection(...)` once per detection view, per frame:

- `core/masker.py:642-645`
- `core/masker.py:894`

The fixed 16-camera detection layout means the reprojection geometry does not change across frames. Recomputing the meshgrid, trig, and remap tables every time is unnecessary.

This is very similar to the current Stage 3 reframer, where `reframe_view(...)` rebuilds:

- pixel grids
- camera rays
- spherical coordinates
- remap tables

on every call in `core/reframer.py:102-145`.

So the central idea is sound: **build remap tables once per geometry and reuse them**.

### 2. Batch YOLO inference is compatible with the current implementation

The report is correct that Pass 1 currently invokes YOLO sequentially in a Python loop:

- `core/masker.py:642-652`

And the backend really does hold a single Ultralytics YOLO model instance:

- `core/backends.py:78-90`

So batching the 16 reframed detection images into one YOLO call is a sensible optimization target. This should reduce Python overhead and improve GPU utilization, especially once the detection images are smaller.

### 3. Highest-confidence box is a strong quality fix

The report correctly identifies that direction estimation still uses the union of all detected boxes:

- `core/masker.py:667-676`

That means a real detection plus a false positive can drag the estimated center away from the person. Replacing that with "pick the highest-confidence person box" is a low-risk change that should improve direction quality without affecting the broader architecture.

---

## Main Gaps In The Report

### 1. The report is too masking-centric for an end-to-end speed goal

The biggest issue with the report is scope.

The current pipeline spends time in multiple stages, not just masking:

- Stage 2 masking: `core/pipeline.py:302-339`
- Stage 3 reframe: `core/pipeline.py:343-364`
- Stage 3.5 overlap masks: `core/pipeline.py:377-413`
- COLMAP stage after that

The report only addresses Stage 2, even though recent timing runs showed Stage 3 reframe is still a meaningful cost center.

That matters because Stage 3 repeats the same class of expensive work:

- For every ERP frame
- For every output view
- It calls `reframe_view(...)`
- And if masking is enabled, it calls `reframe_view(...)` again for the ERP mask

See:

- image reprojection: `core/reframer.py:343-350`
- mask reprojection: `core/reframer.py:366-374`

So when masking is enabled, Stage 3 effectively doubles its reprojection work per view. The current performance report does not mention that at all.

If the goal is "speed up the plugin everywhere possible," the report should explicitly include a **shared reframer optimization track**, not only a masking optimization track.

### 2. The report overstates what `cv2.remap` is buying today

The report says that after caching, the per-frame cost becomes `cv2.remap`, described as "fast, GPU-backed in OpenCV."

That is too strong for the current codebase.

The plugin currently uses plain CPU OpenCV with NumPy arrays:

- `core/reframer.py:137-145`

There is no evidence in the current implementation of `cv2.cuda.remap` or a GPU-backed OpenCV path being used here.

So the optimization is still valid, but the expected-impact language should be corrected to:

- remove repeated CPU-side map construction
- keep only the CPU-side `cv2.remap` call

That is still a worthwhile improvement. It just should not be framed as if OpenCV is already accelerating reprojection on the GPU.

### 3. The shared Pass 2 backprojection map is the riskiest proposal

This is the proposal that needs the most caution.

The current synthetic pass intentionally uses the resolved direction for each frame, then backprojects each tracked synthetic mask using that frame's own rotation:

- direction resolution: `core/masker.py:730-735`
- per-frame synthetic render: `core/masker.py:752-765`
- per-frame backprojection: `core/masker.py:805-819`

That is part of why Pass 2 can now be authoritative: it is following the frame-specific synthetic camera geometry instead of approximating it.

The report's suggestion is to compute one backprojection map using an average direction and reuse it across the clip:

- `docs/2026-04-04-masking-performance-quality-plan.md:180-194`

That may be safe for clips with extremely stable operator direction, but it is not "free." It trades geometric fidelity for speed.

This should therefore be treated as:

- an experimental optimization
- gated behind measured angular spread
- validated against mask drift before becoming default behavior

It should not be grouped mentally with the safer optimizations like remap caching or YOLO batching.

### 4. The report is missing a Stage 3 mask/image remap reuse opportunity

Stage 3 currently does this per view:

1. Reframe ERP image to pinhole
2. Reframe ERP mask to pinhole

That means two calls to the same geometry-heavy reprojection path:

- image: `core/reframer.py:343-350`
- mask: `core/reframer.py:366-374`

Even if you do not introduce a global reframer cache immediately, there is an obvious local optimization:

- build the remap tables once for a given `(yaw, pitch, fov, output_size, erp_w, erp_h)`
- reuse them for both the RGB image and the grayscale mask
- use different interpolation modes only at the remap call site

This is one of the clearest "speed everywhere possible" opportunities, and it is not mentioned in the report.

---

## Response To Each Proposed Change

### 1. Precompute Pass 1 remap tables

**Assessment:** Strongly recommended.

This is the cleanest and safest performance win in the report.

Recommended refinement:

- Do not implement this as a masking-only one-off if you can avoid it
- Instead, consider extracting a reusable remap-table builder/apply path that can also be used by the main reframer later

That gives you a stronger long-term payoff than solving the same math twice in two modules.

### 2. Lower Pass 1 detection resolution to 512

**Assessment:** Worth testing, but not a guaranteed free win.

Current detection size is set here:

- `core/masker.py:891-892`

The report is probably right that 1024 is more than Pass 1 needs for rough direction estimation. But the plan is slightly too confident about the quality remaining "equivalent."

Important current behavior:

- detections below 5% view coverage are ignored for direction estimation
- `core/masker.py:685-691`

That means smaller or more distant people are already close to the rejection line. Dropping resolution may push borderline detections below that threshold sooner.

Recommendation:

- Test 512 after remap caching and YOLO batching are in place
- Keep it as an A/B optimization until you verify direction robustness on harder clips

### 3. Batch YOLO inference

**Assessment:** Recommended.

This is one of the best remaining Stage 2 wins once detection reprojection is cached.

One implementation note the report correctly hints at:

- this reaches into `self._backend._yolo`, which is currently a private attribute
- `core/masker.py:649`
- `core/backends.py:80`

That is not necessarily a problem, because Pass 1 is already explicitly YOLO-specific. But the code should acknowledge that deliberately. There are two reasonable options:

1. Accept the coupling and document it
2. Add a small public batch-detect helper on `YoloSamBackend`

Either is fine; the key thing is not to pretend the abstraction already exists.

### 4. Highest-confidence box for direction estimation

**Assessment:** Recommended and should be moved earlier.

This is a small, low-risk quality improvement that should improve synthetic-pass targeting on difficult clips. It is one of the cheapest changes in the whole report and does not depend on the larger performance work.

If you want a fast path to better real-world results, this should probably ship before or alongside the first performance changes.

### 5. Shared backprojection map for Pass 2

**Assessment:** Experimental only.

This idea may be useful for clips with very stable operator direction, but it is fundamentally different from the other items because it changes geometry assumptions.

Recommendation:

- keep it out of the first performance pass
- only enable it when direction variance is explicitly below a measured threshold
- compare output masks against per-frame backprojection before making it default

---

## Additional Opportunities The Report Should Mention

If the real goal is broader plugin speed, the report should grow to include these items.

### A. Shared remap caching in the main reframer

The most obvious missing optimization outside masking is remap reuse in `core/reframer.py`.

Current Stage 3 behavior:

- for every ERP frame
- for every output view
- recompute reprojection math from scratch

See:

- `core/reframer.py:344-350`
- `core/reframer.py:367-374`

This is directly analogous to the Pass 1 issue the report already identified. A unified remap-cache design would help both:

- Pass 1 detection reframing
- Stage 3 image reframing
- Stage 3 mask reframing

### B. Reuse the same remap tables for image and mask in Stage 3

Even if full global caching is postponed, Stage 3 should not compute separate geometry for:

- the RGB panorama
- the matching ERP mask

Those use the same view geometry. Only interpolation differs:

- image: linear
- mask: nearest

This is an immediate structural optimization opportunity.

### C. Avoid repeated directory creation and small-file overhead where practical

The current reframer writes one JPEG per view and one PNG per mask view:

- image write: `core/reframer.py:361-363`
- mask write: `core/reframer.py:383-386`

This is necessary for the dataset shape, but it is still worth treating output I/O as part of the overall speed story, especially for longer clips. The performance report currently dismisses disk I/O only in the narrow context of SAM2 temp JPEGs. The broader pipeline writes many more files than that.

### D. Consider whether overlap-mask generation is worth optimizing later

The overlap-mask stage is relatively small, but it still:

- copies the entire operator mask tree
- then reopens and rewrites every per-view mask

See:

- `core/pipeline.py:391-410`

This is not an immediate bottleneck compared with masking and reframing, but it is still part of end-to-end runtime and should be considered if the larger wins are exhausted.

---

## Recommended Execution Order

For the current plugin, the most sensible order looks like this:

### Phase 1: Safe, high-value wins

1. Pass 1 detection remap-table caching
2. Batched YOLO inference
3. Highest-confidence box for direction estimation

These are the strongest low-risk changes in the current report.

### Phase 2: Reframer-wide speed work

4. Add shared remap-table caching to `core/reframer.py`
5. Reuse a single view remap for both image and mask reprojection

These are not in the report today, but they should be if the goal is true pipeline-wide speed.

### Phase 3: Measured, optional tradeoffs

6. Test 512px Pass 1 detection resolution
7. Experiment with shared Pass 2 backprojection maps only on low-variance clips

These should be validated rather than assumed.

---

## Recommended Changes To The Report

If `docs/2026-04-04-masking-performance-quality-plan.md` is going to be used as the main decision document, I would revise it in these ways:

1. Change the scope language so it explicitly says whether it is:
   - a masking-stage optimization note, or
   - a full pipeline performance plan

2. Correct the `cv2.remap` description so it does not imply GPU-backed OpenCV acceleration that the current code is not using.

3. Reclassify the shared Pass 2 backprojection map as experimental and threshold-gated.

4. Add a new section for Stage 3 reframer optimization, including:
   - shared remap caching
   - reuse of geometry for image and mask reprojection

5. Move the highest-confidence-box change earlier in the priority list.

---

## Bottom Line

The report is useful and contains several good ideas, but it is best understood as **the start of a masking performance pass**, not the full answer to pipeline speed.

If the goal is to make the plugin broadly faster:

- do items 1, 3, and 4 from the report
- keep item 2 as a tested optimization, not an assumption
- treat item 5 as experimental
- and add a second optimization track for the main reframer, because the current pipeline is still spending a meaningful amount of time there, especially when masks are enabled and every view is reprojected twice.
