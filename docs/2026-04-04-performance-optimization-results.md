# Performance Optimization Results

**Date:** 2026-04-04  
**Plan:** `docs/2026-04-04-masking-performance-quality-plan.md`  
**Test scene family:** `deskTest`  
**Camera behavior:** moving handheld 360 camera  
**Primary preset under test:** `Default` (16 views)

---

## Executive Summary

This optimization pass appears to have worked well.

The important shift is structural:

- Pass 1 is no longer the dominant masking bottleneck
- Stage 3 reframing is materially healthier
- Pass 2 backprojection is now the main remaining hotspot

That is the right outcome for this phase. The obvious repeated geometry work
has been removed, and the pipeline is now exposing the next true expensive
kernel.

Two cautions:

1. The measurements below come from multiple reporting layers and should be read
   **by run**, not as one single additive benchmark.
2. The default preset was also changed during this period. That is a meaningful
   product/config change, but it should be separated mentally from pure
   optimization wins when comparing against older runs.

---

## What Changed

### 0. Substage Timing Instrumentation

**Files:** `core/masker.py`, `core/reframer.py`

Added `_SubstageTimer` to both hot paths. This was required to measure:

- Pass 1 reprojection vs YOLO
- Pass 2 synthetic render vs SAM2 vs backprojection
- Stage 3 image reprojection vs mask reprojection

This instrumentation is now the main source for the detailed timing tables
below.

### 1. Precomputed Pass 1 Remap Tables

**File:** `core/masker.py`

Split `_reframe_to_detection` into:

- `_build_detection_remap(...)`
- `_apply_detection_remap(...)`

The `Masker` caches all 16 detection remap-table pairs on the first frame,
keyed on `(detection_size, erp_w, erp_h)`.

**Observed effect:** `p1_remap_build` fires once, then subsequent frames only
pay remap application cost.

### 2. Batched YOLO Inference

**File:** `core/masker.py`

Restructured `_primary_detection` into:

- Phase A: build the 16 detection images
- Phase B: one batched YOLO call
- Phase C: parse per-view results for direction estimation

**Observed effect:** reduced 16 YOLO launches per frame to 1 batched call.

### 3. Stage 3 Reframe Remap Caching

**File:** `core/reframer.py`

Split `reframe_view(...)` into:

- `_build_reframe_remap(...)`
- `_apply_reframe_remap(...)`

The reframer now builds remap tables once per view geometry and reuses them
across frames. Image and mask reprojection share the same cached geometry; only
interpolation mode differs.

**Observed effect:** removed repeated geometry work across frames and between
image/mask reprojection of the same view.

### 4. Nearest-Neighbor Mask Remap Fix

**File:** `core/reframer.py`

The initial optimized nearest-neighbor mask path used NumPy fancy indexing.
That turned out to be much slower than `cv2.remap(..., INTER_NEAREST)`.

**Observed effect:** mask remap time dropped from roughly `10.6s` to `3.4s`
for the reported 256-call Stage 3 mask benchmark.

### 5. Highest-Confidence Box For Direction

**File:** `core/masker.py`

Replaced union-box direction estimation with the single highest-confidence YOLO
box per view.

**Observed effect:** intended as a quality fix, not a speed fix. Reduces the
risk of false positives dragging the direction estimate away from the actual
operator.

### 6. Lower Pass 1 Detection Resolution

**File:** `core/masker.py`

Changed:

```python
detection_size = min(1024, erp_w // 4)
```

to:

```python
detection_size = min(512, erp_w // 4)
```

**Observed effect:** lowers remap and transfer cost for Pass 1. Quality impact
still needs to be understood through representative runs, not just timing.

### 7. Shared Backprojection Map (Threshold-Gated)

**File:** `core/masker.py`

Added a shared-map optimization for low-angular-variance clips:

- `_BackprojectMap`
- `_build_backproject_map(...)`
- `_direction_angular_spread(...)`

If resolved direction spread stays under 10 degrees, the code can build a
shared ERP-to-fisheye lookup and reuse it for all frames.

**Observed effect in current testing:** did not activate for the handheld test
clip. The measured angular spread was too high.

### 8. Default Preset Change

**Files:** `panels/prep360_panel.py`, `panels/prep360_panel.rml`

Changed the default preset from `Cubemap` (6 views) to `Default` (16 views).

This is included for completeness, but it should be treated as a **related
product/config change**, not as a pure performance optimization.

---

## Benchmark Runs And Measurement Sources

The numbers below come from multiple benchmark views of the pipeline. They
should not be treated as if they were all produced by one identical run unless
explicitly stated.

### Run A: Masking Substage Benchmark

- Clip family: `deskTest`
- Frames: 16
- Preset: `Default` (16 views)
- Source: internal `_SubstageTimer` inside `core/masker.py`
- Purpose: isolate where masking time now goes

### Run B: Reframer Substage Benchmark

- Clip family: `deskTest`
- Frames: 16
- Preset: `Default` (16 views)
- Outputs: 256 image reprojections + 256 mask reprojections
- Source: internal `_SubstageTimer` inside `core/reframer.py`
- Purpose: isolate Stage 3 reprojection and I/O costs

### Run C: Full Pipeline Timing Snapshot

- Clip family: `deskTest`
- Timing table presented as 16-frame `Default` preset snapshot
- Source: stage-level pipeline timing report
- Purpose: understand end-to-end stage balance after optimization

### Registration Outcome Reference

The current write-up also preserves this quality reference:

- `176/176` images registered
- `11/11` complete rig frames
- zero drops

That outcome is useful, but it should be rerun under the same exact benchmark
conditions as the timing tables if this document is going to become the
canonical optimization record.

---

## Environment Metadata

### Captured Here

- test scene family: `deskTest`
- camera behavior: moving handheld 360 camera
- preset: `Default` (16 views)
- approximate frame ranges observed in this phase: 11-16

### Still Worth Recording Explicitly In Future Benchmark Runs

- git commit / branch
- exact clip name
- exact frame count
- ERP resolution
- output size
- GPU / CPU / RAM
- torch version
- CUDA version
- whether `sam2._C` was active
- detection size
- whether shared backprojection activated or was skipped

---

## Before / After Summary

This report contains stronger **post-optimization** measurements than it does
strict apples-to-apples baseline measurements. So the safest summary is
directional:

| Area | Outcome |
|------|---------|
| Pass 1 detection | No longer a major bottleneck |
| Stage 3 reprojection | Significantly improved by remap reuse |
| Stage 3 mask reprojection | Big win after switching nearest-neighbor path to `cv2.remap` |
| Remaining hotspot | Pass 2 backprojection |
| Product impact | Pipeline now appears bottlenecked by deeper geometry work rather than repeated high-level overhead |

The clearest hard numbers currently available are:

- Pass 1 total in Run A: `4.7s`
- Stage 3 total in Run B: `30.2s`
- Full pipeline masking stage in Run C: `59.8s`
- Full pipeline total in Run C: `221.1s`

Those are useful, but should still be read in the context of the run-splitting
notes above.

---

## Measured Results By Run

### Run A: Masking Substage Timing

**Configuration:** 16 frames, `Default` preset

| Substage | Time | Calls | Per-call |
|----------|------|-------|----------|
| `p1_imread` | `1.0s` | 16 | `63ms` |
| `p1_remap_build` | `0.5s` | 1 | `500ms` |
| `p1_remap_apply` | `1.2s` | 16 | `75ms` |
| `p1_yolo_batch` | `2.0s` | 16 | `125ms` |
| `p2_render_fisheye` | `13.7s` | 16 | `856ms` |
| `p2_sam2_tracking` | `6.7s` | 1 | `6.7s` |
| `p2_backproject` | `80.2s` | 16 | `5.0s` |
| **TOTAL** | **`105.4s`** | | |

**Interpretation:**

- Pass 1 total is now `4.7s`, which is only ~4.5% of masking time
- Backprojection now dominates masking time

### Run B: Reframer Substage Timing

**Configuration:** 16 frames, 16 views = 256 outputs

| Substage | Time | Calls | Per-call |
|----------|------|-------|----------|
| `remap_build` | `11.4s` | 1 | `11.4s` |
| `imread` | `1.6s` | 16 | `100ms` |
| `imread_mask` | `0.7s` | 16 | `44ms` |
| `remap_apply_img` | `8.9s` | 256 | `35ms` |
| `imwrite_img` | `2.5s` | 256 | `10ms` |
| `remap_apply_mask` | `3.4s` | 256 | `13ms` |
| `imwrite_mask` | `1.6s` | 256 | `6ms` |
| **TOTAL** | **`30.2s`** | | |

**Interpretation:**

- shared remap caching appears to have paid off
- image reprojection and mask reprojection are now both in a healthier range
- the `INTER_NEAREST` `cv2.remap` change for masks was important

### Run C: Full Pipeline Timing Snapshot

**Configuration:** `Default` preset timing snapshot

| Stage | Time | % |
|-------|------|---|
| Extraction | `74.9s` | 34% |
| Masking | `59.8s` | 27% |
| COLMAP | `48.4s` | 22% |
| Reframe | `26.3s` | 12% |
| Overlap masks | `11.7s` | 5% |
| **TOTAL** | **`221.1s`** | |

**Interpretation:**

- masking is no longer overwhelmingly dominant
- extraction is now a serious pipeline-wide cost center
- after backprojection is improved, extraction may become the next optimization
  target

### Registration Outcome Reference

Current quality reference:

- `176/176` images registered
- `11/11` complete rig frames
- zero drops

This supports the encouraging conclusion that the optimization pass did not
obviously break downstream COLMAP behavior, but it should be rerun under the
same exact benchmark conditions as the timing tables for a cleaner record.

---

## What These Results Prove

The report now supports these conclusions with reasonable confidence:

1. **Pass 1 optimization worked**
   - remap caching + batched YOLO successfully moved Pass 1 out of the critical
     path

2. **Stage 3 optimization worked**
   - remap caching and image/mask geometry reuse materially improved reframing

3. **The next real bottleneck is backprojection**
   - the optimization effort successfully exposed the next expensive kernel

4. **The pipeline is becoming more balanced**
   - now that repeated geometry overhead has been reduced, non-masking stages
     like extraction matter more

---

## Main Remaining Bottleneck: Pass 2 Backprojection

The current hot path is `_backproject_fisheye_mask_to_erp`.

### Why It Is Expensive

For each frame, it currently:

1. Builds the ERP pixel grid
2. Converts ERP pixels to world-space rays
3. Rotates those rays into camera space
4. Filters to the forward hemisphere
5. Projects all surviving rays through `pycolmap.img_from_cam(...)`
6. Applies validity checks and samples the mask

For high-resolution ERP, this means processing millions of rays per frame.

The dominant cost appears to be step 5: the large batched
`pycolmap.img_from_cam(...)` call on many millions of points.

### Why The Shared Backprojection Map Did Not Help Here

The shared-map optimization is only valid when direction spread is low.

In the handheld clip used here, spread was high enough that the optimization
did not activate. So the code path remained on the full per-frame projection.

That does not invalidate the idea. It just means it is only useful for
static or near-static camera setups.

---

## Candidate Next Approaches

### A. Downsampled Backprojection

Compute backprojection on a reduced ERP grid, then upscale the binary mask with
nearest-neighbor interpolation.

**Why it is attractive:**

- simple
- easy to validate
- likely real speedup
- works regardless of camera motion

**Main risk:** may miss very small mask features at extreme resolutions.

### B. Forward Projection

Project only mask pixels from fisheye to ERP rather than asking where every ERP
pixel lands in the fisheye.

**Why it is attractive:**

- potentially huge reduction in processed points for sparse masks

**Main risk:** introduces holes / splat behavior and is a bigger sampling-model
change.

### C. Cache Rotation-Independent World Rays

Precompute ERP-to-world directions once per ERP size and reuse them across
frames.

**Why it is attractive:**

- low risk
- easy companion optimization

**Main downside:** probably modest impact compared with the full projection
cost.

### D. Pure NumPy Equidistant Fisheye Projection

Replace the general-purpose `pycolmap.img_from_cam(...)` call with direct
vectorized NumPy math for the specific ideal equidistant fisheye model actually
used by the synthetic camera.

For camera-space rays `[x, y, z]`:

```text
theta = arctan2(sqrt(x^2 + y^2), z)
phi = arctan2(y, x)
r = f * theta
px = cx + r * cos(phi)
py = cy + r * sin(phi)
```

**Why it is attractive:**

- targets the actual hot path directly
- works for moving cameras
- may also be reusable in other synthetic-camera math paths later

**Main requirement:** must be validated numerically against `pycolmap` before
replacement.

---

## Recommended Next Step

### First Choice: Approach D

This is the strongest next candidate because it:

- attacks the dominant cost directly
- preserves the same overall mapping strategy
- works for moving cameras
- avoids relying on low-variance assumptions

### Fallback: Approach A

If D proves harder than expected, A is the safest practical fallback. It is
simple, measurable, and can still produce a large speedup.

### Later Experiments

- B is worth exploring only after D/A
- C is worth doing opportunistically but is unlikely to be transformative alone

---

## Validation Plan For The Next Round

Before replacing the current backprojection path, the next optimization pass
should add a tiny validation harness that compares candidates against the
current implementation on:

- one stable synthetic case
- one off-center synthetic case
- one real tracked mask from a handheld clip

Recommended metrics:

- runtime
- IoU vs current output
- changed-pixel count
- visual inspection on representative masks

This will keep the next round evidence-based and reduce the risk of optimizing
the wrong thing.

---

## What Information I Still Want In Future Results Reports

For each benchmark run:

- exact clip name
- exact frame count
- exact ERP resolution
- exact output size
- git commit / branch
- GPU / CPU / RAM
- torch / CUDA versions
- whether `sam2._C` was active
- whether shared backprojection was activated or skipped

And for the write-up itself:

- separate runs clearly
- keep timing tables tied to named runs
- keep product/config changes separate from optimization changes
- include a small before/after table when possible

---

## Bottom Line

This looks like a genuinely successful first optimization phase.

The main accomplishment is not just "things got faster." It is that the
pipeline's expensive work has been simplified enough that the next real target
is now obvious: backprojection.

That is good progress.

The next step is to cleanly validate and attack that kernel, with `Approach D`
as the strongest candidate and `Approach A` as the safest fallback.
