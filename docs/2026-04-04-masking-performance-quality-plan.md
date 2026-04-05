# Masking Pipeline — Performance & Quality Improvement Plan

**Date:** 2026-04-04
**Context:** Post-implementation of masking v1 (dedicated 16-camera detection layout, authoritative Pass 2, YOLO-only Pass 1). Pipeline is functionally correct - SAM2 tracks 11/11 frames, masks are directionally right with no false positives. Problem is speed: the pipeline is too slow for real-world use, especially with minutes of footage.
**Source:** Analysis of current implementation in `core/masker.py`, testing observations from session `2847ffdb`.

---

## Scope And Goal

This document is no longer just a masking-stage note. It now covers the
highest-value opportunities to speed up the **end-to-end plugin pipeline**
without changing the overall architecture.

That means:

- **Stage 2 masking** is still the main focus, because it is currently the
  largest single cost center.
- **Stage 3 reframing** is included as a first-class optimization target,
  because it still takes substantial wall-clock time and repeats the same
  reprojection math patterns seen in Pass 1.
- Lower-confidence or geometry-changing ideas are explicitly marked as
  **experimental**, so they do not get mixed in with the safer wins.

Primary goal:

- Reduce end-to-end wall-clock time without regressing mask quality or COLMAP
  registration quality.

---

## Current Pipeline Cost Model

For a clip with **N frames**, the current pipeline does:

### Pass 1: 16-View YOLO Detection (per frame)

Per frame:
1. `cv2.imread` — read ERP from disk
2. `_reframe_to_detection` × 16 — reproject ERP to 16 pinhole detection views
3. YOLO inference × 16 — run YOLOv8s on each view individually
4. Direction computation — union bounding box, CoM, 3D direction

**Cost breakdown per frame:**
- Each `_reframe_to_detection` call at 1024px resolution:
  - Allocates a 1024×1024 float64 meshgrid (8 MB)
  - Computes rotation matrix (`create_rotation_matrix`)
  - Performs element-wise multiply, arctan2, arcsin on the full grid
  - Builds float32 `map_x`/`map_y` arrays (8 MB)
  - Runs `cv2.remap` on the ERP source
- Total per frame: 16 x (grid allocation + trig + remap) + 16 x YOLO inference
- **All 16 grid/trig computations are identical across frames** - the detection layout never changes

For **N=100 frames**: 1,600 redundant remap-table computations.

### Pass 2: Synthetic Fisheye + SAM2

Per frame:
1. `_render_synthetic_fisheye` — ERP → fisheye remap (direction-dependent, 2048×2048)
2. Resize to 512px min dimension + write JPEG to tempdir

Then once for the whole clip:
3. SAM2 `init_state` — reads all numbered JPEGs from disk
4. SAM2 `propagate_in_video` — forward + optional reverse propagation
5. Mask collection and resize back to 2048×2048

Per frame (post-tracking):
6. `_backproject_fisheye_mask_to_erp` — for each ERP pixel, compute world ray, rotate to camera space, project to fisheye pixel, sample mask

**Cost notes:**
- SAM2 model inference is the single largest wall-clock cost in Pass 2 - not optimizable at the plugin level
- The backprojection iterates over every ERP pixel (e.g., 3840x1920 = 7.4M pixels) per frame
- If person direction is stable across frames (common case), the backprojection remap tables are nearly identical frame-to-frame

### Stage 3: ERP -> Pinhole Reframing

Per frame:
1. `cv2.imread` - read the ERP again from disk
2. `reframe_view(...)` once per output view for the RGB image
3. If masking is enabled, `reframe_view(...)` again once per output view for the ERP mask
4. Write one JPEG per view and one PNG per mask view

**Cost notes:**
- The main reframer repeats the same meshgrid + trig + remap-table math
  pattern used in Pass 1.
- With masking enabled, Stage 3 pays for **two reprojections per view**:
  one for the image and one for the mask, even though the view geometry is
  identical.
- This stage is not as expensive as masking, but it is still large enough to
  matter in total wall-clock time.

---

## Proposed Changes

### 1. Precompute Pass 1 Remap Tables

**What:** `DETECTION_LAYOUT` defines 16 fixed (yaw, pitch, fov) camera views. The remap tables (`map_x`, `map_y`) for projecting an ERP image into each of these views depend only on the view geometry and the ERP dimensions — not on the frame content. Currently, `_reframe_to_detection` recomputes these tables from scratch on every call.

**Change:** On the first frame, compute and cache the 16 pairs of `(map_x, map_y)` arrays. On all subsequent frames, go straight to `cv2.remap` using the cached tables.

**Implementation sketch:**
- Add a `_detection_remap_cache: dict[tuple, tuple[np.ndarray, np.ndarray]]` at class or module level
- Key on `(yaw, pitch, fov, out_size, erp_w, erp_h)`
- Split `_reframe_to_detection` into `_build_detection_remap_table(...)` (pure math → map_x, map_y) and `_apply_detection_remap(erp, map_x, map_y, flip_v)` (just cv2.remap + flips)
- Build tables once on first frame, reuse thereafter

**Expected impact:** Eliminates 16 x (N-1) full grid computations. For
N=100 frames at 1024px, that's ~1,584 avoided allocations of 8MB grids + trig
passes. The per-frame cost drops to 16 x `cv2.remap` + 16 x YOLO. This is
still CPU-side OpenCV in the current implementation, but it removes the
repeated map construction cost.

**Risk:** None — pure cache, deterministic output.

### 2. Batch YOLO Inference

**What:** Currently runs YOLO 16 times per frame in a sequential Python loop:
```python
for vi, (yaw, pitch, fov, view_name, flip_v) in enumerate(detection_views):
    ...
    results = self._backend._yolo(image_rgb, stream=True, ...)
```

The `ultralytics` YOLO API supports batch inference - pass a list of images and
get all results in a single GPU call.

**Change:** Collect all 16 reframed detection images into a list, then call
YOLO once with the batch.

**Implementation sketch:**
```python
# Reframe all 16 views first
detection_images = []
for yaw, pitch, fov, view_name, flip_v in detection_views:
    face_img = _apply_detection_remap(erp, cached_maps[...], flip_v)
    detection_images.append(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

# Single batched YOLO call
batch_results = self._backend._yolo(
    detection_images, stream=False, conf=0.35, iou=0.6,
    classes=[0], agnostic_nms=False, max_det=20,
)

# Process results per view
for vi, (result, (yaw, pitch, fov, view_name, flip_v)) in enumerate(
    zip(batch_results, detection_views)
):
    ...
```

**Expected impact:** Reduces GPU kernel launch overhead from 16 per frame to 1.
Actual speedup depends on GPU utilization - if each individual YOLO call
already saturates the GPU, batching won't help much. But for smaller detection
images, individual calls likely underutilize GPU bandwidth, so batching should
give a measurable win.

**Risk:** VRAM usage increases (16 images in GPU memory simultaneously). At
512x512x3 float32, that's still modest relative to model memory.

**Dependency:** Accesses `self._backend._yolo` directly. This is currently a
private attribute of `YoloSamBackend`. If we want this to work cleanly, either
expose it through the `MaskingBackend` protocol or accept the tight coupling
(Pass 1 is already YOLO-specific by design).

### 3. Add Stage 3 Remap-Table Caching And Reuse

**What:** The main reframer repeats the same projection math for every output
view of every ERP frame. When masking is enabled, it does this twice per view:
once for the RGB image and again for the ERP mask.

**Change:** Add a shared remap-table cache to `core/reframer.py`, and reuse the
same `(map_x, map_y)` pair for both:

- RGB image reprojection (`INTER_LINEAR`)
- mask reprojection (`INTER_NEAREST`)

**Implementation sketch:**
- Split `reframe_view(...)` into:
  - `_build_reframe_remap(...) -> map_x, map_y`
  - `_apply_reframe_remap(image, map_x, map_y, mode=...)`
- Key the cache on:
  - `(yaw, pitch, fov, out_size, erp_w, erp_h)`
- Build each view map once, then apply it to both the ERP image and ERP mask

**Expected impact:**
- Removes repeated meshgrid/trig/remap-table work from Stage 3
- Avoids recomputing identical geometry separately for image and mask
- Helps every run, not just masking-stage time

**Risk:** Low - same output geometry, just cached and reused.

### 4. Highest-Confidence Box For Direction Estimation

**What (quality fix, not speed):** Currently, when YOLO returns multiple
detection boxes in a view, the code computes the union bounding box of all
detections:
```python
boxes = np.array(all_boxes)
x1 = boxes[:, 0].min()
y1 = boxes[:, 1].min()
x2 = boxes[:, 2].max()
y2 = boxes[:, 3].max()
box_cx = (x1 + x2) / 2.0
```

If YOLO detects the real person plus a false positive (e.g., a bag, a poster),
the union box center shifts away from the actual person. This degrades
direction estimation.

**Change:** Use the highest-confidence individual YOLO box instead of the
union.

**Implementation sketch:**
```python
# Select highest-confidence box
best_box = None
best_conf = 0.0
for result in results:
    for j in range(len(result.boxes)):
        conf = float(result.boxes.conf[j])
        if conf > best_conf:
            best_conf = conf
            best_box = result.boxes.xyxy[j].cpu().numpy().astype(int)

if best_box is None:
    continue

x1, y1, x2, y2 = best_box
box_cx = (x1 + x2) / 2.0
box_cy = (y1 + y2) / 2.0
box_area = float((x2 - x1) * (y2 - y1))
```

**Expected impact:** More robust direction estimation in scenes with
false-positive YOLO detections. The highest-confidence box is overwhelmingly
likely to be the actual person. Direction weight (box area) now reflects the
real detection, not the inflated union.

**Risk:** In rare multi-person scenes, this picks the most confident person
rather than the union of all people. For this plugin's use case (single
operator masking), that's the correct behavior.

### 5. Lower Pass 1 Detection Resolution (Experimental)

**What:** Currently `detection_size = min(1024, erp_w // 4)`. For direction
estimation via YOLO bounding boxes, this may be overkill. YOLOv8s natively
resizes input toward its own inference size anyway.

**Change:** Test dropping detection resolution to 512px for Pass 1.

**Implementation:** Change `detection_size = min(1024, erp_w // 4)` to `detection_size = min(512, erp_w // 4)` in `process_frames()`.

**Expected impact:**
- 4x fewer pixels in each remap table (512^2 = 262K vs 1024^2 = 1.05M)
- 4x fewer pixels through the numpy trig pipeline (still relevant on first frame or if cache is disabled)
- Faster `cv2.remap` (smaller output image)
- YOLO inference is slightly faster at 512 vs 1024 (less pre-resize work)
- Direction accuracy may still be sufficient for direction-only Pass 1

**Risk:** If the person occupies a very small portion of a 90 degree FOV
detection view at 512px, the YOLO box might be less precise or might fall under
the existing 5% coverage filter. This should be A/B validated, not assumed.

### 6. Shared Backprojection Map For Pass 2 (Experimental)

**What:** `_backproject_fisheye_mask_to_erp` computes a full mapping from every ERP pixel to fisheye pixel coordinates. This involves:
- ERP pixel → world ray (meshgrid + trig)
- World ray → camera space (matrix multiply)
- Camera space → fisheye pixel (pycolmap `img_from_cam`)
- Sample the mask at that pixel

The mapping depends on `R_world_from_cam`, which is derived from the person direction. If the direction is stable across frames (which testing showed — directions were consistent at ~yaw=59-73°, pitch=-66-70°), the remap tables are nearly identical.

**Change:** Optionally compute the backprojection mapping using the average
person direction across the clip, and reuse that mapping for all frames when
the resolved directions are tightly clustered.

**Implementation sketch:**
- After resolving all directions with temporal fallback, compute the mean direction
- Build one `_backproject_fisheye_remap(erp_size, camera, R_mean)` → `(map_erp_to_fish_x, map_erp_to_fish_y, valid_mask)`
- For each frame, sample the tracked mask using the precomputed map instead of calling the full backprojection function

**Expected impact:** Reduces N full-ERP backprojection computations to 1. For a
3840x1920 ERP, each backprojection touches 7.4M pixels. At N=100 frames, this
saves 99 x 7.4M pixel remap computations. It also amortizes the large batched
`pycolmap.img_from_cam(...)` projection step used during backprojection. The
current implementation already calls `img_from_cam(...)` on a full array of
camera-space rays, not one point at a time in Python, so the expected win here
is from avoiding repeated whole-frame projection work, not from eliminating a
Python loop around pybind calls.

**Risk:** If the person direction changes significantly across the clip (e.g.,
camera operator walks around), a single mean direction could shift the
backprojected mask by a few pixels at the edges. Mitigated by:
- The synthetic fisheye has 180° FOV — the person is centered, and the support region is much larger than the mask itself
- The mask is binary and the offset would be sub-pixel for typical direction variation
- If direction variance is high (measurable from the resolved directions), fall back to per-frame backprojection

**Threshold decision needed:** What's the maximum acceptable direction variance
before falling back to per-frame? One approach: compute the angular spread of
all resolved directions. If the max pairwise angle is under some threshold
(e.g., 10 degrees), use the shared map. Otherwise, fall back.

This should stay behind a measured threshold and quality validation. It is not
in the same risk class as remap caching or batched YOLO.

---

## Cache Design Notes

For the caching items above, the implementation should define the cache
contract clearly instead of just "add a dict somewhere."

Recommended cache key shape:

- `(yaw, pitch, fov, out_size, erp_w, erp_h)`

Recommended cache rules:

- cache geometry only, not image data
- share the same geometry cache between image and mask reprojection where
  possible
- invalidate naturally when ERP size, output size, or view geometry changes
- keep cached maps as `float32`, not `float64`

Memory note:

- A remap pair is not free, but it is much cheaper to keep a bounded set of
  geometry maps than to rebuild them repeatedly over a long clip

---

## Benchmark And Success Criteria

Do not treat these changes as done until they are measured.

For each optimization, capture:

1. **Pass 1 time per frame**
2. **Pass 2 time per frame**
3. **Stage 3 image reframe time per frame**
4. **Stage 3 mask reframe time per frame**
5. **Total pipeline wall-clock time**

Instrumentation note:

- The existing pipeline timing report is useful for stage-level totals, but it
  is not granular enough for this plan by itself.
- To evaluate items in this document properly, add lightweight timers inside
  `Masker` and `Reframer` so we can separate:
  - Pass 1 reprojection time vs YOLO time
  - Pass 2 synthetic render time vs SAM2 time vs backprojection time
  - Stage 3 image reframe time vs Stage 3 mask reframe time
- Without those finer-grained timings, we will know whether the whole masking
  or reframe stage got faster, but not which substep actually produced the win.

Run benchmarks on at least:

- one short stable indoor clip
- one longer clip with moderate direction drift
- one difficult clip with small or partially occluded operator detections

Quality guardrails:

- no regression in number of frames with valid person direction
- no regression in final masked frame count
- no visible increase in mask drift or false positives
- no regression in COLMAP registration quality on representative clips

---

## Priority and Dependencies

```
[1] Precompute Pass 1 remap tables (works at current size; cache key includes out_size)
  ↓
[2] Batch YOLO inference ← uses the precomputed remap flow
  ↓
[3] Add Stage 3 remap-table caching + image/mask reuse
  ↓
[4] Highest-confidence box (quality, independent)
  ↓
[5] Lower detection resolution (512px, experimental)
  ↓
[6] Shared backprojection map (experimental, threshold-gated)
```

Interpretation:

- Items 1-3 are the highest-confidence performance work
- Item 4 is a cheap quality improvement that should improve downstream masking
- Item 5 is worth testing, but should be validated rather than assumed
- Item 6 is the most experimental and should come last

---

## Future Candidates Not In This First Pass

These are real opportunities, but they are more architectural than the first
batch of optimizations above.

- **Avoid duplicate ERP decodes across stages.** The masking stage reads every
  ERP frame, and the reframer later reads the same ERP frame again. That is a
  larger design change, but it is worth keeping in mind if Stage 2 + Stage 3
  are refactored more deeply.
- **Avoid reloading ERP masks from disk before Stage 3.** Right now the mask is
  written, then read back for per-view reprojection. A later in-memory handoff
  could reduce disk churn.
- **Stage 3 output write optimization.** The dataset shape requires many small
  JPEG and PNG files. That is not the first bottleneck to attack, but it is
  still part of the end-to-end speed story.
- **Overlap-mask I/O cleanup.** The overlap-mask stage copies and rewrites a
  full mask tree. This is smaller than masking or reframing, but still part of
  total runtime.

---

## What This Plan Does Not Cover Yet

- **SAM2 inference speed.** The model itself is still the dominant cost in
  Pass 2. Reducing it would require a smaller model variant at potential
  quality cost. That should wait until the current quality story is stable.
- **Pass 2 synthetic rendering speed.** Each frame still needs a
  direction-dependent fisheye render. Since direction can change per frame,
  this cannot be cached in the same way as Pass 1.
- **Multi-GPU / async processing.** Not relevant for the plugin's single-machine target.
