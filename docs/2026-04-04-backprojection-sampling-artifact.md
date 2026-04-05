# Backprojection Sampling Artifact — Diagnosis and Fix Plan

**Date:** 2026-04-04
**Status:** Active bug — affects both Default and Cubemap preset masks
**Severity:** High — produces hollow/stippled/missing masks on frames where the person is not centered in the synthetic fisheye

---

## Symptom

ERP masks from Pass 2 (synthetic fisheye → ERP backprojection) show three failure modes:

1. **Stippled/dithered outlines** — the person shape is recognizable but hollow, with a scattered dot pattern instead of a solid fill
2. **Edge-only masks** — head and hands are masked but torso is white (unmasked)
3. **Near-empty masks** — only scattered fragments where the person should be

These failures vary per frame within the same clip. Some frames produce clean, solid masks; others are broken.

## Observed Pattern

- Frames where the person is well-centered in the synthetic fisheye → clean solid masks
- Frames where the person is off-center or near the edge of the fisheye hemisphere → hollow/stippled masks
- The artifact severity correlates with how far from the fisheye center the person appears

## Evidence

From `deskTest` Default preset run (16 frames):

- `deskTest_trim_00008.png` — good: solid person mask, person near fisheye center
- `deskTest_trim_00006.png` — bad: stippled outline, person off-center, left+right edges of ERP (person wraps around 360° boundary)
- `deskTest_trim_00003.png` — bad: near-empty, scattered dots
- `deskTest_trim_00011.png` — bad: hollow outline, head/hands solid but torso empty

## Root Cause

The backprojection function `_backproject_fisheye_mask_to_erp()` uses **point sampling**: for each ERP pixel, it computes the corresponding fisheye pixel coordinate and reads a single pixel from the tracked mask:

```python
px_int = np.clip(np.round(px_py[in_bounds, 0]).astype(int), 0, fish_size - 1)
py_int = np.clip(np.round(px_py[in_bounds, 1]).astype(int), 0, fish_size - 1)
erp_mask[valid_idx] = mask[py_int, px_int]
```

The problem: the mapping from ERP pixels to fisheye pixels is **not uniform**. Near the center of the fisheye (where the synthetic camera points directly at the person), many ERP pixels map to the same fisheye pixel — high sampling density. Near the edge of the fisheye hemisphere (θ approaching 90°), the equidistant projection spreads fisheye pixels over a much larger area of the ERP — low sampling density.

When the person is near the edge of the synthetic fisheye's FOV:
- Adjacent ERP pixels may map to fisheye pixels that are several pixels apart
- Point sampling hits some mask pixels and misses others
- The result is a sparse, stippled pattern — the mask information exists in the fisheye but the ERP grid doesn't sample it densely enough

This is a classic **backward-mapping aliasing problem**: the source (fisheye mask) has higher resolution than the destination (ERP) can sample at that location, and there is no anti-aliasing.

## Why This Wasn't Always Visible

Earlier test runs may have had the person more consistently centered in the synthetic fisheye. The artifact is most visible when:
- The camera moves a lot between frames (person position varies in fisheye space)
- The person is at high pitch (far from equator) — the ERP pixel density drops near the poles, making the sampling mismatch worse
- The synthetic camera size (2048px) is small relative to the ERP (7680×3840)

## The Same Root Cause Also Explains the Cubemap Problem

The cubemap ERP masks showed the same stippled pattern. The cubemap masks were reprojected from the same broken ERP masks. The artifact was not a cubemap-specific bug — it was a backprojection sampling bug visible in all ERP masks, and the cubemap reprojection just made it more obvious.

---

## Candidate Fixes

### Fix A: Dilate the fisheye mask before backprojection

Before sampling, dilate the SAM2 tracked mask by a few pixels. This fills in the gaps that point sampling would otherwise miss.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(mask, kernel, iterations=1)
# ... then backproject mask_dilated instead of mask
```

**Pros:**
- Trivial to implement (2 lines)
- Directly addresses the symptom
- Fast

**Cons:**
- Expands the mask boundary slightly — may mask a few pixels of background
- The dilation amount is a guess unless calibrated to the actual sampling density
- Doesn't fix the fundamental sampling problem

### Fix B: Backproject at higher resolution then downsample

Render the backprojection at 2× or 4× the final ERP resolution, then downsample with max-pooling (any masked pixel in the super-resolution grid → masked in the final grid).

**Pros:**
- Solves the aliasing properly — higher sampling density catches the mask pixels that point sampling misses
- No expansion of the mask boundary

**Cons:**
- 4× the pycolmap projection cost (already the bottleneck at 80s)
- More memory for the super-resolution grid

### Fix C: Forward-project the fisheye mask to ERP (scatter)

Instead of asking "for each ERP pixel, where does it land in the fisheye?", ask "for each white fisheye mask pixel, where does it land in the ERP?" Then paint those ERP pixels.

**Pros:**
- Only processes mask pixels (sparse — maybe 5-15% of the fisheye)
- No aliasing because every mask pixel contributes to the ERP
- Potentially much faster than the current backward mapping

**Cons:**
- Forward projection can leave gaps in the ERP (not every ERP pixel is hit)
- Needs a splat radius or fill step to cover gaps
- More complex implementation

### Fix D: Replace pycolmap with inline numpy math + use cv2.remap with INTER_NEAREST

The current backward mapping uses `pycolmap.img_from_cam()` for point projection, then point-samples the mask. If we replace pycolmap with inline numpy equidistant fisheye math, we can build proper `map_x/map_y` remap tables and use `cv2.remap` with `INTER_NEAREST` (or even `INTER_LINEAR` on a float mask then threshold).

The key insight: once we have the ERP→fisheye coordinate mapping as continuous float arrays, `cv2.remap` handles the sampling internally with proper pixel-center rounding, which may be more robust than our manual `np.round().astype(int)` indexing.

**Pros:**
- Removes pycolmap bottleneck (Approach D from the performance plan)
- Proper remap handling may reduce aliasing
- Much faster

**Cons:**
- Doesn't fundamentally solve the density mismatch — INTER_NEAREST still point-samples
- Would need to combine with Fix A (pre-dilation) for full artifact removal

### Fix E: Pre-dilate + forward fill hybrid

1. Dilate the fisheye mask by a small amount (Fix A)
2. Backproject with the current method
3. Apply morphological closing on the resulting ERP mask to fill small gaps

**Pros:**
- Addresses both the sampling gaps and the boundary noise
- Simple to implement
- No performance regression (dilation and closing are fast)

**Cons:**
- ERP-level morphological closing was previously removed because it bridged false positives across the sphere. But a small closing kernel (3-5px) targeted at filling sampling gaps is different from the old large closing kernel that bridged detections.

---

## Recommended Fix

**Fix A (pre-dilate the fisheye mask) as the immediate fix.** It's 2 lines of code and directly addresses the symptom. The dilation amount should be small — 3-5 pixels at 2048×2048 fisheye resolution. This is not the same as the old ERP morph-close that was removed; it's operating on the fisheye mask before projection, not on the ERP result.

**Fix D (inline numpy projection) as the follow-up performance fix.** This was already planned. When implemented, the remap-based approach may further reduce aliasing.

**Fix B is not recommended** — it makes the bottleneck worse.

**Fix C is architecturally interesting** but too complex for an immediate fix. Worth revisiting after Fix D is done.

---

## Implementation Plan

### Immediate: Pre-dilate fisheye mask (Fix A)

In `_backproject_fisheye_mask_to_erp()` and `_BackprojectMap.apply()`, dilate the input mask before sampling:

```python
BACKPROJECT_DILATE_PX = 3
if BACKPROJECT_DILATE_PX > 0:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * BACKPROJECT_DILATE_PX + 1, 2 * BACKPROJECT_DILATE_PX + 1),
    )
    mask = cv2.dilate(mask, kernel, iterations=1)
```

### Validation

After applying Fix A, re-run the Default preset on the same deskTest clip and compare:
- Frame 3, 6, 11 ERP masks should now be solid instead of stippled
- Frame 8 should remain clean (already working)
- Final pinhole masks should be cleaner across all views
- Registration quality should not regress

### Follow-up: Approach D (inline numpy fisheye projection)

Replace `pycolmap.img_from_cam()` with direct numpy equidistant fisheye math. This is already documented in `docs/2026-04-04-performance-optimization-results.md` as the recommended next performance optimization. It addresses both the performance bottleneck and may further improve sampling quality.
