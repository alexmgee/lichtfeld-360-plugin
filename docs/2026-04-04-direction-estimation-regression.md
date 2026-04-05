# Direction Estimation Regression — Investigation Record

**Date:** 2026-04-04
**Status:** Reverted — testing to confirm fix
**Clip:** `deskTest_trim.mp4` (same clip used for all runs)

---

## What Happened

Two changes were made to Pass 1 direction estimation as part of the performance/quality optimization pass:

1. **Detection resolution lowered** from `min(1024, erp_w // 4)` to `min(512, erp_w // 4)`
2. **Union bounding box replaced with highest-confidence single box** for direction estimation

Both changes were intended as improvements:
- Lower resolution: YOLO internally resizes to ~640px anyway, so 1024px input was thought to be wasted work
- Single best box: prevents false-positive YOLO detections from pulling the direction away from the actual person

## What Went Wrong

After both changes, the Default preset ERP masks showed:
- Stippled/dithered person outlines instead of solid fills
- Hollow masks where only head/hands were solid but torso was white
- Near-empty masks with scattered fragments
- Inconsistent quality across frames in the same clip

The same clip with the same code (before these two changes) produced clean, solid ERP masks.

## Root Cause

Both changes affect the **person direction estimate** — the 3D unit vector that determines where the synthetic fisheye camera points. The direction is the single most important input to Pass 2: it controls what the synthetic camera sees, how well-centered the person is in the fisheye, and therefore how well SAM2 tracks them and how cleanly the backprojection samples the tracked mask.

When the direction is slightly off:
- The person appears off-center in the synthetic fisheye
- The equidistant fisheye projection stretches pixels near the edges
- The ERP→fisheye point sampling becomes sparse in those regions
- The result is the stippled/hollow artifact

The backprojection code itself was unchanged. The sampling weakness was always latent — it just wasn't triggered when the direction was accurate enough to keep the person well-centered.

## Why Each Change May Have Contributed

### Detection resolution: 1024 → 512

At 512px with 90° FOV, a person at distance occupies fewer pixels. YOLO's bounding box coordinates have lower precision — the box center (used for direction computation) is quantized to a coarser grid. This shifts the direction estimate by a small amount per view, which compounds across the weighted average of multiple views.

The 5% coverage filter was supposed to mitigate this (skip small detections), but it gates on area, not on positional precision. A detection that passes the 5% filter at 512px still has half the coordinate precision of the same detection at 1024px.

### Single best box vs union box

The union bounding box of all YOLO detections in a view provides a more spatially averaged center — it smooths out individual box placement noise. The single highest-confidence box removes that averaging. While the highest-confidence box is more likely to be the real person (the intended benefit), its center is also noisier because it's a single sample rather than an average.

In scenes without false positives (like deskTest, where the only person is the operator), the union box was already correct. The single-box change removed useful spatial averaging without providing a benefit.

## What Was Reverted

Both changes reverted on 2026-04-04:

```python
# Detection resolution: back to 1024
detection_size = min(1024, erp_w // 4)

# Direction estimation: back to union bounding box
boxes = np.array(all_boxes)
x1 = boxes[:, 0].min()
y1 = boxes[:, 1].min()
x2 = boxes[:, 2].max()
y2 = boxes[:, 3].max()
box_cx = (x1 + x2) / 2.0
box_cy = (y1 + y2) / 2.0
box_area = float((x2 - x1) * (y2 - y1))
```

## Pending Validation

A Default preset run on the same deskTest clip needs to confirm:
- ERP masks are solid again (no stippling)
- Frames 3, 6, 11 (previously broken) produce clean masks
- Registration quality matches earlier clean runs

## Lessons

1. **Direction accuracy is critical.** The synthetic camera direction is not a "rough localization hint" — it directly determines mask quality through its effect on fisheye centering and backprojection sampling density.

2. **Changes to direction estimation need A/B validation on masks, not just timing.** Both changes were evaluated only for speed and detection count, not for their effect on downstream ERP mask quality.

3. **The backprojection has a latent sampling weakness.** Even with the revert, the backprojection is still vulnerable to off-center persons. The pre-dilation fix (documented in `2026-04-04-backprojection-sampling-artifact.md`) should be applied regardless to make the pipeline more robust.

## If Re-Introducing These Changes Later

### Detection resolution

If 512px detection is revisited:
- Must be validated against ERP mask quality, not just detection count
- Consider testing 768px as a middle ground
- The real bottleneck was remap table computation (now cached), so the performance benefit of lower resolution is smaller than initially estimated

### Highest-confidence box

If single-box direction is revisited:
- Only useful in scenes with actual false positives (multiple detected objects)
- In single-operator scenes, the union box is better because it averages positional noise
- Could be made conditional: use single box only when YOLO detects multiple distinct objects, union box when all detections are of the same person
- Would need a heuristic to distinguish "multiple boxes on one person" from "person + false positive"

---

## Related Documents

- `docs/2026-04-04-backprojection-sampling-artifact.md` — the underlying sampling weakness exposed by these changes
- `docs/2026-04-04-performance-optimization-results.md` — the optimization pass that introduced both changes
- `docs/2026-04-04-masking-performance-quality-plan.md` — the plan that proposed both changes as items 4 and 5
