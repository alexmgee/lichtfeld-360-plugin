# SAM2 Prompt Frame Selection Bug

**Date:** 2026-04-04
**Status:** Active bug — produces wrong masks on some frames
**Severity:** High — SAM2 tracks furniture/room instead of person when prompt frame is bad

---

## Symptom

Some ERP masks from the Default preset show large portions of the room (desk, checkerboard, furniture) masked instead of the person. Other frames in the same clip produce correct person masks.

Example: `deskTest_trim_00004.png` masks the desk and checkerboard target. The person (visible bottom-right in the ERP) is barely touched.

The broken frames are not random — they correlate with distance from the prompt frame in SAM2's temporal propagation. Frames near the prompt frame tend to be correct; frames far from it or where the person is in a very different position tend to fail.

## Root Cause

SAM2 video tracking is prompted with a single center-point click on one frame (the "prompt frame"). The prompt frame selection logic in `_synthetic_pass` is broken.

### The broken code (masker.py, _synthetic_pass):

```python
# Find best_frame_idx (strongest primary detection)
best_idx = 0
best_area = 0
for i, stem in enumerate(frame_order):
    mask = primary_masks.get(stem)
    if mask is not None:
        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best_idx = i
```

### Why it's broken:

Pass 1 was changed to return **empty ERP masks** (`np.zeros(...)`) because Pass 2 is now authoritative for mask shape. Every frame's mask has area 0. So `best_area` stays 0, `best_idx` stays 0, and SAM2 is **always prompted on frame 0** regardless of which frame has the strongest person detection.

### How this causes wrong masks:

1. SAM2 gets a center-point click on frame 0's synthetic fisheye
2. The synthetic camera points at the Pass 1 direction for frame 0
3. If the person isn't well-centered in frame 0's fisheye (because direction estimation was slightly off on that frame, or the person is at the edge of the hemisphere), the center click lands on whatever else is there — desk, floor, wall
4. SAM2 latches onto that object and propagates it forward and backward through all frames
5. The result: masks that track furniture instead of the person

### Why it worked before:

Earlier test runs happened to have frame 0 with a good direction where the person was well-centered in the synthetic fisheye. The bug was always there — it just didn't trigger.

## Proposed Fix

Replace the `best_idx` selection to use Pass 1 detection strength instead of empty mask area.

### What detection strength data is available:

`_primary_detection` already returns `n_detections` — the count of detection layout views that found a person. This is a direct measure of how confidently the person was detected on that frame. More views detecting the person = stronger evidence = better direction estimate = person more likely centered in the synthetic fisheye.

### Implementation:

1. In `process_frames`, collect detection counts per frame alongside directions:
   ```python
   detection_counts: dict[str, int] = {}
   ...
   erp_mask, direction, n_det = self._primary_detection(...)
   detection_counts[stem] = n_det
   ```

2. Add `detection_counts` as a parameter to `_synthetic_pass`

3. Replace the `best_idx` selector:
   ```python
   # Select the frame with the most YOLO detections as prompt frame.
   # More detections = stronger direction = person better centered in fisheye.
   best_idx = 0
   best_count = 0
   for i, stem in enumerate(frame_order):
       count = detection_counts.get(stem, 0)
       if count > best_count:
           best_count = count
           best_idx = i
   ```

4. Print the selected prompt frame for visibility:
   ```python
   print(f"[360] Pass 2: prompting SAM2 on frame {best_idx} "
         f"({frame_order[best_idx]}, {best_count} detections)")
   ```

### Why detection count is a good proxy:

- More detection views seeing the person means the person is clearly visible from multiple angles
- That correlates with a more accurate weighted direction estimate
- A more accurate direction means the synthetic camera points more precisely at the person
- The person is more likely centered in the fisheye on that frame
- SAM2's center-point click is more likely to land on the person

### Alternative considered: use total box area instead of count

Could sum the YOLO box areas across all views for each frame as a richer signal. But detection count is simpler, already returned from `_primary_detection`, and sufficient — a frame with 7/16 views detecting the person is almost certainly better than one with 3/16.

## Files to Change

| File | Change |
|------|--------|
| `core/masker.py` | Collect `detection_counts` in `process_frames`, pass to `_synthetic_pass`, replace `best_idx` logic |

## Verification

After applying the fix:
- Run Default preset on deskTest clip
- Frame 4 ERP mask should show the person, not furniture
- All ERP masks should be solid person shapes
- The prompt frame should be logged — verify it's a frame with high detection count
- Compare against the known-good results from earlier runs

## Broader Implication

The center-point click prompting strategy is fragile. Even with a correct prompt frame, SAM2 can track the wrong object if the person happens to not be at the exact center pixel of the synthetic fisheye. FullCircle uses the same center-click strategy, so this fragility is inherited from the reference implementation. But it means direction accuracy is critical — any improvement to direction estimation directly improves SAM2 prompt reliability.
