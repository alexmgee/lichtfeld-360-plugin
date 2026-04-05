# Masking v1 Testing Report

**Date:** 2026-04-04
**Test scene:** `D:\Capture\deskTest` — desk-mounted 360 camera, person standing above it
**Person location:** Nadir region (~pitch -70°), person visible looking down from above

---

## Test Runs

### Run 1: Low preset (9 views, pre-refactor)

- **Preset:** Low (9 views, 75° FOV)
- **Code state:** Single-pass masker (before two-pass refactor)
- **Result:** Person detected in most frames but many views missed the person entirely. Mask quality inconsistent.

### Run 2: Low preset (10 views, post-refactor, first two-pass test)

- **Preset:** Low (10 views, 75° FOV)
- **Code state:** Two-pass with OR-merge, ERP morph-close postprocessing, FallbackVideoBackend (SAM v2 not confirmed active)
- **Video backend:** Unknown (no diagnostic logging yet)
- **Result:** ERP masks showed large false positives — desk, monitors, wall posters detected as person. Morph-close bridged false positives with real detections into massive blobs. Reframed pinhole masks inherited all false positives.
- **Key finding:** ERP-level morph close amplifies false positives across the sphere

### Run 3: Low preset (12 views, post-refactor, SAM v2 confirmed active)

- **Preset:** Low (12 views, 75° FOV)
- **Code state:** Two-pass with OR-merge, ERP morph-close REMOVED, per-view dilation added in reframer
- **Video backend:** Sam2VideoBackend confirmed active
- **SAM v2 result:** 3/11 frames got tracking masks
- **Direction stability:** Wildly unstable — yaw swinging from -147° to +122° to +104° to +171° between consecutive frames
- **False positives:** Desk and wall regions detected at 0.2%, 0.5%, 2.5% coverage — small but enough to corrupt weighted direction
- **Timing:** 592s masking (70% of 845s total pipeline)
- **Key finding:** False positive detections corrupt the weighted person direction, causing synthetic camera to aim wrong, causing SAM v2 tracking to fail on most frames

### Run 4: Low preset (12 views, 5% direction filter)

- **Preset:** Low (12 views, 75° FOV)
- **Code state:** Added 5% minimum coverage filter for direction computation (detections still contribute to mask, just not to direction)
- **Video backend:** Sam2VideoBackend
- **SAM v2 result:** 2/11 frames got tracking masks (WORSE than run 3)
- **Direction stability:** Still unstable — yaw 73.5° then -123.4° on consecutive frames
- **Filtered detections:** 0.7% and 4.0% correctly skipped for direction
- **Timing:** 519s masking (69% of 752s total)
- **Key finding:** Direction filter helps but doesn't solve the fundamental issue. With only 1-3 views reliably detecting the person per frame, the direction average is dominated by whichever views happen to fire.

### Run 5: Medium preset (16 views)

- **Preset:** Medium (16 views)
- **Result:** Similar false positive rate to Low. More views doesn't help when the views aren't aimed at the person.
- **Key finding:** Adding more reconstruction-optimized views doesn't improve detection quality

---

## Code Changes Made During Testing

| Change | File | Purpose | Result |
|--------|------|---------|--------|
| Removed ERP morph-close postprocessing | `core/masker.py` | Stop bridging false positives across sphere | Reduced blob size but false positives still present |
| Added per-view dilation in reframer | `core/reframer.py` | FullCircle-style edge expansion on pinhole masks | Dilation was expanding WRONG direction (white=keep), fixed to erode |
| Fixed dilation direction (dilate→erode) | `core/reframer.py` | Expand black=remove region, not white=keep | Correct polarity now |
| Added 5% min coverage for direction | `core/masker.py` | Filter false positives from direction computation | Helped but insufficient |
| Fixed torch.tensor warning | `core/backends.py` | Stack numpy array before tensor conversion | Performance warning resolved |
| Added diagnostic print statements | `core/masker.py` | Per-view detection %, direction, backend type, tracking results | Critical for debugging |

---

## Root Causes Identified

### 1. Reconstruction preset used for detection (PRIMARY CAUSE)

The masker uses the active reconstruction preset's view layout for Pass 1 detection. These presets were designed for COLMAP reconstruction quality (coverage, feature matching), not for YOLO person detection. The person at pitch -70° (nadir) is poorly covered by most preset views.

FullCircle uses a dedicated 16-camera detection layout (8 yaw × 2 pitch at ±35°, 90° FOV) independent of their reconstruction cameras. This is the single biggest difference between our results and theirs.

**Impact:** Low detection rate per frame (1-4 views out of 12), unstable person direction, cascade failure through synthetic pass.

### 2. OR-merge never removes false positives

The OR-merge strategy means any false positive from any single view is permanent in the ERP mask. With 12 views per frame, even a low per-view false positive rate compounds quickly.

FullCircle also uses OR-merge but gets away with it because their detection layout produces fewer false positives (person is well-centered and large in multiple views).

**Impact:** Desk, monitors, wall posters appear in ERP masks. False positives survive into reframed pinhole masks.

### 3. SAM v2 tracking yield is very low

Only 2-3 out of 11 frames get useful SAM v2 tracking masks. This is a cascade from the direction instability — the synthetic camera aims wrong, the center click misses the person, SAM v2 has nothing to track.

**Impact:** The synthetic pass adds almost no value. Most frames only have Pass 1 masks.

### 4. Performance

~47 seconds per frame for masking. 12 views × (YOLO inference + SAM v1 encoder + SAM v1 decoder) per frame. SAM v1's `set_image()` re-encodes the image for every view.

**Impact:** 70% of total pipeline time spent on masking.

---

## Proposed Fixes (Not Yet Implemented)

### Fix 1: Dedicated detection layout for Pass 1

Use a hardcoded 16-camera layout (8 yaw × 2 pitch at ±35°, 90° FOV) for detection, independent of the reconstruction preset. Matches FullCircle exactly.

**Expected impact:** More reliable person detection, stable direction, better SAM v2 tracking yield.

### Fix 2: Replace strategy for Pass 2

Within the synthetic camera's hemisphere, replace Pass 1's mask with Pass 2's mask instead of OR-merging. False positives inside the hemisphere get cleaned up.

**Expected impact:** Cleaner masks in the person's region. False positives outside the hemisphere still survive from Pass 1.

### Fix 3: SAM v2 tracking with correct direction

With Fix 1 providing stable direction, SAM v2's center click should land on the person consistently. Tracking should propagate across all frames.

**Expected impact:** Full temporal consistency, tracking yield close to 100%.

### Fix 4: Performance optimization

- Cache SAM v1 image encoding across views that share the same source ERP
- Reduce detection resolution for direction-only views
- Consider running YOLO without SAM v1 for direction estimation (YOLO bounding boxes are sufficient for center-of-mass)

**Expected impact:** Significant reduction in masking time.

---

## Open Questions

1. Will a dedicated 16-camera detection layout produce significantly better results, or is YOLO at conf=0.35 just too noisy for this scene regardless of view layout?
2. Should the detection confidence threshold be raised for the dedicated layout (since it has more overlap and redundancy)?
3. Is the 2048px synthetic fisheye too large for SAM v2's 512px resize? Would a smaller synthetic view retain more detail after resize?
4. Should we investigate SAM v1 ViT-B (smaller, faster) instead of ViT-H for the detection pass, saving ViT-H quality for the synthetic pass?
