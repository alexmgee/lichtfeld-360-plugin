# Cubemap Masking Optimization v1 Execution Plan

**Date:** 2026-04-04  
**Status:** Draft  
**Supersedes:** `2026-04-04-preset-aware-masking-plan.md`  
**Scope:** `cubemap` preset only

---

## Goal

Replace the current cubemap masking path with a cubemap-specific direct per-view masking flow that:

- avoids ERP backprojection artifacts
- avoids the synthetic fisheye + SAM2 path
- writes final per-view masks directly
- leaves the Default preset completely unchanged

---

## In Scope

- `cfg.preset_name == "cubemap"`
- cubemap image masking only
- direct per-view YOLO+SAM v1 masking on final cubemap images
- Pass 1 direction estimation reused as a gating hint
- cubemap-only pipeline branch

## Out of Scope

- changing the Default preset masking flow
- changing non-cubemap presets
- building a generic preset-strategy framework
- adding SAM2 tracking to cubemap
- in-memory reframer/masker fusion

---

## Desired Final Behavior

### Default preset

No change:

1. Extract ERP frames
2. Mask on ERP frames through the current synthetic path
3. Reframe images and masks
4. Run overlap masks / COLMAP as usual

### Cubemap preset

New behavior:

1. Extract ERP frames
2. Reframe ERP frames into cubemap images only
3. Run Pass 1 direction estimation on ERP frames
4. Use direction to gate which cubemap faces need segmentation
5. Run direct per-view segmentation on gated cubemap images
6. Write final masks directly to `masks/{view_name}/{frame_stem}.png`
7. Run overlap masks / COLMAP as usual

No ERP masks are written for cubemap in this path.

---

## Implementation Tasks

### Task 1: Add a cubemap-only branch in `pipeline.py`

When:

```python
cfg.enable_masking and cfg.preset_name == "cubemap"
```

do this order:

1. Reframe images with `mask_dir=None`
2. Run cubemap masking on the reframed images
3. Continue to overlap masks / rig config / COLMAP

For every other preset, keep the current pipeline order unchanged.

### Task 2: Add a public Pass 1 helper in `masker.py`

Add a public method:

```python
def estimate_person_directions(
    self,
    frames_dir: Path,
    progress_callback=None,
) -> dict[str, np.ndarray | None]:
```

Requirements:

- reuse `_primary_detection()` internally
- do not make `pipeline.py` call `_primary_detection()` directly
- return `frame_stem -> direction or None`
- use the existing 16-view detection layout

### Task 3: Add cubemap face gating in `masker.py`

Add a helper that decides whether a cubemap face should be segmented for a given frame.

Inputs:

- face yaw/pitch/fov
- person direction
- configurable margin

Gate rule:

```python
half_diag_deg = np.degrees(np.arctan(np.sqrt(2) * np.tan(np.radians(fov / 2))))
```

For a 90 degree cubemap face, this is about `54.7°`.

Run segmentation when:

```python
angle_between(face_dir, person_dir) < half_diag_deg + margin_deg
```

Use a conservative margin such as `10°` for v1.

### Task 3.5: Lock down v1 constants

Use these values explicitly for the first implementation:

- `gate_margin_deg = 10.0`
- `pass1_detection_size = min(512, erp_w // 4)` to match the current Pass 1 path
- `per_view_erode_kernel = (9, 9)` elliptical kernel to match the current reframer mask finishing

Do not tune these during implementation unless the validation runs show a clear problem.

### Task 4: Add direct cubemap masking on reframed images

Add a new public method:

```python
def process_reframed_views(
    self,
    images_dir: Path,
    masks_dir: Path,
    person_directions: dict[str, np.ndarray | None],
    views: list[tuple[float, float, float, str, bool]],
    progress_callback=None,
) -> MaskResult:
```

Behavior:

1. Walk `images/{view_name}/*.jpg`
2. For each image:
   - parse `view_name`
   - parse `frame_stem`
   - look up person direction for that frame
3. If direction exists:
   - apply cubemap gate for that face
4. If direction is missing:
   - disable gating for that frame
   - treat all 6 faces as eligible
5. If face is eligible:
   - run `self._backend.detect_and_segment()`
6. Convert output to final COLMAP mask polarity
7. Apply the same per-view erosion used by the reframer
8. Write `masks/{view_name}/{frame_stem}.png`

Clarification:

- `detect_and_segment()` returns a detected-region mask, not a final COLMAP keep mask
- treat backend output as "detected operator region"
- convert that into final mask polarity before writing:
  - detected operator region -> black `0` remove
  - background / keep region -> white `255` keep
- apply the per-view erosion after polarity conversion, matching the current reframer behavior

### Task 5: Always write one mask per output image

For cubemap v1, every output image must have a corresponding mask file.

Rules:

- gated-out face -> write an all-white keep mask (`uint8`, same height/width as the output image, every pixel = `255`)
- gated-in face with no detection -> write an all-white keep mask (`uint8`, same height/width as the output image, every pixel = `255`)
- gated-in face with detection -> write real keep/remove mask

Polarity:

- white `255` = keep
- black `0` = remove

Clarification:

- "all-white keep mask" means a normal final output mask file, not a missing file and not an empty array
- erosion is only meaningful for masks with a remove region; a pure all-white keep mask can be written directly

### Task 6: Keep cubemap on the image backend only

For cubemap direct masking:

- set `MaskConfig.enable_synthetic = False`
- do not initialize the video backend
- do not render synthetic fisheye views
- do not run SAM2 tracking
- do not backproject to ERP

### Task 6.5: Define failure behavior

For v1, if the cubemap direct masking path fails, the pipeline should **fail loudly** rather than silently falling back to the old synthetic cubemap path.

Reason:

- this is an intentional cubemap-only replacement path
- silent fallback would make timing and quality evaluation ambiguous
- failing loudly makes bugs obvious during development and testing

### Task 6.6: Define output ownership

For the cubemap branch:

- Stage 3 owns writing `images/`
- the cubemap direct masking step owns writing `masks/`
- overlap-mask generation runs only after direct cubemap masks exist

There should be no mixed ownership of the final `masks/` directory in the cubemap path.

### Task 7: Add cubemap-specific timing labels

Add lightweight timing so the new path can be measured.

Minimum labels:

- `cubemap_p1_direction`
- `cubemap_gate`
- `cubemap_segment`
- `cubemap_write_mask`

If timing is naturally split by image load / decode, record that too.

---

## Expected File Changes

| File | Required change |
|------|-----------------|
| `core/pipeline.py` | Add the cubemap-only routing branch |
| `core/masker.py` | Add public direction-estimation helper, gate helper, and direct cubemap view masking |
| `core/reframer.py` | No required logic change for v1 |
| `core/presets.py` | No required change for v1 |

---

## Acceptance Criteria

### Functional

- When preset is `cubemap`, the pipeline does not generate ERP masks
- When preset is `cubemap`, the pipeline does not initialize or use the synthetic video backend
- When preset is `cubemap`, masks are written directly to `masks/{view_name}/{frame_stem}.png`
- Every cubemap output image has a corresponding mask file
- Every cubemap mask has the same pixel dimensions as its corresponding output image
- The Default preset still follows the current synthetic ERP workflow unchanged

### Quality

- Cubemap masks no longer show the current stippled ERP-backprojection artifact
- Cubemap masks line up with the final cubemap images
- No obvious false-negative holes near cubemap face boundaries caused by over-tight gating
- Registration quality is not materially worse than the current cubemap path

If validation shows recall loss near face boundaries or corners, the gate should be widened or disabled before pursuing more optimization.

### Performance

- Cubemap masking time is measured separately from the Default preset
- The current backprojection bottleneck is eliminated from the cubemap path
- The new cubemap path is fast enough to be practical on real clips, even when some frames fall back to all 6 faces

---

## Validation Runs

Run at least these cases:

1. Cubemap clip where the operator stays mostly on one or two faces
2. Cubemap clip where the operator crosses between neighboring faces
3. Cubemap clip where Pass 1 misses direction on some frames

For each run, record:

- clip name
- frame count
- output size
- backend used
- whether `sam2._C` is active
- total masking time
- cubemap timing breakdown
- count of frames with missing direction
- count of faces gated in
- registration outcome

---

## v1 Safety Rules

- prefer false positives over false negatives in the gate
- if direction is missing, segment all 6 faces for that frame
- if a face is not segmented, still write an all-white keep mask
- do not touch the Default preset workflow while implementing cubemap v1

---

## Follow-Up Work After v1

Only after cubemap v1 is working and benchmarked:

- fuse reframer output and direct masking in memory
- consider per-face SAM2 tracking if temporal flicker becomes a real issue
- decide whether any other preset deserves a similar direct path

---

## Bottom Line

This is a **cubemap-only optimization plan**.

The implementation target is simple:

- keep Default exactly as it is
- for cubemap, reframe images first and then mask the final cubemap views directly

That is the v1 path to better cubemap masks with less complexity than trying to salvage the current ERP backprojection flow.
