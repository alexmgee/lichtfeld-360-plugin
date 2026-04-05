# Preset-Aware Masking Plan

**Date:** 2026-04-04
**Status:** Draft
**Problem:** The masking pipeline currently runs the same two-pass synthetic camera workflow for every preset. This produces poor results for the cubemap preset, where the ERP backprojection creates sparse, stippled masks that look terrible when reprojected into cubemap pinhole views.

---

## Current Architecture (Broken for Cubemap)

Regardless of preset, every masking run does:

1. **Pass 1:** 16-camera detection layout → YOLO bounding boxes → person direction
2. **Pass 2:** Synthetic fisheye aimed at person → SAM2 video tracking → backproject tracked masks to ERP
3. **Stage 3:** Reframer reprojects ERP masks into whatever pinhole views the preset defines

This was designed for the Default (16-camera FullCircle) preset, where:
- The reconstruction views are at ±35° pitch — exactly where the synthetic camera points
- The ERP mask is a good intermediate because the reconstruction views have dense, overlapping coverage in the same band as the detection/synthetic views
- Temporal consistency from SAM2 tracking is valuable across many overlapping views

For cubemap, this pipeline is wrong:
- Cubemap views are at pitch=0° (horizon) and pitch=±90° (poles)
- The person is typically at pitch=-65° or lower — far from the horizon ring
- The ERP backprojection is sparse and aliased (the stippled pattern in the masks)
- Reprojecting that sparse ERP mask into cubemap views at extreme angles makes it worse
- The synthetic camera pipeline adds huge cost (backprojection is 80s) for a result that's worse than simple per-view detection would be

## Proposed Architecture

Split the masking strategy based on preset type:

### Path A: Direct Per-View Masking (Cubemap)

For presets where the reconstruction views are simple, well-spaced pinhole cameras:

1. For each ERP frame, reframe into the preset's own pinhole views (same as Stage 3 does for images)
2. Run YOLO+SAM v1 `detect_and_segment` directly on each pinhole view
3. Output per-view masks directly — no ERP intermediate, no synthetic camera, no backprojection

**Advantages:**
- Masks are pixel-aligned with the output images (same reprojection)
- No ERP intermediate means no sparse backprojection artifacts
- No synthetic camera pass means no 80s backprojection bottleneck
- SAM v1 segmentation is high quality on well-framed pinhole views
- Much simpler and faster

**Disadvantages:**
- No temporal consistency across frames (each frame is independent)
- If the person is barely visible in a cubemap view (edge of FOV), YOLO may miss them
- Each view runs YOLO+SAM independently — 6 detection+segmentation calls per frame instead of 1 synthetic tracking pass

**When to use:**
- Cubemap preset (6 views, pitch=0°/±90°, 90° FOV)
- Any preset where the reconstruction views provide reasonable direct coverage of the scene

### Path B: Synthetic Camera Pipeline (Default / FullCircle-style)

The current pipeline — keep it as-is for the Default preset:

1. Pass 1: 16-camera detection layout → person direction
2. Pass 2: Synthetic fisheye → SAM2 tracking → backproject to ERP
3. Stage 3: Reframer reprojects ERP masks into preset views

**When to use:**
- Default preset (16 views at ±35°)
- Any preset designed around the FullCircle detection layout

## How to Decide Which Path

The decision should be based on the preset, not on a user toggle. The pipeline already knows the preset name and view geometry. Two approaches:

### Option 1: Explicit preset flag

Add a field to `ViewConfig` or `MaskConfig` that indicates which masking strategy to use:

```python
@dataclass
class MaskConfig:
    ...
    masking_strategy: str = "synthetic"  # "synthetic" or "direct"
```

The pipeline sets this based on the preset name:

```python
if cfg.preset_name == "cubemap":
    mask_cfg.masking_strategy = "direct"
else:
    mask_cfg.masking_strategy = "synthetic"
```

### Option 2: Infer from view geometry

Detect whether the preset's reconstruction views overlap well with the detection layout. If the views are at ±35° (like the detection layout), use synthetic. If they're at 0°/±90° (like cubemap), use direct.

This is more flexible but harder to get right. Option 1 is safer for now.

**Recommendation:** Option 1. Explicit, simple, no risk of misclassification.

## Implementation Plan

### Step 1: Add direct masking method to Masker

Add a `process_frames_direct` method (or refactor `process_frames` with a strategy parameter) that:

1. For each frame:
   a. Read the ERP image
   b. Reframe to each preset view (using the existing remap cache from the reframer, or build its own)
   c. Run `detect_and_segment` on each view via the image backend (YOLO+SAM v1)
   d. Write per-view masks directly to the output mask directory

2. Output structure must match what the reframer currently produces:
   ```
   masks/
     00_00/frame001.png
     00_01/frame001.png
     ...
   ```

This bypasses the ERP mask intermediate entirely. The reframer's mask reprojection step is also skipped — the direct masker writes final per-view masks.

### Step 2: Wire the strategy into the pipeline

In `pipeline.py`, the masking stage currently:
1. Runs `masker.process_frames()` → writes ERP masks to `extracted/masks/`
2. Sets `reframe_mask_dir` so the reframer reprojects them

For the direct path:
1. Run the direct masker → writes per-view masks directly to the final `masks/` directory
2. Set `reframe_mask_dir = None` so the reframer does NOT try to reproject ERP masks (there are none)

### Step 3: Handle the output directory structure

The direct masker needs to write masks in the same `masks/{view_name}/{frame_stem}.png` layout that the reframer currently produces. This means the direct masker needs access to:
- The preset's view list (names, yaw, pitch, fov)
- The output directory path (same parent as `images/`)
- The output size

### Step 4: Per-view dilation

The reframer currently applies per-view dilation (erode the KEEP region) after reprojecting ERP masks. The direct masker should apply the same dilation after segmentation, so the mask edge treatment is consistent regardless of path.

## What Changes Per File

| File | Change |
|------|--------|
| `core/masker.py` | Add `process_frames_direct()` or strategy parameter. Direct path: reframe → detect_and_segment per view → dilate → write per-view masks |
| `core/pipeline.py` | Choose masking strategy based on preset. Direct path skips ERP mask writing and reframer mask reprojection |
| `core/presets.py` | Optionally add `masking_strategy` field to `ViewConfig` |

## What Does NOT Change

- The Default preset workflow — completely untouched
- The detection layout (`DETECTION_LAYOUT`) — still used for Pass 1 direction estimation in the synthetic path
- The reframer's image reprojection — unaffected
- COLMAP alignment — unaffected (masks are masks regardless of how they were produced)
- The `_SubstageTimer` instrumentation — add new labels for the direct path

## Mask Polarity

Both paths must produce masks with the same polarity:
- **COLMAP polarity:** white (255) = keep, black (0) = remove
- The current ERP→reframer path inverts in `process_frames` (line: `inverted = ((merged == 0).astype(np.uint8)) * 255`) then the reframer applies per-view dilation
- The direct path must also produce white=keep, black=remove after detection and dilation

## Performance Impact

For cubemap (6 views, N frames):

| | Current (synthetic) | Direct |
|---|---|---|
| Pass 1 detection | 16 YOLO calls/frame (batched) | 0 (skipped) |
| Pass 2 synthetic render | N fisheye renders | 0 (skipped) |
| Pass 2 SAM2 tracking | 1 propagation call | 0 (skipped) |
| Pass 2 backprojection | N × 5s = dominant cost | 0 (skipped) |
| Per-view detection | 0 | 6 × YOLO+SAM v1 per frame |
| Reframer mask reprojection | 6 × N remaps | 0 (skipped) |

The direct path trades the expensive synthetic pipeline (dominated by 80s backprojection) for 6 × YOLO+SAM v1 calls per frame. SAM v1 segmentation at 1920px is roughly 0.5-1s per view, so 6 views ≈ 3-6s per frame. For 51 frames that's ~150-300s of detection — slower per-frame than the batched YOLO-only Pass 1 (which was 2s/frame), but it completely eliminates the 80s backprojection and produces clean, pixel-aligned masks.

Net effect: slower masking detection but much higher quality masks and no backprojection bottleneck.

## Open Questions

1. **Should the direct path use SAM v2 if available?** SAM v2's video predictor could track across frames for temporal consistency on each cubemap view. But that would require running SAM v2 separately for each of the 6 views — 6 tracking passes. Probably not worth it for v1. Keep it simple: YOLO+SAM v1 per-frame per-view.

2. **Should we cache the reframe remap tables between the masker and the reframer?** Both stages reframe ERP→pinhole with the same geometry. Currently they each build their own cache. A shared cache would avoid building tables twice, but adds coupling between stages. Low priority.

3. **What about custom/future presets?** The strategy decision needs to handle presets that aren't "cubemap" or "default". A reasonable heuristic: if the preset has views at pitch > 60° or pitch < -60° (pole-adjacent), use direct. Otherwise use synthetic. But for now, explicit preset-name matching is simpler and safer.
