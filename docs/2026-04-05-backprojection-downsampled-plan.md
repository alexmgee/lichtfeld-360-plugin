# Downsampled Backprojection — Phase 4 Plan

**Date:** 2026-04-05
**Status:** Proposal
**Scope:** Default preset Pass 2 backprojection only
**Prerequisite:** Phase 3 harness is built and working (`dev/backprojection_harness.py`)

---

## Context

Pass 2 backprojection (`_backproject_fisheye_mask_to_erp`) is the single largest remaining cost in the Default masking pipeline. At full ERP resolution (7680×3840), it takes ~5.9s per frame — roughly 95s for a 16-frame clip.

Phase 3's validation harness proved that replacing `pycolmap.img_from_cam()` with inline numpy equidistant math produces bit-identical results but no speedup. The bottleneck is not the projection function — it's the ERP grid construction, trig computation, and matrix rotation over 29.5 million points per frame.

The numpy math replacement is still a valid later cleanup, but it should **not**
be bundled into this experiment. The question in this phase is whether reduced
ERP sampling density is safe. That should be tested in isolation.

## Proposal

Compute the backprojection at reduced ERP resolution, then upscale the binary result mask with nearest-neighbor interpolation.

### Why this works

- The mask is binary (0/1). Nearest-neighbor upscaling avoids grayscale
  interpolation, though it can still introduce coarser silhouette steps at
  boundaries.
- The fisheye mask source is 2048×2048. Backprojecting at full 7680×3840 ERP resolution means ~3.75 ERP pixels per fisheye pixel horizontally — most of those ERP pixels sample the same fisheye pixel. The extra density is wasted.
- At half resolution (3840×1920), the point count drops from 29.5M to 7.4M — a 4× reduction in grid construction, trig, rotation, and projection work.
- The upscale step (`cv2.resize` with `INTER_NEAREST`) is negligible.

### Scale factor choice

| Scale | ERP size | Points | Expected time | Upscale cost |
|-------|----------|--------|---------------|-------------|
| 1.0 (current) | 7680×3840 | 29.5M | ~5.9s | none |
| 0.5 | 3840×1920 | 7.4M | ~1.5s est. | negligible |
| 0.25 | 1920×960 | 1.8M | ~0.4s est. | negligible |

0.5 is the conservative choice — still well above the fisheye mask's own resolution. 0.25 may lose small mask features but could be tested.

---

## Important Scope Constraint

This proposal must account for **both** production Pass 2 branches:

1. direct per-frame backprojection via `_backproject_fisheye_mask_to_erp`
2. shared-map reuse via `_build_backproject_map` when `_synthetic_pass(...)`
   decides directions are stable enough

If the experiment only accelerates the direct path but leaves shared-map
construction untouched, benchmark results may be misleading or behavior may
diverge between clips.

So the experiment must do one of these explicitly:

- apply the reduced-grid idea consistently to both direct backprojection and
  shared-map construction
- or temporarily disable shared-map reuse during the experiment so the results
  are interpretable

The preferred option is the first one.

## Implementation Plan

### Step 1: Add downsampled candidate to the harness

Add a new candidate function `_backproject_downsampled` to `dev/backprojection_harness.py` that:

1. Scales `erp_size` down by a configurable factor (default 0.5)
2. Runs the existing production backprojection math at the reduced resolution
3. Upscales the result with `cv2.resize(..., INTER_NEAREST)` to full ERP size
4. Returns the full-size mask

Run the harness with this candidate alongside the current production implementation. Compare IoU, changed pixels, and runtime.

Do **not** swap in the numpy math replacement in this same harness candidate.
Keep the production projection math unchanged so the experiment isolates only
the grid-density change.

### Step 2: Evaluate harness results

The candidate should:
- Have IoU > 0.99 against the full-resolution reference
- Show a meaningful speedup (target: 3-4× on full-res test)
- Changed pixels should be concentrated near boundaries, not in large interior
  regions
- Keep mask area close to reference (no significant erosion or expansion)

If IoU is too low at 0.5 scale, try 0.75. If the speedup is too small, try 0.25.

Harness evaluation should include:

- fast iteration cases at `3840×1920`
- at least one final signoff case at full `7680×3840`

The reduced-resolution-only harness is not enough for acceptance.

### Step 3: If harness passes, implement in production

Replace or wrap `_backproject_fisheye_mask_to_erp` in `core/masker.py` to use
the downsampled path.

Add a module-level constant for the scale factor:
```python
BACKPROJECT_SCALE = 0.5
```

The production function becomes:
1. Compute reduced ERP dimensions: `(int(erp_w * scale), int(erp_h * scale))`
2. Run existing backprojection math at reduced size
3. `cv2.resize(result, (erp_w, erp_h), interpolation=cv2.INTER_NEAREST)`

If shared-map reuse remains enabled, `_build_backproject_map(...)` must be
updated consistently for the same reduced-grid logic. Otherwise the experiment
should explicitly disable the shared-map branch until both paths are handled.

### Step 4: Validate in LFS

Run the reference Default clip. Compare:
- ERP mask quality (visually)
- Pinhole mask quality
- Registration outcome
- Masking runtime

## Files to modify

| File | Change |
|------|--------|
| `dev/backprojection_harness.py` | Add `_backproject_downsampled` candidate, add test cases at multiple scales |
| `core/masker.py` | Only after harness passes — add scale parameter to `_backproject_fisheye_mask_to_erp`; update or explicitly bypass the shared-map branch during the experiment |

## What NOT to change

- Pass 1 detection logic
- Direction estimation
- SAM2 tracking
- Synthetic fisheye rendering
- Any thresholds or quality parameters
- Reframer
- Pipeline routing
- The numpy math swap in the same patch

## Acceptance criteria

- Harness IoU > 0.99 at chosen scale
- Mask area remains close to reference
- Changed pixels are boundary-localized, not large interior misses
- At least one full-resolution harness signoff case passes
- Reference clip masking runtime drops meaningfully (target: backprojection from ~5.9s/frame to ~1.5s/frame)
- No visible mask quality regression on the reference clip
- Registration unchanged (11/11, 176/176)

## Revert criteria

- IoU below 0.99
- Visible mask artifacts (holes, missing limbs, boundary noise)
- Changed pixels are not boundary-localized
- Shared-map and direct-path behavior diverge in an uncontrolled way
- Registration regression
- Runtime improvement too small to justify the change
