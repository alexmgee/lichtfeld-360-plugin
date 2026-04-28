# Dual Fisheye Pipeline Spec — Interrogation Findings (Round 2)

**Spec under review:** [docs/superpowers/specs/2026-04-22-dual-fisheye-pipeline-design.md](superpowers/specs/2026-04-22-dual-fisheye-pipeline-design.md)
**Date:** 2026-04-27
**Round:** Second pass; first pass produced commit `9bc9fec docs: fix spec issues from review`.

## Context

Line-by-line interrogation of the dual-fisheye spec, validating every claim against (a) the codebase, (b) external references the spec cites, (c) the actual installed pycolmap 4.0.2 surface, and (d) the underlying math. Verification commands and source tags appear inline (`[read: path]`, `[ran: command]`).

Findings are grouped by type:
- **§A. Factual errors** — verifiable as wrong; will produce wrong output if implemented as written
- **§B. Internal contradictions** — spec contradicts itself or referenced files
- **§C. Missing details** — important implementation steps not specified
- **§D. Risky assumptions** — claims that may hold but are unverified
- **§E. Scope ambiguity** — unclear what is or isn't in scope
- **§F. Approved-decision pushback** — challenges to design decisions in §2

Sections marked **[expanded]** were rewritten in round 2 with more detail per user request. New material is **[new]**.

---

## Open clarifications

### Q3 — `transforms_writer.py` vs new `fisheye_transforms.py` location

**Status:** Unresolved. Needs investigation. The spec's §4.8, §7, and §8 don't agree:

- §4.8 paragraph 5: "either in `transforms_writer.py` or in a new `fisheye_transforms.py`" (open)
- §4.8 final paragraph: "needs a new function ... not an extension of the existing `write_transforms_json`" (must be new function, but where?)
- §7 (New Files): only `paired_extractor.py` and `fisheye_circle_mask.py` listed (no `fisheye_transforms.py`)
- §8 (Modified Files): `transforms_writer.py` listed with the change "Fisheye transforms output with per-frame intrinsics and k1-k4"

The factual question to investigate before resolving:

1. Does the existing `transforms_writer.write_transforms_json` need to keep working unchanged for the ERP path? [verified: yes — see B1 below for ERP path code]
2. If yes, is it cleaner to add a sibling function `write_fisheye_transforms` next to it (one file, two functions), or split into a new file?
3. Does `scaffold.py` (which already writes its own transforms_data dict at line 310) belong with `write_transforms_json` or with `write_fisheye_transforms`? Currently it duplicates the JSON-writing logic.

A clean split that avoids duplication would have a single `transforms_writer.py` with three functions: `write_transforms_json` (ERP minimal), `write_erp_scaffold_transforms` (extracted from scaffold.py), `write_fisheye_transforms` (new). But that's a refactor beyond this spec's scope. Easier: add `write_fisheye_transforms` to `transforms_writer.py` as a sibling function, accept the duplication. Spec should commit to one or the other.

### Q6 — Insta360 priors formula

**Status:** Need to figure out before merge. The spec at §5 says:

> Initial priors can use reasonable fisheye defaults (f ~= image_width * 0.27 for ~190 deg FOV, centered principal point, zero distortion)

Two checks:

**Check 1 — does the formula produce a sensible value?**
For a typical Insta360 X3/X4 fisheye image of 5760 px wide:
- Spec heuristic: `f = 5760 * 0.27 = 1555 px`
- Equidistant projection (`r = f * theta`): for image radius ≈ 5760/2 = 2880 px and half-FOV = 95° = 1.658 rad → `f = 2880 / 1.658 = 1737 px`

The two methods differ by ~12%. For Insta360 ONE X (5760×2880 stitched ERP) or ONE X2 (5760×2880), the actual single-fisheye image isn't 5760 wide — it's typically 2880×2880 per lens (Insta360 stores two side-by-side fisheye circles). So the right input to the formula isn't the file width, it's the per-lens image width. With per-lens 2880×2880:
- Spec heuristic: `f = 2880 * 0.27 = 778 px`
- Equidistant: `f = 1440 / 1.658 = 869 px`

Still ~12% apart. The spec heuristic underestimates focal length compared to the equidistant projection model. Either the heuristic is for a different projection (equisolid, stereographic), or it's just a rough rule of thumb.

**Check 2 — what are actual Insta360 numbers?**
[read: docs/specs/SCAFFOLD_IMPLEMENTATION.md:406-411] documents Insta360 stitching defaults:
```python
"insta360": {
    "ih_fov": 190,
    "iv_fov": 190,
    "yaw": -90,
    "roll": 180,
}
```
This describes stitching geometry, not COLMAP priors. The actual fisheye intrinsics for Insta360 cameras are not in any document I can find in this repo. They're embedded in the .insv proprietary trailer per [docs/specs/SCAFFOLD_IMPLEMENTATION.md:381-386].

**Recommendation:** Drop the formula from §5 and replace with a placeholder until a real Insta360 frame is calibrated. If the spec wants a v1 fallback, use the equidistant formula explicitly (`f = (image_width / 2) / radians(half_FOV_deg)`) with a note that it's a starting prior and BA must refine.

The same applies to handling: what if the user picks an Insta360 file but no priors exist? Three options:
1. Refuse the file with a clear error message ("Insta360 priors not yet calibrated — provide --fisheye-calibration").
2. Use the rough formula and warn.
3. Use COLMAP's `default_focal_length_factor` (the existing pinhole fallback) and let BA fully solve.

Spec needs to pick.

---

## §A. Factual Errors

### A1. .osv stream order is reversed in §4.3 [HIGH]

**Spec §4.3 step 1:** "ffmpeg extracts stream 0 (front) and stream 1 (back) from the single container into two temporary video files"

**Reality** [read: docs/dual-fisheye-osv-integration-report.md:111-114]:
```
Stream 0: HEVC 3840x3840 50fps 10-bit  — back lens (away from operator)
Stream 1: HEVC 3840x3840 50fps 10-bit  — front lens (toward operator)
```

[read: docs/dual-fisheye-osv-integration-report.md:129] also documents the convention used by the existing `OSVHandler`: *"Assigning front/back by stream index order (lower = back, higher = front)."*

If implemented literally, every "front" frame is actually back and vice versa. SAM3 operator masking targets the photographer/handle visible in the front lens — with streams swapped, the mask would land on the back, leaving the photographer un-masked in the front frames. Calibration priors (front f=1047.9 vs back f=1044.9) get applied to the wrong lens. Rig pose 180° around Y becomes meaningless because the assumed "front" actually views the rear hemisphere.

This is a single-character spec edit but a silent failure mode — front and back streams have nearly identical motion and dimensions, so no obvious failure signal at runtime.

### A2. .insv rear-lens filename pattern is wrong [HIGH]

**Spec §4.3 step 1:** ".insv: locates the paired `*_00_*.insv` + `*_01_*.insv`"

**Reality** [read: docs/specs/SCAFFOLD_IMPLEMENTATION.md:369-379]:
```
Older cameras (ONE X, ONE X2, X3): Two separate .insv files per recording.
Front lens has _00_ in the filename, rear lens has _10_. Example:
  VID_20231129_224118_00_013.insv   (front)
  VID_20231129_224118_10_013.insv   (rear)

Newer cameras (X4, X5): Single .insv file with two video tracks
  Track 0:v:0 = front fisheye
  Track 0:v:1 = rear fisheye
```

Two errors:
1. The rear file is `_10_`, not `_01_`. A pattern matcher using `_01_` will not find any Insta360 files on disk.
2. The spec entirely misses the X4/X5 single-file case (which is structurally the same as `.osv`). For those models, the spec's pair-detection logic doesn't apply at all — they need the demux path.

### A3. fisheye_circle_margin units mismatch — default value is wrong [HIGH]

**Spec §4.2 PipelineConfig:** `fisheye_circle_margin: float = 0.06  # Circle mask margin (6% default)`

**Spec §4.4 line 104:** "The math: `r_valid = min(w, h) / 2 * (1 - margin / 100)`"

**Source function** [read: d:/Projects/reconstruction-zone/prep360/core/fisheye_reframer.py:1028-1067]:
```python
def generate_fisheye_circle_mask(width, height, fov_degrees=190.0, margin_percent=0.0):
    ...
    r_valid = r_full * (1.0 - margin_percent / 100.0)
```

The source function expects `margin_percent` in percent units (e.g., `5.0` for 5%), not a fraction. The spec's default of `0.06` paired with the `/ 100` in the formula gives `1 - 0.06/100 = 0.9994` — an effective trim of 0.06%, not 6%. The intended 6% trim requires either:

- Default = `6.0` (matching the formula), or
- Drop `/ 100` from the formula and keep default = `0.06` (matching a fraction).

The spec's two phrasings ("~5-6% default" in §2 item 7, and `(1 - margin / 100)` in §4.4) are mutually consistent only if the default is `6.0`.

### A4. Mask-combination polarity bug in §4.4 [HIGH]

**Spec §4.4:** "combined with the SAM3 operator mask via `np.maximum(mask, circle)` for each frame."
**Spec §4.4:** "draw a filled white circle on a black canvas, invert so 0=valid, 1=masked"

**Existing pipeline mask convention** [read: core/overlap_mask.py:90]:
```python
mask = ((closest == idx) * 255).astype(np.uint8)   # 255=own (valid), 0=other (excluded)
```

**Existing combination operation** [read: core/pipeline.py:601]:
```python
combined = cv2.bitwise_and(operator_mask, voronoi_mask)
```

The plugin uses **0=excluded / 255=valid** (the convention COLMAP expects). `bitwise_and` on this scale gives intersection of valid regions: a pixel is valid only if both operator AND voronoi say valid.

The proposed circle mask uses **0=valid / 1=masked** (inverted polarity, lifted directly from `generate_fisheye_circle_mask`'s docstring at fisheye_reframer.py:1036). Mixing the two conventions through `np.maximum`:
- operator mask values: 0 (excluded) or 255 (valid)
- circle mask values: 0 (valid) or 1 (masked)
- np.maximum returns max of each pair pixel-wise

Result is meaningless: a 255 from "operator says valid" looks like "valid" to COLMAP, but a 1 from "circle says masked" looks like ≈0 (still excluded) to COLMAP — wrong direction. And np.maximum of 0 and 0 (operator-excluded + circle-valid) = 0, which COLMAP reads as "excluded" — but the operator said this pixel is excluded for a real reason (e.g., the photographer is here), which conflicts with circle-valid. The polarity gets stripped along with the bit depth.

Two fixes needed:
1. Convert the circle mask to the codebase's convention before combining: `circle_for_colmap = 255 * (1 - circle)` (where `circle` is 0=valid/1=masked, output is 0=excluded/255=valid).
2. Use `cv2.bitwise_and` (intersection of valid regions) rather than `np.maximum`.

The integration report flags this exact conversion at [docs/dual-fisheye-osv-integration-report.md:608]: *"multiply by 255 and invert."* The spec did not absorb that instruction.

### A5. cam_from_rig_translation sign is wrong [HIGH] **[expanded]**

**Spec §4.5:** Back-camera entry includes `"cam_from_rig_translation": [0, 0, 0.025]` (positive Z).

**Multi-method verification:**

#### Method 1 — Algebraic derivation [ran: numpy script]

```python
import numpy as np
# Setup: rig coordinate system anchored at front sensor, OpenCV convention.
# Rig +Z = front camera viewing direction.
# Back is physically BEHIND front (per empirical measurement Z=-24.9 mm
# in [osmo360_rig_calibration_report.md:61]).
p_back_in_rig = np.array([0.0, 0.0, -0.025])   # back position in rig
R_y_180 = np.diag([-1.0, 1.0, -1.0])           # 180 deg around Y

# cam_from_rig.translation = position of rig origin in cam frame.
# Rig origin (= front sensor at [0,0,0]) expressed in back's frame:
#   vec from back to front = [0,0,0] - p_back_in_rig = [0,0,+0.025]
#   apply R_back_from_rig (= R_y_180):
v_back_frame = R_y_180 @ np.array([0, 0, +0.025])  # = [0, 0, -0.025]
```

Result: cam_from_rig translation should be `[0, 0, -0.025]`, not `[0, 0, +0.025]`.

#### Method 2 — Sanity check against pycolmap's `tgt_origin_in_src` [ran: pycolmap]

`Rigid3d.tgt_origin_in_src()` decomposes the transform: for a `cam_from_rig` rigid transform, this returns "cam origin expressed in rig frame" — i.e., where the camera physically sits in rig coordinates.

```python
import pycolmap as p, numpy as np

# spec's value: 180Y rotation + translation [0, 0, +0.026]
T_spec = p.Rigid3d()
T_spec.rotation = p.Rotation3d(np.array([0.0, 1.0, 0.0, 0.0]))   # 180Y in [x,y,z,w]
T_spec.translation = np.array([0.0, 0.0, 0.026])
print(T_spec.tgt_origin_in_src())  # → [0, 0, +0.026]

# corrected value: 180Y rotation + translation [0, 0, -0.026]
T_fix = p.Rigid3d()
T_fix.rotation = p.Rotation3d(np.array([0.0, 1.0, 0.0, 0.0]))
T_fix.translation = np.array([0.0, 0.0, -0.026])
print(T_fix.tgt_origin_in_src())   # → [0, 0, -0.026]
```

Verified output:
- With spec's `+0.026` translation, `tgt_origin_in_src` returns `[0, 0, +0.026]` — meaning **the back camera is at rig position +26 mm in rig +Z**. Rig +Z = front camera viewing direction = "forward of front." So the spec places the back camera **in front of** the front camera. That is physically wrong (the two lenses face opposite directions, with the back lens behind the front along the device's long axis).
- With corrected `-0.026`, `tgt_origin_in_src` returns `[0, 0, -0.026]`. Back camera 26 mm behind front. **Matches reality.**

This is the definitive convention test: pycolmap's own decomposition method shows the spec's translation produces an impossible physical layout.

#### Method 3 — Match against empirical measurement

[read: osmo360_rig_calibration_report.md:61]: *"Offset vector in front camera frame: X=-0.3 mm, Y=+0.4 mm, **Z=-24.9 mm**."*

The Metashape lens-offset tool measured the back camera's position in front camera frame (= rig frame, since front is the reference sensor) and got **negative Z**. That matches Method 1 and Method 2. The spec's positive Z contradicts the empirical measurement that the spec's own §5 cites as the source.

#### Why the calibration report ships the wrong sign anyway

[read: osmo360_rig_calibration_report.md:94]: the JSON example shows `"cam_from_rig_translation": [0, 0, 0.026]` — same sign error.
[read: osmo360_rig_calibration_report.md:73]: explanatory prose: *"Back sensor translation: [0, 0, 0.026] meters in the front sensor's coordinate frame (+Z = forward from front lens)."*

The phrasing "forward from front lens" suggests the author was thinking about the magnitude of the baseline along the device's long axis, not the COLMAP `cam_from_rig` semantics. The wrong sign is in the *output JSON example* but Section 2.2 of the same report has the correct measurement. The error is in the conversion from "back is at -24.9 mm in front frame" to "cam_from_rig_translation field value." Conversion direction was missed.

Critically, [osmo360_rig_calibration_report.md:5.1] explicitly states the rig was discovered by running COLMAP **without** rig constraint, then computing relative poses post-hoc. So no working COLMAP-with-rig-config pipeline ever consumed the JSON in §3.1. The sign error has not been validated end-to-end.

#### Impact estimate

Implementing the spec as written would:
1. Place the back camera 50 mm offset from its true position (twice the 25 mm baseline magnitude, opposite direction).
2. With `ba_refine_sensor_from_rig=False`, BA cannot correct the rig.
3. BA *can* refine each per-camera world pose and 3D point positions to minimize reprojection error. The most likely outcome: the entire reconstruction gets globally rescaled by some compromise factor, producing a self-consistent reconstruction at the wrong real-world scale. Some rotational/skew artifacts are also possible if BA can't find a clean global solution.
4. Downstream Gaussian Splatting training receives a mis-scaled scene. Without ground-truth scale references, the user might never notice — until they try to overlay measurements or merge with other scans.

#### Recommendation

Hard-fix in the spec to `[0, 0, -0.025]` (or whatever exact baseline value is decided — see B3) before any code is written. Update the calibration report's JSON example correspondingly. Add a one-time validation step in CI that constructs the rig from the JSON and asserts `tgt_origin_in_src[2] < 0` for the back sensor.

### A6. `match_vocab_tree` does not exist in pycolmap 4.0.2 [MEDIUM] **[expanded]**

**Spec §4.7:** "The existing Matching Strategy dropdown adds **Vocab Tree** as a third option ... vocab tree requires a vocabulary tree file."

**Existing code** [read: core/colmap_runner.py:651-656]:
```python
elif matcher_name == "vocab_tree":
    if hasattr(pycolmap, "VocabTreePairingOptions"):
        vocab_opts = pycolmap.VocabTreePairingOptions()
        _try_set_attr(vocab_opts, "num_images", 100)
        _try_set_attr(vocab_opts, "num_nearest_neighbors", 5)
        match_kwargs["pairing_options"] = vocab_opts
    fn_name = "match_vocab_tree"
```

**Verified pycolmap 4.0.2 surface** [ran: `python -c "import pycolmap; print(hasattr(pycolmap, 'match_vocab_tree'), hasattr(pycolmap, 'match_vocabtree'))"`]:
```
False True
```

The function is named `match_vocabtree` (no underscore between "vocab" and "tree"), not `match_vocab_tree`. The existing code calls `getattr(pycolmap, "match_vocab_tree", None)` at runtime [core/colmap_runner.py:404], gets `None`, and the wrapper raises `pycolmap.match_vocab_tree is not available in this build` — the user sees a wrong error message ("not available" when actually the *name* is wrong).

**Compounding bug:** Even with the correct function name, `vocab_tree_path` is never set on `VocabTreePairingOptions`. [ran: `print(p.VocabTreePairingOptions().vocab_tree_path)`] → `WindowsPath('.')`. Default is the current directory; pycolmap will fail to load a vocab tree from `.`.

**Required fixes (independent of this spec):**
1. Change `fn_name = "match_vocab_tree"` to `fn_name = "match_vocabtree"` at line 656.
2. Wire a `vocab_tree_path` config field through `ColmapConfig` → `VocabTreePairingOptions.vocab_tree_path`.
3. Add UI for the path (download button + manual path).

The spec proposes (3) without addressing (1) and (2). Shipping a "Vocab Tree" UI option that calls a non-existent function on a config with no tree path is strictly worse than not shipping it at all — the user gets a confusing error and no way forward.

**Recommended sequencing:**
- Land (1) and (2) as a separate small PR before this spec's UI changes.
- Or: include them in this spec's §4.7 changes with explicit callouts and verification steps.

### A7. `constant_rigs=True` is the wrong type [MEDIUM] **[expanded]**

**Spec §4.6:** "`constant_rigs=True` — rig does not change during reconstruction (already present)"

**Existing code** [read: core/colmap_runner.py:710]: `_try_set_attr(pipeline_opts, "constant_rigs", True)`

**Actual pycolmap field type** [ran: `python -c "..."`]:
```
default constant_rigs: set() <class>
after setting to {0,1}: {0, 1}
error setting True (expected): TypeError: incompatible function arguments. ...
    1. (self: pycolmap._core.IncrementalPipelineOptions, arg0: collections.abc.Set[typing.SupportsInt]) -> None
```

The field is `Set[int]` — a set of rig IDs to mark constant during BA. Passing `True` raises TypeError. `_try_set_attr` swallows it. The line is a no-op.

**What actually locks the rig today:** `ba_refine_sensor_from_rig=False` at line 706. That single flag prevents BA from refining any sensor's pose relative to its rig — globally, for all rigs. It is sufficient for the existing ERP path (where the only rig is the virtual ERP rig).

**For the dual-fisheye path:** `ba_refine_sensor_from_rig=False` is also sufficient — same global flag, applied to the new dual-fisheye rig. No `constant_rigs` set is actually needed.

**Recommended fix:**
- Either remove the misleading line entirely and document that `ba_refine_sensor_from_rig=False` does the work, **or**
- After `apply_rig_config`, query the database for the rig ID(s) and set `pipeline_opts.constant_rigs = {rig_id, ...}` properly. This adds defensive belt-and-suspenders but is more code.

The spec inherits the misleading line and treats it as load-bearing. It should pick a fix and update §4.6 to match.

### A8. Pre-existing bug in sequential matcher pairing [MEDIUM] **[new — found during round 2]**

**Existing code** [read: core/colmap_runner.py:644-649]:
```python
if matcher_name == "sequential":
    if hasattr(pycolmap, "SequentialMatchingOptions"):
        pairing_opts = pycolmap.SequentialPairingOptions()
        _try_set_attr(pairing_opts, "loop_detection", True)
        match_kwargs["pairing_options"] = pairing_opts
    fn_name = "match_sequential"
```

[ran: `python -c "import pycolmap; print(hasattr(pycolmap, 'SequentialMatchingOptions'), hasattr(pycolmap, 'SequentialPairingOptions'))"`]:
```
False True
```

The `hasattr` guard checks `SequentialMatchingOptions` — a class that does not exist in pycolmap 4.0.2. The body of the if-block instantiates `SequentialPairingOptions` (which does exist) and sets `loop_detection=True`. But the body never executes because the guard is always False.

**Effect:** Sequential matching runs with default pairing options. `loop_detection=True` is never applied. For 360 video this matters — loop detection is what catches when the camera comes back to a previous location, closing the trajectory loop. Without it, long tracks may drift.

The spec doesn't mention sequential matching directly, but it inherits this bug. In the dual-fisheye path, sequential matching is the default for ordered-frame video.

**Recommended fix:** Change line 645 to `if hasattr(pycolmap, "SequentialPairingOptions"):` (matching what's actually instantiated). One-line fix in colmap_runner.py.

### A9. IMPLEMENTATION.md does not exist [LOW]

**Spec §3, §9:** "(Phase 1 in IMPLEMENTATION.md)"

[ran: `find ... -iname 'IMPLEMENTATION*'`]: no `IMPLEMENTATION.md` in the tree. The intended file is [docs/specs/SCAFFOLD_IMPLEMENTATION.md](specs/SCAFFOLD_IMPLEMENTATION.md), which has a "Phase 1: ERP Scaffold Output Mode" matching the reference. Either fix the path in the spec or rename the file.

---

## §B. Internal Contradictions

### B1. New file vs modified file for the fisheye transforms writer

See Q3 above. Pick one and propagate the choice through §4.8, §7, §8.

### B2. "Approach B" terminology collision

§2 item 3 calls "Approach B" the structural choice of `PipelineJob` conditional branching.
[docs/dual-fisheye-osv-integration-report.md:643-704] uses "Approach A/B/C" for *pipeline strategies* — Approach A = raw fisheye → COLMAP fisheye (which is what this spec actually implements); Approach B = fisheye → reframe → COLMAP pinhole; Approach C = stitch → SphereSfM.

The spec implements integration-report's **Approach A** but labels its structural decision as "Approach B" (a different B, referring to code architecture). Anyone cross-reading the two documents will be confused. Rename the structural option (e.g., "Option 2: conditional branching") to avoid the collision.

### B3. Baseline 25 mm vs 26 mm

§2 item 6 / §4.5 / §5: "Locked at 25mm baseline" / `[0, 0, 0.025]`.
[osmo360_rig_calibration_report.md:17, line 284]: summary says "26 mm"; recommended priors section says "26 mm along +Z."
[osmo360_rig_calibration_report.md:54-65]: empirical measurements all give 25.0 mm (24.9 via lens offset tool).

The spec's 25 mm matches the actual measurements. Fine. But the calibration report is internally inconsistent (summary rounds 25.0 → 26) and the spec doesn't note the discrepancy. One sentence in §5 noting that the 25 mm measured value supersedes the calibration report's "26 mm" recommendation would close this.

### B4. Sharpness scoring assumption contradicts proven prior art [expanded]

**Spec §4.3 step 2:** "scores frames from one stream only (front), since both streams share identical motion blur from the rigid body"

**Why the rationale is incomplete:**

A frame's sharpness score is the result of *several* factors, only one of which is shared between the two lenses on a rigid body:

| Factor | Shared between front and back? |
|---|---|
| Global motion blur (camera moving fast) | **Yes** — both lenses move together |
| Rotational motion blur | **Yes** — same angular velocity |
| Focus | **No** — Osmo 360 / Insta360 lenses have separate (typically fixed but per-unit varying) focus |
| Exposure noise (low-light grain) | **No** — one lens may face the sun, the other into shadow → different sensor gain → different noise floor → different gradient magnitudes |
| Local motion (a moving object) | **No** — visible only on the lens facing it |
| Lens contamination (water drop, smudge) | **No** — happens to one lens at a time |
| Operator/equipment in frame | **Mostly front** — operator's hand, monopod, etc. typically obstruct front |
| Rolling shutter shear | **Mostly shared** — but a fast pan can produce different shear amounts per lens depending on orientation |

The spec's claim is that *motion blur* is identical, which is true. But **sharpness score** is not just motion blur. The Tenengrad / Laplacian metrics measure overall image gradient magnitude, which integrates all of the above factors.

**What the proven prior art does** [read: docs/dual-fisheye-osv-integration-report.md:329-333]:

> Best is ranked by:
>   1. `min(front_score, back_score)` — ensures both lenses are sharp
>   2. `(front_score + back_score) / 2` — tiebreaker
>   3. `-abs(front_score - back_score)` — tiebreaker

Critically, the prior art's `PairedSplitVideoExtractor` (637 lines, used in the existing reconstruction-zone GUI) ranks pairs by `min(front, back)`. This means a candidate frame is only chosen if **both** lenses pass a sharpness threshold. If the back lens has a smudge for a few seconds, the prior art rejects those frames. The spec's "score front only" approach does not — it would happily select a pair where the front is sharp but the back is unusable.

**What can go wrong if only the front is scored:**

- *Differential autofocus drift:* Insta360 X4 has fixed-focus lenses but with per-unit variation; one lens may be slightly mis-focused. The spec's approach picks based on the in-focus lens, leaving the other always blurry.
- *Operator partially blocks front lens, doesn't block back:* the front score includes the operator's clothing (low gradient) reducing it. The spec might reject these frames as "front blurry," missing the fact that the back is fine. *Worse case:* the operator carries a high-contrast object that artificially raises front score; the spec selects these frames; back is mediocre but never measured.
- *Front faces a flat low-contrast surface, back has rich texture:* spec rejects (front low score), wasting good back data.

**Practical fix:** read both streams in the scoring pass, score both. Use `min(front, back)` as the selection criterion (matches prior art). The performance cost is roughly 2× the scoring time — but the existing `PairedSplitVideoExtractor` already does this in a two-pass architecture and the cost is dominated by I/O, not scoring.

If "score front only" is genuinely a perf-versus-quality tradeoff worth taking, the spec should:
1. Frame it as such (not as "physically equivalent to scoring both").
2. Document the failure modes that get accepted (above).
3. Provide a config flag to opt into dual-stream scoring for users who care.

### B5. PER_FOLDER + per-camera priors [expanded — merged with C1, C3 below]

See the "Calibration prior plumbing" expanded section in §C below. B5 is one symptom of the broader plumbing question.

---

## §C. Missing Details

### Calibration prior plumbing (B5 + C1 + C3 unified) [expanded]

The spec has three loosely related items about calibration that, taken together, leave a gap:

**B5 (the contradiction):** §5 publishes a per-camera calibration table (front f=1047.9, back f=1044.9, etc.) but says immediately after that "a family-average prior is used: f=1046, ...". The per-camera table looks load-bearing but the implementation collapses it to one shared value.

**C1 (the missing dispatch):** §4.6 says "`camera_params` receives the per-camera-family intrinsic priors as a formatted string" but doesn't say *who* computes that string or *where* in the code the family lookup happens.

**C3 (the missing schema):** §4.2 introduces `fisheye_calibration_path: Optional[str]` for an override but doesn't document the JSON format expected.

#### Why these matter together: the data flow

The current pinhole flow:

```
PipelineConfig.preset_name + output_size
   ↓ pipeline.py:621-623
infer_shared_pinhole_camera_params(view_fovs, output_size)
   → returns "fx,fy,cx,cy" string
   ↓
ColmapConfig.camera_params = "fx,fy,cx,cy"
   ↓ colmap_runner.py:502
reader_opts.camera_params = "fx,fy,cx,cy"
   ↓
COLMAP feature_extractor reads it as the prior for ALL cameras in PER_FOLDER mode.
```

What the spec proposes for fisheye but doesn't fully wire:

```
File extension (.osv | .insv)
   ↓ detect_input_type()        ← spec defines this
"dual_fisheye"                   ← input_type field
   ↓ ?                           ← dispatch question (C1)
[some lookup] → camera family   ← family ID dispatch (C1)
   ↓ ?                           ← table lookup (B5)
"1046,1046,1915,1919,0,0,0,0"   ← family-average string
   ↓
ColmapConfig.camera_params       ← same destination as ERP
   ↓
reader_opts.camera_params        ← same plumbing as ERP
   ↓
COLMAP applies to BOTH front and back folders identically.
```

The unfilled boxes:

1. **Family ID dispatch.** `detect_input_type()` returns `"dual_fisheye"` but doesn't say which family (DJI, Insta360, etc.). The detection logic at §4.1 uses extension only — `.osv` → DJI Osmo 360, `.insv` → Insta360. So a parallel `detect_camera_family()` function must exist (or `detect_input_type()` must return both pieces of info). The spec doesn't show this.

2. **Family → priors lookup.** A new module is needed. Options:
   - `core/fisheye_priors.py` with a dict: `{"dji_osmo360": (1046, 1046, 1915, 1919, 0, 0, 0, 0), "insta360": (...)}`.
   - Add to `core/presets.py` (presets are already a per-config table).
   - Hardcode in `pipeline.py` next to `infer_shared_pinhole_camera_params`.
   - A new `infer_fisheye_camera_params(family) → str` function that mirrors the pinhole one.

   The spec doesn't pick. Each option has different testability and discoverability.

3. **Per-camera vs family-average.** The §5 table shows different values per lens but PER_FOLDER mode initializes both folders' cameras with the same `camera_params`. After feature extraction, the database has 2 camera records, both with identical intrinsics. BA then refines them independently — front converges toward 1047.9, back toward 1044.9. So the per-camera distinctions in §5 are **not** applied as priors; they're documentation of what BA *should* converge to from the family average. This is fine but the spec presents it ambiguously.

   **If the per-camera priors ARE intended as load-bearing**, the implementation would have to:
   - Run feature extraction with the family-average prior.
   - Open the database with `pycolmap.Database`, query the two cameras, set front's params and back's params individually.
   - Then run matching and mapping.

   This is more code and the spec doesn't propose it. Either drop the per-camera table from §5 (use family average everywhere, document BA convergence target) or add a paragraph to §4.6 about the database-edit path.

4. **Override JSON schema.** [read: osmo360_rig_calibration_report.md:188-201] shows the existing reconstruction-zone schema:
   ```json
   {
     "camera_model": "DJI Osmo 360",
     "front_rotation_deg": 0.0,
     "back_rotation_deg": 180.0,
     "front": {
       "camera_matrix": [[1158, 0, 1920], [0, 1158, 1920], [0, 0, 1]],
       "dist_coeffs": [0, 0, 0, 0],
       "image_size": [3840, 3840],
       "rms_error": -1.0,
       "fov_degrees": 190.0
     },
     "back": { ... }
   }
   ```
   This is `cv2.fisheye` format (`K` matrix + `D` vector). COLMAP wants a flat string `fx,fy,cx,cy,k1,k2,k3,k4`. The override path needs a converter:
   ```python
   def colmap_params_from_cv2_fisheye(K, D) -> str:
       fx, fy = K[0,0], K[1,1]
       cx, cy = K[0,2], K[1,2]
       k1, k2, k3, k4 = D.flatten()
       return f"{fx},{fy},{cx},{cy},{k1},{k2},{k3},{k4}"
   ```
   The spec needs to either (a) accept this format and convert, or (b) define a simpler COLMAP-native schema and document it. Without the schema, an override-from-JSON UI is impossible to build.

5. **Per-unit override semantics.** If the user provides an override JSON with per-camera values, does that mean:
   - Use these values as priors and let BA refine? (Soft override.)
   - Use these values exactly and skip refinement (`ba_refine_focal_length=False`)? (Hard override.)
   - Use them per-camera (set the database) instead of as the single PER_FOLDER prior?

   Each is a different code path. The spec just says "Optional user-provided calibration file override" without picking semantics.

#### Recommended minimal resolution

For v1:
- Add `core/fisheye_priors.py` with a single function `infer_fisheye_camera_params(family: str) -> str` — symmetric with the pinhole equivalent.
- Have `detect_input_type()` return a tuple `(input_type, family)` or split into two functions; pass `family` into `PipelineConfig`.
- Drop the per-camera table from §5; replace with "family-average prior; BA refines per-camera independently because PER_FOLDER mode creates one camera per folder." Use the table values as a documentation-only reference for "what BA should converge to."
- For the override JSON: define schema as the existing reconstruction-zone format (K+D); add a converter; treat it as a soft override (replaces the family default but BA still refines). Document the schema in §4.2 with an example.
- For per-unit per-lens priors (the database-edit path): defer to a future "factory telemetry" feature (the spec already defers this in §2 item 5).

### C2. Mask path layout for COLMAP's `mask_path` setting

[unchanged from round 1]

§4.4: masks at `extracted/masks/front/`, `extracted/masks/back/`. §4.10: final at `output_dir/masks/front/` etc.

[core/colmap_runner.py:514-516]: `reader_opts.mask_path` is a single root dir. COLMAP expects masks at `<mask_path>/<image_relative_path>.png`. For dual fisheye: `mask_path = output_dir/masks/`, then `images/front/frame_0001.jpg` maps to `masks/front/frame_0001.jpg.png` (typical COLMAP convention; suffix conventions vary by version).

[docs/dual-fisheye-osv-integration-report.md:866] open question: *"does it accept a single shared mask per camera (all images in a folder), or does each image need its own mask file?"* — unverified.

Spec needs to: (1) confirm the COLMAP mask filename convention for the installed version, (2) make explicit that `mask_path` is the parent, not per-folder, (3) say what suffix the mask files use.

### C4. Cancellation behavior for demuxed streams

[unchanged from round 1] — cleanup semantics on cancel/error are unspecified.

### C5. `(input_type, output_mode)` branching mechanics [expanded]

**Spec §4.2:** "The branching in `_run_stages` uses `(input_type, output_mode)` together."

**The existing function** [read: core/pipeline.py:270-746]: 477 lines, sequential, with internal branches:

```
_run_stages(cfg)
├── Stage 1: SharpestExtractor.extract()           [single video → frames]
├── Stage 2+3 (branching):
│   ├── if cfg.masking_method == "sam3_cubemap":   [Sam3CubemapMasker → reframe]
│   ├── elif is_cubemap and masking enabled:        [reframe → mask cubemap faces]
│   └── else (default):                             [mask ERP → reframe]
├── Stage 3.5: Voronoi overlap masks (conditional on enable_overlap_masks)
├── Stage 4: write_rig_config(view_config)         [ERP rig only]
├── Stage 5: COLMAP                                 [ColmapRunner with PINHOLE]
└── Stage 6 (branching):
    ├── if cfg.output_mode == "erp":                [export_erp_scaffold]
    └── else:                                        [pinhole dataset]
```

**What the spec adds.** A second axis (`input_type`) plus a fourth output mode (`fisheye`):

| input_type | output_mode | Path |
|---|---|---|
| erp | pinhole | existing default |
| erp | erp | existing scaffold |
| erp | fisheye | (n/a) |
| dual_fisheye | pinhole | deferred per spec §3 |
| dual_fisheye | fisheye | this spec |
| dual_fisheye | erp | (n/a) |

**Two implementation styles:**

**Style 1 — sprinkle if-checks.** Add `if cfg.input_type == "dual_fisheye": ...` checks throughout the existing function. For example, at stage 1:

```python
if cfg.input_type == "dual_fisheye":
    extractor = PairedExtractor(...)
    extract_result = extractor.extract(cfg.video_path, str(extracted_dir), ...)
else:
    extractor = SharpestExtractor()
    extract_result = extractor.extract(cfg.video_path, str(frames_dir), ...)
```

Pros: smallest diff. Existing tests largely unchanged.
Cons: every stage gets an if/elif. The function bloats from ~500 to ~700+ lines. Stage 3.5 needs an "else if dual_fisheye: skip" carve-out. Stage 4 needs to switch between `write_rig_config` and `write_dual_fisheye_rig_config`. Stage 5's `ColmapConfig` instantiation diverges (different `camera_model`, different `camera_params` source). Stage 6 needs another branch for `fisheye` output. The function becomes hard to follow because each stage asks the same question over and over.

**Style 2 — extract per-(input_type, output_mode) functions.** Refactor `_run_stages` into a top-level dispatcher and four (or fewer) leaf functions:

```python
def _run_stages(self, cfg, t0):
    key = (cfg.input_type, cfg.output_mode)
    if key == ("erp", "pinhole"):
        return self._run_erp_pinhole(cfg, t0)
    elif key == ("erp", "erp"):
        return self._run_erp_scaffold(cfg, t0)
    elif key == ("dual_fisheye", "fisheye"):
        return self._run_fisheye_native(cfg, t0)
    else:
        raise ValueError(f"Unsupported pipeline combination: {key}")
```

Each leaf function has the stages it actually runs, in the order it actually runs them, with no branching. The dual_fisheye path naturally skips the reframer and Voronoi stages because those aren't called from `_run_fisheye_native`.

Pros: each path is readable end to end. Adding the deferred (dual_fisheye, pinhole) path later is just adding `_run_fisheye_pinhole`. Easier to test in isolation (stub each leaf separately).
Cons: bigger initial diff. Some shared logic (cancel checks, progress reporting, result assembly) needs to be factored into helpers to avoid duplication.

**Style 3 — pure stage-based composition.** Define abstract stages (`Extractor`, `Masker`, `Aligner`, `Writer`) and compose them per (input_type, output_mode). This is the cleanest long-term but well beyond this spec's scope; mentioned as the direction Style 2 leads toward.

**Recommendation.** Style 2 for this spec. The existing `_run_stages` is already at the limit of being readable; doubling its branching points (Style 1) would push it over. Style 2 is one larger one-time refactor that makes both this work and future paths cleaner. The spec should commit to Style 2 in §4.2 instead of leaving the implementer to choose.

Concrete diff sketch for Style 2:

```
core/pipeline.py
  - Move existing 270-746 body into 3 private methods on PipelineJob:
      _run_erp_pinhole()    ← lines 280-714 main flow today
      _run_erp_scaffold()   ← variant: 666-714 plus shared prefix
      _run_fisheye_native() ← new
  - Top _run_stages() becomes the dispatcher above (~10 lines).
  - Factor shared stage helpers (extract / mask / colmap-prep / colmap-run / cancel)
    into private methods so each leaf is just orchestration.
```

### C6. ALIKED + LightGlue compound enum dispatch

[unchanged from round 1] — `FeatureMatcherType` is `(SIFT|ALIKED) × (BRUTEFORCE|LIGHTGLUE)`. The spec's UI exposes only the matcher axis; the implementation must combine with `feature_type` to produce the right enum value.

### C7. Vocab tree download UX details

[unchanged from round 1]

### C8. ALIKED ONNX model auto-download path

[unchanged from round 1]

### C9. SAM3 image API integration with PairedExtractor

[unchanged from round 1]

---

## §D. Risky Assumptions

[unchanged from round 1; D1 through D6]

D1. Hardcoded per-unit calibration as per-family default.
D2. PER_FOLDER + 25mm baseline + manufacturing tolerance.
D3. .insv detection by extension only.
D4. SIFT on 190° fisheye without explicit benchmark.
D5. GLOMAP + rig constraint (already self-flagged).
D6. Locked calibration report value flips into spec without re-derivation. *(Resolved by A5 deep dive: the value should be flipped to negative.)*

---

## §E. Scope Ambiguity

### E1. Phase ordering with COLMAP 4.0 work [expanded]

**Spec §4.7:** "[the COLMAP 4.0 capabilities] apply to **both** the ERP and dual fisheye paths."

This single sentence collapses two distinct streams of work that have very different shapes:

**Stream A — Dual fisheye pipeline (this spec's primary purpose):**
- Adds new input handling (.osv / .insv demux + paired extraction).
- Adds new rig config generator.
- Adds a fisheye transforms writer.
- Adds new UI (input mode, fisheye-specific settings).
- Inheritance risk: contained to the dual_fisheye path; ERP remains unchanged.

**Stream B — COLMAP 4.0 feature exposure (cross-cutting, §4.7):**
- Adds Feature Type dropdown (SIFT / ALIKED N16 / ALIKED N32) — applies to **both** ERP and dual fisheye paths.
- Adds Matcher Type dropdown (Bruteforce / LightGlue) — applies to both.
- Adds Mapper dropdown (Incremental / Global) — applies to both.
- Adds Vocab Tree to existing Matching Strategy dropdown — applies to both.
- Inheritance risk: every change touches the existing ERP path; regressions in ERP are possible.

**Why the ordering matters:**

If both streams ship in one PR:
- The PR diff is large (probably ~15–20 files modified, mostly UI + colmap_runner + new files).
- The reviewer has to validate that ERP still works AND that fisheye works AND that all 12 (3 features × 2 matchers × 2 mappers, modulo existing combinations) feature-extraction permutations still work.
- A bug in the COLMAP 4.0 wiring (e.g., the existing `match_vocab_tree` typo from A6) gets exposed to ERP users at the same time as fisheye users.

If COLMAP 4.0 ships first:
- Smaller PR, focused on existing ERP path.
- ERP regression surface is the change surface — easier to bound.
- New dual-fisheye PR builds on top, doesn't have to validate as many permutations.
- Total elapsed time slightly longer (two review cycles), but each cycle is faster.

If the dual-fisheye work ships first (with only existing SIFT + Bruteforce + Incremental):
- Dual-fisheye PR is smaller, doesn't touch the matcher/feature/mapper UI.
- COLMAP 4.0 work follows, applies to both paths.
- Risk: dual-fisheye launches without GLOMAP / ALIKED / LightGlue, which might be the optimization the user wanted.

**The §7/§8 file change tables imply combined.** They list `panels/prep360_panel.py`, `panels/prep360_panel.rml`, and `core/colmap_runner.py` as modified — and the changes listed include both fisheye-specific (paired extraction, rig, fisheye margins) and COLMAP-4.0-specific (feature/matcher/mapper dropdowns, vocab tree). Without explicit phase guidance, an implementer would do both at once.

**Three open questions the spec should answer:**

1. **Sequencing.** Single PR (combined) or two-phase (COLMAP 4.0 first, then fisheye)? The combined option makes review heavier; two-phase makes reviewer life easier.

2. **Fisheye-only-applicable items.** Some COLMAP 4.0 items are fisheye-relevant only:
   - ALIKED's open question 2 (ALIKED on fisheye): only fisheye benchmarks need this.
   - GLOMAP's open question 1 (rig compatibility): only rigged paths need this; ERP rig is virtual but works because of `ba_refine_sensor_from_rig=False`.

   These shouldn't ship in the ERP-only phase if it goes first.

3. **Pre-existing bug fixes (A6, A7, A8).** These affect ERP today (sequential matching's loop_detection broken; vocab tree broken in UI; constant_rigs no-op). They're independent of the spec but get fixed as part of the COLMAP 4.0 surface review. Should they be:
   - Fixed in their own surgical PR before either feature stream?
   - Bundled into the COLMAP 4.0 phase?
   - Left to leak into whichever PR happens to touch nearby code?

**Recommendation.** Three sequenced PRs:
1. Surgical bug-fixes for A6 / A7 / A8 (1 file changed, ~10 lines).
2. COLMAP 4.0 feature exposure for the existing ERP path (UI + colmap_runner; ~5 files).
3. Dual fisheye pipeline (this spec, minus §4.7's COLMAP 4.0 cross-cutting work, which is now done in 2).

Total: smaller PRs, easier reviews, regressions bounded per phase.

If the user wants to ship faster as one PR, that's defensible — but the spec should say so explicitly so the reviewer knows what to expect.

### E2. Insta360 priors TBD: does the path work at all in v1?

See Q6 above for the formula analysis. Spec needs to pick: refuse Insta360 in v1, ship with rough prior + warning, or use COLMAP's default_focal_length_factor.

### E3. ".osv has audio + telemetry streams — what happens to them?"

[unchanged from round 1]

### E4. `keep_streams=True` storage location

[unchanged from round 1]

### E5. detect_input_type for `.360` files

[unchanged from round 1]

---

## §F. Approved-Decision Pushback

[unchanged from round 1; F1 through F5]

---

## Summary

| Category | Count | Highest severity |
|---|---|---|
| §A. Factual errors | 9 | A1, A2, A3, A4, A5 (all HIGH) |
| §B. Internal contradictions | 5 | B4 (sharpness assumption disagrees with proven prior art) |
| §C. Missing details | 9 | Calibration plumbing (B5+C1+C3 unified), C5 (branching mechanics) |
| §D. Risky assumptions | 6 | D5 (GLOMAP + rig untested) |
| §E. Scope ambiguities | 5 | E1 (COLMAP 4.0 phase ordering) |
| §F. Approved-decision pushback | 5 | F2 (long-term maintainability) |

### Definitive findings from round 2

**A5 (translation sign):** Verified three ways — algebraic derivation, pycolmap's `Rigid3d.tgt_origin_in_src` decomposition, and the calibration report's own empirical measurement (Z=-24.9 mm in front frame). All agree: `cam_from_rig_translation` should be `[0, 0, -0.025]`, not `[0, 0, +0.025]`. The pycolmap test is the smoking gun — with the spec's value, the back camera is decomposed to a position in front of the front camera, which is physically impossible.

**A6/A7 + new A8:** Three independent pre-existing bugs in colmap_runner.py:
- `match_vocab_tree` (line 656): wrong name, function doesn't exist; correct is `match_vocabtree`.
- `constant_rigs=True` (line 710): wrong type (Set[int] expected), silently no-ops.
- `SequentialMatchingOptions` hasattr (line 645): wrong class name; correct is `SequentialPairingOptions`. As a result, `loop_detection=True` is never set on sequential matching.

All three are caused by the `_try_set_attr` / `hasattr` permissive pattern that swallows API mismatches without warning. Recommendation: tighten the permissive pattern (warn when expected attrs don't exist) and fix the three known cases as a surgical pre-spec PR.

### What I'd block on before implementation

The errors in §A1–A4 are independent of any design discussion — 1-to-5-line spec edits. Fix before merge.

§A5 has a definitive answer now (negative sign). Update the spec and the calibration report's JSON example.

§A6, A7, A8 are pre-existing bugs that the spec would inherit. Fix them in a surgical pre-spec PR (per §E1's recommended sequencing).

The calibration plumbing (B5/C1/C3) needs one resolved decision before code is written: family-average shared prior + BA-refines (recommended), or per-camera priors via database edit (more code, marginal gain).

§C5 should commit to Style 2 (per-path leaf functions) before implementation starts.

§E1 should pick a phase ordering.

### What is fine to defer

§D items can be tracked as risks during implementation. §F pushbacks are food for design debate but not blockers.

### Verification reproducibility

Every claim in this report is sourced. Quick re-checks:

```bash
# Stream order
grep -n "Stream 0\|Stream 1" docs/dual-fisheye-osv-integration-report.md

# .insv pattern
grep -n "_10_\|_00_" docs/specs/SCAFFOLD_IMPLEMENTATION.md

# Mask convention
grep -n "bitwise_and\|255).astype" core/overlap_mask.py core/pipeline.py

# pycolmap surface (existing bugs)
.venv/Scripts/python.exe -c "import pycolmap as p; print('match_vocab_tree:', hasattr(p,'match_vocab_tree')); print('match_vocabtree:', hasattr(p,'match_vocabtree')); print('SequentialMatchingOptions:', hasattr(p,'SequentialMatchingOptions')); print('SequentialPairingOptions:', hasattr(p,'SequentialPairingOptions')); print('constant_rigs type:', type(p.IncrementalPipelineOptions().constant_rigs).__name__)"

# Translation sign math (algebraic)
.venv/Scripts/python.exe -c "
import numpy as np
R = np.diag([-1.0, 1.0, -1.0])
p_back_in_rig = np.array([0.0, 0.0, -0.025])
v_rig = -p_back_in_rig
print('cam_from_rig translation =', R @ v_rig)
"

# Translation sign verification via pycolmap
.venv/Scripts/python.exe -c "
import pycolmap as p, numpy as np
T = p.Rigid3d()
T.rotation = p.Rotation3d(np.array([0.,1.,0.,0.]))  # 180Y in [x,y,z,w]
T.translation = np.array([0.,0.,0.026])
print('spec value places back camera at:', T.tgt_origin_in_src(), '(should be NEGATIVE if back is behind front)')
"
```

Any of these can be re-run to confirm or refute the findings.
