# Dual Fisheye Pipeline Integration

**Date:** 2026-04-22 (updated 2026-05-14)
**Status:** Design approved, updated with COLMAP 4.1 upgrade (Caspar GPU BA, rig fixes) and UX review findings

---

## 1. Problem

The plugin currently only accepts equirectangular (ERP) video as input. Users with dual-fisheye cameras (DJI Osmo 360, Insta360 ONE RS/X4) must stitch to ERP externally before using the plugin. This loses the native fisheye geometry and requires extra tooling.

The goal: accept raw dual-fisheye files (.osv, .insv) directly, feed native fisheye frames through COLMAP with a rig constraint, and output either pinhole crops or fisheye-native transforms for downstream training.

## 2. Design Decisions

These were resolved during brainstorming:

1. **Input detection:** Auto-detect from file extension. User does not choose input type.
2. **Output mode:** User chooses Pinhole or Spherical/Fisheye. Combined with auto-detected input, this gives a four-cell matrix (2 input types x 2 output modes).
3. **Pipeline architecture:** Extend the existing `PipelineJob` with a per-path leaf-functions dispatcher (see §4.2 for implementation details). Not a parallel pipeline, not a single function with intermixed if/elif checks, and not a fully stage-based composition. The "Approach A/B/C" labels in `dual-fisheye-osv-integration-report.md` describe *pipeline strategies* (raw fisheye / reframe / stitch); this design follows that report's Approach A.
4. **Stream retention:** A "Keep demuxed streams" checkbox. Unchecked = temp files cleaned up after extraction. Checked = saved alongside output.
5. **Calibration source:** Hardcoded per-camera-family defaults as COLMAP priors. Optional user-provided calibration file override. No factory telemetry extraction in v1.
6. **Rig constraint (experimental, v1 default off):** v1 runs dual fisheye without rig constraint. Two cameras (front, back) via `CameraMode.PER_FOLDER`, independently solved by BA. The ~10° equator overlap between lenses provides cross-camera features that tie the two solves together.

   A rig-constrained mode is included as an experimental opt-in via `PipelineConfig.use_rig: bool = False`. When enabled, the rig is locked at 25mm baseline, 180° Y rotation, with `ba_refine_sensor_from_rig=False`. The rig path has not been validated end-to-end on real data; the locked baseline value comes from a single-device measurement. Promote to default once validated on production captures (see §9).
7. **Masking:** Same SAM3 backend on fisheye frames. Fisheye circle mask (adjustable margin, ~5-6% default) combined with operator mask.
8. **Camera families:** DJI Osmo 360 (.osv) and Insta360 (.insv) supported. Insta360 priors added incrementally as users test.

## 3. Output Matrix

|                        | Pinhole output              | Spherical/Fisheye output       |
|------------------------|-----------------------------|--------------------------------|
| **ERP input**          | Existing path (working)     | ERP scaffold (incomplete)      |
| **Dual fisheye input** | Not designed yet (see sect. 9) | Fisheye -> COLMAP -> fisheye poses ¹ |

¹ Rig constraint is **optional in v1** (default off). See §2 item 6 and §4.5.

The ERP scaffold cell is a separate effort (Phase 1 in `docs/specs/SCAFFOLD_IMPLEMENTATION.md`). The dual fisheye -> pinhole cell requires a separate design pass for the post-COLMAP reframing step. Neither is part of this spec.

**This spec implements only the dual fisheye -> spherical/fisheye output path.** The dual fisheye -> pinhole path is deferred.

## 4. Architecture

### 4.1 Input Detection

A `detect_input_type()` function determines the input type **and the camera family** from the source file. It returns a tuple `(input_type, family)`:

- `.osv` -> `("dual_fisheye", "dji_osmo360")` — single file, two HEVC video tracks (stream 0 = back, stream 1 = front; verified via ffprobe in `dual-fisheye-osv-integration-report.md`)
- `.insv` (older Insta360 ONE X / X2 / X3) -> `("dual_fisheye", "insta360")` — file pair: front lens has `_00_` in the filename, rear lens has `_10_` (e.g., `VID_..._00_013.insv` + `VID_..._10_013.insv`)
- `.insv` (newer Insta360 X4 / X5) -> `("dual_fisheye", "insta360")` — single file, two HEVC video tracks (same structure as `.osv`); detected by absence of an `_10_` sibling
- `.mp4` / `.mov` / other single-stream video -> `("erp", None)`

This lives in `pipeline.py` as a module-level function. The family ID flows into the calibration prior dispatch (see §4.6 and §5).

### 4.2 PipelineConfig Changes

New fields on `PipelineConfig`:

```python
# Auto-detected from source file
input_type: str = "erp"  # "erp" or "dual_fisheye"

# Dual fisheye options
keep_streams: bool = False  # Save demuxed video streams alongside output
fisheye_calibration_path: Optional[str] = None  # Override calibration JSON
fisheye_circle_margin: float = 6.0   # Circle mask margin in PERCENT (default 6%).
                                     # Passed to generate_fisheye_circle_mask(margin_percent=...).
                                     # Formula: r_valid = min(w, h) / 2 * (1 - margin / 100).
camera_family: Optional[str] = None  # Auto-set by detect_input_type():
                                     # "dji_osmo360" or "insta360" for dual fisheye, None for ERP.

# Rig constraint (experimental — see §2 item 6)
use_rig: bool = False  # Enable locked rig constraint for dual fisheye.
                       # Default False until rig path is validated.
```

The existing `output_mode` field is extended: `"pinhole"` / `"erp"` / `"fisheye"`. For dual fisheye with spherical output, `output_mode` is `"fisheye"`. This avoids overloading `"erp"` with two meanings.

The branching in `_run_stages` uses `(input_type, output_mode)` together. Recognized combinations:
- `("erp", "pinhole")` — existing ERP pinhole path
- `("erp", "erp")` — ERP scaffold path (incomplete, not this spec)
- `("dual_fisheye", "fisheye")` — this spec's primary path
- `("dual_fisheye", "pinhole")` — deferred, not this spec

**Implementation:** refactor `_run_stages` into a top-level dispatcher and per-path leaf methods (the existing 477-line function is already at the limit of readability with three internal branches; adding a fourth via tuple-key if/elif checks would push it over):

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

Move the existing 270–746 body into three private methods:
- `_run_erp_pinhole()` — current default flow
- `_run_erp_scaffold()` — current ERP-output variant
- `_run_fisheye_native()` — new

Factor shared helpers (cancel checks, progress callbacks, result assembly) into private methods so each leaf is straight-line orchestration with no internal `input_type` branching. Each leaf is testable in isolation; future paths (e.g., `_run_fisheye_pinhole` for the deferred `("dual_fisheye", "pinhole")` cell) just add one more leaf.

### 4.3 Extraction Stage

**ERP path (unchanged):** `SharpestExtractor` extracts frames from a single video stream.

**Dual fisheye path:** A new `PairedExtractor` class handles:

1. **Demux** — format-specific:
   - `.osv`: ffmpeg extracts stream 0 (**back**) and stream 1 (**front**) from the single container into two temporary video files. Stream order is documented in `dual-fisheye-osv-integration-report.md` and verified via ffprobe; the lower-indexed stream is the back lens.
   - `.insv` (older Insta360 ONE X / X2 / X3): locates the paired `_00_` (front) and `_10_` (rear) files (already separate streams, no demux needed).
   - `.insv` (newer Insta360 X4 / X5): single file with two video tracks; ffmpeg extracts track 0:v:0 (front) and track 0:v:1 (rear), same flow as `.osv`.
2. **Sharpness scoring** — scores frames on **both** streams independently. Pair selection ranks by `min(front_score, back_score)` to ensure both lenses are sharp (matches the proven `PairedSplitVideoExtractor` selection logic in the reconstruction-zone codebase). Tiebreakers: average score, then `-abs(front - back)`. Scoring "front only" is insufficient because sharpness reflects more than motion blur (per-lens focus, exposure noise, occlusion, scene texture, lens contamination — only motion blur is shared between the two lenses on a rigid body).
3. **Paired extraction** — extracts the same frame indices from both streams, maintaining sync
4. **Output layout:**
   ```
   extracted/
     front/       # front lens frames
     back/        # back lens frames
     masks/
       front/     # front lens masks (if masking enabled)
       back/      # back lens masks (if masking enabled)
   ```

The "Keep demuxed streams" checkbox controls whether intermediate demuxed video files persist after frame extraction or are deleted.

**New file:** `core/paired_extractor.py`

### 4.4 Masking Stage

The masking backend (SAM3) is unchanged. The pipeline adjusts which directories it processes:

- **ERP path:** masks `extracted/frames/` -> `extracted/masks/`
- **Dual fisheye path:** masks `extracted/front/` -> `extracted/masks/front/`, then `extracted/back/` -> `extracted/masks/back/`

For dual fisheye, the masker also generates a **fisheye circle mask** — a binary mask of the valid image circle with an adjustable margin (default 6%). The circle mask generation uses the same `generate_fisheye_circle_mask()` function from Reconstruction Zone's `prep360/core/fisheye_reframer.py`, ported to `core/fisheye_circle_mask.py`. The math: `r_valid = min(w, h) / 2 * (1 - margin / 100)`; draw a filled white circle on a black canvas; the source returns a uint8 array with `0 = valid (inside circle), 1 = masked (outside / margin)`.

**Polarity conversion before combining.** The plugin's existing mask convention (per `core/overlap_mask.py:90` and the bitwise_and combination at `core/pipeline.py:601`) is `0 = excluded / 255 = valid` — matching COLMAP's expected polarity. The circle mask must be converted before combining with the SAM3 operator mask:

```python
# circle: shape (h, w), uint8, 0=valid 1=masked  (from generate_fisheye_circle_mask)
# sam_mask: shape (h, w), uint8, 0=excluded 255=valid  (existing convention)
circle_for_colmap = (1 - circle).astype(np.uint8) * 255   # invert + scale: 0=excluded 255=valid
combined = cv2.bitwise_and(sam_mask, circle_for_colmap)   # intersection of valid regions
```

`bitwise_and` (intersection) means a pixel is valid only if **both** the operator mask AND the circle mask say valid. This is the correct combination for "use this pixel for SIFT" semantics. `np.maximum` was incorrect — it mixed the two polarity conventions and produced meaningless output.

The circle mask margin is exposed as `fisheye_circle_margin` on `PipelineConfig` for tuning.

The fisheye masking step must return a `MaskResult` compatible with the existing `PipelineResult` assembly (which reads `backend_name`, `video_backend_name`, `used_fallback_video_backend`, `video_backend_error`). For dual fisheye, the video backend fields are empty strings / `False` since SAM3 image API is used without video tracking.

### 4.5 Rig Config (conditional on `use_rig`)

The existing `rig_config.py` generates rig JSON for ERP pinhole views (many virtual cameras, shared optical center, zero translation). It is unchanged for the ERP path.

**Default v1 dual fisheye path (`use_rig=False`):** No rig config is written. `images/front/` and `images/back/` are aligned by COLMAP as two independent OPENCV_FISHEYE cameras (PER_FOLDER mode). The existing `ColmapRunner` already skips gracefully when its `rig_config_path` points at a missing file (`core/colmap_runner.py:602-619`). Cross-camera coupling comes from feature matches in the ~10° equator overlap region.

**Experimental v1 rig path (`use_rig=True`):** A new function `write_dual_fisheye_rig_config(output_path, baseline_m, rotation_quat)` writes the two-sensor rig config:

**ERP rig config (existing, unchanged):**
```json
[{
  "cameras": [
    {"image_prefix": "00_00/", "ref_sensor": true},
    {"image_prefix": "00_01/", "cam_from_rig_rotation": [...], "cam_from_rig_translation": [0, 0, 0]},
    ...
  ]
}]
```

**Dual fisheye rig config (new, experimental):**
```json
[{
  "cameras": [
    {"image_prefix": "front/", "ref_sensor": true},
    {
      "image_prefix": "back/",
      "cam_from_rig_rotation": [0, 0, 1, 0],
      "cam_from_rig_translation": [0, 0, -0.025]
    }
  ]
}]
```

The rotation `[0, 0, 1, 0]` is the quaternion (qw=0, qx=0, qy=1, qz=0) for 180° around Y. The translation `[0, 0, -0.025]` is the rig origin's position in the **back camera's** frame (negative Z because back's +Z viewing direction points opposite to front's +Z, and the front sensor sits 25 mm in front's +Z direction = 25 mm in back's -Z direction).

**cam_from_rig_translation derivation:** `cam_from_rig` transforms rig coordinates into camera coordinates. By definition the translation component equals the position of the rig origin (front sensor) expressed in the camera's own coordinate system. For back, this is `[0, 0, -0.025]`. Verified two ways:

1. Algebraic: `R_back_from_rig @ (rig_origin - p_back_in_rig) = diag(-1,1,-1) @ [0,0,+0.025] = [0,0,-0.025]`.
2. pycolmap decomposition: `pycolmap.Rigid3d(rotation, translation).tgt_origin_in_src()` returns the back camera's position in rig frame; with translation `-0.025` this returns `[0, 0, -0.025]` (back behind front, physically correct).

**Sign verification (CI assertion):** before any rig-path code runs, assert:

```python
# Equivalently: pycolmap.Rigid3d(rotation, translation).tgt_origin_in_src()[2] < 0
# — back camera physically sits behind front in rig coordinates.
assert rig.cam_from_rig_translation[2] < 0
```

With the wrong sign this assertion fails and the rig path errors out before COLMAP runs.

A new function `write_dual_fisheye_rig_config(output_path, baseline_m, rotation_quat)` writes the two-sensor rig config. The existing `write_rig_config(view_config, output_path)` is unchanged. The pipeline calls `write_dual_fisheye_rig_config` only when `cfg.use_rig and cfg.input_type == "dual_fisheye"`; otherwise no rig file is written.

### 4.6 COLMAP Stage

`ColmapConfig` changes:

- `camera_model` is already a configurable field (currently defaults to `"PINHOLE"`). For dual fisheye, set to `"OPENCV_FISHEYE"`.
- `camera_params` receives the family-average intrinsic prior as a formatted string (e.g., `"1046,1046,1915,1919,0,0,0,0"` for OPENCV_FISHEYE's 8 params: fx, fy, cx, cy, k1, k2, k3, k4). The string is computed by `infer_fisheye_camera_params(family)` (see "Calibration prior plumbing" below).
- New fields added to `ColmapConfig`:
  - `refine_principal_point: bool = False` — set `True` for fisheye
  - `refine_extra_params: bool = False` — set `True` for fisheye (distortion coefficients)

These new fields are passed through to `IncrementalPipelineOptions` in `_run_pipeline`, replacing the current hardcoded `False` values at lines 708-709.

COLMAP's `CameraMode.PER_FOLDER` assigns one camera (sensor) per subfolder. With `images/front/` and `images/back/`, this naturally gives two cameras initialized from the same `camera_params` string. BA then refines each independently — front converges toward its true focal length (~1047.9 px on the Osmo 360), back toward its own (~1044.9 px). The per-lens values in §5 are documentation of expected BA convergence targets, **not** priors that get directly applied (see Calibration prior plumbing below).

**Calibration prior plumbing.** A new module `core/fisheye_priors.py` mirrors the existing pinhole equivalent:

```python
# core/fisheye_priors.py
FISHEYE_PRIORS = {
    "dji_osmo360": (1046, 1046, 1915, 1919, 0, 0, 0, 0),  # fx, fy, cx, cy, k1-k4
    "insta360":     None,                                   # TBD per camera generation
}

def infer_fisheye_camera_params(family: str) -> Optional[str]:
    """Return COLMAP camera_params string for a known dual fisheye family.

    Returns None if the family has no calibrated prior — caller falls back
    to default_focal_length_factor or the user-provided override.
    """
    prior = FISHEYE_PRIORS.get(family)
    if prior is None:
        return None
    return ",".join(f"{p}" for p in prior)
```

Pipeline data flow (mirrors the existing pinhole flow at `core/pipeline.py:621-623`):

```
detect_input_type(video_path) -> (input_type, family)
   ↓ (set on PipelineConfig)
PipelineConfig.camera_family = "dji_osmo360" | "insta360" | None
   ↓
infer_fisheye_camera_params(cfg.camera_family) -> "fx,fy,cx,cy,k1,k2,k3,k4" or None
   ↓
ColmapConfig.camera_params = <prior string>
   ↓
ColmapRunner sets reader_opts.camera_params (already wired at colmap_runner.py:502)
   ↓
COLMAP applies SAME prior to both front/ and back/ folders;
BA refines each independently.
```

**Override JSON format.** When `cfg.fisheye_calibration_path` is set, the pipeline loads the JSON and converts it to the COLMAP string. The schema mirrors reconstruction-zone's `DualFisheyeCalibration`:

```json
{
  "camera_model": "DJI Osmo 360",
  "front": {
    "camera_matrix": [[1047.9, 0, 1917.6], [0, 1047.9, 1919.9], [0, 0, 1]],
    "dist_coeffs": [0.0559, 0.0114, -0.0095, 0.0005],
    "image_size": [3840, 3840]
  },
  "back": { ... same shape ... }
}
```

Conversion (in `core/fisheye_priors.py`):

```python
def colmap_params_from_cv2_fisheye(K, D) -> str:
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    k1, k2, k3, k4 = (float(d) for d in D)
    return f"{fx},{fy},{cx},{cy},{k1},{k2},{k3},{k4}"
```

The override is applied as a **soft prior**: it replaces the family default `camera_params`, but BA still refines (`ba_refine_focal_length=True`, `ba_refine_extra_params=True`). Per-lens distinctions in the override (front vs back values) are flattened to the family average for the PER_FOLDER prior — the same caveat as the hardcoded family default. Per-lens hard priors via database edit are deferred to v2.

Rig BA settings (apply only when a rig config file is present):
- `ba_refine_sensor_from_rig=False` — rig geometry is locked (already present)
- `ba_refine_focal_length=True` — each sensor's intrinsics refined independently (already present)
- `ba_refine_principal_point` — `False` for pinhole (existing default), `True` for fisheye (new)
- `ba_refine_extra_params` — `False` for pinhole (existing default), `True` for fisheye (new, refines k1-k4)

Note: the existing `_try_set_attr(pipeline_opts, "constant_rigs", True)` at `core/colmap_runner.py:710` is a no-op (the field is `Set[int]`, not bool — `_try_set_attr` swallows the type error). `ba_refine_sensor_from_rig=False` alone does the rig-lock work. The misleading line should be removed or rewritten to set a real `Set[int]` of rig IDs after `apply_rig_config`.

| Setting                      | ERP (pinhole)  | Dual fisheye, no rig (v1 default) | Dual fisheye, rig (experimental) |
|------------------------------|----------------|-----------------------------------|----------------------------------|
| `camera_model`               | `"PINHOLE"`    | `"OPENCV_FISHEYE"`                | `"OPENCV_FISHEYE"`               |
| `refine_principal_point`     | `False`        | `True`                            | `True`                           |
| `refine_extra_params`        | `False`        | `True`                            | `True`                           |
| Rig config file written      | yes (ERP rig)  | **no**                            | yes (dual fisheye rig)           |
| `ba_refine_sensor_from_rig`  | `False`        | (n/a — no rig)                    | `False`                          |

When `use_rig=False`, no rig config file is written. `ColmapRunner` is invoked as today; it detects the missing rig config and skips the rig-application step (`core/colmap_runner.py:602-619`) — no runner code change needed.

### 4.7 COLMAP 4.1 Upgrade & Feature Exposure (Cross-Cutting)

The plugin upgrades from pycolmap 4.0.2 to **pycolmap 4.1.0-dev1** (released 2026-05-14, lyehe/build_gpu_colmap v4.1.0-dev1). This is not optional — 4.1.0-dev1 includes rig correctness fixes that directly affect the pipeline:

- **`refine_sensor_from_rig` applied through all global mapper stages** — GLOMAP now properly respects rig constraints
- **Fixed missing rig pose manifold in generalized absolute pose refinement** — rig BA correctness fix
- **Clipped extreme pixels in fisheye undistortion** — fisheye-specific improvement
- **Caspar GPU-accelerated bundle adjustment** — 5-20x faster BA on supported camera models (currently SimpleRadial and Pinhole; OPENCV_FISHEYE falls back to Ceres)

Source: colmap/colmap#4018 (merged 2026-05-05), built from upstream commit `6cfbc040` (2026-05-10).

The following capabilities are wired into the UI as part of this work. These apply to **both** the ERP and dual fisheye paths.

#### New UI controls

| Control | Section | Options | Default |
|---------|---------|---------|---------|
| Feature Type | Reframe & Alignment | SIFT, ALIKED N16, ALIKED N32 | SIFT |
| Matcher Type | Output Quality | Bruteforce, LightGlue | Bruteforce |
| Mapper | Reframe & Alignment | Incremental, Global (GLOMAP) | Incremental |
| BA Solver | Reframe & Alignment | Ceres, Caspar (GPU) | Caspar when available |

**Caspar BA availability:** Caspar currently supports SimpleRadial and Pinhole camera models only. For the ERP pinhole path, Caspar is the default when available (5-20x faster BA). For the fisheye path (OPENCV_FISHEYE), the solver automatically falls back to Ceres — no user action needed. The UI should indicate which solver is active. As Caspar adds more camera models in future COLMAP releases, fisheye will benefit automatically.

The existing Matching Strategy dropdown adds **Vocab Tree** as a third option alongside Sequential and Exhaustive. Vocab tree requires a vocabulary tree file — the plugin provides a download button for the standard 32K tree from demuc.de, or the user provides a custom path.

#### colmap_runner.py changes

**Feature extraction:**
- Route `FeatureExtraction.type` to `aliked_n16rot` / `aliked_n32` when selected
- Use `AlikedExtraction.max_num_features` instead of `SiftExtraction` for ALIKED
- ONNX models auto-download on first use (ALIKED, LightGlue)

**Feature matching:**
- The pycolmap `FeatureMatcherType` enum is **compound** — every value combines a feature detector and a matching algorithm. Verified against pycolmap 4.0.2: `FeatureMatcherType` has exactly four members:

  | (feature_type, matcher_type) | `FeatureMatcherType` enum value |
  |---|---|
  | `("sift", "bruteforce")`       | `SIFT_BRUTEFORCE`        |
  | `("sift", "lightglue")`        | `SIFT_LIGHTGLUE`         |
  | `("aliked_n16rot", "bruteforce")` or `("aliked_n32", "bruteforce")` | `ALIKED_BRUTEFORCE` |
  | `("aliked_n16rot", "lightglue")` or `("aliked_n32", "lightglue")`   | `ALIKED_LIGHTGLUE`  |

- The implementation must dispatch on the (feature_type, matcher_type) pair and set `FeatureMatchingOptions.type` to the corresponding enum value. The UI exposes the two axes independently; the enum mapping happens in `colmap_runner.py` before invoking the matcher.

**Mapping:**
- Call `pycolmap.global_mapping()` instead of `pycolmap.incremental_mapping()` when GLOMAP selected.

**New `ColmapConfig` fields:**
```python
feature_type: str = "sift"        # "sift", "aliked_n16rot", "aliked_n32"
matcher_type: str = "bruteforce"  # "bruteforce", "lightglue"
mapper: str = "incremental"       # "incremental", "global"
vocab_tree_path: Optional[str] = None  # Path to vocab tree file when matcher == "vocab_tree"
```

#### Open questions requiring testing

1. **GLOMAP + rig constraints:** The 4.1.0-dev1 release includes "applied `refine_sensor_from_rig` through all global mapper stages" which strongly suggests GLOMAP now works with rigs. Needs verification on an actual dual-fisheye dataset before enabling as a recommended option for the rig path.

2. **ALIKED on fisheye frames:** ALIKED is a learned feature detector trained on perspective images. It may perform poorly on 190° fisheye frames with heavy radial distortion. SIFT may remain the better choice for fisheye. Needs benchmarking.

3. **Vocab tree file management:** The standard 32K tree from demuc.de needs to be downloadable from within the plugin. The UI should show a download button when vocab tree is selected but no tree file is present.

4. **Caspar solver stability:** The PR discussion (colmap/colmap#4018) notes that Caspar can diverge on certain scenes (exhibition_hall case). The plugin should fall back to Ceres gracefully if Caspar BA fails.

#### Reference

See `docs/colmap-4.0-upgrade-reference.md` for the full API mapping, parameter key changes, and gotchas. The 4.1 upgrade is additive to the 4.0 reference.

### 4.8 Transforms Output

**Dual fisheye -> Pinhole output:** Deferred. See section 9.

**Dual fisheye -> Spherical/Fisheye output:** The transforms writer produces per-frame entries with fisheye-native intrinsics. Each frame entry carries its own camera parameters because front and back lenses have different calibrations:

```json
{
  "camera_model": "OPENCV_FISHEYE",
  "applied_transform": [[1,0,0,0],[0,-1,0,0],[0,0,-1,0]],
  "ply_file_path": "pointcloud.ply",
  "frames": [
    {
      "file_path": "images/front/frame_0001.jpg",
      "transform_matrix": [[...], [...], [...], [...]],
      "w": 3840,
      "h": 3840,
      "fl_x": 1048.2,
      "fl_y": 1048.2,
      "cx": 1917.5,
      "cy": 1919.8,
      "k1": 0.056,
      "k2": 0.011,
      "k3": -0.010,
      "k4": 0.001
    },
    {
      "file_path": "images/back/frame_0001.jpg",
      "transform_matrix": [[...], [...], [...], [...]],
      "w": 3840,
      "h": 3840,
      "fl_x": 1044.9,
      "fl_y": 1044.9,
      "cx": 1911.7,
      "cy": 1917.9,
      "k1": 0.057,
      "k2": 0.008,
      "k3": -0.007,
      "k4": 0.000
    }
  ]
}
```

The per-frame intrinsics pattern follows `scaffold.py`'s existing approach (each frame entry has `w`, `h`, `fl_x`, `fl_y`, `cx`, `cy`), extended with `k1`-`k4` for the fisheye distortion.

The coordinate conversion (COLMAP OpenCV -> LFS OpenGL + Y pre-comp) uses the same chain as `scaffold.py` and `transforms_writer.py`.

**New function `write_fisheye_transforms` added to `transforms_writer.py`** (sibling to existing `write_transforms_json`, not an extension of it). It reads the COLMAP reconstruction, extracts per-image poses and per-sensor intrinsics, converts coordinates, and writes the JSON directly (mirroring how `scaffold.py` writes its own `transforms_data` dict at line 310).

The existing `write_transforms_json` does not support `applied_transform` or per-frame intrinsics, so it remains unchanged and unused by the fisheye path. The two functions live side-by-side; future cleanup could refactor both plus `scaffold.py`'s inline writer into a shared helper, but that's out of scope here.

### 4.9 Overlap Masks

The existing Voronoi overlap mask stage (pipeline.py stage 3.5) is **skipped** for the dual fisheye path. It partitions the image space between virtual cameras to prevent duplicate features — this only applies to the ERP pinhole path where multiple overlapping views are extracted from one panorama. With two physical fisheye sensors, COLMAP's `skip_image_pairs_in_same_frame` (already set in `colmap_runner.py`) handles the equivalent concern by preventing matching between front/back images of the same frame.

### 4.10 Output Directory Layout

**Dual fisheye -> Spherical output:**
```
output_dir/
  images/
    front/          # fisheye frames (moved from extracted/)
    back/
  masks/            # (if masking enabled)
    front/
    back/
  sparse/0/         # COLMAP reconstruction
  rig_config.json   # only present when use_rig=True (experimental)
  transforms.json   # fisheye-native with per-frame intrinsics
  pointcloud.ply
```

## 5. Per-Camera-Family Calibration Defaults

Family-average values used as COLMAP `camera_params` prior in PER_FOLDER mode. COLMAP initializes both `front/` and `back/` cameras from the **same** family-average string, then refines each independently during BA. Per-lens distinctions in the tables below are documentation of expected BA convergence targets — not priors that get directly applied. (See §4.6 "Calibration prior plumbing" for the data flow.)

### DJI Osmo 360

Source: Empirical measurement on unit 95SXN7S02213TB, 2026-04-19 through 2026-04-21. Metashape 2.3 two-sensor alignment with scale bars (0.0% error). 248 frame pairs. Full report: `osmo360_rig_calibration_report.md`.

**Family-average prior (used as COLMAP `camera_params`):**
- `f = 1046, cx = 1915, cy = 1919, k1 = k2 = k3 = k4 = 0`
- Image size: 3840×3840

**Per-lens BA convergence targets (documentation only, not priors):**

| Parameter | Front | Back |
|-----------|-------|------|
| f (px)    | 1047.9 | 1044.9 |
| cx (px)   | 1917.6 | 1911.7 |
| cy (px)   | 1919.9 | 1917.9 |
| k1        | 0.0559 | 0.0572 |
| k2        | 0.0114 | 0.0076 |
| k3        | -0.0095 | -0.0072 |
| k4        | 0.0005 | 0.0001 |

Rig geometry (used only when `use_rig=True`):
- Rotation: 180° around Y (quaternion `[0, 0, 1, 0]`)
- Baseline: 25mm — back camera at `Z = -24.9 mm` in front camera frame (per `osmo360_rig_calibration_report.md` §2.2). The summary section of that report says "26 mm" but the empirical measurements all cluster at 24.9–25.0 mm; spec uses 25 mm.
- `cam_from_rig_translation` for back: `[0, 0, -0.025]` (rig origin in back camera's frame; negative because back's +Z viewing direction points opposite to front's +Z)
- Both locked during BA when rig is enabled

In v1 default mode (`use_rig=False`), the rig geometry above is **not** used — front and back are aligned as independent cameras and BA recovers the per-pair baseline from feature matches in the equator overlap.

### Insta360

**v1 status: no calibrated prior.** `infer_fisheye_camera_params("insta360")` returns `None`. The pipeline falls back to COLMAP's `default_focal_length_factor` (the existing pinhole fallback path at `core/pipeline.py:621-628` already handles this), or to a user-provided override JSON via `cfg.fisheye_calibration_path`.

Producing a calibrated prior requires running the equivalent of the Osmo 360 calibration workflow (Metashape with scale bars on a real Insta360 capture) and recording the family-average values. Until then, the path runs but BA must converge from a generic starting point — slower and more failure-prone, especially on low-texture scenes.

Two follow-up tasks before promoting Insta360 from "fallback only" to "calibrated":
1. Capture and calibrate a representative Insta360 X3/X4/X5 unit; produce per-lens K + D values in the same format as the Osmo 360 report.
2. Add the family-average values to `FISHEYE_PRIORS` in `core/fisheye_priors.py`.

The previously published rough heuristic `f ≈ image_width * 0.27` does not match the equidistant projection model (`f = (image_width / 2) / radians(half_FOV_deg)` gives ~12% larger values). Do not use the heuristic; either ship without a prior or wait for real calibration data.

## 6. Pipeline Flow (Dual Fisheye Path)

```
User provides .osv or .insv file
  |
  v
detect_input_type() -> "dual_fisheye"
  |
  v
PairedExtractor:
  1. Demux (format-specific) -> front.mp4, back.mp4
  2. Score sharpness on front stream
  3. Extract paired frames -> extracted/front/, extracted/back/
  4. Clean up temp streams (unless keep_streams=True)
  |
  v
Masking (if enabled):
  1. Generate fisheye circle mask (6% default margin)
  2. SAM3 operator detection on front frames -> extracted/masks/front/
  3. SAM3 operator detection on back frames -> extracted/masks/back/
  4. Combine circle + operator masks
  |
  v
Copy frames to images/front/, images/back/
  |
  v
If use_rig=True (experimental):
    Write dual fisheye rig config -> rig_config.json
    (otherwise: skip; ColmapRunner detects missing file and skips rig step)
  |
  v
COLMAP (OPENCV_FISHEYE, PER_FOLDER, rig constraint when use_rig=True):
  1. Feature extraction (two sensors, fisheye priors)
  2. Apply rig config (only if file present)
  3. Feature matching
  4. Incremental mapping (refine intrinsics + distortion;
     locked rig only when use_rig=True)
  |
  v
Output (fisheye mode only in this spec):
    Move fisheye frames from extracted/ to images/
    Write fisheye transforms.json with per-frame intrinsics
    Export sparse point cloud
```

## 7. New Files

| File | Purpose | v1 status |
|------|---------|-----------|
| `core/paired_extractor.py` | Demux + paired sharpness scoring + synchronized frame extraction | required |
| `core/fisheye_circle_mask.py` | Fisheye image circle mask generation with adjustable margin | required |
| `write_dual_fisheye_rig_config` (added to existing `core/rig_config.py`, see §8) | Writer for the two-sensor rig JSON | **experimental** — only invoked when `use_rig=True` |

## 8. Modified Files

| File | Changes |
|------|---------|
| `core/pipeline.py` | `PipelineConfig` new fields (incl. `use_rig: bool = False`), `_run_stages` branching on `input_type` |
| `core/colmap_runner.py` | `ColmapConfig` new fields (`refine_principal_point`, `refine_extra_params`, `feature_type`, `matcher_type`, `mapper`), route ALIKED/LightGlue/GLOMAP, unhardcode BA refinement bools |
| `core/rig_config.py` | Adds `write_dual_fisheye_rig_config(output_path, baseline_m, rotation_quat)` (only invoked when `use_rig=True`); existing `write_rig_config` unchanged |
| `core/transforms_writer.py` | Fisheye transforms output with per-frame intrinsics and k1-k4 |
| `panels/prep360_panel.py` | UI: "Keep demuxed streams" checkbox, fisheye circle margin slider, feature type / matcher type / mapper dropdowns, vocab tree option + download. (Optional: experimental "Use rig constraint" checkbox if exposed to users.) |
| `panels/prep360_panel.rml` | UI markup for all new controls |

## 9. What This Spec Does NOT Cover

- **ERP scaffold output** (ERP input -> Spherical output). Separate effort, Phase 1 in `docs/specs/SCAFFOLD_IMPLEMENTATION.md`.
- **ERP stitching from fisheye.** The original Phase 2 spec proposed stitching dual fisheye to ERP. This design replaces that approach entirely — raw fisheye goes directly to COLMAP.
- **Default rig-constrained alignment.** v1 ships dual fisheye **without** rig constraint by default; rig-locked alignment is opt-in via `use_rig=True` and remains experimental until validated end-to-end on real Osmo 360 / Insta360 captures. Pending tasks before promotion: validate the corrected `cam_from_rig_translation` sign on a real reconstruction, measure reconstruction quality vs the no-rig baseline, evaluate per-unit baseline tolerance.
- **Factory telemetry extraction.** Per-unit calibration from .osv proto messages is interesting but not needed when COLMAP refines from generic priors.
- **Insta360 calibration values.** Added incrementally as users test.
- **Dual fisheye -> Pinhole reframing details.** The post-COLMAP step that extracts pinhole crops from fisheye frames needs its own design pass (which pinhole layout, what FOV, how to handle the ~190 deg fisheye FOV vs typical pinhole ~90 deg).
- **GLOMAP + rig constraint verification.** 4.1.0-dev1 includes the fix ("refine_sensor_from_rig through all global mapper stages") but still needs testing on actual data before recommending.
- **Caspar solver stability.** Caspar can diverge on certain scenes (colmap/colmap#4018 discussion). Needs graceful fallback to Ceres.
- **ALIKED on fisheye benchmarking.** Whether learned features perform well on 190° fisheye frames needs comparison against SIFT.
