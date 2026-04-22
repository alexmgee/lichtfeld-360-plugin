# Dual Fisheye Pipeline Integration

**Date:** 2026-04-22
**Status:** Design approved, pending implementation plan

---

## 1. Problem

The plugin currently only accepts equirectangular (ERP) video as input. Users with dual-fisheye cameras (DJI Osmo 360, Insta360 ONE RS/X4) must stitch to ERP externally before using the plugin. This loses the native fisheye geometry and requires extra tooling.

The goal: accept raw dual-fisheye files (.osv, .insv) directly, feed native fisheye frames through COLMAP with a rig constraint, and output either pinhole crops or fisheye-native transforms for downstream training.

## 2. Design Decisions

These were resolved during brainstorming:

1. **Input detection:** Auto-detect from file extension. User does not choose input type.
2. **Output mode:** User chooses Pinhole or Spherical/Fisheye. Combined with auto-detected input, this gives a four-cell matrix (2 input types x 2 output modes).
3. **Pipeline architecture:** Extend the existing `PipelineJob` with conditional branching (Approach B), not a parallel pipeline or stage-based composition.
4. **Stream retention:** A "Keep demuxed streams" checkbox. Unchecked = temp files cleaned up after extraction. Checked = saved alongside output.
5. **Calibration source:** Hardcoded per-camera-family defaults as COLMAP priors. Optional user-provided calibration file override. No factory telemetry extraction in v1.
6. **Rig constraint:** Locked at 25mm baseline, 180 deg Y rotation. `constant_rigs=True`, `ba_refine_sensor_from_rig=False`. COLMAP refines per-sensor intrinsics independently but does not touch the rig geometry.
7. **Masking:** Same SAM3 backend on fisheye frames. Fisheye circle mask (adjustable margin, ~5-6% default) combined with operator mask.
8. **Camera families:** DJI Osmo 360 (.osv) and Insta360 (.insv) supported. Insta360 priors added incrementally as users test.

## 3. Output Matrix

|                        | Pinhole output              | Spherical/Fisheye output       |
|------------------------|-----------------------------|--------------------------------|
| **ERP input**          | Existing path (working)     | ERP scaffold (incomplete)      |
| **Dual fisheye input** | Fisheye -> COLMAP -> pinhole | Fisheye -> COLMAP -> fisheye poses |

The ERP scaffold cell is a separate effort (Phase 1 in IMPLEMENTATION.md) and is not part of this spec.

## 4. Architecture

### 4.1 Input Detection

A `detect_input_type()` function determines the input type from the source file:

- `.osv` -> `"dual_fisheye"` (single file, two HEVC video tracks)
- `.insv` -> `"dual_fisheye"` (file pair: `*_00_*.insv` + `*_01_*.insv`, already separate streams)
- `.mp4` / `.mov` / other single-stream video -> `"erp"`

This lives in `pipeline.py` or a small utility module.

### 4.2 PipelineConfig Changes

New fields on `PipelineConfig`:

```python
# Auto-detected from source file
input_type: str = "erp"  # "erp" or "dual_fisheye"

# Dual fisheye options
keep_streams: bool = False  # Save demuxed video streams alongside output
fisheye_calibration_path: Optional[str] = None  # Override calibration JSON
fisheye_circle_margin: float = 0.05  # Circle mask margin (5% default)
```

The existing `output_mode` field (`"pinhole"` / `"erp"`) is reused. For dual fisheye with spherical output, `output_mode` is `"erp"` (meaning "native camera model, not pinhole crops"). The naming could be improved later but the semantics are: pinhole = COLMAP outputs pinhole cameras, erp = output uses the source camera model.

### 4.3 Extraction Stage

**ERP path (unchanged):** `SharpestExtractor` extracts frames from a single video stream.

**Dual fisheye path:** A new `PairedExtractor` class handles:

1. **Demux** — format-specific:
   - `.osv`: ffmpeg extracts stream 0 (front) and stream 1 (back) from the single container into two temporary video files
   - `.insv`: locates the paired `_00_` and `_01_` files (already separate, no demux needed)
2. **Sharpness scoring** — scores frames from one stream only (front), since both streams share identical motion blur from the rigid body
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

For dual fisheye, the masker also generates a **fisheye circle mask** — a binary mask of the valid image circle with an adjustable margin (default ~5-6%). This is combined (bitwise AND) with the SAM3 operator mask for each frame. The circle mask generation is a simple geometric operation (draw a filled circle on a black image, erode by margin percentage).

The circle mask margin is exposed as `fisheye_circle_margin` on `PipelineConfig` for tuning.

### 4.5 Rig Config

The existing `rig_config.py` generates rig JSON for ERP pinhole views (many virtual cameras, shared optical center, zero translation). For dual fisheye, it generates a simpler rig with two physical sensors and a real translational offset.

**ERP rig config (existing):**
```json
[{
  "cameras": [
    {"image_prefix": "00_00/", "ref_sensor": true},
    {"image_prefix": "00_01/", "cam_from_rig_rotation": [...], "cam_from_rig_translation": [0, 0, 0]},
    ...
  ]
}]
```

**Dual fisheye rig config (new):**
```json
[{
  "cameras": [
    {"image_prefix": "front/", "ref_sensor": true},
    {
      "image_prefix": "back/",
      "cam_from_rig_rotation": [0, 0, 1, 0],
      "cam_from_rig_translation": [0, 0, 0.025]
    }
  ]
}]
```

The rotation `[0, 0, 1, 0]` is the quaternion (qw=0, qx=0, qy=1, qz=0) for 180 deg around Y. The translation `[0, 0, 0.025]` is 25mm along +Z in the front camera's frame.

The `write_rig_config` function gains an alternate code path: when called with a dual-fisheye config (not a `ViewConfig`), it writes the two-sensor rig directly. This could be a second function (`write_dual_fisheye_rig_config`) or a branch within the existing function.

### 4.6 COLMAP Stage

`ColmapConfig` changes:

- `camera_model` is already a configurable field (currently defaults to `"PINHOLE"`). For dual fisheye, set to `"OPENCV_FISHEYE"`.
- `camera_params` receives the per-camera-family intrinsic priors as a formatted string (e.g., `"1046,1046,1920,1920,0,0,0,0"` for OPENCV_FISHEYE's 8 params: fx, fy, cx, cy, k1, k2, k3, k4).

COLMAP's `CameraMode.PER_FOLDER` assigns one camera (sensor) per subfolder. With `images/front/` and `images/back/`, this naturally gives two independent sensors that COLMAP calibrates separately during BA.

Rig BA settings (already present in `colmap_runner.py`):
- `ba_refine_sensor_from_rig=False` — rig geometry is locked
- `constant_rigs=True` — rig does not change during reconstruction
- `ba_refine_focal_length=True` — each sensor's intrinsics refined independently
- `ba_refine_principal_point` and `ba_refine_extra_params` — may need to be `True` for fisheye (currently `False` for pinhole). The fisheye model benefits from refining distortion coefficients during BA.

The pipeline sets these differently for dual fisheye vs ERP:

| Setting                      | ERP (pinhole)  | Dual fisheye        |
|------------------------------|----------------|---------------------|
| `camera_model`               | `"PINHOLE"`    | `"OPENCV_FISHEYE"`  |
| `ba_refine_principal_point`  | `False`        | `True`              |
| `ba_refine_extra_params`     | `False`        | `True`              |

### 4.7 Transforms Output

**Dual fisheye -> Pinhole output:** After COLMAP aligns fisheye images and reconstructs the scene, pinhole crops are extracted from the fisheye frames using the recovered poses. The existing `transforms_writer.py` handles pinhole output as-is. (This path requires an additional reframing step after COLMAP — extracting pinhole views from the fisheye images using the now-known poses and calibrations.)

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

**New file or extension:** A `write_fisheye_transforms` function, either in `transforms_writer.py` or in a new `fisheye_transforms.py`. It reads the COLMAP reconstruction, extracts per-image poses and per-sensor intrinsics, converts coordinates, and writes the JSON.

### 4.8 Output Directory Layout

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
  rig_config.json
  transforms.json   # fisheye-native with per-frame intrinsics
  pointcloud.ply
```

**Dual fisheye -> Pinhole output:**
```
output_dir/
  images/           # pinhole crops extracted from fisheye after COLMAP
    00_00/
    00_01/
    ...
  masks/            # (if masking enabled)
  sparse/0/         # COLMAP reconstruction
  rig_config.json
  transforms.json   # pinhole intrinsics
```

## 5. Per-Camera-Family Calibration Defaults

Hardcoded defaults used as COLMAP priors. COLMAP refines from these during BA.

### DJI Osmo 360

Source: Empirical measurement on unit 95SXN7S02213TB, 2026-04-19 through 2026-04-21. Metashape 2.3 two-sensor alignment with scale bars (0.0% error). 248 frame pairs. Full report: `osmo360_rig_calibration_report.md`.

| Parameter | Front | Back |
|-----------|-------|------|
| f (px)    | 1047.9 | 1044.9 |
| cx (px)   | 1917.6 | 1911.7 |
| cy (px)   | 1919.9 | 1917.9 |
| k1        | 0.0559 | 0.0572 |
| k2        | 0.0114 | 0.0076 |
| k3        | -0.0095 | -0.0072 |
| k4        | 0.0005 | 0.0001 |
| Image size | 3840x3840 | 3840x3840 |

Rig geometry:
- Rotation: 180 deg around Y
- Baseline: 25mm along +Z (front camera frame)
- Both locked during BA

For COLMAP's single-prior-per-folder mode, a family-average prior is used: f=1046, cx=1915, cy=1919, k1-k4=0. COLMAP refines per-sensor from there.

### Insta360

Priors TBD. Will be added incrementally as users test with specific Insta360 models. Initial priors can use reasonable fisheye defaults (f ~= image_width * 0.27 for ~190 deg FOV, centered principal point, zero distortion) and let COLMAP refine.

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
  1. Generate fisheye circle mask (5-6% margin)
  2. SAM3 operator detection on front frames -> extracted/masks/front/
  3. SAM3 operator detection on back frames -> extracted/masks/back/
  4. Combine circle + operator masks
  |
  v
Copy frames to images/front/, images/back/
  |
  v
Write dual fisheye rig config -> rig_config.json
  |
  v
COLMAP (OPENCV_FISHEYE, PER_FOLDER, rig constraint):
  1. Feature extraction (two sensors, fisheye priors)
  2. Apply rig config
  3. Feature matching
  4. Incremental mapping (locked rig, refine intrinsics + distortion)
  |
  v
Output:
  if output_mode == "pinhole":
    Extract pinhole crops from fisheye using recovered poses
    Write pinhole transforms.json
  else:  # spherical/fisheye
    Write fisheye transforms.json with per-frame intrinsics
    Export sparse point cloud
```

## 7. New Files

| File | Purpose |
|------|---------|
| `core/paired_extractor.py` | Demux + paired sharpness scoring + synchronized frame extraction |
| `core/fisheye_circle_mask.py` | Fisheye image circle mask generation with adjustable margin |

## 8. Modified Files

| File | Changes |
|------|---------|
| `core/pipeline.py` | `PipelineConfig` new fields, `_run_stages` branching on `input_type` |
| `core/colmap_runner.py` | `ColmapConfig` fisheye-aware BA settings (refine principal point + extra params) |
| `core/rig_config.py` | Dual fisheye rig config generation (two sensors, 25mm baseline, 180 deg Y) |
| `core/transforms_writer.py` | Fisheye transforms output with per-frame intrinsics and k1-k4 |
| `panels/prep360_panel.py` | UI: "Keep demuxed streams" checkbox, fisheye circle margin slider |
| `panels/prep360_panel.rml` | UI markup for new controls |

## 9. What This Spec Does NOT Cover

- **ERP scaffold output** (ERP input -> Spherical output). Separate effort, Phase 1 in IMPLEMENTATION.md.
- **ERP stitching from fisheye.** The original Phase 2 spec proposed stitching dual fisheye to ERP. This design replaces that approach entirely — raw fisheye goes directly to COLMAP.
- **Factory telemetry extraction.** Per-unit calibration from .osv proto messages is interesting but not needed when COLMAP refines from generic priors.
- **Insta360 calibration values.** Added incrementally as users test.
- **Dual fisheye -> Pinhole reframing details.** The post-COLMAP step that extracts pinhole crops from fisheye frames needs its own design pass (which pinhole layout, what FOV, how to handle the ~190 deg fisheye FOV vs typical pinhole ~90 deg).
