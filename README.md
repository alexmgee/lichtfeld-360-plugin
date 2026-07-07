*This is a work in progress.*

# 360 Plugin for LichtFeld Studio

An end-to-end 360° capture pipeline for LichtFeld Studio. Drop in a video from any popular 360° camera — DJI Osmo (.osv), Insta360 (.insv), or any pre-stitched equirectangular .mp4 — and the plugin handles everything needed to produce a training-ready Gaussian Splatting dataset: frame extraction with sharpness-aware selection, automatic detection and masking of the camera operator and unwanted objects using Meta's SAM 3, geometric alignment via COLMAP 4.1 with GPU-accelerated feature extraction (SIFT or ALIKED) and learned matching (LightGlue), and direct import back into LichtFeld Studio for training.

The plugin works entirely inside LichtFeld Studio's UI. No command-line tools, no manual COLMAP runs, no separate masking step. Select a video, choose your settings, click Process.

## How It Works

The pipeline runs in six stages, each configurable from the plugin panel:

1. **Ingest** — Dual fisheye containers (.osv, .insv) are automatically demuxed into separate lens streams and stitched to equirectangular using ffmpeg with camera-family-specific parameters. Pre-stitched ERP video and pre-split front/back .mp4 pairs are also accepted.

2. **Extract** — Frames are pulled from the video at a configurable FPS. Four sharpness modes control how much analysis goes into selecting the best frame from each interval — from no analysis (fastest) to full Tenengrad/Laplacian scoring of every candidate frame (sharpest results).

3. **Mask** — When SAM 3 is installed and masking is enabled, the plugin automatically detects and masks the camera operator (and any other objects specified by prompt keywords) from every extracted frame. For ERP/Pinhole modes, detection runs on pinhole crops and results are back-projected and merged into a full-resolution ERP mask. For fisheye modes, detection runs directly on the raw fisheye frames.

4. **Reframe** — Each equirectangular frame is reprojected into multiple pinhole perspective views arranged in a virtual camera rig. Built-in presets range from 6 views (cubemap) to 24 views (ultra-dense Fibonacci spiral). These pinhole crops give COLMAP clean perspective images to work with, which is critical because COLMAP cannot align equirectangular images directly.

5. **Align** — All pinhole views are fed to COLMAP 4.1 for structure-from-motion reconstruction. The plugin configures COLMAP with rig constraints that lock the geometric relationship between virtual cameras from the same panorama, since they share an exact optical center. This produces a single consistent reconstruction across the full sequence. Feature extraction uses SIFT or ALIKED (N16/N32) with Bruteforce or LightGlue matching. Sequential or exhaustive pair selection is available, with a configurable overlap window for sequential mode.

6. **Output** — The final dataset is written in one of five output modes (see below) and can be imported directly into LichtFeld Studio for Gaussian Splatting training.

## Supported Input

| Format | Source | Notes |
|--------|--------|-------|
| ERP video (.mp4) | Any pre-stitched equirectangular video | Standard 2:1 aspect ratio |
| .OSV container | DJI Osmo 360 | Dual fisheye, camera family auto-detected |
| .INSV container | Insta360 cameras | Dual fisheye, camera family auto-detected |
| Front + back .mp4 | Pre-split fisheye lens pair | Two-file mode for graded or externally processed footage |

## Output Modes

### ERP

Native equirectangular reconstruction using COLMAP's `EQUIRECTANGULAR` camera model. Feeds ERP frames directly without pinhole scaffolding — faster than ERP (Scaffold), but generally lower accuracy.

### ERP (Pinhole)

Standard COLMAP pinhole dataset. Each source frame produces multiple perspective crops (6–24 depending on preset). The output is a conventional COLMAP sparse reconstruction with per-view images, masks, and a rig config.

### ERP (Scaffold)

Designed for training with 3DGUT, which can consume equirectangular images directly. The plugin uses the pinhole crops only as temporary scaffolding: COLMAP aligns them to recover the camera trajectory, then the plugin extracts the rig-origin pose for each station, applies pitch correction and auto-orientation (aligning camera up to +Y), converts coordinates from COLMAP's OpenCV convention to LichtFeld's OpenGL convention, and writes a transforms.json referencing the original full-resolution ERP frames with `"camera_model": "EQUIRECTANGULAR"`. The scaffolding is deleted after export by default — a "Keep pinhole scaffolding" checkbox retains it for inspection.

### Fisheye

Reconstructs the dual fisheye pair **natively** using COLMAP's `OPENCV_FISHEYE` camera model — the front and back lens streams are aligned directly without reframing, and each lens is calibrated independently by bundle adjustment. Working on the raw lenses registers substantially more of the sequence than reframing to pinhole first.

A **Training output** selector then controls which dataset is written from that single native reconstruction:

| Training output | Writes | For |
|-----------------|--------|-----|
| **Native (fisheye)** | The fisheye dataset (`OPENCV_FISHEYE`) | 3DGUT / fisheye-capable trainers |
| **Pinhole** | Pinhole crops **derived from the native poses** — no second COLMAP run | Standard 3DGS |
| **Both** | Both datasets, side by side | Either trainer |

The **Pinhole** option renders pinhole crops from the raw fisheye using the COLMAP-refined intrinsics and propagates each crop's pose from its lens's *measured* pose. Because the crops inherit the native reconstruction's registration (rather than being re-aligned as a rig of assumed geometry), the resulting pinhole dataset covers far more frames than the legacy Fisheye (Pinhole) mode below.

### Fisheye (Pinhole)

The original direct-pinhole path: reframes each dual fisheye lens into 8 pinhole crops (16 total per source frame) and reconstructs them directly in COLMAP with pinhole camera models and rig constraints. Kept for compatibility, but for a pinhole dataset from fisheye input the **Fisheye** mode's **Pinhole** training output is now recommended — reconstructing the reframed crops directly registers noticeably fewer frames than the native path.

## Installation

### From Plugin Manager

1. Open **Plugin Manager** in LichtFeld Studio
2. Search for **360 Plugin** or paste the repo URL: `alexmgee/lichtfeld-360-plugin`
3. Click **Install**

The base runtime (OpenCV, numpy, pycolmap, ffmpeg) installs automatically. SAM 3 masking is optional and requires a one-time setup from inside the plugin.

### Manual

```bash
git clone https://github.com/alexmgee/lichtfeld-360-plugin.git ~/.lichtfeld/plugins/lichtfeld-360-plugin
```

Then restart LichtFeld Studio.

## Usage

1. Open the **360 Plugin** tab in the right panel
2. Load your video:
   - **Select 360° Video** for a single file (ERP, .osv, or .insv)
   - **Select Front + Back Lens Videos** for pre-split fisheye .mp4 pairs
   - **Re-run COLMAP on Existing Output** to re-align previously extracted frames with different COLMAP settings
3. Choose an **Output Path** and **Output Mode**. For the **Fisheye** mode, a **Training output** selector (Native / Pinhole / Both) chooses which dataset(s) to write from the native reconstruction.
4. Configure each pipeline stage using the collapsible sections:

   **Frame Extraction** — `FPS` sets the extraction rate. `Sharpness` controls frame selection quality (None / Basic / Better / Best). The blur metric can be switched between Tenengrad and Laplacian.

   **Masking** — Enable masking and enter prompt keywords (e.g. `person, tripod`). SAM 3 detects and masks matching objects in every frame.

   **Reframe & Alignment** — `Preset` selects the virtual camera rig layout (Pinhole mode). `Features` chooses the extractor (SIFT, ALIKED N16 rot, ALIKED N32). `Matcher Type` chooses the matcher (Bruteforce or LightGlue). `Matcher` selects pair strategy (Sequential with configurable overlap, or Exhaustive). `Mapper` selects Incremental (supports rig constraints) or Global/GLOMAP (faster, no rig support).

   **Output Quality** — `Crop Size` sets pinhole view resolution. `COLMAP Preset` (Low / Normal / High) adjusts match limits, and `Max. Matches` sets the per-pair cap directly.

5. Click **Process & Import** to run the pipeline and load the result into LichtFeld Studio, or **Process Only** to create the dataset without importing

## Masking

Masking is optional. The plugin works without it, but masking significantly improves training quality by removing the camera operator, tripod, shadow, and other unwanted objects from the dataset.

The masking system uses Meta's SAM 3 (Segment Anything Model 3) with text-prompted detection. How masking works depends on the output mode:

**ERP and Pinhole modes** — The equirectangular frame is reframed into pinhole views at detection resolution, and SAM 3 runs on each pinhole view independently. All per-view detections are then back-projected into ERP space and OR-merged into a single high-resolution ERP mask — if any view detects the operator at a given ERP pixel, that pixel is masked, even in views where detection missed them. The merged ERP mask is then reprojected into each output pinhole view during the final reframing stage.

**Fisheye and Fisheye (Pinhole) modes** — Each dual fisheye lens is masked independently. SAM 3 runs on the raw fisheye frames from both the front and back lenses. A circular mask is also applied to exclude the dark border outside the fisheye image circle — the `Circle Margin` setting (visible when masking is enabled in fisheye modes) controls how aggressively this border is trimmed. For Fisheye (Pinhole) mode, the fisheye masks are reprojected into the pinhole crops during reframing, just like the ERP path.

### SAM 3 Setup

SAM 3 is a gated HuggingFace model. First-time setup is guided from inside the plugin:

1. Request access to `facebook/sam3` on HuggingFace
2. Create a HuggingFace token with `Read` access
3. Paste the token into the plugin and click **Verify Access**
4. Click **Install SAM 3**

Access verification is cached to disk so it persists across sessions.

## COLMAP Integration

The plugin uses COLMAP 4.1 (via pycolmap with GPU-accelerated wheels on Windows) for structure-from-motion alignment. All COLMAP configuration is exposed through the plugin UI — no command-line interaction is needed.

### Feature Extraction

| Extractor | Default Max Features | Description |
|-----------|--------------------:|-------------|
| SIFT | 8,192 | Scale-invariant feature transform. Classic, robust, widest compatibility. |
| ALIKED N16 (rot) | 2,048 | Learned keypoint detector with rotation invariance. Faster extraction than SIFT. |
| ALIKED N32 | 2,048 | Larger ALIKED variant with more capacity. |

When switching extractors, the max features slider automatically adjusts to the appropriate default for the selected extractor.

### Matching

| Matcher Type | Description |
|--------------|-------------|
| Bruteforce | Standard nearest-neighbor descriptor matching. Fast and reliable. |
| LightGlue | Learned feature matcher from COLMAP 4.1. Generally produces more precise correspondences than brute-force, especially for challenging viewpoint changes. |

All 6 combinations of extractor and matcher are supported. The required ONNX model files (aliked-n16rot, aliked-n32, aliked-lightglue, sift-lightglue, bruteforce-matcher) are bundled in `lib/` and validated before each COLMAP run.

### Pair Selection

- **Sequential** matches each image against its temporal neighbors. The **Overlap** slider (2–20) controls the neighborhood size. Faster for video sequences with smooth motion.
- **Exhaustive** tests all possible image pairs. Slower but more robust for irregular capture patterns or scenes that need global loop closure.
- **Loop Closure** can be enabled alongside sequential matching to add vocabulary-tree-based global pair candidates, helping close loops in sequences that revisit earlier viewpoints.

### Mapper

| Mapper | Description |
|--------|-------------|
| Incremental | Standard COLMAP incremental mapper. Supports rig constraints, which are required for pinhole-reframed 360° data where multiple virtual cameras share the same optical center. |
| Global (GLOMAP) | COLMAP 4.1's global SfM mapper. Faster for large datasets but does not support rig constraints. A warning is shown when selected with rig-dependent output modes. |

### Match Budget

| Preset | Max Matches / Pair | Use Case |
|--------|-------------------:|----------|
| Fast | 8,192 | Quick iterations on easier scenes |
| Balanced | 16,384 | General purpose |
| Default | 32,768 | Recommended starting point |
| High | 65,536 | Difficult scenes that need dense correspondence |
| Custom | User-defined | Manual tuning |

## Reframing Presets

Each output mode uses a different reframing strategy to convert the source footage into pinhole views for COLMAP alignment.

### Pinhole Mode

The user-selectable preset controls the virtual camera rig. All presets produce 90° field-of-view pinhole views:

| Preset | Views | Layout |
|--------|------:|--------|
| Cubemap | 6 | 4 horizon faces plus top and bottom |
| Low | 12 | Golden-angle spiral from zenith to nadir |
| Medium | 16 | 8+8 two-ring layout at ±35° with staggered upper ring |
| High | 20 | Golden-angle spiral from zenith to nadir |
| Ultra | 24 | Golden-angle spiral from zenith to nadir |

### ERP Mode

Uses a dedicated internal 8-view staggered scaffold preset (4 views at -35° pitch, 4 at +35°, upper ring offset by 45°). This layout is optimized for pose recovery rather than view coverage — the pinhole crops are temporary scaffolding that gets deleted after COLMAP extracts the rig poses.

### Fisheye (Pinhole) Mode

Each dual fisheye lens is reframed into 8 pinhole crops (16 views total per source frame), using the fisheye-specific calibration for each camera family. The front and back lens rigs are combined into a single COLMAP rig constraint with a baseline offset between the two optical centers.

## Output Structure

### Pinhole / Fisheye (Pinhole)

```text
output_dir/
├── extracted/
│   ├── frames/          source equirectangular frames
│   └── masks/           ERP-resolution masks (when masking enabled)
├── images/              pinhole views, camera-first layout
│   ├── 00_00/
│   │   ├── frame_0001.jpg
│   │   └── frame_0002.jpg
│   ├── 00_01/
│   └── ...
├── masks/               per-view masks (when masking enabled)
├── sparse/0/            COLMAP sparse reconstruction
├── transforms.json
├── rig_config.json
├── database.db
├── colmap_debug.log
└── timing.json
```

The `images/` directory uses camera-first naming: folder prefix = virtual sensor ID, shared filename across folders = rig frame. This convention is required by COLMAP's rig constraint workflow.

### ERP

```text
output_dir/
├── images/              original ERP frames
├── masks/               ERP masks (when masking enabled)
├── pointcloud.ply       auto-oriented sparse point cloud
└── transforms.json      camera_model: EQUIRECTANGULAR
```

All pinhole scaffolding is removed after pose extraction. Enable "Keep pinhole scaffolding" to retain the intermediate crops as `pinhole_images/` and `pinhole_masks/`.

### Fisheye

The **Training output** selector determines the layout. Each dataset lands in its own subfolder and is self-contained — its own COLMAP `sparse/0`, `transforms.json`, images, masks, and point cloud. **Native** writes `native/` only; **Pinhole** writes `pinhole/` only; **Both** writes both.

```text
output_dir/
├── native/                        (Native or Both)
│   ├── images/ front/ back/       fisheye lens frames
│   ├── masks/ front/ back/        (when masking enabled)
│   ├── sparse/0/                  COLMAP reconstruction (OPENCV_FISHEYE)
│   ├── pointcloud.ply
│   └── transforms.json
└── pinhole/                       (Pinhole or Both)
    ├── images/                    flat pinhole crops (view_frame.jpg)
    ├── masks/
    ├── sparse/0/                  COLMAP model (PINHOLE; poses derived from native)
    ├── pointcloud.ply
    └── transforms.json
```

Point LichtFeld Studio at the specific subfolder you want to train (`native/` or `pinhole/`).

## Dependencies

Installed automatically unless noted:

- **opencv-python-headless** — frame extraction, blur scoring, image processing
- **numpy** — coordinate math, rotation matrices
- **pycolmap** — COLMAP 4.1 Python bindings. On Windows, GPU-enabled wheels from [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap). On Linux, CPU build from PyPI.
- **static-ffmpeg** — bundled ffmpeg for video decoding and fisheye stitching
- **SAM 3** — optional, gated HuggingFace model. Installed from inside the plugin after access verification.

ONNX model files for ALIKED, LightGlue, and Bruteforce matching are bundled in `lib/`.

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm. GPU-enabled `pycolmap` wheels provided by [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) by lyehe.

## License

GPL-3.0-or-later
