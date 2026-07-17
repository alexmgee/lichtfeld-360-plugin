*This is a work in progress.*

# 360 Plugin for LichtFeld Studio

An end-to-end 360° capture pipeline for LichtFeld Studio. Drop in a video from any popular 360° camera — DJI Osmo (.osv), Insta360 (.insv), or any pre-stitched equirectangular .mp4 — and the plugin handles everything needed to produce a training-ready Gaussian Splatting dataset: frame extraction with sharpness-aware selection, automatic detection and masking of the camera operator and unwanted objects using Meta's SAM 3, geometric alignment via COLMAP 4.1 with GPU-accelerated feature extraction (SIFT or ALIKED) and learned matching (LightGlue), and direct import back into LichtFeld Studio for training.

The plugin works entirely inside LichtFeld Studio's UI. No command-line tools, no manual COLMAP runs, no separate masking step. Select a video, choose your settings, click Process.

## How It Works

The pipeline runs in six stages, each configurable from the plugin panel:

1. **Ingest** — Dual fisheye containers (.osv, .insv) are automatically demuxed into separate lens streams and stitched to equirectangular using ffmpeg with camera-family-specific parameters. Pre-stitched ERP video and pre-split front/back .mp4 pairs are also accepted.

2. **Extract** — Frames are pulled from the video at a configurable FPS. Four sharpness modes control how much analysis goes into selecting the best frame from each interval — from no analysis (fastest) to full Tenengrad/Laplacian scoring of every candidate frame (sharpest results).

3. **Mask** — When SAM 3 is installed and masking is enabled, the plugin automatically detects and masks the camera operator (and any other objects specified by prompt keywords) from every extracted frame. For ERP/Pinhole modes, detection runs on pinhole crops and results are back-projected and merged into a full-resolution ERP mask. For fisheye modes, detection runs directly on the raw fisheye frames.

4. **Reframe** — In the Pinhole output modes, each equirectangular frame is reprojected into multiple pinhole perspective views arranged in a virtual camera rig. Built-in presets range from 6 views (cubemap) to 24 views (ultra-dense Fibonacci spiral). The native modes skip this stage for the reconstruction itself — COLMAP 4.1 aligns equirectangular and fisheye frames directly — but reframing is still used for ERP masking detection and for deriving a Pinhole training output from a native solve.

5. **Align** — The images are fed to COLMAP 4.1 for structure-from-motion reconstruction: pinhole views in the Pinhole modes, the source ERP or fisheye frames in the native modes. In the Pinhole modes the plugin configures COLMAP with rig constraints that lock the geometric relationship between virtual cameras from the same panorama, since they share an exact optical center. This produces a single consistent reconstruction across the full sequence. Feature extraction uses SIFT or ALIKED (N16/N32) with Bruteforce or LightGlue matching. Sequential or exhaustive pair selection is available, with a configurable overlap window for sequential mode.

6. **Output** — The final dataset is written in one of four output modes (see below) and can be imported directly into LichtFeld Studio for Gaussian Splatting training.

## Supported Input

| Format | Source | Notes |
|--------|--------|-------|
| ERP video (.mp4) | Any pre-stitched equirectangular video | Standard 2:1 aspect ratio |
| .OSV container | DJI Osmo 360 | Dual fisheye, camera family auto-detected |
| .INSV container | Insta360 cameras | Dual fisheye, camera family auto-detected |
| Front + back .mp4 | Pre-split fisheye lens pair | Two-file mode for graded or externally processed footage |
| Image folder | Already-extracted frames | Equirectangular, or dual fisheye (one folder or two); skips extraction |

### Select Image Folder

You can skip video extraction entirely and point the plugin at a folder of
frames you already have. Click **Image Folder** on the input screen,
then choose a projection:

- **Equirectangular**: one folder of 360 frames.
- **Fisheye**: a dual-lens set, as either **One** folder (files named
  `front...` and `back...` so each pair matches, e.g. `front_0001.jpg` and
  `back_0001.jpg`) or **Two** folders (front and back). Pick the **Camera**
  family so calibration matches. Staged pairs are renamed to matched index
  names (`000001.jpg`, each file keeping its own extension) so each
  front/back pair shares a basename, and for Native and Both outputs the
  native dataset's `paired_extraction_manifest.json` (beside its `images/`
  folder, i.e. in `colmap/` or `colmap/native/`) maps every staged name back
  to your original file.

**Masks**: **Generate with SAM 3**, **Use pre-existing masks** (available for
Equirectangular + Pinhole output; point at a folder of masks named to match
your frames), or **None**.

**Training output** (shown for **Native**, both projections): **Native**,
**Pinhole**, or **Both**. **Pinhole** and **Both** derive the pinhole crops from
the native reconstruction's poses — no second COLMAP run.

**Where the output goes.** One rule covers every image-folder run:

| Training output | Datasets | Where |
|-----------------|----------|-------|
| Native | 1 | `<output>/colmap/` |
| Pinhole | 1 | `<output>/colmap/` |
| Both | 2 | `<output>/colmap/native/` + `<output>/colmap/pinhole/` |

Each dataset is self-contained — its masks live inside it (`colmap/masks/`),
because `transforms.json` references them relatively. Pinhole output
additionally keeps the source-projection masks at `<output>/masks/` as a
reusable deliverable: the equirectangular masks for ERP input, or the lens
masks for fisheye input — named after your original images and mirroring your
source folder layout (one folder → flat, front + back → two folders).

**What happens to your source folder.** **Native** output *absorbs* the source:
your frames become the dataset's images. Equirectangular frames are read where
they sit and moved into the dataset once COLMAP succeeds; fisheye frames are
copied into the dataset before the solve (the two-lens intrinsics need them
there) and the originals removed only after it succeeds. That removal takes only
image files — the frames themselves are preserved inside the dataset — and
anything else in the folder (a sidecar file, a subfolder) is never deleted;
it just blocks removal of the emptied folder, which stays in place. **Pinhole** output never touches the
source at all: the native solve it needs runs in a temporary workspace that is
discarded afterwards.

Your source folder may live inside the Output Path (e.g. `<output>/source/`), as
long as it isn't where the run writes: `colmap/`, `masks/`, or `metadata/`.

## Output Modes

Output mode is two independent choices: the **projection** (equirectangular or fisheye, detected from your input) and the **Output Mode** dropdown (**Native** or **Pinhole**). The four combinations are the modes below.

### ERP

Native equirectangular reconstruction using COLMAP's `EQUIRECTANGULAR` camera model. Feeds ERP frames straight to COLMAP with no pinhole reframing step.

A **Training output** selector (Native / Pinhole / Both) controls what is
written from the single native reconstruction, exactly as for image folders:
one dataset lands in `<output>/colmap/`; **Both** writes
`colmap/native/` + `colmap/pinhole/`, with the pinhole crops' poses propagated
from the native solve (no second COLMAP run). **Pinhole** ships the propagated
pinhole dataset at `colmap/`; with **Keep frames & masks** on (the default) it
also keeps the extracted ERP frames at `<output>/images/` and the ERP masks at
`<output>/masks/` as reusable deliverables, and unchecking it ships only
`colmap/`.

### ERP (Pinhole)

Standard COLMAP pinhole dataset. Each source frame produces multiple perspective crops (6–24 depending on preset). The output is a conventional COLMAP sparse reconstruction with per-view images, masks, and a rig config — packaged under `<output>/colmap/`. For video input, the **Keep frames & masks** checkbox (on by default) also keeps the extracted ERP frames at `<output>/images/` and ERP masks at `<output>/masks/` as reusable deliverables; unchecking it ships only `colmap/`. An image-folder source is read in place and left untouched.

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

The original direct-pinhole path: reframes each dual fisheye lens into 8 pinhole crops (16 total per source frame), aligns the two front-lens reference views in COLMAP under a mini-rig constraint, and propagates the remaining crops' poses from that solve. Kept for compatibility, but for a pinhole dataset from fisheye input the **Fisheye** mode's **Pinhole** training output is now recommended — reconstructing the reframed crops directly registers noticeably fewer frames than the native path.

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
2. Load your input:
   - **360° Video** for a single file (ERP, .osv, or .insv)
   - **Front + Back Lens Video** for pre-split fisheye .mp4 pairs
   - **Image Folder** for already-extracted frames (see [Select Image Folder](#select-image-folder))
3. Choose an **Output Path** and **Output Mode**. For the **ERP** and **Fisheye** modes, a **Training output** selector (Native / Pinhole / Both) chooses which dataset(s) to write from the native reconstruction.
4. Configure each pipeline stage using the collapsible sections:

   **Frame Extraction** — `FPS` sets the extraction rate. `Sharpness` controls frame selection quality (None / Basic / Better / Best). The blur metric can be switched between Tenengrad and Laplacian. Check **Extract all frames** to skip scoring entirely and save every frame, which suits timelapses and image sequences where each frame matters; the FPS, Sharpness, and Blur Metric controls grey out and the estimate shows the true frame total.

   **Masking** — Enable masking and enter prompt keywords (e.g. `person, tripod`). SAM 3 detects and masks matching objects in every frame.

   **Reframe & Alignment** — `Preset` selects the virtual camera rig layout (Pinhole mode). `Features` chooses the extractor (SIFT, ALIKED N16 rot, ALIKED N32). `Matcher Type` chooses the matcher (Bruteforce or LightGlue). `Matcher` selects pair strategy (Sequential with configurable overlap, or Exhaustive). `Mapper` selects Incremental (supports rig constraints) or Global/GLOMAP (faster, no rig support). `BA Solver` picks the bundle-adjustment backend: Hybrid (default), Caspar, Ceres GPU, or Ceres CPU.

   **Output Quality** — `Crop Size` sets pinhole view resolution. `COLMAP Preset` (Normal / High / Custom) snaps max features, max image size, and the match budget to per-mode defaults; `Match Limit` (Fast / Balanced / Default / High / Custom) picks the per-pair match budget, and `Max. Matches` sets the cap directly.

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

The Advanced disclosure also offers **Affine + DSP** (SIFT only, off by default): enables affine shape estimation and domain-size pooling, which improve matching under strong perspective and scale variation at a significant CPU cost — these covariant detectors bypass GPU SIFT extraction.

### Matching

| Matcher Type | Description |
|--------------|-------------|
| Bruteforce | Standard nearest-neighbor descriptor matching. Fast and reliable. |
| LightGlue | Learned feature matcher from COLMAP 4.1. Generally produces more precise correspondences than brute-force, especially for challenging viewpoint changes. |

All 6 combinations of extractor and matcher are supported. The required ONNX model files (aliked-n16rot, aliked-n32, aliked-lightglue, sift-lightglue, bruteforce-matcher) are bundled in `lib/` and validated before each COLMAP run.

### Pair Selection

- **Sequential** matches each image against its temporal neighbors. The **Overlap** slider (2–20) controls the neighborhood size. Faster for video sequences with smooth motion.
- **Exhaustive** tests all possible image pairs. Slower but more robust for irregular capture patterns or scenes that need global loop closure.
- **Loop Closure** can be enabled alongside sequential matching to add vocabulary-tree-based global pair candidates, helping close loops in sequences that revisit earlier viewpoints. Every mode honors it except the ERP direct-pinhole solves (video and image folder), which force it off — their rig-constrained sequential pairs are sufficient.
- **Guided Matching** (off by default) re-matches geometrically verified pairs under their estimated epipolar geometry, recovering extra correspondences on difficult pairs at the cost of a second matching pass.

### Mapper

| Mapper | Description |
|--------|-------------|
| Incremental | Standard COLMAP incremental mapper. Supports rig constraints, which are required for pinhole-reframed 360° data where multiple virtual cameras share the same optical center. |
| Global (GLOMAP) | COLMAP 4.1's global SfM mapper. Faster for large datasets but does not support rig constraints. A warning is shown when selected with rig-dependent output modes. |

### Match Limit

| Preset | Max Matches / Pair | Use Case |
|--------|-------------------:|----------|
| Fast | 8,192 | Quick iterations on easier scenes |
| Balanced | 16,384 | General purpose |
| Default | 32,768 | Recommended starting point |
| High | 65,536 | Difficult scenes that need dense correspondence |
| Custom | User-defined | Manual tuning |

## Reframing Presets

The Pinhole modes use reframing to convert the source footage into pinhole views for COLMAP alignment; the native modes reconstruct the source frames directly and reframe only for ERP masking detection or a Pinhole training output.

### Pinhole Mode

The user-selectable preset controls the virtual camera rig. All presets produce 90° field-of-view pinhole views:

| Preset | Views | Layout |
|--------|------:|--------|
| Cubemap | 6 | 4 horizon faces plus top and bottom |
| Low | 12 | Golden-angle spiral from zenith to nadir |
| Medium | 16 | 8+8 two-ring layout at ±35° with staggered upper ring |
| High | 20 | Golden-angle spiral from zenith to nadir |
| Ultra | 24 | Golden-angle spiral from zenith to nadir |

### Fisheye (Pinhole) Mode

Each dual fisheye lens is reframed into 8 pinhole crops (16 views total per source frame), using the fisheye-specific calibration for each camera family. COLMAP aligns only the two front-lens reference views, joined by a mini-rig constraint; every other view's pose is propagated from the reference solve.

## Output Structure

### Pinhole (direct COLMAP solve, video input)

The `images/` + `masks/` deliverables below are kept when **Keep frames &
masks** is on (the default) and dropped when it's off (only `colmap/` ships).
Image-folder Pinhole runs read the source in place and never promote an
`images/` folder (masks still export to `<output>/masks/`).

```text
output_dir/
├── images/                       extracted ERP frames (kept deliverable)
├── masks/                        ERP masks (when masking enabled)
├── extraction_manifest.json
├── colmap/                       THE dataset — point LFS here
│   ├── images/                   pinhole views, camera-first layout
│   │   ├── 00_00/ 00_01/ ...       folder = virtual sensor, shared
│   │   │                            filename across folders = rig frame
│   ├── masks/                    per-view masks (when masking enabled)
│   ├── sparse/0/                 COLMAP sparse reconstruction
│   ├── rig_config.json
│   ├── database.db
│   └── colmap_debug.log
└── metadata/
```

The camera-first naming inside `colmap/images/` is required by COLMAP's rig
constraint workflow.

### ERP (Native / Pinhole / Both via Training output)

```text
output_dir/                        Training output = Native
├── colmap/                        THE dataset
│   ├── images/                    the ERP frames
│   ├── masks/                     ERP masks (when masking enabled)
│   ├── sparse/0/  database.db
│   ├── extraction_manifest.json
│   ├── pointcloud.ply
│   └── transforms.json            camera_model: EQUIRECTANGULAR
└── metadata/
```

**Both** nests two self-contained datasets instead: `colmap/native/` (as
above) plus `colmap/pinhole/` (flat view-prefixed crops `00_00_frame.jpg`,
matching masks, propagated `sparse/0`, PINHOLE `transforms.json`).
**Pinhole** ships only the propagated dataset at `colmap/` and keeps the
extracted ERP frames at `output_dir/images/` + masks at `output_dir/masks/`
as deliverables.

### Fisheye

For video input, the **Training output** selector determines the layout (image-folder runs follow the unified `colmap/` rule in [Select Image Folder](#select-image-folder)). **Native** and **Both** write self-contained datasets in their own subfolders; **Pinhole** ships a single dataset in `colmap/`, matching the ERP (Pinhole) layout.

**Native** / **Both** — each dataset self-contained (its own `sparse/0`, `transforms.json`, images, masks, point cloud):

```text
output_dir/
├── native/                        (Native or Both)
│   ├── images/ front/ back/       fisheye lens frames
│   ├── masks/ front/ back/        (when masking enabled)
│   ├── sparse/0/                  COLMAP reconstruction (OPENCV_FISHEYE)
│   ├── pointcloud.ply
│   └── transforms.json
└── pinhole/                       (Both only)
    ├── images/                    flat pinhole crops (view_frame.jpg)
    ├── masks/
    ├── sparse/0/                  COLMAP model (PINHOLE; poses derived from native)
    ├── pointcloud.ply
    └── transforms.json
```

**Pinhole** — single dataset in `colmap/`; with **Keep frames & masks** on (default) the extracted lens frames + masks are also kept at the output root (unchecking ships only `colmap/`):

```text
output_dir/
├── images/ front/ back/           extracted lens frames (Keep frames & masks)
├── masks/ front/ back/            lens masks (Keep frames & masks)
└── colmap/                        THE dataset — point LFS here
    ├── images/                    flat pinhole crops (view_frame.jpg)
    ├── masks/
    ├── sparse/0/                  COLMAP model (PINHOLE; poses derived from native)
    ├── pointcloud.ply
    ├── transforms.json
    └── colmap_debug.log
```

Point LichtFeld Studio at `colmap/` (Pinhole), or the `native/` / `pinhole/` subfolder you want to train.

## Dependencies

Installed automatically unless noted:

- **opencv-contrib-python** — frame extraction, blur scoring, image processing (CPU wheel by default; an opt-in CUDA build for GPU-accelerated extraction is available from the panel, see [GPU Extraction](#gpu-extraction-optional))
- **numpy** — coordinate math, rotation matrices
- **pycolmap** — COLMAP 4.1 Python bindings. On Windows, GPU-enabled wheels from [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap). On Linux, CPU build from PyPI.
- **static-ffmpeg** — bundled ffmpeg for video decoding and fisheye stitching
- **SAM 3** — optional, gated HuggingFace model. Installed from inside the plugin after access verification.

ONNX model files for ALIKED, LightGlue, and Bruteforce matching are bundled in `lib/`.

### GPU Extraction (optional)

On machines with an NVIDIA GPU, the Frame Extraction panel shows a
one-click **Enable GPU Extraction** button. It decodes video on the GPU
(NVDEC) and scores sharpness with CUDA, much faster than the CPU path on
long or high-resolution clips.

- Enabling downloads about 1.2 GB of GPU runtime and needs one restart to
  take effect. About 2.9 GB of disk is used while it is active.
- The CPU build is always the default. If a plugin update resets
  extraction to CPU, the panel shows a one-click **Re-enable**.
- Nothing installs without your click, and only machines with an NVIDIA
  GPU ever see the button.

This ships as beta until it is certified on machines without CUDA
installed.

#### How it works (maintainer note)

The CPU wheel pinned in the lock is the safety floor. When GPU extraction
is enabled, the plugin stages the CUDA OpenCV build and NVIDIA runtime
DLLs, then swaps them in at the next load before anything imports OpenCV
(a running LichtFeld Studio holds the module locked). The installed CUDA
build is registered under the baseline pin's version so the host's
dependency sync preserves it instead of reverting to CPU; the NVIDIA DLLs
live in the plugin's own `lib/gpu/` and are never touched by the sync. If
a sync ever removes the CUDA build, the panel detects it and offers
re-enable, so the plugin never breaks.

To restore the CPU build manually (rarely needed): with LichtFeld Studio
closed, delete `.venv/Lib/site-packages/cv2`, the
`opencv_contrib_python-*.dist-info` folder next to it, and the plugin's
`lib/gpu/` folder. The next launch reinstalls the CPU build automatically.

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm. GPU-enabled `pycolmap` wheels provided by [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) by lyehe.

## License

GPL-3.0-or-later
