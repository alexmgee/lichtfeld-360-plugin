# 360 Plugin for LichtFeld Studio

Process 360° video into COLMAP-aligned datasets ready for Gaussian Splatting — directly inside LichtFeld Studio.

## What It Does

Takes a 360° equirectangular video and produces a complete COLMAP dataset:

1. **Extract** frames from your video — four sharpness levels from instant (interval-only) to thorough (full blur analysis with scene detection)
2. **Mask** the camera operator automatically — with ERP masks, per-view masks, and SAM2 video tracking on the Default path when video tracking is installed
3. **Reframe** each equirectangular frame into pinhole perspective views — currently either the Default 16-view layout or the 6-view Cubemap layout
4. **Align** all views using COLMAP — sequential or exhaustive matching with rig-aware constraints
5. **Import** the result directly into LichtFeld Studio for training

## Installation

### From Plugin Manager (recommended)

1. Open **Plugin Manager** in LichtFeld Studio
2. Search for **360 Plugin** or paste the repo URL: `alexmgee/lichtfeld-360-plugin`
3. Click **Install**

Dependencies are installed automatically on first load.

### Manual

```bash
git clone https://github.com/alexmgee/lichtfeld-360-plugin.git ~/.lichtfeld/plugins/lichtfeld-360-plugin
```

Then restart LichtFeld Studio.

## Usage

1. Open the **360 Plugin** tab in the right panel
2. Click **Select 360° Video** and choose your equirectangular video file
3. Choose an **Output Path**
4. Adjust settings by section:

   **Frame Extraction**
   `FPS` controls how often frames are extracted from the video, and `Sharpness` controls how carefully the plugin searches each interval for the best frame.

   **Reframe & Alignment**
   `Preset` controls which virtual camera layout is generated from each 360 frame, and `Matcher` controls how COLMAP searches for matching image pairs across the sequence.

   **Output Quality**
   `Crop Size` sets the resolution of each pinhole view, `Match Limit` chooses one of the built-in COLMAP matching tiers, and `Max. Matches` sets the actual per-pair correspondence cap COLMAP may keep.
5. Click **Process & Import** to process and load the result, or **Process Only** to just create the dataset

After every run, the panel's **Run Diagnostics** block shows:

- stage timings
- matcher, match-limit tier, and max-match setting
- mask backend and video backend used during the run
- extracted frames vs written images
- registered rig frames and images
- per-view registration counts
- the failing error message and log path if COLMAP aborts

## Output

The plugin creates a standard COLMAP dataset layout:

```
output_dir/
├── extracted/
│   ├── frames/       source equirectangular frames
│   └── masks/        ERP masks for the Default preset when masking is enabled
├── images/           reframed pinhole views grouped by virtual camera
│   ├── 00_00/
│   │   ├── frame_0001.jpg
│   │   └── frame_0002.jpg
│   ├── 00_01/
│   └── ...
├── masks/            final per-view masks grouped by virtual camera when masking is enabled
├── sparse/
│   └── 0/            COLMAP reconstruction
├── timing.json       timing and registration summary
├── colmap_debug.log  detailed COLMAP log
├── database.db
└── rig_config.json
```

The `images/` tree is camera-first on purpose: COLMAP's rig workflow uses the
folder prefix to identify the virtual sensor, and the shared filename across
folders to group the images into rig frames.

## Current Presets

| Preset | Views | Description |
|--------|-------|-------------|
| Default | 16 | Two-ring FullCircle-style layout at `±35°` pitch with `90°` FOV. This is the highest-confidence preset and the current mainline path. |
| Cubemap | 6 | 4 horizon faces plus top and bottom at `90°` FOV. This is the faster preset with less reconstruction redundancy than Default. |

## Masking

When masking is enabled, 360 Plugin produces:

- ERP masks during the Default preset path
- final per-view masks in `masks/`
- backend reporting in the run diagnostics summary

Current masking behavior:

- **Default** uses the full ERP masking path and, when video tracking is installed, runs SAM2-backed video propagation
- **Cubemap** masks the 6 final cubemap faces directly and skips the heavier Default synthetic path

The diagnostics summary now records which backends were actually used during the run so you can tell whether SAM2 video tracking was active or whether the plugin fell back to the simpler image-only path.


## Extraction Sharpness

Sharpness controls how much work the plugin does to choose the best frame from each time interval.

In every mode except **None**, the plugin looks at candidate frames first and only saves the winner. Higher tiers inspect more frames before deciding which one to keep, and the scene-aware tiers can also split an interval around a cut so both sides get a representative frame.

| Sharpness | Analysis Method | Extraction Result |
|---------|-----------------|-------------------|
| None | No sharpness analysis | Saves exactly 1 frame per interval |
| Low | Tests 3 candidate frames per interval with a lightweight Laplacian sharpness score at 480 px | Saves the sharpest of those 3 candidates |
| Medium | Runs FFmpeg `blurdetect` plus scene scoring on a subsampled analysis stream at about 5× the target extraction rate, using a 640 px analysis width | Saves the sharpest frame per interval, with possible extra splits at scene changes |
| High | Runs FFmpeg `blurdetect` plus scene scoring on every source frame, using a 1280 px analysis width | Saves the strongest frame choice it can find, with possible extra splits at scene changes |

The default sharpness is **High**. Drop to **Medium** or **Low** if you need faster extraction.

## COLMAP Matching

The **Matcher** setting controls which image pairs COLMAP attempts to compare
before mapping begins. The default is **Exhaustive**.

- **Sequential** matches nearby frames in order. It is usually faster and works
  well for normal video-like motion where each frame mostly overlaps with the
  next few frames.
- **Exhaustive** tries all candidate image pairs. It is slower, but it can
  recover harder scenes where nearby-frame matching is not enough to keep the
  reconstruction connected.

The plugin exposes COLMAP's matching cap in two linked controls:

- **Match Limit** chooses a preset tier
- **Max. Matches** shows the actual `max_num_matches` value COLMAP will use

Choosing a built-in tier updates **Max. Matches** automatically. Selecting
**Custom** lets you tune the numeric limit directly.

This matters because COLMAP will otherwise clamp feature correspondences per
image pair. On reframed 360 datasets, overly small limits can starve the
reconstruction graph and cause entire rig frames to drop out.

Available tiers:

| Match Limit | Max. Matches / Pair | Intended Use |
|------|---------------------|--------------|
| Fast | 8,192 | Faster runs on easier scenes |
| Balanced | 16,384 | Middle ground for general use |
| Default | 32,768 | COLMAP's default and the best starting point for most scenes |
| High | 65,536 | Harder scenes that need more match coverage |
| Custom | User defined | Manual tuning |


## Dependencies

- opencv, numpy, pycolmap, static-ffmpeg (installed automatically)
- On Windows, pycolmap is GPU-accelerated via [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) (CUDA 12.8, installed automatically)
- On Linux, pycolmap falls back to the CPU version from PyPI


## Diagnostics

Successful and failed runs both write debug artifacts into the output folder:

- `timing.json` — structured timing and registration summary
- `colmap_debug.log` — detailed COLMAP stage log

These same results are surfaced in the panel's **Run Diagnostics** summary so you can
inspect registration behavior without digging through the Python console.

## Repo Scope

This public repository is intentionally kept focused on the plugin code and the
files needed to run it inside LichtFeld Studio.

Internal planning notes, investigation writeups, and developer-only harnesses
may exist locally but are not required for using or sharing the plugin.

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm. Parts of the masking approach and reconstruction-layout comparisons were informed by the FullCircle / Reconstruction Zone work. GPU-accelerated pycolmap wheels from [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) by lyehe.

## License

GPL-3.0-or-later
