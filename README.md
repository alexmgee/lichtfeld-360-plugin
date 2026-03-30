# 360 Camera Plugin for LichtFeld Studio

Process 360° video into COLMAP-aligned datasets ready for Gaussian Splatting — directly inside LichtFeld Studio.

## What It Does

Takes a 360° equirectangular video and produces a complete COLMAP dataset:

1. **Extract** frames from your video — four sharpness levels from instant (interval-only) to thorough (full blur analysis with scene detection)
2. **Reframe** each equirectangular frame into pinhole perspective views — five presets from Cubemap (6 views) to Full Sphere (26 views per frame)
3. **Align** all views using COLMAP — sequential or exhaustive matching with rig-aware constraints
4. **Import** the result directly into LichtFeld Studio for training

## Installation

### From Plugin Manager (recommended)

1. Open **Plugin Manager** in LichtFeld Studio
2. Search for **360 Camera** or paste the repo URL: `alexmgee/lichtfeld-360-plugin`
3. Click **Install**

Dependencies are installed automatically on first load.

### Manual

```bash
git clone https://github.com/alexmgee/lichtfeld-360-plugin.git ~/.lichtfeld/plugins/lichtfeld-360-plugin
```

Then restart LichtFeld Studio.

## Usage

1. Open the **360 Camera** tab in the right panel
2. Click **Select 360° Video** and choose your equirectangular video file
3. Choose an **Output Path**
4. Adjust settings by section:

   **Frame Extraction**
   `FPS` controls how often frames are extracted from the video, and `Sharpness` controls how carefully the plugin searches each interval for the best frame.

   **Reframe & Alignment**
   `Preset` controls how many pinhole views are generated from each 360 frame, and `Matcher` controls how COLMAP searches for matching image pairs across the sequence.

   **Output Quality**
   `Crop Size` sets the resolution of each pinhole view, `Match Limit` chooses one of the built-in COLMAP matching tiers, and `Max. Matches` sets the actual per-pair correspondence cap COLMAP may keep.
5. Click **Process & Import** to process and load the result, or **Process Only** to just create the dataset

After every run, the panel's **Run Diagnostics** block shows:

- stage timings
- matcher, match-limit tier, and max-match setting
- extracted frames vs written images
- registered rig frames and images
- per-view registration counts
- the failing error message and log path if COLMAP aborts

## Output

The plugin creates a standard COLMAP dataset layout:

```
output_dir/
├── extracted/
│   └── frames/       source equirectangular frames
├── images/           reframed pinhole views grouped by virtual camera
│   ├── 00_00/
│   │   ├── frame_0001.jpg
│   │   └── frame_0002.jpg
│   ├── 00_01/
│   └── ...
├── sparse/
│   └── 0/            COLMAP reconstruction
├── database.db
└── rig_config.json
```

The `images/` tree is camera-first on purpose: COLMAP's rig workflow uses the
folder prefix to identify the virtual sensor, and the shared filename across
folders to group the images into rig frames.

## Reframing Presets

| Preset | Views | Coverage |
|--------|-------|----------|
| Cubemap | 6 | 4 horizon, 1 top, 1 bottom |
| Balanced | 9 | 6 horizon, 2 below, zenith |
| Standard | 13 | 8 horizon, 4 below, zenith |
| Dense | 17 | 8 horizon, 8 below, zenith |
| Full | 26 | 8 above, 8 horizon, 8 below, zenith, nadir |

## Extraction Sharpness

Sharpness controls how much work the plugin does to choose the best frame from each time interval.

In every mode except **None**, the plugin looks at candidate frames first and only saves the winner. Higher tiers inspect more frames before deciding which one to keep, and the scene-aware tiers can also split an interval around a cut so both sides get a representative frame.

| Sharpness | Analysis Method | Extraction Result |
|---------|-----------------|-------------------|
| None | No sharpness analysis | Saves exactly 1 frame per interval |
| Fast | Tests 3 candidate frames per interval with a lightweight Laplacian sharpness score at 480 px | Saves the sharpest of those 3 candidates |
| Normal | Runs FFmpeg `blurdetect` plus scene scoring on a subsampled analysis stream at about 5× the target extraction rate, using a 640 px analysis width | Saves the sharpest frame per interval, with possible extra splits at scene changes |
| Maximum | Runs FFmpeg `blurdetect` plus scene scoring on every source frame, using a 1280 px analysis width | Saves the strongest frame choice it can find, with possible extra splits at scene changes |

### What Each Tier Means

- **None** — The plugin saves exactly 1 frame at each target interval and does no sharpness analysis at all. This is the fastest option, but it is also the most likely to keep a blurry frame if the camera is moving.
- **Fast** — The plugin checks 3 candidate frames around each interval and scores them with a lightweight Laplacian sharpness check at 480 px, then keeps the sharpest one. This gives you a quick sharpness boost over **None** without adding much extra processing time.
- **Normal** — The plugin runs FFmpeg `blurdetect` plus scene scoring on a lighter analysis stream at about 5× your target extraction rate, using a 640 px analysis width. It then picks the sharpest frame in each interval, and it can split intervals at scene changes, so difficult footage may produce slightly more than 1 saved frame per target interval.
- **Maximum** — The plugin runs FFmpeg `blurdetect` plus scene scoring on every source frame in the clip, using a 1280 px analysis width. It then picks the sharpest frame from each interval or scene-split sub-interval. This is the slowest option, but it gives the plugin the best chance of avoiding weak or blurry selections when motion, cuts, or changing content make the video harder to sample cleanly.

## Dependencies

- opencv, numpy, pycolmap, static-ffmpeg (installed automatically)
- On Windows, pycolmap is GPU-accelerated via [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) (CUDA 12.8, installed automatically)
- On Linux, pycolmap falls back to the CPU version from PyPI

## COLMAP Match Limit

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

Guidance:

- Start with **Default** for most 360 video.
- Try **High** if COLMAP logs repeated `Clamping features...` warnings or
  drops too many whole rig frames.
- Lower the match limit if you need to save time or memory.

## Diagnostics

Successful and failed runs both write debug artifacts into the output folder:

- `timing.json` — structured timing and registration summary
- `colmap_debug.log` — detailed COLMAP stage log

These same results are surfaced in the panel's **Last Run** summary so you can
inspect registration behavior without digging through the Python console.

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm. GPU-accelerated pycolmap wheels from [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) by lyehe.

## License

GPL-3.0-or-later
