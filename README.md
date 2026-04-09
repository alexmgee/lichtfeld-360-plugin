(THIS IS A WORK IN PROGESS)

# 360 Plugin for LichtFeld Studio

Process 360° video into COLMAP-aligned datasets ready for Gaussian Splatting, directly inside LichtFeld Studio.

## What It Does

Takes a 360° equirectangular video and produces a complete COLMAP dataset:

1. **Extract** frames from your video — three sharpness modes from instant fixed-interval extraction to full-frame scoring
2. **Mask** the camera operator automatically — equirectangular and pinhole masks using SAM 3 when masking setup and is enabled
3. **Reframe** each equirectangular frame into pinhole perspective views — presets from 12 to 24 camera outputs including cubemap
4. **Align** all views using COLMAP — sequential or exhaustive matching with rig-aware constraints
5. **Import** the result directly back into LichtFeld Studio for training

## Installation

### From Plugin Manager

1. Open **Plugin Manager** in LichtFeld Studio
2. Search for **360 Plugin** or paste the repo URL: `alexmgee/lichtfeld-360-plugin`
3. Click **Install**

The base runtime installs automatically. SAM 3 is optional and is installed from inside the plugin after access is verified.

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
   `FPS` controls how often frames are extracted from the source video. `Sharpness` controls how much analysis the plugin does before choosing each extracted frame.

   **Masking**  
   If SAM 3 is installed, masking can be enabled and targeted with comma-separated prompt keywords such as `person`.

   **Reframe & Alignment**  
   `Preset` controls the final output camera rig. `Matcher` controls how COLMAP searches for image pairs across the sequence.

   **Output Quality**  
   `Crop Size` sets the resolution of each pinhole view. `Match Limit` chooses a built-in COLMAP matching tier, and `Max. Matches` sets the per-pair correspondence cap.

5. Click **Process & Import** to process and load the result, or **Process Only** to create the dataset without importing it

## Masking

Masking is optional; the plugin still works without it if SAM 3 has not been set up yet.

SAM 3 is a gated HuggingFace model. Guided first-time setup inside the plugin is:

1. Request access to `facebook/sam3`
2. Open the HuggingFace **User Tokens** page and create a token with `Read` access
3. Paste the token into the plugin and click **Verify Access**
4. Click **Install SAM 3**

Once SAM 3 is installed, masking is enabled by default in the plugin.

When masking is enabled, the plugin writes:

- ERP masks in `extracted/masks/`
- final per-view masks in `masks/`

## Current Presets

All built-in presets use 90° pinhole views. The preset controls the final output rig:

| Preset | Views | Description |
|--------|------:|-------------|
| Cubemap | 6 | 4 horizon faces plus top and bottom |
| Low | 12 | Fibonacci-spiral freeview layout from zenith to nadir |
| Medium | 16 | Fibonacci-spiral freeview layout from zenith to nadir |
| High | 20 | Fibonacci-spiral freeview layout from zenith to nadir |
| Ultra | 24 | Fibonacci-spiral freeview layout from zenith to nadir |

## Extraction Sharpness

Sharpness controls how much work the plugin does before selecting the frame to keep from each interval. The current plugin exposes three modes: `None`, `Basic`, and `Best`.

| Sharpness | Analysis Method | Extraction Result |
|-----------|-----------------|-------------------|
| None | No sharpness analysis | Saves one frame at each extraction interval |
| Basic | Tests about 10 candidate frames per interval using OpenCV blur scoring | Saves the strongest candidate from each interval |
| Best | Scores every frame in the video using OpenCV blur scoring | Saves the strongest frame choice it can find |

`Best` is the default and gives the strongest frame selection, but takes the most time.  
The blur metric can be switched between **Tenengrad** and **Laplacian**:

- **Tenengrad** measures Sobel gradient energy. It generally favors strong edges and is the default, more noise-robust choice.
- **Laplacian** measures Laplacian variance after a light Gaussian blur. It can be a useful alternate signal when you want a slightly different edge-response profile.

In both cases, higher scores mean sharper frames.

## COLMAP Matching

The **Matcher** setting controls which image pairs COLMAP attempts to compare before mapping begins.

- **Sequential** matches nearby frames in order. It is usually faster and works well for normal video-like motion.
- **Exhaustive** tries all candidate image pairs. It is slower, but can recover harder scenes where nearby-frame matching is not enough to keep the reconstruction connected.

The plugin also exposes COLMAP's matching cap in two linked controls:

- **Match Limit** chooses a built-in tier
- **Max. Matches** shows the actual `max_num_matches` value COLMAP will use

Available tiers:

| Match Limit | Max. Matches / Pair | Intended Use |
|-------------|---------------------:|--------------|
| Fast | 8,192 | Faster runs on easier scenes |
| Balanced | 16,384 | Middle ground for general use |
| Default | 32,768 | Best starting point for most scenes |
| High | 65,536 | Harder scenes that need more match coverage |
| Custom | User defined | Manual tuning |

## Output

The plugin creates a standard COLMAP dataset layout:

```text
output_dir/
├── extracted/
│   ├── frames/       source equirectangular frames
│   └── masks/        ERP masks when masking is enabled
├── images/           reframed pinhole views grouped by virtual camera
│   ├── 00_00/
│   │   ├── frame_0001.jpg
│   │   └── frame_0002.jpg
│   ├── 00_01/
│   └── ...
├── masks/            final per-view masks grouped by virtual camera when masking is enabled
├── sparse/
│   └── 0/            COLMAP reconstruction
├── timing.json       timing and run summary
├── colmap_debug.log  detailed COLMAP log
├── database.db
└── rig_config.json
```

The `images/` tree is camera-first on purpose: COLMAP's rig workflow uses the
folder prefix to identify the virtual sensor, and the shared filename across
folders to group images into rig frames.

## Dependencies

- `opencv-python-headless`, `numpy`, `pycolmap`, and `static-ffmpeg` are installed automatically
- On Windows, `pycolmap` uses GPU-enabled wheels from [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap)
- On Linux, `pycolmap` falls back to the CPU build from PyPI
- SAM 3 is installed from inside the plugin after HuggingFace access is verified

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm. GPU-enabled `pycolmap` wheels provided by [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap) by lyehe.

## License

GPL-3.0-or-later
