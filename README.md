# 360 Camera Plugin for LichtFeld Studio

Process 360° video into COLMAP-aligned datasets ready for Gaussian Splatting — directly inside LichtFeld Studio.

## What It Does

Takes a 360° equirectangular video and produces a complete COLMAP dataset:

1. **Extract** the sharpest frames from your video — four quality levels from instant (interval-only) to thorough (full blur analysis with scene detection)
2. **Reframe** each equirectangular frame into pinhole perspective views — five presets from Cubemap (6 views) to Full Sphere (26 views per frame)
3. **Align** all views using COLMAP — sequential or exhaustive matching with rig-aware constraints
4. **Import** the result directly into LichtFeld Studio for training

No command line. No manual COLMAP setup. One panel, one click.

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
3. Adjust settings:
   - **FPS** — how many frames per second to extract (0.1–5.0)
   - **Quality** — extraction sharpness level (None / Fast / Normal / Maximum)
   - **Preset** — reframing coverage (Cubemap 6 views → Full 26 views)
   - **Crop Size** — output resolution per pinhole view (960–1920 px)
4. Click **Process & Import** to process and load the result, or **Process Only** to just create the dataset

## Output

The plugin creates a standard COLMAP dataset layout:

```
output_dir/
├── extracted/
│   └── frames/       source equirectangular frames
├── images/           reframed pinhole perspective views
├── sparse/
│   └── 0/            COLMAP reconstruction
├── database.db
└── rig_config.json
```

## Reframing Presets

| Preset | Views | Coverage |
|--------|-------|----------|
| Cubemap | 6 | 4 horizon, 1 top, 1 bottom |
| Balanced | 9 | 6 horizon, 2 below, zenith |
| Standard | 13 | 8 horizon, 4 below, zenith |
| Dense | 17 | 8 horizon, 8 below, zenith |
| Full | 26 | 8 above, 8 horizon, 8 below, zenith, nadir |

## Extraction Quality

All quality levels analyze the video *before* extracting — only the winning frames are saved to disk.

| Quality | Analysis | Extraction |
|---------|----------|------------|
| None | No analysis | Saves one frame per interval directly |
| Fast | Reads 3 candidates per interval into memory, scores sharpness (Laplacian) | Saves only the sharpest candidate |
| Normal | Runs blur analysis on a subsampled stream (~5 candidates/interval) | Extracts only the winners |
| Maximum | Runs blur analysis on every frame at native FPS with scene-aware chunking | Extracts only the winners |

## Dependencies

- opencv, numpy, pycolmap, static-ffmpeg (installed automatically)

## Credits

COLMAP integration adapted from [Lichtfeld-COLMAP-Plugin](https://github.com/shadygm/Lichtfeld-COLMAP-Plugin) by shadygm.

## License

GPL-3.0-or-later
