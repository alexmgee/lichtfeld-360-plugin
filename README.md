# 360 Plugin for LichtFeld Studio

Prepare a 360 equirectangular video for Gaussian splatting directly inside LichtFeld Studio.

The plugin extracts frames, optionally masks the camera operator with SAM 3, reframes the video into a virtual camera rig, runs COLMAP alignment, and imports the finished dataset back into LichtFeld.

## Features

- Import-ready 360 video processing inside LichtFeld Studio
- Optional SAM 3 masking
- Multiple output rig presets
- COLMAP-ready dataset generation
- Process-only or process-and-import workflows

## Installation

### Plugin Manager

1. Open the LichtFeld Studio Plugin Manager.
2. Install `alexmgee/lichtfeld-360-plugin`.
3. Restart LichtFeld Studio if needed.

### Manual Install

Clone the repo into your LichtFeld plugins folder:

```bash
git clone https://github.com/alexmgee/lichtfeld-360-plugin.git ~/.lichtfeld/plugins/lichtfeld-360-plugin
```

## SAM 3 Masking

Masking is optional. The plugin works without masking if SAM 3 has not been set up yet.

SAM 3 is a gated HuggingFace model. First-time setup in the plugin is:

1. Request access to `facebook/sam3`
2. Open the HuggingFace user tokens page and create a token with `Read` access
3. Paste the token into the plugin and click `Verify Access`
4. Click `Install SAM 3`

Once SAM 3 is installed, masking is ready in the plugin and enabled by default.

## Presets

The preset controls the final output camera rig:

| Preset | Views |
| --- | ---: |
| Cubemap | 6 |
| Default | 16 |
| Low | 10 |
| Medium | 14 |
| High | 20 |
| Ultra | 24 |

## Basic Workflow

1. Open the `360 Plugin` tab in LichtFeld Studio.
2. Select a 360 video.
3. Choose an output path.
4. Pick the output preset and quality settings.
5. Enable masking if you want SAM 3 masks.
6. Click `Process & Import` or `Process Only`.

## Output Structure

The plugin writes a standard dataset layout similar to:

```text
output_dir/
├── extracted/
│   ├── frames/
│   └── masks/
├── images/
├── masks/
├── sparse/0/
├── database.db
├── rig_config.json
├── timing.json
└── colmap_debug.log
```

- `extracted/frames` contains ERP source frames
- `extracted/masks` contains ERP masks when masking is enabled
- `images` contains the reframed multiview dataset
- `masks` contains per-view masks when masking is enabled
- `masking_diagnostics.json` is written when masking diagnostics are enabled

## Notes

- The plugin targets Python `3.12`
- `pycolmap` is installed automatically
- Windows builds are configured for GPU-enabled `pycolmap` wheels
- SAM 3 is installed from inside the plugin after access is verified

## License

GPL-3.0-or-later
