# Changelog

All notable changes to the 360 Plugin are documented here.

## [Unreleased]

### Added
- "Extract all frames" toggle in the Frame Extraction section. When
  enabled, every frame of the video is decoded and saved with no sharpness
  scoring and no interval sampling, for timelapses and other sources where
  each frame matters. Applies to both single 360 videos and dual-fisheye
  inputs. The FPS, Sharpness, and Blur Metric controls grey out and the
  estimate shows the true frame total from the video metadata. ([#3])
- Loading a video that is still being written to disk (a copy or export in
  progress) is now blocked with a clear message to wait and reload, instead
  of failing later with a confusing error. All-frames extraction also stops
  with a clear message when it produces far fewer frames than the video
  reports, which usually means an incomplete or corrupt source.
- One-click opt-in GPU-accelerated frame extraction (NVDEC decode + CUDA
  blur scoring) for machines with an NVIDIA GPU. The plugin vendors the
  required runtime DLLs, so no system CUDA installation is needed. The CPU
  baseline stays the default and the automatic fallback. Enabling
  downloads about 1.2 GB and takes one restart to activate; if a plugin
  update ever resets extraction to CPU, the panel offers one-click
  re-enable. Ships as beta until certified on machines without CUDA
  installed.

### Changed
- The Output Mode dropdown is now just Native or Pinhole. The projection
  (equirectangular or fisheye) is detected from your input, so the same two
  choices apply to both, in place of the previous five-entry list. Switching
  Native or Pinhole now also applies the matching COLMAP matcher and mapper
  corrections consistently, including on auto-detected fisheye input. ([#3])

### Removed
- The "ERP (Scaffold)" output mode and its "Keep pinhole scaffolding"
  checkbox. COLMAP aligns equirectangular frames directly now, so the pinhole
  scaffold workaround is no longer needed. The ERP mode (Native
  equirectangular) produces the equivalent transforms.json. ([#3])

## [0.2.0] - 2026-07-10

### Fixed
- The plugin no longer requires the CUDA build of OpenCV. The baseline
  install now uses the standard CPU wheel from PyPI
  (opencv-contrib-python 4.13.0.92), which loads on any Windows machine.
  Previously the pinned CUDA wheel required CUDA 13 runtime DLLs that
  most systems don't have, which broke video selection and processing
  entirely. The version is deliberately newer than the old CUDA wheel's
  (4.13.0.90) so existing installs converge automatically on their next
  dependency sync. ([#6], [#8])
- Selecting a video no longer fails with a misleading "No video loaded"
  when a heavy dependency (OpenCV / torch / pycolmap) is broken:
  input-type detection now lives in a dependency-free module, and
  pipeline start reports the real import error in the panel. ([#6])

- The SAM 3 runtime no longer disappears on every LichtFeld Studio
  restart. The host runs a plain dependency sync when loading the plugin,
  which uninstalls anything that only an optional extra pulls in — so an
  installed SAM 3 runtime was silently removed on every restart and
  showed "Runtime: Missing" again. The runtime packages (plain PyPI,
  ungated) now ship in the base dependencies; the gated model weights
  are unchanged and still require HuggingFace token + access via the
  panel. Likely the root cause behind "runtime/weights missing" loops
  reported in [#2] and [#5].

### Added
- `gpu-opencv` optional dependency placeholder: GPU-accelerated frame
  extraction (CUDA OpenCV) is now opt-in instead of mandatory. A
  panel-driven one-click install is planned in a follow-up; the CPU
  baseline works everywhere without it.

[#2]: https://github.com/alexmgee/lichtfeld-360-plugin/issues/2
[#5]: https://github.com/alexmgee/lichtfeld-360-plugin/issues/5
[#6]: https://github.com/alexmgee/lichtfeld-360-plugin/issues/6
[#8]: https://github.com/alexmgee/lichtfeld-360-plugin/issues/8
