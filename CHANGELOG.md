# Changelog

All notable changes to the 360 Plugin are documented here.

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
