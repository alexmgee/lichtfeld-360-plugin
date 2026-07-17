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
- "Select Image Folder" input. Instead of a video, you can point the plugin
  at a folder of already-extracted frames. Choose Equirectangular for a
  single folder of 360 frames, or Fisheye for a dual-lens set (one folder of
  front... / back... files, or two separate front and back folders). Masking
  offers Generate with SAM 3, Use pre-existing masks (Equirectangular +
  Pinhole only), or None. Extraction is skipped and the pipeline runs
  masking, reframing, and COLMAP on your images. ([#3])
- **Training output** for image folders, on both projections: Native,
  Pinhole, or Both. Pinhole and Both derive the pinhole crops from the
  native reconstruction's poses rather than running COLMAP a second time,
  so the crops inherit the native registration. ([#3])
- **Training output for video ERP input** (Native / Pinhole / Both), the same
  option as image folders: Both writes the native dataset plus propagated
  pinhole crops; Pinhole ships only the propagated dataset and keeps the
  extracted ERP frames (`<output>/images/`) and the ERP masks
  (`<output>/masks/`) as reusable deliverables. ([#3])

### Changed
- **BREAKING: video ERP output moved under `<output>/colmap/`.** Both ERP
  modes now follow the same unified rule as image-folder runs. Native: the
  dataset (images, masks, sparse, transforms.json, pointcloud.ply) that
  previously landed at the output root lives in `colmap/`; Training output
  Both → `colmap/native/` + `colmap/pinhole/`. Pinhole (direct solve): the
  COLMAP dataset (crops, per-view masks, sparse, database, rig config) is
  packaged under `colmap/` instead of being spread across the output root,
  and the extracted ERP frames + masks are kept at `<output>/images/` and
  `<output>/masks/` instead of being buried in `extracted/`. The `extracted/`
  work folder no longer outlives any run (the extraction manifest moves
  beside the frames). Update anything that pointed at
  `<output>/transforms.json` to `<output>/colmap/transforms.json`; in-app
  auto-import follows the new locations automatically. ([#3])
- Every image-folder run now follows one output rule: a single dataset lands
  in `<output>/colmap/`, and Training output = Both writes two datasets side
  by side in `<output>/colmap/native/` and `<output>/colmap/pinhole/`. Each
  dataset is self-contained, with its masks inside it. Previously the fisheye
  image-folder paths inherited two different hard-coded layouts from the
  video pipeline and landed in inconsistent places. ([#3])
- Native output now absorbs your source folder: the frames become the
  dataset's images and the emptied folder is removed, but only ever after
  COLMAP succeeds and the frames are safely inside a dataset being kept. The
  removal takes only image files — a sidecar file or subfolder stops it and
  leaves the folder alone. Pinhole output never touches your source; the
  native solve it needs runs in a temporary workspace that is discarded.
  ([#3])
- Image-folder runs no longer report "Frame Extraction" while reading your
  folder; progress now reads "Reading Image Folder". The fisheye Folders
  choice is an inline One folder / Front + back control instead of a
  dropdown that overlapped the row beneath it. ([#3])
- Choosing an image folder now defaults the Output Path to that folder's
  PARENT (picking `Folder/Images` outputs to `Folder/`), matching the
  supported source-in-a-subfolder layout instead of inventing a sibling
  `_LFS360` folder. Loading a video no longer auto-fills the Output Path at
  all -- you choose it. An already-set Output Path is never overwritten.
  ([#3])
- Fisheye image-folder polish from QA: the kept `<output>/masks/` deliverable
  (Pinhole output) is named after your original images and mirrors your
  source layout (one folder → flat, front + back → two folders); the staging
  manifest records each pair's original filenames alongside the staged
  `000001.jpg` names; absorbing a two-folder source no longer leaves the
  emptied parent folder behind; and the completion summary reports the
  frame/registration counts for image-folder runs instead of zeros. ([#3])
- The Output Mode dropdown is now just Native or Pinhole. The projection
  (equirectangular or fisheye) is detected from your input, so the same two
  choices apply to both, in place of the previous five-entry list. Switching
  Native or Pinhole now also applies the matching COLMAP matcher and mapper
  corrections consistently, including on auto-detected fisheye input. ([#3])
- **Keep frames & masks** (the former "Keep extracted data" checkbox,
  relabeled and now **on by default**) governs whether a Pinhole run keeps the
  extracted source frames + masks as `<output>/images/` + `<output>/masks/`
  deliverables next to the `colmap/` dataset. It covers the three video paths
  that ship a pinhole dataset: **ERP (Pinhole)**, native **Fisheye → Training
  output = Pinhole**, and **ERP (Native) → Training output = Pinhole**.
  Unchecking ships only the dataset. The native Fisheye → Pinhole path
  previously deleted the native fisheye frames and masks outright; it now
  keeps them (default) and ships its dataset in `<output>/colmap/` (unified
  with ERP (Pinhole)) instead of `<output>/pinhole/`, always preserving the
  COLMAP solve log. ([#3])
- The transient `extracted/` work directory is now **always removed** at the
  end of every run, including fisheye runs — it no longer survives when frames
  are retained (retention promotes them to `<output>/images/` + `<output>/masks/`
  instead of leaving an `extracted/` folder behind). ([#3])

### Removed
- The **"Fisheye (Pinhole)"** output mode — the legacy direct-reframe path
  that cut each lens into 8 pinhole crops, solved
  two front reference views under an inline mini-rig, and propagated the other
  14 by rig geometry. Fisheye input now has exactly one pinhole route: the
  **Fisheye** mode with **Training output = Pinhole** (or **Both**), which
  solves the raw lenses natively and derives the crops from the COLMAP-refined
  poses — registering substantially more frames. Fisheye no longer uses the
  Native/Pinhole processing axis at all: the Output Mode dropdown is hidden for
  fisheye input, and its output is chosen solely by the Training output
  selector (Native / Pinhole / Both). ([#3])
- The **Insta360 calibration JSON picker** (the "Calibration" file field) and
  its `dual_fisheye_calibration_path` setting, which existed only to override
  the reframing intrinsics of the retired mode. The surviving native path
  builds its calibration from the COLMAP-refined reconstruction instead. The
  inline mini-rig, the rig-propagated transforms writer, and the dual-fisheye
  calibration provider module were removed along with it. ([#3])
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
