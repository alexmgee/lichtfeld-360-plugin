# Masking Layer v1 — Design Spec

**Date:** 2026-04-02
**Status:** Draft
**Author:** Alex Gee + Claude

## Problem

The PanoSplat converts 360 equirectangular video into COLMAP-aligned pinhole datasets for 3D Gaussian Splatting. Without masking, the camera operator is always present in the 360 footage and baked into the resulting splat. This makes the plugin a novelty rather than a production tool.

## Goals

- Equip the basic plugin user with automatic operator masking — no ML expertise required.
- Ship a zero-friction default masking backend (YOLO + SAM v1) that requires no gating, no tokens, no external approvals.
- Offer SAM 3.1 as an optional upgrade for users who want better quality and are willing to go through the gating process.
- Integrate cubemap decomposition for detection (proven in Reconstruction Zone).
- Add closest-camera anti-overlap masks (from FullCircle) to eliminate duplicate COLMAP feature extraction.
- Unblock the python3.dll issue so masking (both tiers) can run in-process within LichtFeld Studio.

## Non-Goals (v1)

- Selfie stick / nadir / camera mount masking (deferred — Reconstruction Zone covers this for power users).
- VOS temporal propagation across frames.
- Alpha matting or shadow detection.
- Preset redesign (follow-up workstream, informed by overlap analysis below).

---

## Architecture

### Pipeline Stages

The masking layer modifies the existing pipeline. The stage order remains: Extract → Mask → Reframe → COLMAP, but masking internals change significantly.

```
Stage 1 — Frame Extraction (0-20%)
  360 video → sharpest ERP frames
  SharpestExtractor — unchanged

       ↓ ERP frames (e.g. 7680x3840)

Stage 2 — Operator Masking (20-45%)  ← NEW
  2a. Cubemap decompose: ERP → 6 perspective faces @ min(1024, w/4)
  2b. SAM3 per-face detection: text-prompted, per-image mode
  2c. Merge + postprocess:
      - CUBEMAP FAST PATH: if preset=cubemap, resize face masks
        to output resolution → write directly as pinhole masks.
        Skip ERP merge and Stage 3 mask reprojection.
      - STANDARD PATH: cubemap2equirect() merge → morph close +
        flood-fill at full ERP resolution → pass to Stage 3.

       ↓ ERP frames + ERP masks (standard path)
       ↓ pinhole images + pinhole masks (cubemap fast path)

Stage 3 — Reframe to Pinhole (45-60%)
  Reframer reprojects images AND masks to pinhole views.
  Existing mask_dir flow — unchanged.

       ↓ pinhole images + pinhole masks

Stage 3.5 — Closest-Camera Mask (new, lightweight)
  Precomputed Voronoi mask per view (purely geometric).
  AND with operator mask: features extracted only where
  (no person) AND (this camera owns the pixel).
  Computed once per preset, reused for every frame.

       ↓ combined masks

Stage 4-5 — Rig Config + COLMAP (60-95%)
  Rig constraints → feature extraction (with mask_path) → matching → mapping.
  COLMAP extracts features only from unmasked (white) regions.
  Unchanged except receives mask_path.
```

### Cubemap Fast Path

When the user selects the cubemap preset, the 6 detection faces and the 6 output views share the same geometry (same orientations, same 90 FOV). The only difference is resolution. This enables a shortcut:

1. SAM3 detects on 6 cubemap faces at detection resolution (~1024px).
2. Resize face masks to the user's configured output resolution.
3. Write directly to `masks/{view_id}/{frame_id}.png`.
4. Skip `cubemap2equirect()` merge entirely.
5. Skip reframer mask reprojection entirely.

For non-cubemap presets (low/medium/high), the detection faces don't align with output views, so the full decompose → ERP merge → reframe path is required.

### Closest-Camera Anti-Overlap Mask

Ported from FullCircle (`omni2perspective.py:148-153`). For each pixel in each pinhole view, determines which camera center has the most similar viewing direction. If this camera is closest → white. If another camera is closer → black.

This is a spherical Voronoi partition — each camera "owns" its nearest region on the sphere. Boundaries are clean, no duplicate features.

**Properties:**
- Purely geometric — computed from camera rotations, no ML.
- Precomputed once per preset configuration, reused for all frames.
- Combined with operator mask via AND before passing to COLMAP.
- Benefit scales with overlap: cubemap (0% overlap) gets no benefit; high preset (32.7% overlap) gets significant benefit.

**Implementation:** For each view, during reframing, compute `closest_camera = argmax(ray · cam_centers)` and generate a binary mask where `closest_camera == this_camera`. The Voronoi mask is precomputed once per preset (one mask per view). At mask output time, AND it with each frame's operator mask to produce the final per-frame combined mask written to `masks/{view_id}/{frame_id}.png`. The Voronoi mask itself is not stored as a file — it's held in memory and applied during mask generation.

For the cubemap preset (zero overlap), skip Voronoi mask computation entirely — all pixels are already owned by exactly one camera.

---

## Mask Conventions

Three consumers need masks in compatible formats. All are verified from source code.

### Polarity

White (255) = keep, black (0) = remove. Consistent across COLMAP, LFS, and the existing reframer.

### Directory Layout

```
output/
  images/
    00_00/frame_001.jpg
    00_01/frame_001.jpg
    ...
  masks/
    00_00/frame_001.png    ← must match image dimensions exactly
    00_01/frame_001.png
    ...
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

### COLMAP

- `reader_opts.mask_path` points to the `masks/` directory.
- COLMAP expects masks to mirror the `images/` subdirectory structure.
- Features extracted only from white regions.
- Source: `colmap_runner.py:514-516`.

### LichtFeld Studio

- Searches for masks in: `masks/`, `mask/`, `segmentation/`, `dynamic_masks/` (in order).
- Matches masks to images by stem name across `.png`, `.jpg`, `.jpeg` extensions.
- Recursive subdirectory search via `MaskDirCache`.
- **Validates that mask dimensions exactly match image dimensions** — returns `MASK_SIZE_MISMATCH` error if not.
- Has `mask_invert` CUDA kernel but expects white=keep from COLMAP datasets.
- Source: `filesystem_utils.hpp:45-54`, `colmap.cpp:830-843`.

### Reframer Output

- Already writes to `masks/{view_id}/{frame_id}.png`.
- Thresholds with `mask_persp > 0` and scales to 0/255.
- Source: `reframer.py:340-381`.

---

## Detection Strategy

### Two-Tier Detection Backend

The plugin ships with two masking backends. The default tier requires zero external approvals. The premium tier is an opt-in upgrade.

#### Default: YOLO + SAM v1 (zero friction)

- **Detection:** YOLOv8 bounding box detection via `ultralytics`. Person = COCO class 0 (conf=0.35, iou=0.6). Other target classes (camera, tripod, etc.) also available as COCO classes.
- **Segmentation:** SAM v1 (`segment-anything`) refines bounding boxes to pixel-accurate masks. Weights auto-download from Meta's public URL — no gating required.
- **Install:** `uv pip install ultralytics segment-anything` into the plugin venv. Automatic on first use, no user action beyond enabling masking.
- **VRAM:** ~2-3 GB.
- **Reference implementation:** FullCircle's `mask_perspectives.py` uses this exact YOLO+SAM v1 approach.
- **python3.dll:** May sidestep the blocker entirely — `ultralytics` and `segment-anything` may not pull in `psutil`/`pycocotools`. Needs verification.

#### Premium: SAM 3.1 (opt-in upgrade)

- **Detection + Segmentation:** Unified text-prompted detection and segmentation. No YOLO needed.
- **Quality:** Better segmentation, especially for ambiguous or partially-occluded subjects.
- **Target flexibility:** Any text prompt (not limited to COCO classes).
- **Install:** Requires Meta license approval → HuggingFace gated model access → HF token → `uv pip install sam3` → weight download. The setup wizard (see First-Time Setup UX) guides users through this process.
- **VRAM:** ~6-8 GB.
- **python3.dll:** Required — SAM 3.1's import chain pulls in `psutil` and `pycocotools` which need the stable ABI DLL.

#### Backend Interface

Both backends implement the same interface: given a perspective image, return a binary mask of detected objects. The cubemap decomposition, postprocessing, and closest-camera mask logic are backend-agnostic — they call the backend per face and handle everything else.

```
class MaskingBackend(Protocol):
    def detect_and_segment(self, image: np.ndarray, targets: list[str]) -> np.ndarray:
        """Return binary mask (0/1 uint8) of detected objects."""
        ...
```

The pipeline selects the backend based on what's installed: SAM 3.1 if available, otherwise YOLO+SAM v1.

### Why Cubemap Decomposition

Both detection backends perform poorly on raw equirectangular images due to distortion. The Reconstruction Zone's approach — decompose to 6 cubemap faces, detect on undistorted perspective images, merge back — is battle-tested across 1200+ frames and applies equally to both backends.

**Source:** `reconstruction_pipeline.py:2052-2116` — `_process_equirect_cubemap()`.

### What Gets Ported

From Reconstruction Zone's `reconstruction_pipeline.py`:

1. **CubemapProjection class** (~120 lines)
   - `equirect2cubemap()` — splits ERP into 6 perspective faces.
   - `cubemap2equirect()` — merges face masks back to ERP space.
   - `_face_to_xyz()` / `_xyz_to_face()` — coordinate transforms.
   - Face size: `min(1024, w // 4)`.

2. **Postprocessing** (~30 lines)
   - Morphological close (`cv2.morphologyEx` with `MORPH_CLOSE`) to bridge small gaps.
   - Flood-fill from corners to fill enclosed holes.
   - Applied once at full ERP resolution after cubemap merge — **not** per-face. Per-face dilation at 1024px can't close gaps that are only a few pixels wide at face scale. The combined mask at full resolution (e.g. 7680x3840) makes gaps large enough to bridge.
   - Mask values throughout: 0/1 uint8. Threshold with `mask > 0`, **not** `mask > 127`.

### From FullCircle

1. **YOLO + SAM v1 detection pattern** — `mask_perspectives.py`. YOLO bounding boxes → SAM segmentation → union of all person masks.
2. **Closest-camera mask computation** — `omni2perspective.py:148-153`.
   - `closest_camera = argmax(rays_in_omni @ cam_centers.T)`
   - `mask = (closest_camera == cam_idx) * 255`

---

## python3.dll Resolution

### The Blocker

LFS bundles Python 3.12 via vcpkg. The build ships `python312.dll` but not `python3.dll` (the stable ABI shim). SAM 3.1's import chain pulls in `psutil` and `pycocotools`, which link against `python3.dll` and crash with "DLL load failed."

**What works in plugin venvs:** torch, scipy, open3d (link against `python312.dll` directly).
**What fails:** psutil, pycocotools (link against `python3.dll`).

### The Fix

Bundle `python3.dll` from a standard Python 3.12 install with the plugin. Call `os.add_dll_directory()` at plugin init to make it discoverable.

This pattern is already proven: `tests/conftest.py` uses the same approach (`_configure_windows_dll_search`) to enable OpenCV imports during testing.

**Implementation:**
1. Copy `python3.dll` from `C:/Python312/python3.dll` into plugin directory (e.g. `lib/python3.dll`).
2. At plugin init (`__init__.py` or `plugin.py`), call `os.add_dll_directory(str(plugin_root / "lib"))`.
3. Must execute before any import of torch, sam3, or their transitive dependencies.

**Verification needed:**
1. Test that the default tier (YOLO + SAM v1) works **without** `python3.dll` — if `ultralytics` and `segment-anything` don't transitively import `psutil`/`pycocotools`, the DLL bundling is only needed for the premium tier.
2. Test that importing psutil and pycocotools succeeds after adding the DLL directory, from within LFS's embedded Python — needed for the SAM 3.1 premium tier.

If verification 1 confirms the default tier doesn't need `python3.dll`, the DLL can be bundled lazily (only copied/loaded when the user upgrades to SAM 3.1).

---

## First-Time Setup UX

### Default Tier: YOLO + SAM v1

Masking is opt-in. When the user enables masking for the first time:

1. Plugin detects that `ultralytics` and `segment-anything` are not installed.
2. Shows a brief "Installing masking dependencies..." progress bar.
3. Runs `uv pip install ultralytics segment-anything torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128` into the plugin venv.
4. SAM v1 weights auto-download on first detection run (~2.4 GB, one-time).
5. Done. Masking works.

No tokens, no accounts, no external approvals. One-click install.

### Premium Tier: SAM 3.1 Upgrade

An "Upgrade to SAM 3.1" option in the masking settings triggers the setup wizard. SAM 3.1 is a gated model requiring an external approval chain:

1. **Meta license** — Accept Meta's license agreement for SAM 3.1.
2. **HuggingFace account** — Create or log into HF.
3. **Gated model access** — Request and receive access to `facebook/sam3.1` on HuggingFace (linked to Meta approval).
4. **HF token** — Generate an access token.
5. **Install SAM3** — `uv pip install sam3` into the plugin venv (torch already installed from default tier).
6. **Download weights** — `snapshot_download("facebook/sam3.1")`.

The setup wizard walks through each step with clear instructions, links, and progress indicators. The wizard only appears when the user explicitly opts into the upgrade.

### State Machine

`setup_checks.py` tracks setup state via `MaskingSetupState`:
- Default tier: check `has_torch`, `has_yolo`, `has_sam1` (new fields).
- Premium tier: check `has_token`, `has_sam3`, `has_weights` (existing fields).
- `active_backend` returns `"sam3"` if premium is installed, `"yolo_sam1"` if default is installed, `None` if neither.

### Panel UI

When masking is enabled:
- If no backend installed → show default tier install button (one click).
- If default tier installed → show target checkboxes + "Upgrade to SAM 3.1" link.
- If premium tier installed → show target checkboxes + "Using SAM 3.1" indicator.

### Target Selection

Simple checkboxes for what to mask. Default: "person" checked. Additional options: "camera", "tripod", etc.

- **Default tier:** Targets map to COCO class IDs (person=0, etc.). Limited to the 80 COCO classes.
- **Premium tier:** Targets map to SAM 3.1 text prompts. Flexible — any text description works.

Both are presented identically to the user. The checkbox labels are the same; only the backend interpretation differs. This is distinct from geometric nadir/mount removal (which is a v1 non-goal).

No other controls exposed. No confidence slider, no dilation slider, no postprocessing options. Sensible defaults handle everything.

---

## Preset Overlap Analysis (Informing Follow-Up)

Measured overlap for all four presets (spherical coverage analysis at 360x180 resolution):

| Preset | Views | FOV | Coverage | Overlap (2+ views) | Max overlap |
|--------|-------|-----|----------|---------------------|-------------|
| Cubemap | 6 | 90 | 89.7% | 0.0% | 1 |
| Low | 9 | 75 | 88.4% | 3.9% | 2 |
| Medium | 14 | 65 | 81.6% | 15.6% | 2 |
| High | 18 | 65 | 98.7% | 32.7% | 3 |

**Key findings:**
- Cubemap has zero overlap by design — closest-camera mask adds no value here.
- Medium has the worst profile: only 81.6% coverage despite 14 views, with 15.6% overlap. This suggests suboptimal view placement.
- High has heavy overlap (32.7%) but near-complete coverage (98.7%). Biggest beneficiary of closest-camera masks.
- With closest-camera masks eliminating duplicate features, the optimization shifts: **maximize coverage with minimum FOV** (tighter FOV = sharper SIFT features = better COLMAP), rather than using wider FOV for overlap-based matching signal.

**Follow-up:** Redesign presets to minimize overlap while maximizing coverage. This is a sphere-packing optimization problem and is its own workstream. The masking pipeline adapts automatically to any preset configuration.

---

## Files Modified

### New Files
- `core/cubemap_projection.py` — CubemapProjection class (port from Reconstruction Zone).
- `core/overlap_mask.py` — Closest-camera Voronoi mask computation (port from FullCircle).
- `core/backends.py` — `MaskingBackend` protocol + YOLO+SAM v1 and SAM 3.1 backend implementations.
- `lib/python3.dll` — Bundled stable ABI DLL (may be deferred if default tier doesn't need it).

### Modified Files
- `core/masker.py` — Replace video-mode SAM3 with cubemap decomposition + backend-agnostic per-face detection + ERP merge + postprocessing. Add cubemap fast path. Backend selection (YOLO+SAM v1 vs SAM 3.1) based on what's installed.
- `core/pipeline.py` — Update Stage 2 to use new masker. Add Stage 3.5 for closest-camera mask. Adjust progress allocation (masking: 20-45%).
- `core/setup_checks.py` — Add default tier checks (`has_yolo`, `has_sam1`). Add `active_backend` property. Keep existing SAM 3.1 checks.
- `panels/prep360_panel.py` — Re-enable masking UI. Default tier one-click install. SAM 3.1 upgrade wizard. Target class checkboxes.
- `panels/prep360_panel.rml` — Add masking section markup.
- `plugin.py` or `__init__.py` — Add `os.add_dll_directory()` for bundled `python3.dll` at init (may be conditional on SAM 3.1 tier).

### Reference Code (Not Modified, Ported From)
- `d:/Projects/reconstruction-zone/reconstruction_gui/reconstruction_pipeline.py` — CubemapProjection, postprocessing.
- `D:/Data/fullcircle/masking/omni2perspective.py` — Closest-camera mask computation.
- `D:/Data/fullcircle/masking/mask_perspectives.py` — YOLO + SAM v1 detection pattern.

---

## Development Approach

Push the current plugin to GitHub as the clean baseline. Back up a copy to `D:\Projects\LFSplugins\` for safekeeping. Develop masking-v1 directly in the live plugin directory (`~/.lichtfeld/plugins/PanoSplat/`). LFS auto-scans this directory (`manager.py:discover()` loads any subdirectory with a `pyproject.toml`), so developing in-place means we can test in LFS at every step. The GitHub copy serves as the rollback point if needed.

---

## Open Questions

1. **Default tier DLL independence** — Does YOLO + SAM v1 work without `python3.dll`? If `ultralytics` and `segment-anything` don't transitively import `psutil`/`pycocotools`, the DLL bundling becomes a SAM 3.1-only concern. Test early.

2. **SAM v1 weight download UX** — SAM v1 weights (~2.4 GB) auto-download on first use. Should the plugin pre-download during install, or let it happen lazily on first mask? Lazy is simpler but the first run will be slow.

3. **SAM3 model size for plugin users** — The current code uses `build_sam3_multiplex_video_predictor()` which loads the default (largest) model. Should we offer a smaller model option for users with limited VRAM? Future enhancement.

4. **Closest-camera mask for cubemap** — Resolved: skip entirely for cubemap preset (zero overlap, all pixels already owned by exactly one camera).
