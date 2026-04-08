# SAM 3 Cubemap Masking — Design Spec

**Date:** 2026-04-08
**Status:** Approved design, pending implementation
**Scope:** New independent masking path + SAM 3.1 video tracking groundwork
**Restore point:** `ebfc024` (commit before any SAM 3 changes)

---

## Summary

Add SAM 3 text-prompted masking as an independent alternative to the existing FullCircle (YOLO+SAM v1+SAM v2) masking pipeline. The SAM 3 path uses cubemap decomposition to mask ERP frames directly, eliminating the need for direction estimation, synthetic fisheye views, and video tracking. Users choose between the two methods in the panel UI.

Additionally, lay structural groundwork for replacing SAM v2 video tracking with SAM 3.1's multiplex video predictor in the FullCircle path — stub only, no implementation.

---

## Architecture: Approach B — Parallel Masker Classes

The SAM 3 path lives in a new `core/sam3_masker.py` module, completely independent of the existing `Masker` class. The pipeline orchestrator (`pipeline.py`) selects which masker to instantiate based on the user's method choice. Both produce identical output: `masks/{view_id}/{frame_id}.png` in the COLMAP-compatible layout.

This avoids touching the FullCircle path (which is actively being iterated) and allows the SAM 3 path to be developed, tested, and compared independently.

---

## Section 1: Dependency & Install Path

### pyproject.toml

New optional extra alongside the existing `video-tracking`:

```toml
[project.optional-dependencies]
video-tracking = ["sam2==1.1.0", "huggingface-hub==1.9.0"]
sam3-masking = ["sam3>=0.1.3", "huggingface-hub>=1.9.0"]
```

### setup_checks.py

`install_premium_tier()` changes from `uv add sam3` to:

```python
_run_uv_command([
    "sync", "--locked", "--no-dev", "--extra", "sam3-masking",
], on_output=on_output)
```

This matches the pattern used by `install_video_tracking()` — locked sync, no runtime mutation of pyproject.toml.

The existing eager weight download after install (lines 521-529) must be preserved — `install_premium_tier` should still call `download_model_weights()` after the sync completes.

Everything else in setup_checks.py stays: HF token verification, access checking, weight download via `snapshot_download("facebook/sam3")`, the `MaskingSetupState` dataclass.

### Sam3Backend API fixes (core/backends.py)

The existing `Sam3Backend` class has API mismatches vs. the real SAM 3 package. Fixes based on the working `reconstruction_gui/test_cubemap_sam3.py`:

1. **Import path** — `test_cubemap_sam3.py` uses `from sam3 import build_sam3_image_model` (top-level re-export); `backends.py` line 48 uses `from sam3.model_builder import build_sam3_image_model`. Verified identical — `sam3/__init__.py` re-exports from `model_builder` via `__all__`. Change to top-level import for consistency with the working reference.
2. **`build_sam3_image_model()`** — no `device` arg. Current code at line 406 passes `device=self._device`. Fix: call without args, then `.to(device)` and `.eval()` on the returned model.
3. **`Sam3Processor`** — takes `confidence_threshold` as constructor arg. Current code at line 407 creates `Sam3Processor(self._model)` with no threshold. Fix: `Sam3Processor(model, confidence_threshold=self._confidence_threshold)` where `_confidence_threshold` is a constructor parameter (default `0.3`).
4. **`reset_all_prompts(state)`** — must be called between prompt strings to clear prior detections. Current `detect_and_segment()` at lines 429-451 iterates prompts without calling this. Fix: add `self._processor.reset_all_prompts(state)` before each `set_text_prompt` call.
5. **`output["scores"]`** — use native scores in `batch_detect_boxes` instead of hardcoded `1.0` at line 473. When one mask produces multiple contours, assign the mask-level score to each contour box.
6. **Flash Attention 3 fallback** — detect FA3 availability, monkey-patch `sam3.model.decoder` to allow MATH attention backend if FA3 is missing (from `sam3_pipeline.py` lines 148-169)

### HF token panel flow

Stays as-is. The existing buttons (Open HuggingFace Signup, Open SAM 3 Model Page, Open Tokens Page) and the token verify flow are already built for this purpose.

---

## Section 2: Sam3CubemapMasker

New file: `core/sam3_masker.py`

### Input

- Extracted ERP frames (from sharpest_extractor output)
- Active `ViewConfig` preset (low/medium/high/cubemap)
- Output directory
- User-configured text prompts (default: `["person", "tripod"]`)
- Confidence threshold

### Pipeline per frame

1. **Cubemap decomposition** — `CubemapProjection.equirect2cubemap(erp_frame)` produces 6 cube faces. Uses `core/cubemap_projection.py` which is already ported from reconstruction-zone.
   - `face_size = min(1024, w // 4)` (matches reconstruction-zone convention)

2. **SAM 3 per-face detection + segmentation** — For each of the 6 cube faces:
   - Convert BGR→RGB→PIL
   - `processor.set_image(pil_img)` → returns inference state
   - For each text prompt: `processor.reset_all_prompts(state)`, then `processor.set_text_prompt(state=state, prompt=text)`
   - Collect `output["masks"]` and `output["scores"]`
   - Convert mask tensors to binary uint8, resize if needed, union all detections per face

3. **Reassemble** — `CubemapProjection.cubemap2equirect(face_masks, (w, h))` merges 6 face masks back to ERP space.

4. **Mask polarity inversion** — SAM 3 outputs white=detected-object (white=person). COLMAP convention used by this plugin is white=keep. Invert the mask: `mask = 1 - mask` so that detected objects become black (masked out) and background becomes white (kept). This matches the existing FullCircle path's output polarity.

5. **Mask dilation** — Apply configurable dilation to the inverted ERP mask (reuse existing dilation logic). Dilation expands the masked-out region slightly to catch edge artifacts.

6. **Reframe to pinhole views** — Reframe the ERP mask into the active preset's pinhole views using the reframer. Output to `masks/{view_id}/{frame_id}.png`.

### What it does NOT do

- No direction estimation (Pass 1)
- No synthetic fisheye views
- No video tracking / temporal continuity
- No multi-pass architecture

### Reference implementation

The working pattern is in `reconstruction_gui/reconstruction_pipeline.py`:
- `CubemapProjection` class (line 1207)
- `_process_equirectangular()` method (line 2047)

And `reconstruction_gui/test_cubemap_sam3.py` for the proven SAM 3 API usage.

### Progress reporting

Reports progress per-frame via the same `_update(stage, pct, msg)` callback the existing masker uses. The pipeline's 20-35% masking allocation applies identically.

---

## Section 3: Pipeline Integration & UI

### pipeline.py

- `PipelineConfig` gets `masking_method: str = "fullcircle"`
- In the masking stage: if `masking_method == "sam3_cubemap"`, instantiate `Sam3CubemapMasker`; otherwise use existing `Masker`
- Progress allocation unchanged

### setup_checks.py — `masking_ready` gate change

**Critical:** The current `masking_ready` property (line 57) requires `has_sam2`:

```python
def masking_ready(self) -> bool:
    return self.active_backend is not None and self.has_sam2
```

The panel uses this to toggle `show_masking_install` vs `show_masking_controls` (line 211). This means SAM 3 installed without SAM v2 would show the "install" state, blocking the user.

Fix: The panel's conditional logic must bypass `masking_ready` when the SAM 3 method is selected. The Method dropdown should always be visible. When "SAM 3 Cubemap" is selected, the panel checks `premium_tier_ready` directly (torch + sam3 + weights) rather than `masking_ready`. When "FullCircle" is selected, the existing `masking_ready` gate applies unchanged.

Additionally, the install completion handler at `prep360_panel.py` line 788 sets `self._masking_available = self._setup_state.masking_ready`. This must also be updated for the SAM 3 path — when SAM 3 is the selected method and `premium_tier_ready` is true, `_masking_available` should be set to `True` regardless of `masking_ready`.

### prep360_panel.py

- New `masking_method_idx` binding (0 = FullCircle, 1 = SAM 3 Cubemap)
- Method dropdown at top of masking section — **always visible regardless of install state**
- Conditional display logic routes on method selection first, then install state:
  - FullCircle selected → existing `show_masking_install` / `show_masking_controls` flow unchanged (gates on `masking_ready`)
  - SAM 3 selected + not installed → HF setup flow (gates on `premium_tier_ready`)
  - SAM 3 selected + installing → progress bar with live log output
  - SAM 3 selected + installed → SAM 3 controls (enable, diagnostics, prompts field, backend info)
- `mask_prompts_str` (already exists as binding) becomes visible in SAM 3 ready state
- `PipelineConfig` receives `masking_method` from the panel's method selection

### prep360_panel.rml

- Method dropdown added at top of masking section content
- New conditional blocks: `show_masking_sam3_setup`, `show_masking_sam3_installing`, `show_masking_sam3_ready`
- Existing `show_masking_install` / `show_masking_controls` blocks preserved for FullCircle path

### UI States (see planning.pen mockups)

1. **Not installed** — Method dropdown (FullCircle default), status "Not installed", green install button
2. **SAM 3 selected, setup** — Guided HF flow: create account → request access → paste token → verify
3. **SAM 3 installing** — Token verified badge, progress bar, live install log
4. **SAM 3 ready** — Method dropdown, enable checkbox, diagnostics checkbox, backend info, prompts text field

---

## Section 4: SAM 3.1 Video Tracking Groundwork

Structural preparation only. No functional implementation, no UI surface.

### pyproject.toml

One extra covers both: `sam3-masking`. The video predictor ships in the same `sam3` package.

### backends.py

- `HAS_SAM3_VIDEO = False` flag
- Conditional import: `from sam3.model_builder import build_sam3_multiplex_video_predictor`
- `Sam3VideoBackend` stub class implementing `VideoTrackingBackend` protocol:
  - `initialize()` → raises `NotImplementedError("SAM 3.1 video tracking not yet implemented")`
  - `track_sequence()` → raises `NotImplementedError`
  - `cleanup()` → no-op
- `get_video_backend()` selection unchanged — SAM v2 stays active

### setup_checks.py

- `MaskingSetupState` gets `has_sam3_video: bool` field
- `_check_sam3_video_installed()` checks the import
- No new install function — auto-detected when sam3-masking is installed

### User-facing distinction

None. There is no "SAM 3" vs "SAM 3.1" in the UI. The user sees "SAM 3 Cubemap" as their masking method. Whether the internals use the image API or the video predictor is an implementation detail handled later.

---

## Implementation Prerequisites

- **Restore point:** `ebfc024` committed on local `main`
- **sam3 not currently installed** in plugin venv — first step is `uv sync --extra sam3-masking` after adding the extra to pyproject.toml
- **Flash Attention 3** may not be available — the FA3 detection + monkey-patch fallback from reconstruction-zone must be included

## Key Risks

1. **SAM 3 `device` parameter** — README doesn't show it. Using `.to(device)` after construction (confirmed working in test_cubemap_sam3.py)
2. **VRAM pressure** — SAM 3 is 0.9B params. Loading alongside YOLO+SAM v1+SAM v2 would be excessive. The SAM 3 path should be exclusive — don't load FullCircle backends when SAM 3 is active.
3. **Mask quality unknown on plugin presets** — SAM 3 cubemap masking is proven on reconstruction-zone's ERP frames but not on this plugin's specific preset geometries. Needs real-world validation.
4. **pyproject.toml lock file** — adding `sam3>=0.1.3` may require `uv lock` to regenerate uv.lock. Must verify sam3's torch version requirements are compatible with the existing `torch>=2.11.0` pin.
5. **SAM 3 import path** — `test_cubemap_sam3.py` uses `from sam3 import build_sam3_image_model` (top-level); `backends.py` uses `from sam3.model_builder import build_sam3_image_model`. Verified: `sam3/__init__.py` re-exports `build_sam3_image_model` from `model_builder` via `__all__`. Both resolve identically. Use the top-level import for consistency with the working reference.
6. **`masking_ready` gate** — Current `MaskingSetupState.masking_ready` requires `has_sam2`. The SAM 3 path must not depend on SAM v2 being installed. Panel routing must check `premium_tier_ready` for the SAM 3 path, not `masking_ready`. See Section 3 for details.

## Visual Reference

Architecture diagram and UI mockups in `docs/planning.pen`.
