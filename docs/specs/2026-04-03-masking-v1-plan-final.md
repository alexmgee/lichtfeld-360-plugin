# PanoSplat Masking v1 — Final Implementation Plan

> **Date:** 2026-04-03
> **Supersedes:** All prior plans (v1 through v8)
> **Spec:** `2026-04-02-masking-layer-v1-design.md` (to be updated in Milestone 2)
> **Review:** Incorporates feedback from all inspection documents through `docs/2026-04-03-masking-v1-plan-v8-final-inspection.md`

---

## Context

The plugin has a working single-pass masking pipeline (per-view detection → ERP OR-merge → postprocess). This implements FullCircle's Steps 1-3. What's missing is Steps 4-6: the synthetic camera aimed at the detected person, with SAM v2 video tracking for temporal consistency.

The previous plan (v2) treated SAM v2 integration as a known quantity. An independent inspection of the FullCircle repo and plugin codebase identified that:

- FullCircle runs SAM v2 via an external tool (`thirdparty/sam-ui`), not as a directly imported backend
- The `sam2` PyPI package has extra dependencies (`hydra-core`, `iopath`, `omegaconf`) untested in our embedded Python environment
- The plugin's UI, setup checks, and design spec still reflect the old two-tier model and need updating

We've since checked out the `sam-ui` submodule and inspected the actual code. The SAM v2 API is clean and usable as an imported library — the "external tool" framing was overstated. The real risk is packaging, not API design.

**This plan splits work into two tracks:** geometry/pipeline refactor (proven, no new deps) and SAM v2 integration (needs validation). The geometry track delivers value on its own — even without SAM v2, aiming a camera at the person improves masks.

**Intentional v1 choice — primary masking views:** FullCircle uses a dedicated 16-camera masking layout (8 yaw × 2 pitch bands, 90° FOV) optimized for coarse person localization. The plugin instead derives Pass 1 masking views from the active reconstruction preset (cubemap: 6 views, low: 9, medium: 14, high: 18). This couples masking quality to the user's preset choice but avoids introducing a separate hidden view layout for v1. If testing shows this produces noticeably weaker localization than FullCircle's fixed layout, a dedicated `masking_primary` preset can be introduced as a follow-up without changing the two-pass architecture.

---

## Architecture: Two-Pass Masking

```
PASS 1 — Primary Detection (existing + center-of-mass recording)
  For each ERP frame:
    1. Reframe to N preset views at detection resolution
    2. Detect person on each view (YOLO+SAM v1 or SAM 3 image mode)
    3. Record center-of-mass + mask area per detection → 3D direction
    4. Back-project all detections to ERP, OR-merge
    → Store: primary ERP mask + weighted person direction per frame

PASS 2 — Synthetic Camera + Second-Pass Detection (NEW)
  A. Resolve person direction per frame (temporal fallback for gaps)
     If NO frame in the entire clip has a valid direction, skip Pass 2
     entirely — preserve primary ERP masks as the final result.
  B. Render synthetic OPENCV_FISHEYE views: ERP → fisheye aimed at person
  C. Detect on synthetic views:
     - If SAM v2 available: video tracking (temporal propagation)
     - If SAM 3 available + preferred: SAM 3 video predictor
     - Fallback: per-frame detect_and_segment using image backend
  D. Back-project each synthetic fisheye mask to ERP
  E. OR-merge with primary ERP mask from Pass 1

PASS 3 — Postprocess + Save (existing)
  For each frame:
    1. Morph close + flood fill on final merged ERP mask
    2. Invert to COLMAP polarity (white=keep, black=remove)
    3. Save to extracted/masks/
```

---

## Fisheye Synthetic Camera

Ideal equidistant fisheye via pycolmap's OPENCV_FISHEYE model with zero distortion:

```python
SYNTHETIC_SIZE = 2048
SYNTHETIC_FOCAL = SYNTHETIC_SIZE / 2 / (np.pi / 2)  # ≈651.9
SYNTHETIC_CENTER = SYNTHETIC_SIZE / 2                # 1024.0

camera = pycolmap.Camera(
    camera_id=0,
    model=pycolmap.CameraModelId.OPENCV_FISHEYE,
    width=SYNTHETIC_SIZE, height=SYNTHETIC_SIZE,
    params=[SYNTHETIC_FOCAL, SYNTHETIC_FOCAL,
            SYNTHETIC_CENTER, SYNTHETIC_CENTER,
            0.0, 0.0, 0.0, 0.0],
)
```

With k1-k4=0, OPENCV_FISHEYE reduces to pure equidistant projection: r = f·θ. This gives exactly 180° FOV (full hemisphere) on a 2048×2048 image. No hardware calibration needed.

**Projection** uses pycolmap's `cam_from_img()` / `img_from_cam()`. Important detail: `cam_from_img()` returns 2D normalized camera-plane coordinates `[x/z, y/z]`, not a 3D ray. To form a camera-space direction, append `z=1` to get `[u, v, 1]` and optionally normalize. `img_from_cam()` accepts 3D camera-space points, and the round-trip `img_from_cam([u, v, 1])` is exact up to floating-point precision. The center pixel maps to `[0,0]` (forward = +Z). `[1,0]` corresponds to a 45° ray, not the 90° hemisphere edge — the 90° horizon is the asymptote where `z → 0+`, so it is not represented by a finite normalized coordinate.

**Rotation** uses `look_at_camZ()` ported from FullCircle's `lib/cam_utils.py:66-75`. Camera +Z points at the person. This is a different convention than our reframer (camera -Z = forward) — the two conventions are isolated in separate functions and never mixed.

**Reference code:**
- `D:/Data/fullcircle/masking/omni2synthetic.py` — ERP→fisheye rendering
- `D:/Data/fullcircle/masking/synthetic2omni.py` — fisheye→ERP back-projection
- `D:/Data/fullcircle/masking/lib/cam_utils.py` — `look_at_camZ`, coordinate utils

---

## SAM v2 Video Tracking — Verified API

Source: `D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py` (now checked out locally).

**Package:** `sam2` on PyPI, version 1.1.0. Requires `torch>=2.5.1` (we have 2.11.0 ✓).

**Model:** `facebook/sam2.1-hiera-large` from HuggingFace.

**Imports:**
```python
from sam2.build_sam import build_sam2_video_predictor_hf
from sam2.sam2_video_predictor import SAM2VideoPredictor
```

**Headless flow** (from `tracking_gui.py:550-624`):
```python
# 1. Resize frames to 512px min dimension, write as numbered JPEGs
for i, path in enumerate(jpeg_file_paths):
    image = open_image(path)
    image = image.resize(target_size_from_min_dimension(image, 512))
    image.save(temp_frames_path / f"{i:04d}.jpg", quality=100)

# 2. Build predictor and init state from frame directory
predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-large", device=device)
inference_state = predictor.init_state(str(temp_frames_path))

# 3. Add one click prompt at person center on frame 0
points = np.array([[click_x, click_y]], dtype=np.float32)
labels = np.array([1], np.int32)  # 1 = positive
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state, frame_idx=0, obj_id=0,
    points=points, labels=labels,
)

# 4. Propagate forward through all frames
for out_frame_idx, out_obj_ids, out_mask_logits in propagate_in_whole_video(
    predictor, inference_state, start_frame_idx=0,
    window=1500, total_frames=total_frames, reverse=False,
):
    masks = (torch.sigmoid(out_mask_logits[:, 0]) > 0.5).float().cpu().numpy()
    # ... save masks
```

**Key constraint:** `init_state()` reads frames from a directory of numbered JPEG files on disk. Cannot use in-memory arrays. Our plan writes synthetic views to a tempdir, which is cleaned up after tracking completes.

**Prompt strategy:** FullCircle clicks at the center of the synthetic fisheye image (1440,1440 for 2880×2880). Since our synthetic camera is aimed directly at the person, the person IS at the center. The click coordinates must be in the **resized SAM v2 temp-frame** coordinate space, not the original `SYNTHETIC_SIZE` space. Compute `cx, cy` from the actual dimensions of the resized JPEG written to the tempdir (i.e., center of the resized image). Because the synthetic camera is aimed at the person, the person remains centered after resizing.

**Resize / back-projection geometry:** SAM v2 works on frames resized to 512px minimum dimension (matching FullCircle's `tracking_gui.py:567`). The output masks are at this reduced resolution. Before back-projecting to ERP, masks MUST be resized back to the original synthetic camera resolution (`SYNTHETIC_SIZE × SYNTHETIC_SIZE`) using nearest-neighbor interpolation. FullCircle does exactly this in `synthetic2omni.py:57` (`cv2.resize(img_r, (W_fi, H_fi))`) before back-projection. The back-projection math always uses the original 2048×2048 synthetic camera model — never a resized camera.

**Windows / HuggingFace caveat (confirmed in preflight):** In the plugin venv on this machine, `build_sam2_video_predictor_hf()` only succeeded after setting `HF_HUB_DISABLE_XET=1`. The default Xet download path failed because `hf_xet` hit a DLL-load error on Windows. Treat this as a likely environment requirement unless B1 proves the final LichtFeld runtime no longer needs it. Also note that the optional `sam2._C` extension emitted a warning during propagation and SAM2 skipped one post-processing step, but basic prompting and video propagation still worked. If `%TEMP%` or the default temp directory proves unreliable in the LichtFeld runtime on Windows, use a plugin-controlled workspace-local temp root for the numbered SAM2 frame directory instead of relying on OS temp.

---

## Tier Structure

### Default Tier: YOLO + SAM v1 (+ optional SAM v2)

| Model | Package | Job | Required? |
|-------|---------|-----|-----------|
| YOLOv8s | `ultralytics` | Bounding box detection | Yes |
| SAM v1 ViT-H | `segment-anything` | Pixel-accurate segmentation | Yes |
| SAM v2 | `sam2` | Video tracking on synthetic views | Optional enhancement |

Without SAM v2: Pass 2 uses the image backend (YOLO+SAM v1) per-frame on synthetic views. Still beneficial — person is centered and well-scaled.

With SAM v2: Pass 2 uses temporal video tracking. Masks are temporally consistent across frames.

### Premium Tier: SAM 3

Replaces all default-tier models. Uses SAM 3's image API for Pass 1, SAM 3's video predictor for Pass 2.

### Capability Levels (new concept for setup_checks + UI)

| Level | What's installed | Pass 1 | Pass 2 | Label |
|-------|-----------------|--------|--------|-------|
| 0 | Nothing | — | — | "Not installed" |
| 1 | YOLO + SAM v1 | Per-view detection | Per-frame fallback on synthetic views | "Masking ready" |
| 2 | YOLO + SAM v1 + SAM v2 | Per-view detection | SAM v2 video tracking on synthetic views | "Masking ready (video tracking)" |
| 3 | SAM 3 | Text-prompted detection | SAM 3 video predictor | "Masking ready (SAM 3)" |

---

## Coordinate Convention Notes

| | Our Reframer | FullCircle / Synthetic Camera |
|--|-------------|-------------------------------|
| Camera forward | -Z | +Z |
| Rotation meaning | world-to-camera (rows = [right, up, -fwd]) | world-from-camera (cols = [x, y, z]) |
| pycolmap used? | No (custom projection math) | Yes (`cam_from_img` / `img_from_cam`) |

The synthetic camera functions use FullCircle's convention. The existing reframer is unchanged. The bridge between them is `_pixel_com_to_3d_direction()` which converts from reframer-convention pixel coords to a convention-agnostic world-space 3D unit vector.

---

## Track A: Geometry & Pipeline Refactor (no new dependencies)

### Task A1: Fisheye Projection Functions

Pure geometry — no ML dependencies. Foundation for everything.

**Files:** Create `tests/test_synthetic_camera.py`, modify `core/masker.py`

New functions in `core/masker.py`:
- `_create_synthetic_camera(size)` — pycolmap OPENCV_FISHEYE, ideal equidistant
- `_render_synthetic_fisheye(erp, camera, R_syn)` — ERP→fisheye via `cam_from_img` + remap
- `_backproject_fisheye_mask_to_erp(mask, erp_size, camera, R_syn)` — fisheye→ERP via `img_from_cam` + remap

**Steps:**
- [ ] A1.1: Write test — create camera, verify `cam_from_img()` returns 2D normalized coordinates `[x/z, y/z]`, then verify `img_from_cam([u, v, 1])` round-trips exactly (within floating-point tolerance). Include spot checks: center pixel → `[0,0]`, a 45° sample ray → `[1,0]`, and note that the 90° hemisphere edge is asymptotic rather than a finite normalized value.
- [ ] A1.2: Write test — render known ERP, verify fisheye output is circular, center maps to look-at direction
- [ ] A1.3: Write test — backproject known blob, verify ERP location
- [ ] A1.4: Write round-trip test — render→mask→backproject, verify coverage consistency
- [ ] A1.5: Implement the three functions
- [ ] A1.6: Run tests

### Task A2: Direction Computation Helpers

Pure math — center-of-mass, pixel→3D, weighted average, temporal fallback, look_at.

**Files:** Extend `tests/test_synthetic_camera.py`, modify `core/masker.py`

New functions:
- `_compute_detection_com(mask)` → `(cx, cy)` or None. Via `cv2.moments`. Ref: `mask_perspectives.py:138-144`
- `_pixel_com_to_3d_direction(cx, cy, fov, yaw, pitch, size, flip_v)` → 3D unit vector. Must invert fliplr/flipud + rotation from `_reframe_to_detection()`. Ref: `perspective2omni.py:390-410`
- `_compute_weighted_person_direction(directions_and_weights)` → unit vector or None. Weights = mask area. Ref: `omni2synthetic.py:92-99`
- `_temporal_fallback_direction(frame_idx, all_directions)` → unit vector or None. Search nearest frames forward/backward. Returns None only if the entire clip has no valid direction. Ref: `omni2synthetic.py:62-89`
- `_look_at_rotation(center_dir)` → 3×3 world_from_cam matrix. Ref: `cam_utils.py:66-75`
- `_direction_to_yaw_pitch(direction)` → `(yaw_deg, pitch_deg)`. Trivial.

**Steps:**
- [ ] A2.1: Write test — `_compute_detection_com` with known blob
- [ ] A2.2: Write test — `_pixel_com_to_3d_direction` using the exact optical-center sample of the flipped detection image (for an even-sized view this is the left-of-middle center pixel in x, e.g. `(size/2 - 1, size/2)` for a 1024px view). Verify yaw=0, pitch=0 maps to `[0,0,1]`, yaw=90, pitch=0 maps to `[1,0,0]`, and `flip_v` correctly inverts top/bottom recovery.
- [ ] A2.3: Write test — `_compute_weighted_person_direction` with two equal-weight directions
- [ ] A2.4: Write test — `_temporal_fallback_direction` borrows from nearest non-None frame
- [ ] A2.5: Write test — `_look_at_rotation` for known directions. Verify `R @ [0,0,1] ≈ center_dir`.
- [ ] A2.6: Write test — `_direction_to_yaw_pitch` for cardinal directions
- [ ] A2.7: Implement all helpers
- [ ] A2.8: Run tests

### Task A3: VideoTrackingBackend Protocol + FallbackVideoBackend

**Files:** Modify `core/backends.py`, extend `tests/test_backends.py`

```python
class VideoTrackingBackend(Protocol):
    def initialize(self) -> None: ...
    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]: ...
    def cleanup(self) -> None: ...
```

**`initial_mask` parameter:** Reserved for future backends that accept a mask prompt instead of (or in addition to) a point click. `Sam2VideoBackend` ignores it — it uses a center-point click on `initial_frame_idx` instead. `FallbackVideoBackend` also ignores it — it runs per-frame detection independently. A future SAM 3 video backend or a mask-prompted SAM v2 flow could use it to seed tracking from a primary-pass detection mask rather than a point click. Keep it in the protocol now to avoid a breaking interface change later.

```python
class FallbackVideoBackend:
    """Wraps a MaskingBackend. Calls detect_and_segment per-frame."""
    def __init__(self, image_backend: MaskingBackend, targets: list[str]):
        ...
```

**Ownership rule:** `FallbackVideoBackend` can wrap either:

- the already-initialized Pass 1 image backend for the normal Track A fallback path, in which case ownership stays with `Masker` and `FallbackVideoBackend.initialize()` / `cleanup()` should be no-ops, or
- a freshly created image backend for runtime recovery after a real video backend failure, in which case the fallback backend owns that wrapped backend for the duration of the retry and must clean it up when finished.

Factory:
```python
def get_video_backend(
    preference: str | None = None,
    fallback_image_backend: MaskingBackend | None = None,
    targets: list[str] | None = None,
) -> VideoTrackingBackend | None:
```

**Steps:**
- [ ] A3.1: Write test — FallbackVideoBackend with mock image backend
- [ ] A3.2: Write test — `get_video_backend()` returns FallbackVideoBackend when no SAM v2/v3
- [ ] A3.3: Implement protocol, FallbackVideoBackend, factory
- [ ] A3.4: Run all tests (all existing tests must still pass)

### Task A4: Refactor Masker to Two-Pass

The core integration. `process_frames()` goes from single-pass to two-pass. External interface unchanged.

**Files:** Modify `core/masker.py`, extend `tests/test_masker.py`

Updated `MaskConfig`:
```python
@dataclass
class MaskConfig:
    # ... existing fields unchanged ...
    enable_synthetic: bool = True
    synthetic_size: int = 2048
```

Refactored `Masker`:
- `_primary_detection(erp)` → `(erp_mask, person_direction_or_None)` — extracted from current per-frame loop, adds CoM recording
- `_synthetic_pass(frame_files, primary_masks, directions, frame_order)` → `dict[str, np.ndarray]` — render synthetic fisheye, run the currently selected video backend, backproject, and merge. Owns the `best_frame_idx` selection (e.g. frame with largest primary mask area) and passes it to the video backend as `initial_frame_idx`, but it does **not** own backend recovery/retry.
- `process_frames(...)` — Phase 1 → Phase 2 → Phase 3. Interface unchanged. Owns backend teardown, runtime fallback, and retry behavior: if the real video backend fails during `initialize()` or `track_sequence()`, log the error, clean up VRAM/tempdir, create and initialize a fresh image backend, wrap it in `FallbackVideoBackend`, and retry Pass 2. The pipeline should not fail because SAM v2 crashed — the user still gets the primary masks plus per-frame fallback detection on the synthetic views.
- `__init__` and `initialize()`/`cleanup()` manage video backend lifecycle

**Steps:**
- [ ] A4.1: Extract `_primary_detection()` with CoM recording
- [ ] A4.2: Implement `_synthetic_pass()` using fisheye projection + video backend
- [ ] A4.3: Refactor `process_frames()` to three-phase
- [ ] A4.4: Update `Masker.__init__`, `initialize()`, `cleanup()` for video backend
- [ ] A4.5: Add `enable_synthetic`, `synthetic_size` to MaskConfig
- [ ] A4.6: Write test — with FallbackVideoBackend, verify two-pass produces output
- [ ] A4.7: Write test — verify synthetic pass skipped when enable_synthetic=False
- [ ] A4.8: Run ALL tests — all existing tests must still pass

### Task A5: Integration Test

**Files:** Extend `tests/test_masking_integration.py`

- [ ] A5.1: Test with synthetic ERP containing a known blob — verify blob is masked
- [ ] A5.2: Test synthetic camera direction — verify it points at the blob after primary detection
- [ ] A5.3: Test no-detection clip — feed an ERP with no detectable person (e.g. solid color). Verify Pass 2 is skipped entirely, primary ERP masks are preserved as output, and no errors are raised.
- [ ] A5.4: Test video backend failure — mock the video backend to raise during `track_sequence()`. Verify the masker falls back to FallbackVideoBackend, still produces output masks, and cleans up VRAM/tempdir without leaking resources.
- [ ] A5.5: Run full suite

### Task A6: Update core/__init__.py Exports

- [ ] A6.1: Export `VideoTrackingBackend`, `FallbackVideoBackend`, `get_video_backend`
- [ ] A6.2: Run all tests

### Task A7: Minimum Product-Facing Updates

If Track A ships before Track B, the plugin must truthfully describe what the user has. This task adds the minimum plumbing — the full polished UI progression remains in Track B.

**Files:** Modify `core/setup_checks.py`, `panels/prep360_panel.py`

- [ ] A7.1: Add `capability_level` property to `MaskingSetupState` (levels 0 and 1 only — level 2/3 detection comes in Track B). Must preserve existing SAM 3 reporting: if `premium_tier_ready` is true, the panel must still show "Using SAM 3" — do not collapse or hide that state.
- [ ] A7.2: Update panel status text for the non-SAM 3 path to say "Masking ready" (not just "Using YOLO + SAM v1") when capability_level >= 1. Leave SAM 3 reporting path untouched.
- [ ] A7.3: Add short interim note to design spec header acknowledging two-pass architecture exists (full rewrite stays in Task C1)

---

## Track A Milestone: Shippable Without SAM v2

At the end of Track A:
- Two-pass masker structure is implemented and tested
- Synthetic fisheye camera math works correctly
- Pass 2 runs using the existing image backend (FallbackVideoBackend)
- No new dependencies — `sam2` is NOT required
- Masks are already better because the synthetic camera is aimed at the person

This is a **shippable improvement** that doesn't depend on SAM v2 at all.

---

## Track B: SAM v2 Integration (after Track A)

### Task B1: SAM v2 Validation Spike

Before writing any backend code, prove that SAM v2 works in our environment.

- [ ] B1.1: Test SAM v2 installation in the plugin venv. Start with `uv add sam2` and document whether it works cleanly or whether another method is required (e.g. `uv pip install --no-deps`, install from Git, etc.). Note: extra deps include `hydra-core`, `iopath`, `omegaconf` — check for conflicts.
- [ ] B1.2: Test import: `from sam2.build_sam import build_sam2_video_predictor_hf`
- [ ] B1.3: Test model download: `build_sam2_video_predictor_hf("facebook/sam2.1-hiera-large")`. If Windows download fails through HuggingFace Xet, retry with `HF_HUB_DISABLE_XET=1` and document whether that override is required in production.
- [ ] B1.4: Test init_state with a small frame directory (5-10 test images as numbered JPEGs)
- [ ] B1.5: Test add_new_points_or_box + propagate on test frames
- [ ] B1.6: Test within LichtFeld Studio runtime (python3.dll, embedded Python). Explicitly verify whether OS tempdir behavior is reliable there; if not, confirm a workspace-local temp root works for numbered SAM2 frame directories.
- [ ] B1.7: Document findings — does it work? Any workarounds needed? Explicitly record whether `HF_HUB_DISABLE_XET=1` is required, and whether the missing `sam2._C` post-processing warning has any observable quality impact in the real runtime.

**If B1 fails:** SAM v2 becomes a deferred enhancement. Track A's FallbackVideoBackend is the shipping behavior. No changes needed.

**If B1 succeeds:** Proceed to B2.

### Task B2: Sam2VideoBackend Implementation

**Files:** Modify `core/backends.py`, extend `tests/test_backends.py`

```python
class Sam2VideoBackend:
    """SAM v2 video tracking on synthetic views.

    API (confirmed from D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py):
      - build_sam2_video_predictor_hf("facebook/sam2.1-hiera-large")
      - predictor.init_state(str(frames_dir))  # numbered JPEGs on disk
      - predictor.add_new_points_or_box(state, frame_idx, obj_id, points, labels)
      - propagate_in_whole_video(predictor, state, ...)  # yields per-frame masks
    """
```

Flow:
1. Write synthetic fisheye frames to tempdir as `{i:04d}.jpg` (resized to 512px min dimension, matching FullCircle)
2. `predictor.init_state(str(tempdir))`
3. Use the caller-provided `initial_frame_idx` (selected by `_synthetic_pass()` from the strongest primary detection) and compute click coords from the resized temp-frame dimensions for that frame
4. `predictor.add_new_points_or_box(state, initial_frame_idx, 0, [[cx, cy]], [1])`
5. Propagate forward AND backward from `initial_frame_idx`, passing `window=total_frames` (or equivalent full-sequence span) so propagation covers the entire clip in both directions. The helper uses `window` to compute bounds: `forward_end = min(start + window, total - 1)`, `reverse_end = max(start - window, 0)`. A too-small window silently truncates coverage.
6. Collect masks by `out_frame_idx` — preallocate `results: list[np.ndarray | None]` for all frames, write each callback's mask into `results[out_frame_idx]`. Bidirectional propagation yields results in processing order (forward then reverse), not frame order, and the prompt frame may appear twice — index-based storage handles both correctly.
7. Threshold at 0.5, resize each mask to `SYNTHETIC_SIZE` before returning
8. Clean up tempdir — use `tempfile.TemporaryDirectory()` context manager or `try/finally` so cleanup runs on both success and failure. Backend cleanup (predictor teardown, VRAM release) must also be in the `finally` path, matching the existing pattern in `core/pipeline.py:322-341` and `core/masker.py:239-250`.

**Implementation note from preflight:** If model download occurs on Windows/LichtFeld and `hf_xet` still fails to load, set `HF_HUB_DISABLE_XET=1` before the first `build_sam2_video_predictor_hf()` call so HuggingFace falls back to the standard download path. If OS tempdir behavior is flaky in the real runtime, store the numbered JPEG sequence under a plugin-controlled workspace-local temp root instead of `%TEMP%`. Treat the missing `sam2._C` warning as a known caveat to evaluate, not an automatic blocker, since prompting and propagation still worked in preflight.

**Implementation note from preflight:** If model download occurs on Windows/LichtFeld and `hf_xet` still fails to load, set `HF_HUB_DISABLE_XET=1` before the first `build_sam2_video_predictor_hf()` call so HuggingFace falls back to the standard download path. Also treat the missing `sam2._C` warning as a known caveat to evaluate, not an automatic blocker, since prompting and propagation still worked in preflight.

**Steps:**
- [ ] B2.1: Add `HAS_SAM2` probe to backends.py
- [ ] B2.2: Implement `Sam2VideoBackend`
- [ ] B2.3: Update `get_video_backend()` to prefer Sam2VideoBackend when available
- [ ] B2.4: Write mocked interface test
- [ ] B2.5: Run all tests

### Task B3: Setup Checks + Installation

**Files:** Modify `core/setup_checks.py`

```python
@dataclass
class MaskingSetupState:
    # Existing
    has_torch: bool = False
    has_yolo: bool = False
    has_sam1: bool = False
    has_token: bool = False
    has_access: bool = False
    has_sam3: bool = False
    has_weights: bool = False
    # New
    has_sam2: bool = False

    @property
    def default_tier_ready(self) -> bool:
        """Pass 1 works. Pass 2 uses fallback."""
        return self.has_torch and self.has_yolo and self.has_sam1

    @property
    def video_tracking_ready(self) -> bool:
        """Pass 2 uses SAM v2 temporal tracking."""
        return self.default_tier_ready and self.has_sam2

    @property
    def capability_level(self) -> int:
        """0=nothing, 1=pass-1 only, 2=video tracking, 3=SAM 3"""
        if self.premium_tier_ready:
            return 3
        if self.video_tracking_ready:
            return 2
        if self.default_tier_ready:
            return 1
        return 0
```

SAM v2 is installed separately from the default tier — NOT added to `pyproject.toml` as a hard dependency. Instead:
- `install_default_tier()` installs YOLO + SAM v1 + torch (unchanged)
- New `install_video_tracking(on_output=None)` installs SAM v2 using the validated method from Task B1 (may be `uv add sam2`, `uv pip install --no-deps ...`, or another approach — B1 determines what works)

**Steps:**
- [ ] B3.1: Add `_check_sam2_installed()` and `has_sam2` field
- [ ] B3.2: Add `video_tracking_ready` and `capability_level` properties
- [ ] B3.3: Add `install_video_tracking()` function
- [ ] B3.4: Write tests for new properties
- [ ] B3.5: Run all tests

### Task B4: Panel UI Updates

**Files:** Modify `panels/prep360_panel.py`, `panels/prep360_panel.rml`

The panel should reflect actual masking capability:

| Capability level | Status text | Action |
|-----------------|-------------|--------|
| 0 | "Install masking" | One-click install button |
| 1 | "Masking ready" | Show "Enable video tracking" link |
| 2 | "Masking ready (video tracking)" | Show "Upgrade to SAM 3" link |
| 3 | "Masking ready (SAM 3)" | No action needed |

- [ ] B4.1: Add capability_level-aware status text
- [ ] B4.2: Add "Enable video tracking" install action (calls `install_video_tracking()`)
- [ ] B4.3: Test in LFS (manual)

### Task B5: SAM 3 Video Backend (Premium Tier)

Only after SAM v2 path is stable.

- [ ] B5.1: Research SAM 3 video predictor API — verify it follows SAM v2's pattern
- [ ] B5.2: Implement `Sam3VideoBackend`
- [ ] B5.3: Update `get_video_backend()` to prefer Sam3VideoBackend when preference="sam3"
- [ ] B5.4: Write mocked test
- [ ] B5.5: Run all tests

---

## Track B Milestone: Full FullCircle Pipeline

At the end of Track B:
- SAM v2 video tracking works on synthetic views
- Users can optionally install SAM v2 for better temporal consistency
- UI accurately reflects capability level
- Premium tier (SAM 3) video tracking also available

---

## Task C: Update Design Spec

After both tracks are complete:

- [ ] C1: Rewrite `docs/specs/2026-04-02-masking-layer-v1-design.md` to reflect:
  - Two-pass architecture
  - OPENCV_FISHEYE synthetic camera
  - Three capability levels (not two tiers)
  - SAM v2 as optional video tracking enhancement
  - Revised pipeline stage descriptions

---

## VRAM Management

YOLO + SAM v1 + SAM v2 simultaneously would exceed 8GB VRAM. Solution:

```python
# In process_frames():
# Pass 1 uses self._backend (YOLO + SAM v1)
primary_masks, directions = self._run_pass_1(frame_files)

# If using a real video backend (not fallback), free Pass 1 VRAM first
if isinstance(self._video_backend, (Sam2VideoBackend, Sam3VideoBackend)):
    self._backend.cleanup()
    self._backend = None

# Pass 2 uses self._video_backend
try:
    synthetic_masks = self._synthetic_pass(...)
except Exception:
    # Video backend failed at runtime — clean it up fully
    if self._video_backend is not None:
        self._video_backend.cleanup()
        self._video_backend = None
    # Re-initialize a fresh image backend for fallback
    fallback_backend = get_backend(self.config.backend_preference)
    fallback_backend.initialize()
    self._video_backend = FallbackVideoBackend(fallback_backend, self.config.targets)
    synthetic_masks = self._synthetic_pass(...)  # retry with fallback
finally:
    # Clean up whatever backend is active
    if self._video_backend is not None:
        self._video_backend.cleanup()
```

For FallbackVideoBackend: the image backend it wraps must stay alive. Only clean up the Pass 1 image backend when a real video backend (Sam2/Sam3) is selected — if the video backend then fails, re-initialize a fresh image backend for the fallback path.

---

## Progress Allocation (within masking stage 20-45%)

| Sub-phase | Progress | Description |
|-----------|----------|-------------|
| Pass 1: Primary detection | 20% → 32% | Per-view detection on all frames |
| Pass 2a: Render synthetic views | 32% → 35% | ERP → fisheye for all frames |
| Pass 2b: Video tracking / fallback detection | 35% → 40% | SAM v2 or per-frame fallback |
| Pass 2c: Backproject + merge | 40% → 42% | Fisheye masks → ERP, OR-merge |
| Pass 3: Postprocess + save | 42% → 45% | Morph close, flood fill, invert, write |

---

## Verification

1. **Unit tests:** `.venv/Scripts/pytest.exe tests/ -v` — all pass including new
2. **Clear pycache:** `rm -rf core/__pycache__ panels/__pycache__ __pycache__` (Git Bash) or `Remove-Item -Recurse -Force core\__pycache__, panels\__pycache__, __pycache__` (PowerShell)
3. **Manual test in LFS:** Run on `D:\Capture\deskTest` with Low preset + masking
4. **Compare masks:** ERP masks should show better person coverage vs previous run
5. **Verify synthetic aim:** Debug output confirming synthetic camera points at person
6. **Temporal consistency** (Track B only): Consecutive frame masks smooth, no flicker

---

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Direction math convention mismatch | Synthetic camera aims wrong | Task A2.2 tests this explicitly with known view geometries |
| pycolmap fisheye round-trip accuracy | Projection errors in render/backproject | Task A1.1 verifies cam_from_img/img_from_cam round-trip |
| SAM v2 packaging fails in plugin env | No video tracking | Track A ships without it; FallbackVideoBackend is the baseline |
| SAM v2 extra deps (hydra-core, iopath, omegaconf) conflict | Install failure | Task B1.1 validates before any code is written |
| VRAM pressure with 3 models loaded | OOM crash | Sequential cleanup in process_frames — never load all 3 at once |
| Spec/UI/plan disagree on "default tier" | User confusion | Task C1 reconciles all documents after implementation |
