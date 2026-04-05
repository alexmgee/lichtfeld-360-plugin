# PanoSplat Masking v1 — Updated Implementation Plan

> **Date:** 2026-04-03
> **Supersedes:** `docs/specs/2026-04-02-masking-v1-plan.md`
> **Spec:** `docs/specs/2026-04-02-masking-layer-v1-design.md` (to be updated after implementation)

---

## Context

The current masker implements FullCircle's Steps 1-3 and 7-8 (detect on preset views → OR-merge to ERP → postprocess → reframer reprojects to pinhole masks). All 116 tests pass and the basic pipeline works.

**What's missing:** FullCircle's Steps 4-6 — the synthetic camera step. After the initial per-view detection, FullCircle computes WHERE the person is on the sphere (weighted center-of-mass from all detections), aims a synthetic camera directly at them, runs a second-pass detection with SAM v2 video tracking for temporal consistency, and OR-merges the result back. This dramatically improves mask quality because the person is centered and well-scaled in the synthetic view.

**Key decision (from user):** The default tier IS the FullCircle method: YOLO + SAM v1 (initial detection) + SAM v2 (synthetic view video tracking). The premium tier (SAM 3) replaces all three models. The synthetic camera uses OPENCV_FISHEYE projection (matching FullCircle), not wide-angle pinhole.

---

## Architecture: Two-Pass Masking

```
PASS 1 — Primary Detection (existing + center-of-mass recording)
  For each ERP frame:
    1. Reframe to N preset views at detection resolution
    2. Detect person on each view (YOLO+SAM v1 or SAM 3 image mode)
    3. Record center-of-mass + area per detection → 3D direction
    4. Back-project all detections to ERP, OR-merge
    → Store: primary ERP mask + weighted person direction per frame

PASS 2 — Synthetic Camera + Video Tracking (NEW)
  A. Resolve person direction per frame (temporal fallback for gaps)
  B. Render synthetic OPENCV_FISHEYE views: ERP → fisheye aimed at person
  C. Run video tracking on synthetic sequence:
     - Default tier: SAM v2 video predictor
     - Premium tier: SAM 3 video predictor
     - Fallback: per-frame detect_and_segment (if no video backend)
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

FullCircle uses OPENCV_FISHEYE at 2880x2880 with hardware-calibrated distortion coefficients. We don't have hardware — we synthesize from ERP. We use an **ideal equidistant fisheye** (zero distortion):

```python
# OPENCV_FISHEYE with k1=k2=k3=k4=0 → equidistant projection: r = f*θ
# For 180° FOV on 2048x2048:
#   θ_max = π/2 (90° from center), r_max = 1024 (half image)
#   f = r_max / θ_max = 1024 / (π/2) ≈ 651.9
SYNTHETIC_SIZE = 2048
SYNTHETIC_FOCAL = SYNTHETIC_SIZE / 2 / (np.pi / 2)  # ~651.9
SYNTHETIC_CENTER = SYNTHETIC_SIZE / 2                # 1024.0

synthetic_camera = pycolmap.Camera(
    camera_id=0,
    model=pycolmap.CameraModelId.OPENCV_FISHEYE,
    width=SYNTHETIC_SIZE, height=SYNTHETIC_SIZE,
    params=[SYNTHETIC_FOCAL, SYNTHETIC_FOCAL,
            SYNTHETIC_CENTER, SYNTHETIC_CENTER,
            0.0, 0.0, 0.0, 0.0],  # no distortion
)
```

**Projection functions** (use pycolmap's built-in methods):
- `camera.cam_from_img(pixel_coords)` → normalized camera rays (for reading FROM fisheye)
- `camera.img_from_cam(cam_rays)` → pixel coords (for projecting TO fisheye)

**ERP → fisheye rendering:**
1. Generate ray directions for all fisheye pixels via `camera.cam_from_img()` + normalize
2. Apply radial validity mask (circular lens region)
3. Rotate rays to world space: `xyz_world = R_syn @ rays_cam.T`
4. Convert world rays to ERP coordinates: `θ = atan2(x, z)`, `φ = asin(y)` → `(u, v)`
5. `cv2.remap()` to sample ERP image

**Fisheye mask → ERP back-projection:**
1. For each ERP pixel, compute world ray direction
2. Rotate to synthetic camera space: `rays_cam = R_syn.T @ rays_world.T`
3. Keep only forward-pointing rays (`z > 0`)
4. Project to fisheye pixel coords via `camera.img_from_cam()`
5. Apply radial validity mask
6. Sample the synthetic mask at those pixel coords

**Rotation matrix:** Port `look_at_camZ()` from `D:/Data/fullcircle/masking/lib/cam_utils.py:66-75`. This returns `world_from_cam` (cam +Z aligned to center_dir). For our back-projection we need `cam_from_world = world_from_cam.T`.

**Reference files:**
- `D:/Data/fullcircle/masking/omni2synthetic.py` — ERP→fisheye rendering (lines 107-124)
- `D:/Data/fullcircle/masking/synthetic2omni.py` — fisheye→ERP back-projection (lines 44-90)
- `D:/Data/fullcircle/masking/lib/cam_utils.py` — `look_at_camZ`, `get_cam_ray_dirs`, coordinate transforms

---

## Tier Structure (Updated)

### Default Tier: YOLO + SAM v1 + SAM v2

Three models, three jobs:
| Model | Package | Job | Pass |
|-------|---------|-----|------|
| YOLOv8s | `ultralytics` | Bounding box detection (person = COCO class 0) | Pass 1 |
| SAM v1 ViT-H | `segment-anything` | Refine bounding boxes to pixel masks | Pass 1 |
| SAM v2 | `sam2` | Video tracking on synthetic views (temporal propagation) | Pass 2 |

Dependencies: `ultralytics`, `segment-anything`, `sam2`, `torch`, `torchvision`

### Premium Tier: SAM 3

Replaces all three default-tier models:
| Capability | SAM 3 API |
|------------|-----------|
| Detection + segmentation | `build_sam3_image_model()` + `Sam3Processor.set_text_prompt()` |
| Video tracking | SAM 3 video predictor (inherits SAM 2's temporal propagation) |

Dependencies: `sam3`, `torch`, `torchvision`

### Graceful Degradation

If SAM v2 is not installed, Pass 2 falls back to per-frame `detect_and_segment()` on synthetic views (no temporal consistency, but still beneficial because the camera is aimed at the person). The masker never hard-fails due to missing SAM v2.

---

## File Changes

### New Functions in Existing Files

#### `core/masker.py` — Major refactor

**New helpers (pure math, no ML deps):**

| Function | Purpose | Reference |
|----------|---------|-----------|
| `_compute_detection_com(mask)` | Center-of-mass of detection mask via `cv2.moments` | FullCircle `mask_perspectives.py:138-144` |
| `_pixel_com_to_3d_direction(cx, cy, fov, yaw, pitch, size, flip_v)` | Convert pixel CoM to world-space 3D unit direction. Must invert fliplr/flipud + rotation from `_reframe_to_detection()`. | FullCircle `perspective2omni.py:390-410` |
| `_compute_weighted_person_direction(directions_and_weights)` | Weighted average of 3D directions → normalized unit vector. Weights = detection mask area. | FullCircle `omni2synthetic.py:92-99` |
| `_temporal_fallback_direction(frame_idx, all_directions)` | Search nearest frames forward/backward for valid direction | FullCircle `omni2synthetic.py:62-89` |
| `_look_at_rotation(center_dir)` | `world_from_cam` rotation with cam +Z aligned to center_dir. Port of `look_at_camZ()`. | `cam_utils.py:66-75` |
| `_direction_to_yaw_pitch(direction)` | Convert 3D unit direction to (yaw_deg, pitch_deg) | Trivial: `atan2(dx, dz)`, `asin(dy)` |
| `_create_synthetic_camera()` | Create pycolmap OPENCV_FISHEYE camera with ideal equidistant params | See fisheye section above |
| `_render_synthetic_fisheye(erp, camera, R_syn)` | ERP → fisheye view via remap. Uses `cam_from_img` for ray generation. | `omni2synthetic.py:107-124` |
| `_backproject_fisheye_mask_to_erp(mask, erp_size, camera, R_syn)` | Fisheye mask → ERP via remap. Uses `img_from_cam` for projection. | `synthetic2omni.py:44-90` |

**Refactored methods on `Masker`:**

| Method | Purpose |
|--------|---------|
| `_primary_detection(erp)` | Extract current per-view loop. Returns `(erp_mask, person_direction_or_None)`. Adds CoM recording. |
| `_synthetic_pass(frame_files, primary_masks, person_directions, frame_order)` | Full Pass 2: resolve directions → render synthetic views → video track → backproject → return synthetic ERP masks. |
| `process_frames(...)` | Refactored to three-phase: primary → synthetic → postprocess+save. Interface unchanged. |

**Updated `MaskConfig`:**
```python
@dataclass
class MaskConfig:
    # ... existing fields unchanged ...
    enable_synthetic: bool = True           # Enable Pass 2 synthetic camera
    synthetic_size: int = 2048              # Fisheye image resolution
```

**Updated `Masker.__init__` and `initialize`:**
```python
class Masker:
    def __init__(self, config=None):
        self.config = config or MaskConfig()
        self._backend: MaskingBackend | None = None
        self._video_backend: VideoTrackingBackend | None = None

    def initialize(self):
        self._backend = get_backend(self.config.backend_preference)
        # ... existing error handling ...
        self._backend.initialize()
        if self.config.enable_synthetic:
            self._video_backend = get_video_backend(
                self.config.backend_preference,
                fallback_image_backend=self._backend,
            )
            if self._video_backend is not None:
                self._video_backend.initialize()

    def cleanup(self):
        if self._video_backend is not None:
            self._video_backend.cleanup()
            self._video_backend = None
        # ... existing backend cleanup ...
```

#### `core/backends.py` — Add video tracking backends

**New module-level probe:**
```python
HAS_SAM2 = False
try:
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    HAS_SAM2 = True
except ImportError:
    pass
```

**New protocol:**
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

**New classes:**

| Class | Description |
|-------|-------------|
| `Sam2VideoBackend` | SAM v2 video predictor. Writes frames to tempdir, initializes video state, adds initial mask as prompt, propagates forward+backward. |
| `Sam3VideoBackend` | SAM 3 video predictor. Same pattern but uses SAM 3's API. |
| `FallbackVideoBackend` | Wraps a `MaskingBackend` instance. `track_sequence()` calls `detect_and_segment()` per-frame. No temporal consistency. |

**New factory:**
```python
def get_video_backend(
    preference: str | None = None,
    fallback_image_backend: MaskingBackend | None = None,
) -> VideoTrackingBackend | None:
    """Return best available video tracking backend.

    Priority: Sam3VideoBackend (if sam3 preferred + available)
            > Sam2VideoBackend (if sam2 available)
            > FallbackVideoBackend (wraps image backend)
    """
```

#### `core/setup_checks.py` — Add SAM v2 state

```python
@dataclass
class MaskingSetupState:
    # ... existing fields ...
    has_sam2: bool = False  # NEW

    @property
    def default_tier_ready(self) -> bool:
        # Base functionality: YOLO + SAM v1 (Pass 1 works without SAM v2)
        return self.has_torch and self.has_yolo and self.has_sam1

    @property
    def default_tier_video_ready(self) -> bool:  # NEW
        """Full FullCircle pipeline including SAM v2 video tracking."""
        return self.default_tier_ready and self.has_sam2
```

Add `_check_sam2_installed()`, update `check_masking_setup()`, update `install_default_tier()` to include `sam2` in the `uv add` command.

#### `pyproject.toml` — Add sam2 dependency

```toml
dependencies = [
    # ... existing ...
    "sam2>=1.0",
]
```

#### `core/__init__.py` — Add new exports

Export `VideoTrackingBackend`, `get_video_backend`, `Sam2VideoBackend`, `FallbackVideoBackend`.

---

## Task Breakdown

### Task 1: Fisheye Projection Functions

Pure geometry — no ML dependencies. Foundation for everything else.

**Files:**
- Create: `tests/test_synthetic_camera.py`
- Modify: `core/masker.py` (add functions only, don't refactor process_frames yet)

**Steps:**
- [ ] 1.1: Write tests for `_create_synthetic_camera()` — verify pycolmap Camera creation with OPENCV_FISHEYE, verify `cam_from_img` / `img_from_cam` round-trip on known pixel coords
- [ ] 1.2: Write tests for `_render_synthetic_fisheye()` — render a synthetic test pattern ERP, verify output is circular (fisheye lens region), verify center pixel maps to the look-at direction
- [ ] 1.3: Write tests for `_backproject_fisheye_mask_to_erp()` — create a synthetic mask with a known blob, backproject, verify the blob appears at the expected ERP location
- [ ] 1.4: Write round-trip test — render ERP→fisheye→mask→backproject→ERP, verify coverage area is consistent
- [ ] 1.5: Implement `_create_synthetic_camera()`, `_render_synthetic_fisheye()`, `_backproject_fisheye_mask_to_erp()` in masker.py
- [ ] 1.6: Run tests, verify all pass

### Task 2: Direction Computation Helpers

Pure math — center-of-mass, pixel→3D, weighted average, temporal fallback, look_at.

**Files:**
- Extend: `tests/test_synthetic_camera.py`
- Modify: `core/masker.py`

**Steps:**
- [ ] 2.1: Write tests for `_compute_detection_com()` — known blob with known centroid
- [ ] 2.2: Write tests for `_pixel_com_to_3d_direction()` — center pixel of a 0°/0° view should return [0, 0, 1] (forward). Center pixel of a 90°/0° view should return [1, 0, 0] (right). Must correctly invert fliplr.
- [ ] 2.3: Write tests for `_compute_weighted_person_direction()` — two directions at equal weight should return their normalized average
- [ ] 2.4: Write tests for `_temporal_fallback_direction()` — frame with None should borrow from nearest non-None frame (forward or backward)
- [ ] 2.5: Write tests for `_look_at_rotation()` — rotation for [0,0,1] should be identity-ish. Rotation for [1,0,0] should rotate 90° around Y. Verify `R @ [0,0,1] ≈ center_dir`.
- [ ] 2.6: Write test for `_direction_to_yaw_pitch()` — [0,0,1]→(0,0), [1,0,0]→(90,0), [0,1,0]→(0,90)
- [ ] 2.7: Implement all direction helpers
- [ ] 2.8: Run tests, verify all pass

### Task 3: VideoTrackingBackend Protocol + FallbackVideoBackend

**Files:**
- Modify: `core/backends.py`
- Extend: `tests/test_backends.py`

**Steps:**
- [ ] 3.1: Write test for `FallbackVideoBackend` — given a mock image backend that returns a fixed mask, verify `track_sequence()` returns that mask for each frame
- [ ] 3.2: Write test for `get_video_backend()` — with no SAM v2/v3 installed, should return FallbackVideoBackend when a fallback is provided
- [ ] 3.3: Add `VideoTrackingBackend` protocol, `FallbackVideoBackend`, `get_video_backend()` to backends.py
- [ ] 3.4: Add `HAS_SAM2` probe
- [ ] 3.5: Run tests, verify all pass (including all 116 existing tests)

### Task 4: Refactor Masker to Two-Pass

The core integration. Changes process_frames from single-pass to two-pass.

**Files:**
- Modify: `core/masker.py`
- Extend: `tests/test_masker.py`

**Steps:**
- [ ] 4.1: Extract `_primary_detection()` from the current per-frame loop. Add CoM recording + direction computation. The method returns `(erp_mask, weighted_person_direction_or_None)`.
- [ ] 4.2: Implement `_synthetic_pass()` — given primary masks + person directions + frame files, render synthetic fisheye views, run video tracking, backproject, return synthetic ERP masks.
- [ ] 4.3: Refactor `process_frames()` to: Phase 1 (primary) → Phase 2 (synthetic, if enabled) → Phase 3 (postprocess+save). Interface unchanged.
- [ ] 4.4: Update `Masker.__init__` and `initialize()` / `cleanup()` to handle video backend lifecycle.
- [ ] 4.5: Add `enable_synthetic` and `synthetic_size` to `MaskConfig`.
- [ ] 4.6: Write test — with FallbackVideoBackend (mock), verify process_frames produces output. Verify synthetic pass runs when enable_synthetic=True. Verify it's skipped when enable_synthetic=False.
- [ ] 4.7: Run ALL tests (116+), verify nothing broke. The masker's external interface hasn't changed so pipeline/panel tests should pass.

### Task 5: SAM v2 Video Backend

The actual SAM v2 integration.

**Files:**
- Modify: `core/backends.py`
- Modify: `core/setup_checks.py`
- Modify: `pyproject.toml`
- Extend: `tests/test_backends.py`, `tests/test_setup_checks.py`

**Steps:**
- [ ] 5.1: Research SAM v2's video predictor API — verify import paths (`sam2.build_sam.build_sam2_video_predictor`), verify it expects frames as a directory of numbered images or supports in-memory, verify the prompt/propagate API.
- [ ] 5.2: Implement `Sam2VideoBackend` — writes synthetic frames to tempdir (cleaned up after), initializes predictor, adds initial mask prompt on strongest-detection frame, propagates forward+backward, reads back masks.
- [ ] 5.3: Add `_check_sam2_installed()` and `has_sam2` field to setup_checks.py.
- [ ] 5.4: Update `install_default_tier()` to include `sam2` in `uv add`.
- [ ] 5.5: Add SAM v2 checkpoint download to install step (eager, like SAM v1).
- [ ] 5.6: Update `pyproject.toml` — add `sam2` to dependencies.
- [ ] 5.7: Update `get_video_backend()` to return `Sam2VideoBackend` when available.
- [ ] 5.8: Write interface test for `Sam2VideoBackend` (mocked, since sam2 may not be in test env).
- [ ] 5.9: Write test for `default_tier_video_ready` property in setup_checks.
- [ ] 5.10: Run all tests.

### Task 6: SAM 3 Video Backend (Premium Tier)

**Files:**
- Modify: `core/backends.py`

**Steps:**
- [ ] 6.1: Research SAM 3's video predictor API — verify it inherits SAM 2's pattern.
- [ ] 6.2: Implement `Sam3VideoBackend`.
- [ ] 6.3: Update `get_video_backend()` to prefer `Sam3VideoBackend` when preference="sam3".
- [ ] 6.4: Write interface test (mocked).
- [ ] 6.5: Run all tests.

### Task 7: Update `core/__init__.py` Exports

- [ ] 7.1: Export `VideoTrackingBackend`, `get_video_backend`, `FallbackVideoBackend`, `Sam2VideoBackend`.
- [ ] 7.2: Run all tests.

### Task 8: Integration Test

**Files:**
- Extend: `tests/test_masking_integration.py`

**Steps:**
- [ ] 8.1: Write integration test with synthetic ERP data — create a test ERP image with a known colored blob at a known location, run the full two-pass masker (with FallbackVideoBackend), verify the blob is masked in the output ERP mask.
- [ ] 8.2: Write test verifying synthetic camera is aimed at the right direction — after primary detection, check that `_direction_to_yaw_pitch()` of the computed direction points toward the blob.
- [ ] 8.3: Run full test suite.

### Task 9: Update Design Spec

- [ ] 9.1: Update `docs/specs/2026-04-02-masking-layer-v1-design.md` to reflect: two-pass architecture, OPENCV_FISHEYE synthetic camera, three-model default tier (YOLO + SAM v1 + SAM v2), updated pipeline stage descriptions.

---

## VRAM Management

Loading YOLO + SAM v1 + SAM v2 simultaneously may exceed 8GB VRAM. Mitigation:

1. After Pass 1 completes, call `self._backend.cleanup()` to free YOLO + SAM v1 VRAM.
2. Initialize `self._video_backend` (SAM v2) only when Pass 2 starts.
3. After Pass 2, clean up video backend.
4. This means the image backend is NOT available during Pass 2 — the `FallbackVideoBackend` path needs the image backend alive. Solution: `FallbackVideoBackend` keeps a reference and is initialized before the image backend is cleaned up. OR: only clean up the image backend when a proper video backend (SAM v2/v3) is available.

Concretely in `process_frames()`:
```python
# Pass 1 uses self._backend (YOLO + SAM v1)
# After Pass 1:
if isinstance(self._video_backend, (Sam2VideoBackend, Sam3VideoBackend)):
    self._backend.cleanup()  # free VRAM for video tracker
# Pass 2 uses self._video_backend
# After Pass 2:
self._video_backend.cleanup()
```

For FallbackVideoBackend: keep image backend alive (it wraps it).

---

## Progress Allocation (within masking stage 20-45%)

| Sub-phase | Progress | Description |
|-----------|----------|-------------|
| Pass 1: Primary detection | 20% → 32% | Per-view detection on all frames |
| Pass 2a: Render synthetic views | 32% → 35% | ERP → fisheye for all frames |
| Pass 2b: Video tracking | 35% → 40% | SAM v2/v3 on synthetic sequence |
| Pass 2c: Backproject + merge | 40% → 42% | Fisheye masks → ERP, OR-merge |
| Pass 3: Postprocess + save | 42% → 45% | Morph close, flood fill, invert, write |

---

## Coordinate Convention Notes

Our reframer uses a **different convention** than FullCircle. Both are correct but must not be mixed:

| | Our Reframer | FullCircle | Synthetic Camera (new) |
|--|-------------|------------|----------------------|
| Camera forward | -Z | +Z | +Z (matches FullCircle) |
| Rotation matrix | w2c (rows = [right, up, -fwd]) | w_from_c (cols = [x, y, z]) | w_from_c (FullCircle style) |
| ERP longitude | `θ = atan2(wx, wz)` | `lon = atan2(x, -z)` | Use FullCircle convention |
| ERP latitude | `φ = asin(wy)` | `lat = asin(-y)` | Use FullCircle convention |
| pycolmap camera | Not used | `cam_from_img` / `img_from_cam` | `cam_from_img` / `img_from_cam` |

The synthetic camera functions (`_render_synthetic_fisheye`, `_backproject_fisheye_mask_to_erp`) use **FullCircle's convention** since they use pycolmap's fisheye model which expects camera +Z = forward. The existing `_reframe_to_detection` and `_backproject_mask_to_erp` keep our reframer convention unchanged.

The bridge between them is `_pixel_com_to_3d_direction()` which converts from reframer-convention pixel coords to a world-space 3D direction (convention-agnostic unit vector).

---

## Verification

1. **Unit tests:** `.venv/Scripts/pytest.exe tests/ -v` — all tests pass including new ones
2. **Clear pycache:** `rm -rf core/__pycache__ panels/__pycache__ __pycache__`
3. **Manual test in LFS:** Run on `D:\Capture\deskTest` with Low preset + masking enabled
4. **Compare masks:** ERP masks in `extracted/masks/` should show better person coverage vs. previous run (especially in views where YOLO missed the person)
5. **Verify temporal consistency:** Check consecutive frame masks — with SAM v2 tracking, the mask region should be smooth across frames (no flickering in/out)
6. **Verify synthetic aim:** Add temporary debug output to verify the synthetic camera direction points at the person (can be removed after validation)

---

## Key Risks

1. **SAM v2 API surface** — The `sam2` PyPI package API may differ from the GitHub repo documentation. Step 5.1 (research) must verify exact import paths and predictor API before implementation.
2. **pycolmap fisheye round-trip accuracy** — `cam_from_img` and `img_from_cam` with OPENCV_FISHEYE model at k1-k4=0 should be exact for equidistant projection, but needs verification (Task 1.1).
3. **Direction computation correctness** — `_pixel_com_to_3d_direction` must exactly invert the fliplr + rotation in `_reframe_to_detection()`. Off-by-one or sign errors would aim the synthetic camera wrong. Task 2.2 tests this explicitly.
4. **VRAM pressure** — See VRAM Management section above.
