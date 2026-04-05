# Masking v1 Preflight Checklist

**Date:** 2026-04-03
**Scope:** Preflight verification for `docs/specs/2026-04-03-masking-v1-plan-final.md`
**Goal:** Confirm the highest-risk assumptions before starting the full Track A / Track B implementation

---

## Exit Criteria

The plan is ready to execute when:

- synthetic fisheye projection math is verified in the plugin environment
- direction math is consistent with the current reframer conventions
- backend lifecycle and fallback behavior are proven with a small mocked flow
- Track A's fallback-only path is shown to be viable on a tiny synthetic sequence
- setup/UI capability expansion is confirmed to fit the current plugin state model
- SAM v2 B1 viability is either proven or explicitly downgraded to a deferred enhancement

---

## Checklist

### 1. Synthetic Camera Math

**Purpose:** Confirm the pycolmap fisheye assumptions the plan depends on.

- [x] 1.1 Verify `pycolmap` version in the plugin venv
- [x] 1.2 Verify `cam_from_img()` returns 2D normalized camera-plane coordinates
- [x] 1.3 Verify `img_from_cam()` round-trips exactly from `[u, v, 1]`
- [x] 1.4 Verify center pixel maps to forward `+Z`
- [x] 1.5 Verify a 45° sample ray maps to `[1, 0]`
- [x] 1.6 Verify the 90° hemisphere edge is asymptotic, not a finite normalized value
- [x] 1.7 Verify ERP → synthetic fisheye → ERP round-trip behavior on a tiny synthetic image

**Pass condition:** The camera model behaves exactly as required by the plan and the render/backproject path is numerically sane.

### 2. Direction Math / Reframer Contract

**Purpose:** Confirm the bridge between the current pinhole reframer and the new synthetic-camera world-space direction math.

- [x] 2.1 Verify `_reframe_to_detection()` conventions against `create_rotation_matrix()`
- [x] 2.2 Verify left/right/top/bottom pixel movement maps to the expected world directions
- [x] 2.3 Verify `flip_v=True` behavior is correctly inverted in direction recovery
- [x] 2.4 Verify center-of-mass to world-direction recovery on known synthetic masks
- [x] 2.5 Verify `_look_at_rotation()` sends camera `+Z` to the resolved direction

**Pass condition:** There is no unresolved coordinate-system mismatch between existing masking views and the synthetic camera.

### 3. Backend Lifecycle / Fallback Contract

**Purpose:** Confirm the plan's runtime ownership model before refactoring the real masker.

- [x] 3.1 Mock a primary image backend with `initialize()` / `detect_and_segment()` / `cleanup()`
- [x] 3.2 Mock a video backend that succeeds through `initialize()` and `track_sequence()`
- [x] 3.3 Mock a video backend that fails during `initialize()`
- [x] 3.4 Mock a video backend that fails during `track_sequence()`
- [x] 3.5 Verify the control flow is: Pass 1 backend cleanup → video backend failure → fresh image backend init → `FallbackVideoBackend` retry
- [x] 3.6 Verify final cleanup always runs and no backend instance is leaked

**Pass condition:** The recovery path described in the final plan is implementable without conflicting lifecycle ownership.

### 4. Track A Dry Run

**Purpose:** Prove the two-pass architecture works even without SAM v2.

- [x] 4.1 Build a tiny synthetic ERP sequence with an obvious target region
- [x] 4.2 Simulate primary-pass direction selection
- [x] 4.3 Render synthetic fisheye frames aimed at the target
- [x] 4.4 Run fallback per-frame masking on the synthetic views
- [x] 4.5 Backproject and OR-merge the synthetic masks with the primary masks
- [x] 4.6 Verify the output is meaningfully better than primary-only or at least not worse

**Pass condition:** Track A is demonstrably shippable as a fallback-only improvement.

### 5. Setup / UI Contract

**Purpose:** Confirm the existing setup state can be expanded cleanly to the capability model in the plan.

- [x] 5.1 Verify current `MaskingSetupState` fields and derived properties
- [x] 5.2 Verify current panel status text paths for YOLO+SAM v1 and SAM 3
- [x] 5.3 Verify capability levels `0/1/2/3` can be added without regressing existing SAM 3 reporting
- [x] 5.4 Verify the new install action split is compatible with the existing `install_default_tier()` flow

**Pass condition:** The capability-level UI plan fits the current plugin structure without surprising refactors.

### 6. SAM v2 B1 Viability

**Purpose:** Decide whether Track B is truly available or should remain deferred.

- [x] 6.1 Test `sam2` installation method in the plugin venv
- [x] 6.2 Test import of `build_sam2_video_predictor_hf`
- [x] 6.3 Test model construction and download
- [x] 6.4 Test `init_state()` with numbered JPEG frames on disk
- [x] 6.5 Test prompt + propagation on a tiny frame set
- [ ] 6.6 Test behavior inside LichtFeld Studio's embedded runtime

**Pass condition:** `sam2` is usable enough to justify implementing Track B now.

### 7. Failure-Mode Matrix

**Purpose:** Verify the plan degrades cleanly on difficult or broken runs.

- [x] 7.1 No detections in the entire clip
- [x] 7.2 Sparse detections with temporal gaps
- [x] 7.3 Video backend init failure
- [x] 7.4 Video backend track failure
- [ ] 7.5 Tempdir cleanup on exception
- [ ] 7.6 Cancellation during masking stage

**Pass condition:** None of the expected failure paths force a whole-pipeline failure unless absolutely necessary.

---

## Working Log

### Completed

- Synthetic camera math passed in the plugin venv with `pycolmap 4.0.2`.
- Confirmed `cam_from_img()` returns 2D normalized coordinates `[x/z, y/z]`.
- Confirmed `img_from_cam([u, v, 1])` round-trips with `0.0` max reprojection error on sample points.
- Confirmed center pixel maps to `[0, 0]`, a 45° sample ray maps to `[1, 0]`, and the near-horizon coordinate is large (`~651.898`) rather than finite at `1`.
- Confirmed a tiny ERP → fisheye → ERP prototype preserved a centered blob exactly in both pixel count (`1009`) and centroid (`(256.0, 128.0)`).
- Direction / reframer contract passed numerically against the current `create_rotation_matrix()` math.
- Confirmed the exact optical-center sample for an even-sized, horizontally flipped detection image is the left-of-middle center pixel in x (for a `1024px` view, `(511, 512)` maps exactly to `[0, 0, 1]`).
- Confirmed yaw=90 at that optical-center sample maps exactly to `[1, 0, 0]`, pitch=45 maps to `[0, 0.7071, 0.7071]`, and `flip_v` correctly swaps top/bottom recovery.
- Confirmed the FullCircle-style `look_at_camZ()` convention maps camera `+Z` onto the requested world-space direction.
- Backend lifecycle / retry contract passed in a mocked control-flow prototype for success, init-failure, and track-failure scenarios.
- Confirmed the intended recovery sequence is viable: Pass 1 image backend cleanup → video backend failure → fresh image backend init → `FallbackVideoBackend` retry → final cleanup.
- Tightened the final plan so `process_frames()` explicitly owns runtime recovery and `FallbackVideoBackend` ownership is spelled out for both the shared Track A path and the fresh recovery path.
- Track A dry run passed on a tiny synthetic three-frame sequence.
- Confirmed the resolved directions tracked the expected left / center / right movement, the synthetic fisheye target stayed centered on every frame, and the final merged ERP mask improved IoU over the coarse primary-only pass on all frames (`0.857/0.909/0.857` → `1.0/1.0/1.0`).
- Setup / UI contract review passed: the current `MaskingSetupState` is small and capability levels can be added cleanly on top of `default_tier_ready`, `premium_tier_ready`, and `active_backend`.
- Confirmed the current panel text paths are localized to a few spots (`show_masking_install`, `show_masking_controls`, `_get_masking_backend_text`, and the default-tier install handler), so the capability-level transition should be a contained change rather than a panel rewrite.
- SAM v2 B1 viability is mostly green in the plugin venv:
- `sam2==1.1.0` and `huggingface-hub==1.9.0` installed successfully with LichtFeld's bundled `uv.exe`.
- Import of `build_sam2_video_predictor_hf` and `SAM2VideoPredictor` succeeded.
- Model construction succeeded on CPU after setting `HF_HUB_DISABLE_XET=1`; without that override, HuggingFace download failed because `hf_xet` hit a Windows DLL-load error.
- `init_state()` succeeded on a workspace-local numbered JPEG directory.
- `add_new_points_or_box()` succeeded.
- `propagate_in_video()` succeeded on a tiny 5-frame sequence and returned masks for all frames.
- Failure-mode coverage is partially cleared:
- Confirmed the all-empty direction case returns `None` for every frame, which matches the plan's “skip Pass 2 and preserve primary masks” behavior.
- Confirmed sparse-gap fallback borrows the nearest valid direction in the expected order.
- Confirmed video backend init and track failures both degrade cleanly through the mocked recovery path.

### Open Risks

- Final SAM v2 behavior still needs confirmation inside the real LichtFeld Studio runtime path, not just the plugin venv shell.
- `cv2` import from the bare plugin venv shell failed with a DLL load error during the first spike; this did not block the `pycolmap` verification, but it may matter when running standalone shell-side prototypes outside the normal plugin/LichtFeld runtime.
- HuggingFace Xet is currently a concrete Windows caveat: fresh model download required `HF_HUB_DISABLE_XET=1` to avoid an `hf_xet` DLL-load failure.
- SAM2 emitted a warning that the optional `sam2._C` extension was unavailable, so post-processing was skipped. Propagation still worked in the preflight, but this should be documented and sanity-checked during implementation.
- Tempdir cleanup-on-exception and masking-stage cancellation still need explicit preflight coverage or implementation-time tests.

### Recommended Execution Order

1. Synthetic camera math
2. Direction math / reframer contract
3. Backend lifecycle / fallback contract
4. Track A dry run
5. Setup / UI contract
6. SAM v2 B1 viability
7. Failure-mode matrix
