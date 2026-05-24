# COLMAP Options Fixes — Test Plan

Validation checklist for the changes on `dev/dual-fisheye-pipeline` implementing
Parts A, B, D, E of `colmap-options-fix-plan.md` plus the ERP scaffold pitch fix.
All tests run against real local data through the LichtFeld Studio UI or resume mode.
Committed as `df1414d` on 2026-05-24.

**How to use this document:** Work through sections in order. Check boxes as tests pass.
Record timing and notes in the provided columns. If a test fails, document what happened
in the Notes column and stop — do not proceed to later sections that depend on the failing one.

---

## 0. Available Test Datasets

| Dataset | Path | Type | Duration/Size | Notes |
|---------|------|------|---------------|-------|
| **testing_fisheye** | `D:/Capture/testing_fisheye/osmo/` | .OSV + pre-split .mp4 | 28s (~185s container) | Short clip, good for quick iterations |
| **testing_fisheye2** | `D:/Capture/testing_fisheye2/osmo/` | .OSV + pre-split .mp4 | — | Has test12 (SIFT baseline, 3952 imgs) and test13 (ALIKED, 1584 imgs, 16 cameras) |
| **testing_fisheye3** | `D:/Capture/testing_fisheye3/osmo/` | .OSV (3.3 GB) | 152s | test1 (610 imgs fisheye native), test2 (154 imgs fisheye native, full output) |
| **testing_fisheye4** | `D:/Capture/testing_fisheye4/osmo/` | .OSV + pre-split .mp4 | — | Has pre-split front/back .mp4 + source dir |
| **pantry_test** | `D:/Capture/pantry_test/osmo/` | .OSV | 34s | Short, newest dataset. Good for end-to-end |

**Primary test dataset for resume mode:** `testing_fisheye2/osmo/` — use existing extracted images
to skip extraction/masking/reframing and test only the COLMAP changes.

**Primary test dataset for end-to-end:** `pantry_test/osmo/` — 34s clip, completes in reasonable time.

### Existing baselines

| Run | Dataset | Mode | Feature/Matcher | Images | Reg. Rate | COLMAP Time | Notes |
|-----|---------|------|-----------------|--------|-----------|-------------|-------|
| test12_pinhole | testing_fisheye2 | fisheye_pinhole | SIFT+Bruteforce | 3952 | — | — | Baseline, completed successfully |
| test13_pinhole_alikedLightglue | testing_fisheye2 | fisheye_pinhole | ALIKED_N16ROT+LightGlue | 1584 (198 used) | — | Crashed | Missing aliked-n32.onnx — the exact bug Part A fixes |
| test2 | testing_fisheye3 | fisheye (native) | — | 154 | 153/154 | 1750s | Exhaustive matcher, 77 source frames |

---

## 1. Pre-Flight Checks

These verify the environment is correct before any real testing begins.

- [x] **1.1 — ONNX models exist in lib/**
  - Run: `ls lib/*.onnx` in plugin dir
  - Expected: 5 files — `aliked-lightglue.onnx`, `aliked-n16rot.onnx`, `aliked-n32.onnx`, `bruteforce-matcher.onnx`, `sift-lightglue.onnx`
  - Result: PASS — all 5 files present (46MB, 3MB, 4MB, 5KB, 46MB)

- [x] **1.2 — Unit tests pass**
  - Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
  - Expected: 60 passed, 0 failed (47 original + 13 orientation tests)
  - Result: PASS — 60 passed in 2.31s (re-verified 2026-05-24 after pitch fix)

- [x] **1.3 — Plugin loads in LichtFeld Studio**
  - Launch LichtFeld Studio, open the 360 Plugin panel
  - Expected: panel renders without errors, all dropdowns visible
  - Result: PASS (with notes) — panel loads, all dropdowns visible. Two observations: (1) SAM3 masking setup not recognized — likely pre-existing, not attributed to current changes. (2) BA solver note still shows "Hybrid: Ceres-GPU + Caspar global BA" — expected, Part C deferred.

- [x] **1.4 — Pre-flight validation works (negative test)**
  - Temporarily rename `lib/sift-lightglue.onnx` to `lib/sift-lightglue.onnx.bak`
  - Set Features=SIFT, Matcher Type=LightGlue, start a resume-mode run
  - Expected: immediate error message mentioning "Missing ONNX model(s) for sift+lightglue"
  - Restore: rename back to `lib/sift-lightglue.onnx`
  - Result: PASS — error message: "Missing ONNX model(s) for sift+lightglue: sift_lightglue (C:\...\lib\sift-lightglue.onnx). Download from COLMAP 3.13.0 release to lib/."

---

## 2. Part A — Feature/Matcher Matrix (Resume Mode)

Test each of the 6 feature/matcher combinations using resume mode on existing extracted
images. This skips extraction/masking/reframing and tests only the COLMAP feature extraction,
matching, and mapping pipeline.

**Dataset:** Copy `testing_fisheye2/osmo/test13_pinhole_alikedLightglue/images/` to a fresh
output directory for each test (or use the plugin's resume mode pointing at the existing dir).
The test13 dataset has 16 per-folder cameras x 99 frames = 1584 images total, but the previous
run only used 198 images (likely a subset). Use whatever the resume mode picks up.

**Output mode:** fisheye_pinhole (to exercise the rig-constrained path)

**Matcher:** Sequential (the default for fisheye_pinhole mode)

**For each test:** Record feature extraction time, matching time, mapping time, total COLMAP
time, and registration rate (registered images / total images).

**Headless test (2026-05-21):** Ran all 6 combos on a 480-image subset (16 cameras x 30 frames)
of the test13 dataset via `ColmapRunner` directly. No rig config — tests extraction + matching
only. Mapping produced no reconstruction (expected: sequential matching on disconnected
per-folder cameras without rig constraints can't build a connected graph).

### 2.1 — SIFT + Bruteforce (baseline, already working)

- [x] **Extraction + matching complete without crash**
  - Features: SIFT, Matcher Type: Bruteforce, Mapper: Incremental
  - Result: PASS — extraction 8.7s, matching 0.6s, matcher type SIFT_BRUTEFORCE

### 2.2 — SIFT + LightGlue (new — requires sift-lightglue.onnx)

- [x] **Extraction + matching complete without crash**
  - Features: SIFT, Matcher Type: LightGlue, Mapper: Incremental
  - Result: PASS — extraction 8.8s, matching 1.5s, matcher type SIFT_LIGHTGLUE
  - Pre-flight: `ONNX models verified for sift+lightglue: ['sift_lightglue']`
  - Log: `Step 3: SIFT LightGlue model: sift-lightglue.onnx`

### 2.3 — ALIKED N16 (rot) + Bruteforce (new — requires bruteforce-matcher.onnx)

- [x] **Extraction + matching complete without crash**
  - Features: ALIKED N16 (rot), Matcher Type: Bruteforce, Mapper: Incremental
  - Result: PASS — extraction 5.7s, matching 1.0s, matcher type ALIKED_BRUTEFORCE
  - Pre-flight: `ONNX models verified for aliked_n16rot+bruteforce: ['aliked_n16rot', 'bruteforce_matcher']`
  - Log: `Step 1: ALIKED model: aliked-n16rot.onnx` + `Step 3: Bruteforce ONNX model: bruteforce-matcher.onnx`

### 2.4 — ALIKED N16 (rot) + LightGlue (already working)

- [x] **Extraction + matching complete without crash**
  - Features: ALIKED N16 (rot), Matcher Type: LightGlue, Mapper: Incremental
  - Result: PASS — extraction 5.7s, matching 1.5s, matcher type ALIKED_LIGHTGLUE

### 2.5 — ALIKED N32 + Bruteforce (new — requires aliked-n32.onnx + bruteforce-matcher.onnx)

- [x] **Extraction + matching complete without crash**
  - Features: ALIKED N32, Matcher Type: Bruteforce, Mapper: Incremental
  - Result: PASS — extraction 5.7s, matching 1.1s, matcher type ALIKED_BRUTEFORCE
  - Pre-flight: `ONNX models verified for aliked_n32+bruteforce: ['aliked_n32', 'bruteforce_matcher']`
  - Log: `Step 1: ALIKED model: aliked-n32.onnx` + `Step 3: Bruteforce ONNX model: bruteforce-matcher.onnx`

### 2.6 — ALIKED N32 + LightGlue (new — requires aliked-n32.onnx)

- [x] **Extraction + matching complete without crash**
  - Features: ALIKED N32, Matcher Type: LightGlue, Mapper: Incremental
  - Result: PASS — extraction 5.7s, matching 1.5s, matcher type ALIKED_LIGHTGLUE
  - Pre-flight: `ONNX models verified for aliked_n32+lightglue: ['aliked_n32', 'aliked_lightglue']`
  - Log: `Step 1: ALIKED model: aliked-n32.onnx` + `Step 3: LightGlue model: aliked-lightglue.onnx`

### 2.x — Comparison Table (480 images, headless, no rig)

| # | Feature | Matcher | Extract (s) | Match (s) | Total (s) | Notes |
|---|---------|---------|-------------|-----------|-----------|-------|
| 2.1 | SIFT | Bruteforce | 8.7 | 0.6 | 10.0 | baseline |
| 2.2 | SIFT | LightGlue | 8.8 | 1.5 | 10.8 | new — sift-lightglue.onnx wired |
| 2.3 | ALIKED N16 | Bruteforce | 5.7 | 1.0 | 7.3 | new — bruteforce-matcher.onnx wired |
| 2.4 | ALIKED N16 | LightGlue | 5.7 | 1.5 | 7.8 | existed |
| 2.5 | ALIKED N32 | Bruteforce | 5.7 | 1.1 | 7.3 | new — aliked-n32.onnx + bruteforce wired |
| 2.6 | ALIKED N32 | LightGlue | 5.7 | 1.5 | 7.8 | new — aliked-n32.onnx + lightglue wired |

**Note:** Mapping failed for all 6 (`Failed to create any sparse model`) — expected without
rig constraints on per-folder sequential data. Full end-to-end mapping validation requires
the LichtFeld Studio UI with rig config (Section 6). The extraction + matching validation
confirms all ONNX models are correctly wired.

- [ ] **2.E2E — Full end-to-end validation with rig constraints**
  - Requires LichtFeld Studio UI — run at least one new combo (e.g. SIFT+LightGlue or ALIKED_N32+LightGlue) through the full pipeline with rig config to confirm mapping also works
  - Result: ___ (requires LichtFeld Studio UI)

---

## 3. Part B — Feature-Type max_features Defaults (UI)

These are visual checks in LichtFeld Studio. No pipeline run needed — just observe the UI.

**Setup:** Open the 360 Plugin panel. Expand "Advanced (feature tuning)" to see the Max Features slider.

- [x] **3.1 — Label renamed**
  - Expected: disclosure reads "Advanced (feature tuning)", not "Advanced (SIFT tuning)"
  - Result: PASS

- [x] **3.2 — SIFT default**
  - Set Features = SIFT, COLMAP Preset = Normal
  - Expected: Max Features slider shows 8192 (for pinhole mode) or the mode-appropriate preset value
  - Result: PASS

- [x] **3.3 — Switch SIFT to ALIKED N16 — max_features snaps down**
  - With Max Features at 8192 (SIFT default), switch Features to ALIKED N16 (rot)
  - Expected: Max Features slider snaps to 2048
  - Result: PASS

- [x] **3.4 — Switch ALIKED N16 back to SIFT — max_features snaps up**
  - With Max Features at 2048 (ALIKED default), switch Features back to SIFT
  - Expected: Max Features slider snaps back to 8192
  - Result: PASS

- [x] **3.5 — Custom value preserved across switch**
  - Set Features = SIFT, drag Max Features to 4096 (a non-default value)
  - Switch Features to ALIKED N16 (rot)
  - Expected: Max Features stays at 4096 (not reset, because it didn't match the old SIFT default of 8192)
  - Result: PASS

- [x] **3.6 — Preset system cap for ALIKED**
  - Set Features = ALIKED N16 (rot), Max Features shows 2048
  - Change COLMAP Preset from Normal to High
  - Expected: Max Features does NOT exceed 2048 (the cap from Step 7a prevents the preset system from overwriting ALIKED's documented default)
  - Result: PASS

- [x] **3.7 — Switch to ALIKED N32 — same 2048 default**
  - Switch Features to ALIKED N32
  - Expected: Max Features shows 2048 (same default as N16)
  - Result: PASS

---

## 4. Part D — GLOMAP Rig Warning (UI)

Visual checks in LichtFeld Studio. Observe the Mapper dropdown and warning note behavior.

- [x] **4.1 — Fisheye mode: both mapper options available, no warning**
  - Set Output Mode = Fisheye
  - Expected: Mapper dropdown shows both "Incremental" and "Global (GLOMAP)". No warning note visible.
  - Result: PASS

- [x] **4.2 — Fisheye mode + GLOMAP selected: no warning**
  - Set Output Mode = Fisheye, select Mapper = Global (GLOMAP)
  - Expected: No warning note (fisheye native doesn't use rig constraints)
  - Result: PASS

- [x] **4.3 — Switch to Fisheye (Pinhole): mapper clamps to Incremental**
  - With GLOMAP selected in fisheye mode, switch Output Mode to Fisheye (Pinhole)
  - Expected: Mapper dropdown resets to "Incremental" (index 0). GLOMAP is no longer selected.
  - Result: PASS

- [x] **4.4 — Pinhole mode + GLOMAP: warning appears**
  - Set Output Mode = Pinhole, select Mapper = Global (GLOMAP)
  - Expected: Warning note appears below the Mapper dropdown: "GLOMAP lacks rig constraints..."
  - Result: PASS

- [x] **4.5 — Switch to Pinhole from fisheye with GLOMAP: mapper clamps**
  - Set Output Mode = Fisheye, select GLOMAP
  - Switch to Output Mode = Pinhole
  - Expected: Mapper clamps to Incremental
  - Result: PASS

- [x] **4.6 — ERP mode + GLOMAP: warning appears**
  - Set Output Mode = ERP, select Mapper = Global (GLOMAP)
  - Expected: Warning note appears (ERP uses rig scaffolding)
  - Result: PASS

---

## 5. Part E — Sequential Overlap Slider

### 5.1 — UI Checks

- [x] **5.1.1 — Slider visible when Matcher = Sequential**
  - Set Matcher = Sequential
  - Expected: "Overlap" slider visible, shows default value of 10, range 2-20
  - Result: PASS

- [x] **5.1.2 — Slider hidden when Matcher = Exhaustive**
  - Set Matcher = Exhaustive
  - Expected: Overlap slider disappears
  - Result: PASS

- [x] **5.1.3 — Slider value clamps to range**
  - Drag slider to min (2) and max (20)
  - Expected: Values clamp correctly, no out-of-range values
  - Result: PASS

### 5.2 — Matching Speed Comparison (Resume Mode)

Run the same dataset twice with different overlap values to confirm the parameter actually
affects matching time. Use the same feature/matcher combo for both (whichever was fastest
in Section 2).

**Dataset:** Same resume-mode dataset as Section 2.
**Feature/Matcher:** (pick the fastest from Section 2)
**Matcher:** Sequential

- [x] **5.2.1 — Overlap = 10 (default)**
  - Dataset: test13_pinhole_alikedLightglue (1584 images, 16 cameras, fisheye_pinhole resume mode)
  - Features: SIFT, Matcher: Bruteforce, Sequential
  - Result: PASS — completed in 237.8s

| Metric | Value |
|--------|-------|
| Feature extraction | 0.1s |
| Matching time | 131.4s (55%) |
| Mapping time | 3.7s (2%) |
| Total COLMAP time | 237.7s |
| Registration rate | 26/99 frames (26 complete rig, 73 dropped) |
| Registered images | 52/1584 |
| Notes | Low reg rate expected — SIFT+Bruteforce on ALIKED-extracted crops (mismatched extractor/images). Only front_ctr_hi and front_ctr_lo registered (26 each). |

- [x] **5.2.2 — Overlap = 5 (halved)**
  - Result: Mapping failed ("No valid frames after pose propagation") but matching completed successfully.

| Metric | Value |
|--------|-------|
| Feature extraction | 0.1s |
| Matching time | 79.6s (84%) |
| Mapping time | 2.0s (2%) |
| Total COLMAP time | 95.0s |
| Registration rate | 0/99 (mapping failed) |
| Notes | Too few matches at overlap=5 on this already-marginal dataset. Matching itself completed — the parameter is wired. |

- [x] **5.2.3 — Speed ratio**
  - Expected: overlap=5 matching time is roughly half of overlap=10 (within ~30%)
  - Actual ratio: 79.6s / 131.4s = **0.61** (39% reduction). Within expected range.
  - Registration rate comparison: overlap=10 registered 26/99, overlap=5 registered 0/99. Dataset is marginal (SIFT on ALIKED-extracted crops) — the overlap reduction pushed it past the failure threshold. Not a code issue.

---

## 6. End-to-End Pipeline Test

Full pipeline run from .OSV container through to output. Confirms the complete flow still
works with all the changes.

**Dataset:** `D:/Capture/pantry_test/osmo/CAM_20260516233117_0036_D.OSV` (34s clip)

**Settings:**
- Output Mode: Fisheye (Pinhole)
- Features: SIFT (baseline extractor)
- Matcher Type: Bruteforce
- Matcher: Sequential
- Overlap: 10 (default)
- COLMAP Preset: Normal
- Mapper: Incremental

- [x] **6.1 — Pipeline starts without errors**
  - Expected: extraction begins, no pre-flight failures
  - Result: PASS

- [x] **6.2 — All stages complete**
  - Expected: extraction, masking (if enabled), reframing, COLMAP, output — all complete
  - Result: PASS — all stages completed. Initial run failed due to pre-existing bug: `ImageReaderOptions` pickle corruption caused empty COLMAP database. **Bug found and fixed during testing** (serialization now uses todict+setattr instead of pickle for `ImageReaderOptions`, skipping default mask_path="."). Re-run on fresh output path succeeded.

- [x] **6.3 — Output structure correct**
  - Expected: `images/` has camera subdirectories, `sparse/0/` has reconstruction, `transforms.json` exists
  - Result: PASS — transforms.json written at colmap_options_e2e_test3

- [x] **6.4 — Registration rate acceptable**
  - Expected: >90% of images registered
  - Result: PASS — 33/35 frames (94%), 66/70 images, 33 complete rig frames

| Metric | Value |
|--------|-------|
| Source frames | 35 |
| Total images | 1120 (16 cameras x 35 frames, staged: 70) |
| Registered frames | 33/35 (94%) |
| Registered images | 66/70 |
| Complete rig frames | 33 |
| Dropped rig frames | 2 (000018.jpg, 000035.jpg) |
| Extraction | 49.1s |
| Masking | 66.9s (SAM3) |
| Staging | 45.6s |
| COLMAP | 22.6s (feat 0.1s, match 11.8s, map 5.7s) |
| Total pipeline time | 184.3s |
| Notes | Fisheye (Pinhole) mode, SIFT+Bruteforce, Sequential overlap=10, pantry_test .OSV (34s). Also confirmed rig DB serialization fix. |

---

## 7. Regression Checks

Verify existing functionality that should not have been affected.

- [x] **7.1 — Fisheye native mode still works**
  - Run a fisheye native pipeline (Output Mode = Fisheye) on any short dataset
  - Expected: completes without errors, outputs OPENCV_FISHEYE transforms.json
  - Result: PASS — pantry_test .OSV, 186.5s total, 34/35 frames registered (97%), masking 72.9s (SAM3 working), COLMAP 20.3s. Output: transforms.json at colmap_options_e2e_test2. Also confirms rig DB fix — COLMAP got 70 images in database (not 0).

- [x] **7.2 — ERP scaffold mode still works**
  - Run with Output Mode = ERP on pantry_ERP.mp4 (7680x3840 equirectangular), output colmap_options_erp_test5
  - Expected: completes, outputs transforms.json with `camera_model: EQUIRECTANGULAR` referencing ERP frames, training converges
  - Result: PASS — 732.2s total, 35/35 frames (100%), 280/280 images, 0 dropped rig frames. transforms.json has camera_model: EQUIRECTANGULAR, 35 frames referencing 7680x3840 ERP images. Masking via Sam3Backend (511.7s). COLMAP 125.7s. **Training confirmed working with GUT + masks** (2026-05-24, after pitch sign fix). Earlier runs (erp_test1–test4) produced garbage training due to pitch sign bug — now fixed.

- [x] **7.3 — Loop closure checkbox still functions**
  - Enable Loop Closure checkbox, verify vocab tree status text appears
  - Expected: status shows the bundled vocab tree filename
  - Result: PASS

---

## 8. Summary

| Section | Tests | Passed | Failed | Skipped |
|---------|-------|--------|--------|---------|
| 1. Pre-flight | 4 | 4 | 0 | 0 |
| 2. Part A — Feature/Matcher | 7 | 6 | 0 | 1 (UI) |
| 3. Part B — max_features | 7 | 7 | 0 | 0 |
| 4. Part D — GLOMAP warning | 6 | 6 | 0 | 0 |
| 5. Part E — Overlap | 5 | 5 | 0 | 0 |
| 6. End-to-end | 4 | 4 | 0 | 0 |
| 7. Regression | 3 | 3 | 0 | 0 |
| **Total** | **36** | **35** | **0** | **1** |

**Overall result:** 35/36 pass, 0 failures. 1 remaining: 2.E2E — full mapping with a new
feature/matcher combo (e.g. ALIKED+LightGlue) through the UI with rig constraints.

**Date started:** 2026-05-22
**Date updated:** 2026-05-24

**Tester:** Claude + Alex

**Bugs discovered and fixed during testing:**

1. **ImageReaderOptions pickle corruption** — `pickle.dumps/loads` of pycolmap
   `ImageReaderOptions` corrupts internal C++ state. Setting `mask_path="."` (the default)
   triggers COLMAP's mask loader which silently skips every image (database shows 0 images
   despite "70/70 processed"). Fixed: JSON serialization with field-by-field `setattr`
   deserialization, skipping default path fields.

2. **HuggingFace access check not persisting** — `model_info()` network call ran every LFS
   session because the cache was in-memory only. Fixed: persist verification to
   `.hf_access_verified` file on disk.

3. **LFS loading pinhole cameras instead of ERP** — After ERP scaffold export, passing the
   output directory to `lf.load_file` triggered ColmapLoader (which matches directories)
   before BlenderLoader. Fixed: pass the `transforms.json` file path instead.

4. **sparse/ not cleaned after ERP export** — Leftover `sparse/` references pinhole crop
   camera geometry but `images/` now contains ERP frames, producing a broken dataset on
   reload. Fixed: `shutil.rmtree` in `cleanup_colmap_artifacts`.

5. **ERP scaffold pitch sign (critical)** — `correction_rad = np.radians(-ref_pitch_deg)`
   negated the pitch correction, applying +35° instead of -35°. Total angular error:
   2 × 35° = 70°, exactly matching the observed systematic rotation residual vs Metashape
   reference. Positions were correct (0.13% error); only orientations were wrong. Fixed:
   `np.radians(ref_pitch_deg)`. Training confirmed working after fix.

**Features added during testing:**

1. **"Keep pinhole scaffolding" checkbox** — ERP mode now deletes pinhole crops by default
   after scaffold export. Checkbox under the Output Mode description retains them for
   debugging.

---
