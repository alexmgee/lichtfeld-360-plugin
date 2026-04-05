# Default Masking Regression — Investigation Plan

**Date:** 2026-04-04
**Status:** Active — masks are still broken after reverting all optimization changes

---

## Situation

The Default preset masking pipeline produced good ERP masks at the end of the previous session (2847ffdb). In this session, we made optimization changes, saw broken masks, and reverted all optimization changes back to match the previous session's code structure. The masks are still broken.

This means the regression is NOT caused by the code changes we made in this session. Something else changed between the two sessions.

## What Has Been Ruled Out

1. **Batched YOLO** — reverted to per-view YOLO calls. Masks still bad.
2. **Detection resolution (512 vs 1024)** — reverted to 1024. Masks still bad.
3. **Single-best-box vs union box** — reverted to union box. Masks still bad.
4. **Remap caching** — removed from _primary_detection. Masks still bad.
5. **Pre-dilation (BACKPROJECT_DILATE_PX)** — removed. Masks still bad.
6. **SAM2 prompt frame selection** — fixed (detection count instead of empty mask area). Masks still bad.
7. **`_C.pyd` hole-filling** — attempted to disable but `ensure_sam2_c_extension()` re-installed it. Test was inconclusive. Needs proper testing.

## What Has NOT Been Checked

### 1. Package versions

Did `uv sync` run between sessions and update packages? The `pyproject.toml` was modified multiple times during this session (name changes, description changes, torch source pinning), and LichtFeld runs `uv sync` on any pyproject.toml change.

**Check:** Run from the plugin venv:
```powershell
.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); import sam2; print('sam2', sam2.__version__); import ultralytics; print('ultralytics', ultralytics.__version__); import segment_anything; print('sam1 ok')"
```

Compare against expected versions:
- torch: 2.11.0+cu128 (was this before)
- sam2: 1.1.0
- ultralytics: 8.4.33 or similar

If torch or sam2 were updated, that could change model behavior.

### 2. SAM2 model weights

Did SAM2 re-download model weights? HuggingFace model weights are cached in `~/.cache/huggingface/`. If they were updated or corrupted, SAM2's tracking behavior would change.

**Check:**
```powershell
dir C:\Users\alexm\.cache\huggingface\hub\models--facebook--sam2.1-hiera-large\
```

Look at file dates — are any newer than the previous session?

### 3. `_C.pyd` effect (inconclusive test)

The rename test was inconclusive because `ensure_sam2_c_extension()` re-copied `_C.pyd` from `lib/`. To properly test:

**Steps:**
1. Rename BOTH copies:
   ```powershell
   ren .venv\Lib\site-packages\sam2\_C.pyd _C.pyd.disabled
   ren lib\_C.pyd _C.pyd.disabled
   ```
2. Run test_masking.py to a new output directory
3. Verify the `cannot import name '_C' from 'sam2'` warning appears in output
4. Compare masks against previous runs

### 4. Torch CUDA behavior

Did the CUDA runtime or torch CUDA behavior change? The `pyproject.toml` now pins torch to the `pytorch-cu128` index explicitly, which may have pulled a different torch build than what was installed before.

**Check:**
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda); print(torch.backends.cudnn.version())"
```

### 5. The actual synthetic fisheye images

We haven't inspected what SAM2 is actually seeing and producing. Save the intermediate synthetic fisheye images and SAM2 tracked masks to disk.

**Steps:**
- Add debug output to `_synthetic_pass` that saves each rendered fisheye image and each SAM2 tracked mask
- Inspect whether the person is centered in the fisheye (direction quality)
- Inspect whether SAM2's tracked masks are clean before backprojection

This separates "SAM2 tracking the wrong thing" from "backprojection destroying good masks."

### 6. The `setup_checks.py` changes

The `_run_uv_command` function was modified during this session to use `UV_CACHE_DIR` and `install_default_tier` was changed to use `uv sync --locked --no-dev` instead of `uv add`. If LichtFeld ran the install flow at any point, this could have changed the venv state.

**Check:** Review whether the venv was modified:
```powershell
dir .venv\Lib\site-packages\torch\version.py
```
Check the file date against the previous session date.

## Recommended Investigation Order

1. **Package versions** — quickest check, highest impact if something changed
2. **`_C.pyd` proper disable** — rename both copies, verify warning appears
3. **Synthetic fisheye intermediates** — save and inspect what SAM2 sees
4. **Torch/CUDA versions** — check for CUDA runtime changes
5. **SAM2 model weights dates** — check for re-downloads

## Success Criteria

The investigation is complete when we can either:
- Identify the specific environmental change that caused the regression and fix/revert it
- OR reproduce the good masks by restoring the exact environment state from the previous session

## Related Documents

- `docs/2026-04-04-default-masking-regression-causality-report.md`
- `docs/2026-04-04-sam2-prompt-frame-selection-bug.md`
- `docs/2026-04-04-backprojection-sampling-artifact.md`
- `docs/2026-04-04-direction-estimation-regression.md`
