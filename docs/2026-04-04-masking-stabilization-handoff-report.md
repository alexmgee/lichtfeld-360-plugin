# Default Masking Stabilization — Handoff Report

**Date:** 2026-04-04  
**Audience:** Claude / future maintainers  
**Purpose:** Provide one comprehensive update on how the plugin was brought back to a stable working masking state, what changed, what was verified, and what the current safe baseline is before further optimization work.

---

## Executive Summary

The apparent Default preset masking regression was **primarily caused by environment drift**, not by an unresolved masking-code regression.

The important end state is now:

- Default masking quality is back to the prior good level.
- Full plugin test runs now produce good ERP masks and good pinhole masks again.
- The plugin venv is restored to the known-good CUDA + SAM2 runtime.
- The environment has been backed up externally.
- The SAM2/video-tracking path is now modeled as a locked optional feature in the project.
- `uv sync --locked --extra video-tracking` now preserves the working runtime.
- The repo has been checkpointed cleanly in git.

At this point, the plugin is back in a stable working state and is ready for careful re-application of the previous optimization work.

---

## What Went Wrong

Two problem families became entangled during investigation:

### 1. Real masking-quality concerns

During the optimization period, several suspected regressions were investigated:

- Pass 1 detection resolution changes
- direction-estimation changes
- prompt-frame-selection behavior
- backprojection quality

Some of these were real concerns at different points in the session history, but they did **not** turn out to be the main reason masking looked catastrophically worse in the final observed state.

### 2. Environment drift

The plugin venv was mutated by multiple mechanisms during the same period:

- `uv sync`
- `uv pip install`
- plugin setup/runtime helper behavior
- ad hoc recovery attempts
- `_C.pyd` handling experiments

That drift caused the active runtime to stop matching the earlier known-good masking environment.

This was the critical confounder.

---

## Key Root Cause Findings

## 1. Ownership/ACL issues were real but not the final blocker

Earlier in the session there were genuine Windows ownership/ACL problems:

- `Access denied` behavior in the `uv` cache
- mixed ownership under `.venv`
- runtime trouble around copied binary artifacts like `_C.pyd`

Those were repaired successfully and were no longer the main reason for bad mask quality.

### What was done

- workspace/plugin ACL repair was performed
- `.venv`, `tmp`, `%LOCALAPPDATA%\\uv\\cache`, and `%USERPROFILE%\\.config\\git` were normalized back to user ownership
- a reusable repair script was created:
  - [dev/fix_workspace_acl_v2.ps1](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/fix_workspace_acl_v2.ps1)

### Outcome

- permission failures stopped being the primary blocker
- `git` and `uv` became usable again

---

## 2. The venv had drifted away from the known-good masking runtime

At one point during investigation, the plugin environment was no longer equivalent to the previously good masking session.

Important observed failures included:

- `sam2` briefly stopped existing as a healthy installed package state
- CUDA-enabled `torch` / `torchvision` were replaced with CPU builds
- masking tests were then being interpreted from a different runtime than the one that produced the earlier good masks

### Concrete bad runtime state that was observed

- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`
- `torch.version.cuda == None`
- `torch.cuda.is_available() == False`

This meant the runtime was no longer a valid comparison point against the earlier successful masking runs.

---

## 3. Reverting optimization code alone was not enough

Some important masking code paths had already been reverted or corrected during the investigation:

- Pass 1 detection size restored to `min(1024, erp_w // 4)`
- union-box direction path restored
- SAM2 prompt-frame selection changed to use `detection_counts`

Those code corrections mattered, but they did **not** fix the observed regression while the environment remained drifted.

This was the key clue that the runtime had become the dominant problem.

---

## Recovery Work That Was Performed

## 1. Stabilization reporting and planning

The following planning/reference docs were created to stop the work from drifting:

- [2026-04-04-default-masking-stabilization-report-and-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-default-masking-stabilization-report-and-plan.md)
- [2026-04-04-known-good-masking-env-state.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-known-good-masking-env-state.md)
- [2026-04-04-masking-env-drift-report-and-recovery-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-env-drift-report-and-recovery-plan.md)
- [2026-04-04-sam2-lock-in-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-sam2-lock-in-plan.md)

These became the basis for the controlled recovery sequence.

---

## 2. Known-good runtime restored

The environment was restored back to the intended masking runtime:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`
- `sam2.build_sam` importable

### Validation command used

```powershell
& 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv\Scripts\python.exe' -c "import torch, torchvision, importlib.util as u; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('sam2_build_sam', u.find_spec('sam2.build_sam') is not None)"
```

### Validated result

- `torch 2.11.0+cu128`
- `torch_cuda 12.8`
- `cuda_available True`
- `torchvision 0.26.0+cu128`
- `sam2_build_sam True`

---

## 3. Runtime snapshot backed up externally

To protect the recovered working baseline, the live `.venv` was copied to:

- [C:\Users\alexm\.lichtfeld\plugins\backup\lichtfeld-360-plugin-venv-masking-good-2026-04-04](/c:/Users/alexm/.lichtfeld/plugins/backup/lichtfeld-360-plugin-venv-masking-good-2026-04-04)

This is the emergency fallback snapshot of the recovered env.

---

## 4. Masking-only validation

A controlled masking-only rerun was performed using:

- [dev/test_masking.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/dev/test_masking.py)

against:

- `D:\Capture\deskTest\default_test\extracted\frames`

with output written to a fresh directory.

### Outcome

- masks looked “much much better”
- quality returned to the prior expected baseline

This was the first strong confirmation that the environment restoration had fixed the apparent regression.

---

## 5. Full plugin validation

After the masking-only recovery succeeded, a full plugin test run was also performed.

### Outcome

- ERP masks looked good
- pinhole masks looked good

This confirmed that the restored runtime also behaved correctly in the real app path, not just the isolated masking runner.

---

## 6. SAM2 lock-in work completed

The next goal was not just “make the env work again,” but “make it stay working.”

### Problem before lock-in

`uv sync --locked --dry-run` no longer wanted to strip CUDA `torch`, but it **did** want to strip:

- `sam2`
- and the rest of the video-tracking/Hugging Face support stack

This meant the recovered env was still only *currently working*, not yet *project-reproducible*.

### What was changed

#### `pyproject.toml`

The project now models video tracking as an explicit optional feature:

```toml
[project.optional-dependencies]
video-tracking = [
    "sam2==1.1.0",
    "huggingface-hub==1.9.0",
]
```

This was needed because `sam2.build_sam` imports Hugging Face functionality at runtime, but that requirement was not sufficiently represented by the original project state.

#### `core/setup_checks.py`

`install_video_tracking()` was changed from:

```text
uv add sam2
```

to a locked sync path:

```text
uv sync --locked --no-dev --extra video-tracking
```

This is the critical architectural change that moved video tracking from an ad hoc env mutation to a lock-backed feature.

#### `uv.lock`

The lockfile was regenerated so the `video-tracking` extra is now represented in the project’s locked dependency state.

---

## 7. Lock-backed sync verified

After the metadata and installer changes:

```powershell
uv sync --locked --extra video-tracking
```

now works against the live env.

### Important practical result

The environment now remains on:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- CUDA enabled
- `sam2.build_sam` importable

and the masking smoke test still passes afterward.

### Remaining dry-run churn

The only remaining dry-run churn is the harmless direct-wheel naming normalization:

- `pycolmap==4.0.2` ↔ `pycolmap==4.0.2+cuda`

This is not a functional masking issue.

---

## Final Technical State

## Current live environment

- CUDA torch restored
- SAM2 importable
- video-tracking extra lock-backed
- full plugin masking verified

## Current safe commands

### Base/default plugin env

```powershell
uv sync --locked --no-dev
```

### Plugin env with SAM2 video tracking

```powershell
uv sync --locked --no-dev --extra video-tracking
```

This is now the canonical distinction.

---

## Current Git State

Two important checkpoint commits were created:

### Major recovery checkpoint

- `ec158a1`
- `Checkpoint masking stabilization and environment lock-in`

This commit captures:

- the masking/pipeline/setup state
- env hardening changes
- SAM2 lock-in changes
- helper scripts
- tests
- docs
- `uv.lock`
- bundled SAM2 artifacts in `lib/`

### Verification checkpoint

- `93452b3`
- `Document verified masking baseline and lock-backed env`

This commit records the final successful outcome in the docs:

- masking quality restored
- env drift identified as the primary cause
- lock-backed `video-tracking` sync validated

---

## Main Conclusion

The most important conclusion to carry forward is:

> the apparent Default preset masking regression was primarily an environment problem, not a proof that the restored masking code path was still broken.

More specifically:

- environment drift caused the runtime to diverge from the known-good session
- once CUDA + SAM2 were restored, the quality returned
- once the `video-tracking` extra was lock-backed, that good runtime became project-reproducible

So the plugin is now back in a stable state.

---

## What Should Happen Next

The next phase should **not** be more emergency recovery work.

It should be:

1. Keep the current stabilized baseline intact.
2. Re-apply the earlier optimization and quality improvements **carefully**.
3. Make one change at a time.
4. After each change:
   - run masking-only validation
   - inspect a fixed set of known-sensitive frames
   - capture timing
   - keep the change only if both speed and quality hold

Recommended order for re-application:

1. safest performance-only work first
   - Pass 1 remap caching
   - batched YOLO
2. Stage 3 caching/reuse work
3. quality-sensitive changes only with explicit validation
   - direction-estimation changes
   - prompting changes
   - backprojection changes

---

## Related Reference Docs

Most important:

- [2026-04-04-default-masking-stabilization-report-and-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-default-masking-stabilization-report-and-plan.md)
- [2026-04-04-sam2-lock-in-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-sam2-lock-in-plan.md)
- [2026-04-04-known-good-masking-env-state.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-known-good-masking-env-state.md)
- [2026-04-04-masking-env-drift-report-and-recovery-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-env-drift-report-and-recovery-plan.md)
- [2026-04-04-performance-optimization-results.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-performance-optimization-results.md)
- [2026-04-04-masking-performance-quality-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-performance-quality-plan.md)

Contextual background:

- [2026-04-04-default-masking-regression-causality-report.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-default-masking-regression-causality-report.md)
- [2026-04-04-masking-regression-investigation-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-regression-investigation-plan.md)
- [2026-04-04-masking-regression-investigation-plan-addendum.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-regression-investigation-plan-addendum.md)
