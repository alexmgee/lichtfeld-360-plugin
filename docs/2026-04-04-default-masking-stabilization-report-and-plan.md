# Default Masking Stabilization — Consolidated Report And Plan

**Date:** 2026-04-04  
**Status:** Baseline restored / lock-in planning active  
**Scope:** Default preset masking only  
**Purpose:** Freeze the current understanding of the regression, separate code-state facts from environment drift, and define the exact recovery/investigation order before any more implementation changes.

---

## Executive Summary

The Default preset masking investigation originally involved **two overlapping problems**:

1. **A real masking-quality regression** is being observed in current outputs.
2. **The runtime environment has drifted**, so code-level conclusions are not yet trustworthy.

The most important current facts are now:

- The earlier ownership/ACL problem appears to be repaired and is no longer the main blocker.
- The current code already contains some important revert/fix work:
  - Pass 1 detection size is back to `min(1024, erp_w // 4)`.
  - Pass 1 direction uses the union-box path again.
  - Pass 2 prompt-frame selection uses `detection_counts`, not empty mask area.
- The masking runtime has now been restored to the known-good CUDA + SAM2 baseline:
  - `torch 2.11.0+cu128`
  - `torchvision 0.26.0+cu128`
  - `torch.version.cuda == 12.8`
  - `torch.cuda.is_available() == True`
  - `sam2.build_sam` importable
- A masking-only rerun from the restored environment returned to the prior good mask quality.
- A subsequent full plugin test run also produced good ERP masks and good pinhole masks.
- Therefore, environment drift was the primary cause of the apparent masking regression.

So the disciplined next move is:

> keep the restored environment stable and finish locking the SAM2 path into project metadata / lockfiles so future `uv sync` operations preserve the working setup.

---

## Why This Document Exists

The investigation has been spread across multiple reports:

- regression-causality notes
- environment drift notes
- known-good env notes
- prompt-frame bug notes
- backprojection artifact notes

This document is meant to be the single reference that answers:

- what is known for certain
- what is no longer the main blocker
- what code state is live right now
- what environment state is live right now
- what order future work must follow

---

## Confirmed Current Facts

## 1. Ownership / ACL issues were real, but are no longer the primary blocker

The repo, `.venv`, `tmp`, user `uv` cache, and user git config were repaired back to user ownership.

Recent evidence:

- `git status --short` runs without git-config permission failures
- `uv sync --dry-run` runs without `Access is denied`

That does **not** mean ownership can never regress again, but it does mean the current masking investigation should not default to ACLs as the main explanation.

---

## 2. The current code is not the raw pre-investigation state

Several important code paths have already been reverted or corrected relative to the earlier optimization period.

### Pass 1 detection size

Current code:

```python
detection_size = min(1024, erp_w // 4)
```

Observed in:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L820)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1305)

### Pass 1 direction path

The union-box direction logic is back in the codebase.

### Pass 2 prompt frame selection

Current code now selects the prompt frame using `detection_counts`, not `mask.sum()` from empty ERP masks.

Observed in:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1089)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1121)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1160)
- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1176)

### Why this matters

This means the investigation cannot keep assuming:

- “the live code still has the old 512px regression”
- or “the live code still always prompts SAM2 on frame 0”

Those were real issues in the investigation history, but they are **not** the only live explanation now.

---

## 3. SAM2 backend availability depends on package health

The real SAM2 video backend is only enabled if this import succeeds:

```python
from sam2.build_sam import build_sam2_video_predictor_hf
```

Observed in:

- [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py#L254)
- [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py#L256)
- [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py#L472)

If that import fails:

- `HAS_SAM2 = False`
- the runtime falls back to `FallbackVideoBackend`

So environment/package state directly affects the algorithm path being exercised.

---

## 4. The runtime baseline is restored again

Current verified environment state:

- `torch 2.11.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`
- `torchvision 0.26.0+cu128`
- `sam2.build_sam` importable

This matches the known-good masking target.

---

## 5. The repo metadata still targets CUDA torch on Windows

[pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml) currently encodes:

- `torch` from the `pytorch-cu128` index on Windows
- `torchvision` from the `pytorch-cu128` index on Windows

So the current CPU torch state is **not** best explained by “the repo is configured for CPU torch.”

It is best explained by **venv drift from mixed package-management paths**.

---

## Known-Good Target State

This is the masking environment target we should treat as the baseline:

### Core runtime

- Python `3.12.x`
- `torch 2.11.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`
- `torchvision 0.26.0+cu128`

### SAM2 stack

- `sam2 1.1.0`
- `sam2.build_sam` importable
- support packages present:
  - `huggingface-hub 1.9.0`
  - `hydra-core 1.3.2`
  - `iopath 0.1.10`
  - `omegaconf 2.3.0`
  - `portalocker 3.2.0`
  - `pywin32 311`
  - `shellingham 1.5.4`
  - `tqdm 4.67.3`
  - `typer 0.24.1`
  - `httpcore 1.0.9`
  - `httpx 0.28.1`
  - `hf-xet 1.4.3`

### Existing image masking stack

- `ultralytics 8.4.33`
- `segment-anything 1.0`
- `pycolmap 4.0.2`

---

## What Happened

The cleanest current interpretation is:

1. The optimization period likely **triggered or exposed** masking weakness.
2. Several suspect code paths have since been reverted or corrected.
3. The environment drifted during investigation.
4. Once the runtime was restored to the CUDA + SAM2 baseline, both masking-only and full-plugin tests returned to the previous good quality.

So “reverting the code did not restore the masks” was misleading while the environment was still drifted.

The strongest current conclusion is:

> environment drift was the primary cause of the apparent regression.

---

## What Has Been Ruled Out, And What Has Not

## Ruled out as the *sole* explanation

These are not enough by themselves to explain the current situation:

- ACL/ownership problems
- the old 512px Pass 1 detection-size change
- the old single-best-box direction change
- the old prompt-frame-selection-by-empty-mask bug

Why:

- ACLs appear repaired
- the reverted/fixed code paths are already back in place
- masks are still bad
- the environment is still drifted

## Still worth keeping in mind

- SAM2 vs fallback backend mismatch during some earlier test runs
- remaining backprojection weakness
- remaining geometry/tracking weakness even after the prompt-frame fix
- `_C.pyd` influence on the result

These are no longer the primary explanation for the previously observed quality drop, but they may still matter in future quality work.

---

## Frozen Workflow Rules

Now that the baseline is restored:

1. Do **not** change masking code.
2. Do **not** casually run `uv sync`.
3. Do **not** casually run `uv pip install`.
4. Do **not** mix runtime repair with algorithm debugging.
5. Treat environment validation as a prerequisite to any new mask-quality conclusion.

---

## Recovery And Investigation Plan

## Phase 1 — Freeze

Goal:

- stop introducing new variables

Rules:

- no masking code edits
- no venv mutation
- no new runtime experiments

Status:

- this phase is active now

---

## Phase 2 — Restore the minimal drifted runtime pieces

Goal:

- restore only the packages that are confirmed drifted

Minimal restore target:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`

Important constraint:

- do **not** broadly reinstall unrelated packages if they already match the target state

Planned narrow restore command:

```powershell
$env:UV_CACHE_DIR='C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\tmp\uv-cache-runtime'
& 'C:\Users\alexm\LichtFeld-Studio\build\bin\uv.exe' pip install `
  --python '.\.venv\Scripts\python.exe' `
  --reinstall `
  --torch-backend cu128 `
  torch==2.11.0 torchvision==0.26.0
```

Status:

- completed successfully

---

## Phase 3 — Validate the runtime only

Goal:

- confirm the environment matches the known-good target before any masking rerun

Validation command:

```powershell
& '.\.venv\Scripts\python.exe' -c "import torch, torchvision, importlib.util as u; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('sam2_build_sam', u.find_spec('sam2.build_sam') is not None)"
```

Validated result:

- `torch 2.11.0+cu128`
- `torch_cuda 12.8`
- `cuda_available True`
- `torchvision 0.26.0+cu128`
- `sam2_build_sam True`

Optional deeper check:

```powershell
& '.\.venv\Scripts\python.exe' -c "from sam2.build_sam import build_sam2_video_predictor_hf; print('sam2 ok')"
```

---

## Phase 4 — Run one controlled Default masking test

Goal:

- generate one trustworthy comparison run from the restored environment

Requirements:

- same clip family
- one new output directory
- no code changes between validation and test
- capture console output and resulting masks

Outcome:

- masking-only rerun returned to prior good quality
- full plugin rerun also returned good ERP and pinhole masks

---

## Phase 5 — Finish environment lock-in so the baseline stays reproducible

Current next step:

1. lock the SAM2 path into project metadata
2. keep the installer on a locked sync path for video tracking
3. avoid future environment drift during masking work

---

## Decision Gates

This plan is intended to separate approval moments cleanly.

### Gate 1 — Env-only restore

Status: passed

### Gate 2 — Env validation only

Status: passed

### Gate 3 — One masking rerun

Status: passed

### Gate 4 — Lock-in work

Status: active

---

## Bottom Line

The apparent Default masking regression is now much better understood.

What is clear is:

- ownership is no longer the main blocker
- some earlier code suspects have already been reverted/fixed
- the runtime was restored to the known-good CUDA + SAM2 baseline
- masking-only and full-plugin reruns returned to the prior good quality
- the primary cause of the apparent regression was environment drift
- the remaining work is to lock SAM2 into the project-managed environment

So the next disciplined move is not more masking-code thrash.

It is:

> finish the SAM2 lock-in work so the current good baseline becomes reproducible from project metadata and lockfiles.

---

## Related Documents

- [2026-04-04-masking-regression-investigation-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-regression-investigation-plan.md)
- [2026-04-04-masking-regression-investigation-plan-addendum.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-regression-investigation-plan-addendum.md)
- [2026-04-04-masking-env-drift-report-and-recovery-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-env-drift-report-and-recovery-plan.md)
- [2026-04-04-known-good-masking-env-state.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-known-good-masking-env-state.md)
- [2026-04-04-default-masking-regression-causality-report.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-default-masking-regression-causality-report.md)
