# Known-Good Masking Environment State

**Date:** 2026-04-04  
**Status:** Reference snapshot  
**Purpose:** Freeze one known-good masking package state and define one restore path.

---

## Why This Exists

The plugin venv has drifted because multiple mechanisms have been changing it:

- `uv sync`
- `uv pip install`
- plugin/runtime setup helpers
- ad hoc recovery attempts

This document defines one masking-focused environment state that we can treat as the reference point.

---

## Last Known-Good Masking State

This is the last explicitly verified healthy masking state from this session family:

### Core runtime

- Python: `3.12.x`
- torch: `2.11.0+cu128`
- torch CUDA: `12.8`
- `torch.cuda.is_available() == True`
- torchvision: `0.26.0+cu128`

### SAM2 stack

- sam2: `1.1.0`
- huggingface-hub: `1.9.0`
- hydra-core: `1.3.2`
- iopath: `0.1.10`
- omegaconf: `2.3.0`
- portalocker: `3.2.0`
- pywin32: `311`
- shellingham: `1.5.4`
- tqdm: `4.67.3`
- typer: `0.24.1`
- httpcore: `1.0.9`
- httpx: `0.28.1`
- hf-xet: `1.4.3`

### Image masking stack

- ultralytics: `8.4.33`
- segment-anything: `1.0`

### Supporting geometry/runtime pieces

- pycolmap: `4.0.2`
- opencv-python: `4.13.0.92`

---

## What Is Broken Right Now

At the moment this reference was written, the venv had drifted to:

- torch: `2.11.0+cpu`
- torchvision: `0.26.0+cpu`
- `torch.version.cuda == None`
- `torch.cuda.is_available() == False`

Important nuance:

- the SAM2 package itself is currently installed again
- the main drift is that torch/torchvision are on CPU builds instead of the known-good CUDA builds

So the minimal restore should target **torch** and **torchvision**, not re-randomize the whole environment.

---

## Exact Restore Target

The goal is to restore only the drifted packages to:

- torch `2.11.0+cu128`
- torchvision `0.26.0+cu128`

while keeping the currently restored SAM2 stack in place.

---

## Single Approved Recovery Method

Use one explicit restore command, not a mixed workflow:

```powershell
$env:UV_CACHE_DIR='C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\tmp\uv-cache-runtime'
& 'C:\Users\alexm\LichtFeld-Studio\build\bin\uv.exe' pip install `
  --python '.\.venv\Scripts\python.exe' `
  --reinstall `
  --torch-backend cu128 `
  torch==2.11.0 torchvision==0.26.0
```

This is intentionally narrow:

- restore only the packages that drifted
- force the CUDA backend explicitly
- avoid a broad reinstall of unrelated packages

---

## Validation After Restore

```powershell
& '.\.venv\Scripts\python.exe' -c "import torch, torchvision, importlib.util as u; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('sam2_build_sam', u.find_spec('sam2.build_sam') is not None)"
```

Expected:

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

## Workflow Rule Going Forward

For masking debugging, do **not** mix these casually:

- `uv sync`
- `uv pip install`
- runtime/plugin auto-install flows

Instead:

1. Treat this document as the masking env reference.
2. If the env drifts, restore it with the exact narrow command above.
3. Do not run plain `uv sync` during masking debugging unless the lockfile has been intentionally updated to encode this exact state.

---

## Practical Summary

The working rule is:

> one known-good masking env, one narrow restore path, no casual sync/install mixing while debugging mask quality.
