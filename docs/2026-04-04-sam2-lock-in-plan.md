# SAM2 Lock-In Plan

**Date:** 2026-04-04  
**Status:** Metadata/installer path updated; verification in progress  
**Scope:** Make the currently working SAM2 masking environment reproducible from project metadata and lockfiles  
**Non-goal:** No code changes or environment mutations are performed by this document

---

## Executive Summary

The current masking environment is now working again:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`
- `sam2.build_sam` importable

The project has now been updated to model SAM2 as an explicit optional feature, and the installer path has been redirected away from `uv add sam2`.

However, the environment is **not yet perfectly no-op against the lockfile** because the current live venv still contains some manually drifted package versions and dev packages.

The original proof of the gap was the dry-run:

```powershell
uv sync --locked --dry-run
```

It did **not** try to replace CUDA `torch` / `torchvision`, which was good.

But it **did** try to remove the entire SAM2/video-tracking stack:

- `sam2`
- `huggingface-hub`
- `hydra-core`
- `iopath`
- `omegaconf`
- `portalocker`
- `pywin32`
- `tqdm`
- `typer`
- `shellingham`
- `httpcore`
- `httpx`
- `hf-xet`

That meant the project had:

- **CUDA lock-in:** mostly working
- **SAM2 lock-in:** not working yet

After the metadata and installer changes in this document:

- `uv lock` now includes the `video-tracking` extra
- `uv sync --locked --no-dev --extra video-tracking --dry-run` no longer proposes removing:
  - `sam2`
  - `huggingface-hub`
  - `hydra-core`
  - `iopath`
  - `omegaconf`
  - `portalocker`
  - `pywin32`
  - `tqdm`
  - `typer`
  - `shellingham`
  - `httpcore`
  - `httpx`
  - `hf-xet`

The remaining dry-run churn is now limited to:

- dev packages excluded by `--no-dev`
- version normalization of some packages that were manually installed during recovery
- benign naming normalization on the direct `pycolmap` wheel

So the main SAM2 lock-in gap has been closed at the project-definition level.

---

## Problem Statement

Right now the project uses two different environment-management models:

### Base/default install path

[core/setup_checks.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/setup_checks.py) uses:

```text
uv sync --locked --no-dev
```

This is good because it is lock-backed and reproducible.

### Video-tracking install path

The same file currently uses:

```text
uv add sam2
```

for the video-tracking step.

That was the root mismatch.

`uv add sam2` mutates the environment after the locked base install, which means:

- the environment can work
- but `uv sync --locked` will not preserve that state unless the project metadata/lockfile explicitly model it

That is exactly what the dry-run proved.

---

## What “Locked In” Must Mean

The project should satisfy all of these at once:

1. A clean sync path can reproduce the working masking environment.
2. `uv sync --locked` does not silently strip SAM2 after it has been intentionally installed.
3. Default plugin users do not have to install SAM2 unless they choose the video-tracking feature.
4. The plugin setup flow does not rely on ad hoc `uv add` to establish a critical runtime feature.
5. The `_C.pyd` install step remains a post-install binary step, but the Python package state for SAM2 must already be present and locked before that step runs.

---

## Recommendation

Model SAM2 as an **explicit optional project feature** and make the installer request that feature through a locked sync path.

In practical terms:

- keep the base dependencies as the default plugin environment
- add a dedicated optional dependency set for video tracking
- lock that feature into `uv.lock`
- make `install_video_tracking()` use a locked sync for that feature, not `uv add sam2`

This keeps:

- base installs lean
- video tracking intentional
- the final environment reproducible

---

## Recommended Project Shape

## 1. Keep base dependencies focused on the core plugin

The base dependency list should remain the default plugin runtime:

- OpenCV
- NumPy
- pycolmap
- ffmpeg helper
- YOLO/SAM image masking stack
- Windows CUDA torch / torchvision mapping

This is the environment required for the plugin to function without SAM2 video tracking.

## 2. Add a dedicated video-tracking optional dependency set

The SAM2 stack should be modeled explicitly in project metadata, for example as:

```toml
[project.optional-dependencies]
video-tracking = [
  "sam2==1.1.0",
]
```

Important note:

- the transitive packages like `huggingface-hub`, `hydra-core`, `iopath`, `omegaconf`, `portalocker`, and the rest should normally resolve through `sam2`
- they do **not** need to be manually duplicated unless resolution proves that one or more of them must be pinned directly

The goal is to make the **feature** explicit, not to hand-copy every transitive package unless needed.

## 3. Lock the optional feature into `uv.lock`

After the optional feature exists in project metadata, the lockfile must be regenerated so that:

- base env is locked
- Windows CUDA torch remains locked
- `video-tracking` is also locked as a reproducible extra

After that, the important dry-run should become:

```powershell
uv sync --locked --extra video-tracking --dry-run
```

Expected result:

- no uninstall of `sam2`
- no uninstall of the transitive SAM2 stack
- no downgrade of CUDA torch

---

## Installer Behavior Plan

## Current behavior

`install_video_tracking()` currently behaves like:

1. mutate env with `uv add sam2`
2. copy/install `_C.pyd`

## Planned behavior

`install_video_tracking()` should instead behave like:

1. locked sync of the optional video-tracking feature
2. validate `sam2.build_sam` importability
3. install/copy `_C.pyd`
4. validate `sam2._C` importability in the normal runtime order

Conceptually:

```text
uv sync --locked --no-dev --extra video-tracking
```

followed by the existing `_C.pyd` install/verification logic.

This is better because:

- the Python package state becomes lock-backed
- the binary `_C.pyd` step becomes a narrow post-install patch, not a substitute for package installation

---

## Why Optional Feature Is Better Than Making SAM2 Base

Making `sam2` a base dependency would solve the reproducibility problem, but it would also:

- force all users into the heavier video-tracking dependency path
- increase install time and install complexity
- make the default plugin environment larger than necessary

That is not ideal because video tracking is still a feature tier, not the minimum plugin requirement.

So the cleaner design is:

- base env for core plugin behavior
- optional locked video-tracking feature for SAM2

---

## Current Verification Result

The important verification step now behaves the way we wanted:

```powershell
uv sync --locked --no-dev --extra video-tracking --dry-run
```

Current interpretation:

- success: it no longer strips the SAM2 runtime stack
- expected remaining churn: dev packages and manual version normalization

That means the project is now modeling the video-tracking feature correctly enough for future locked installs.

---

## Exact Verification Criteria For Success

The SAM2 lock-in plan is complete only when all of these are true.

## Environment creation / sync

A clean or existing plugin environment can run:

```powershell
uv sync --locked --no-dev --extra video-tracking
```

without:

- uninstalling `sam2`
- uninstalling the SAM2 transitive stack
- downgrading CUDA torch on Windows

## Validation

After that sync:

```powershell
python -c "import torch, torchvision, importlib.util as u; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torchvision.__version__); print(u.find_spec('sam2.build_sam') is not None)"
```

must show:

- `torch 2.11.0+cu128`
- CUDA `12.8`
- `True`
- `torchvision 0.26.0+cu128`
- `True`

## Dry-run reproducibility

Then:

```powershell
uv sync --locked --no-dev --extra video-tracking --dry-run
```

should be effectively a no-op for:

- `sam2`
- its support packages
- `torch`
- `torchvision`

## Runtime

The plugin should then be able to:

- select `Sam2VideoBackend`
- import `sam2.build_sam`
- load `_C.pyd` successfully after the existing install step

---

## Proposed Implementation Sequence

This is the order the actual implementation should follow later.

### Step 1

Add a `video-tracking` optional dependency section to [pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml) with `sam2==1.1.0`.

### Step 2

Regenerate [uv.lock](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/uv.lock) so the optional feature is lock-backed.

### Step 3

Dry-run:

```powershell
uv sync --locked --no-dev --extra video-tracking --dry-run
```

Confirm it no longer strips the SAM2 stack.

### Step 4

Update [core/setup_checks.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/setup_checks.py) so `install_video_tracking()` uses the locked optional-feature sync instead of `uv add sam2`.

### Step 5

Keep the existing `_C.pyd` install step, but treat it as a post-install binary patch only.

### Step 6

Validate the full runtime path once from a clean or intentionally reset env.

---

## What Not To Do

To avoid repeating the same drift:

- do not continue relying on `uv add sam2` as the main installer path
- do not use casual `uv pip install` commands as the normal way to maintain the masking environment
- do not treat a currently working venv as “locked in” unless the corresponding `uv sync --locked ... --dry-run` also agrees

---

## Practical Bottom Line

Right now you have:

- a working live env
- a backup of that working env

But you do **not** yet have:

- a project definition that can reproduce the full working SAM2 stack on demand

The exact gap is now known.

The fix is:

> model SAM2 as a locked optional feature and make the installer use that locked feature path instead of `uv add sam2`.
