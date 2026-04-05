# Marketplace Name And Env Regression Report

Date: 2026-04-04

## Executive Summary

Two separate issues became entangled:

1. Marketplace naming was edited through `pyproject.toml`, but that is not the correct mechanism for the current GitHub-backed marketplace path.
2. A later `uv sync` repaired the environment structurally but synced it to the current project dependency state, which restored CPU-only `torch` and removed the real `sam2` package.

The plugin did not "randomly" break because of a display-name edit by itself. The environment broke because `uv sync` reconciled `.venv` to a project state that does not currently persist a Windows CUDA PyTorch selection.

## What Was Verified

### 1. Local plugin metadata uses `project.name`

LichtFeld's local plugin manager parses `pyproject.toml` and requires:

- `project.name`
- `project.version`
- `project.description`

Relevant local host code:

- [manager.py](/C:/Users/alexm/LichtFeld-Studio/build/src/python/lfs_plugins/manager.py#L238)
- [installer.py](/C:/Users/alexm/LichtFeld-Studio/build/src/python/lfs_plugins/installer.py#L763)

`installer.py` also uses `project.name` as the final installed plugin directory name.

### 2. GitHub marketplace entries do not use this repo's `pyproject.toml` name

For GitHub-backed marketplace entries, LichtFeld resolves metadata from the GitHub repo API:

- [marketplace.py](/C:/Users/alexm/LichtFeld-Studio/build/src/python/lfs_plugins/marketplace.py#L207)

That code uses:

- the GitHub repo name
- the GitHub repo description

not the local plugin's `pyproject.toml`.

### 3. Registry entries do support a separate display name

The registry model supports:

- `name`
- `display_name`

Relevant local host code:

- [registry.py](/C:/Users/alexm/LichtFeld-Studio/build/src/python/lfs_plugins/registry.py#L100)

So the desired model is valid:

- package/plugin slug: `lichtfeld-360-plugin`
- marketplace display label: `360 Plugin`

But that display label needs to come from the registry path, not from `pyproject.toml`.

### 4. The current `uv sync` result is CPU-only

Direct verification in the plugin venv showed:

```text
torch 2.11.0+cpu
torch cuda version None
cuda available False
torchvision 0.26.0+cpu
pycolmap 4.0.2
```

That means the current environment is not in a healthy GPU state for the plugin's masking stack.

### 5. `sam2` is not actually installed anymore

This needed a careful distinction:

- `importlib.util.find_spec("sam2")` returns a result
- but `sam2` resolves only as a namespace package
- `sam2.__file__` is `None`
- the only files left in `site-packages/sam2` are `_C.pyd` and its backup
- importing `sam2.build_sam` now fails with `ModuleNotFoundError`

Current observed contents of `site-packages/sam2`:

- `_C.pyd`
- `_C.pyd.bak_20260404_050709`

So the earlier "`sam2 installed: True`" result was misleading. The package directory still exists, but the actual Python package contents are gone.

### 6. The lock state explains the CPU fallback

Current project metadata still declares generic PyTorch dependencies:

- [pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml)

```toml
"torch>=2.11.0"
"torchvision>=0.26.0"
```

Current lock state resolves them from plain PyPI:

- [uv.lock](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/uv.lock#L1060)
- [uv.lock](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/uv.lock#L1087)

Observed lock details:

- `torch 2.11.0`
- `torchvision 0.26.0`
- source registry: `https://pypi.org/simple`

There is no persistent project-level CUDA torch selection encoded in the current manifest/lock pair.

### 7. ACL issues were real, but they were not the CUDA root cause

Earlier, `.venv` had ownership/ACL inconsistencies caused by mixed principals touching files in the workspace. That blocked `uv` operations. Once ownership was repaired, `uv sync` could run again.

That ACL problem explains the earlier access-denied behavior.

It does **not** explain why `torch` became CPU-only.

The CPU regression came from sync resolution, not from file ownership.

## What Most Likely Happened

### Step 1. Marketplace-facing text was edited

This likely happened in or around `pyproject.toml`, because that is the most obvious local metadata file.

### Step 2. That edit did not actually target the correct marketplace name path

For the current GitHub-backed path, the marketplace name comes from GitHub repo metadata, not from `pyproject.toml`.

So the name edit was the wrong lever for the stated goal.

### Step 3. The environment then had to be reconciled

Because of earlier ACL issues and subsequent troubleshooting, `uv sync` was run.

### Step 4. `uv sync` restored the environment to the current project state

That project state currently resolves:

- CPU-only `torch`
- CPU-only `torchvision`
- no actual `sam2` package install

So the env went from "working but drifted" to "consistent but wrong."

## Why This Felt Sudden

From the user perspective, this looked like:

- edit a plugin name/description
- then everything falls apart

But the actual chain was:

- metadata confusion
- ACL repair
- env sync
- CPU-only resolution
- stale `sam2` namespace leftovers making the break harder to see

That is why the failure felt disconnected from the action that triggered the cleanup.

## Current Safe Naming Model

The current safe choice is:

- `project.name = "lichtfeld-360-plugin"`
- intended public display name = `360 Plugin`

That slug has already been restored in:

- [pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml)

This keeps the local plugin/package identity packaging-safe and directory-safe.

## What Should Be Done Next

### Immediate recovery

Restore the venv to a working GPU state by reinstalling:

- CUDA-backed `torch`
- CUDA-backed `torchvision`
- the actual `sam2` package

This was not completed in this pass because the install command required approval and was not run.

### Naming / marketplace path

Treat `360 Plugin` as a registry display name, not as the package slug.

### Process guardrail

Do not run `uv sync` as part of marketplace text edits unless the project's torch backend selection is already encoded durably.

## Practical Conclusions

1. The package/plugin slug should remain `lichtfeld-360-plugin`.
2. The public display name should be `360 Plugin`.
3. The current GitHub marketplace path cannot provide that exact display label from `pyproject.toml`.
4. The current env is genuinely broken for GPU use right now.
5. The current `sam2` state is not healthy; only the copied `_C.pyd` artifact remains.
6. The next repair step is environment restoration, not more metadata editing.
