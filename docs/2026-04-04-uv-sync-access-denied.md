# uv sync Failure — Access Denied on .venv Pycache

**Date:** 2026-04-04
**Status:** Blocking — plugin cannot load
**Severity:** Critical

---

## Symptom

Plugin fails to load in LichtFeld Studio with:

```
Load failed: uv sync failed: Resolved 87 packages in 940ms error: failed to remove directory
'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv\Lib\site-packages\h11\__pycache__'
Access is denied. (os error 5)
```

The plugin was loading and running correctly immediately before this error appeared. LichtFeld Studio has been restarted multiple times — the error persists.

## Trigger

The error began after editing `pyproject.toml` (description text and name field). Any change to `pyproject.toml` causes LichtFeld to re-run `uv sync` on the next plugin load. The sync is failing because it cannot remove a `__pycache__` directory inside the venv's `site-packages`.

The `pyproject.toml` content changes themselves are not the cause. The issue is that `uv sync` cannot modify the venv.

## What Was Verified

- The venv is functional. `torchvision` imports correctly from the venv Python (`0.26.0+cu128`).
- Running `uv sync` manually from the command line with LichtFeld closed reproduces the same error.
- The locked directory is `.venv/Lib/site-packages/h11/__pycache__/` — `ls` returns "Permission denied".
- `uv sync` is attempting to reinstall `torch` and `torchvision` (downloading both), then fails when trying to remove `h11/__pycache__`.

## What Has Not Been Verified

- Whether the `h11/__pycache__` directory has corrupted ACLs or is owned by a different user/process.
- Whether another process is holding a lock on files in that directory.
- Whether the previous failed `uv sync` attempt (when `name = "360 Plugin"` was rejected as invalid) left the venv in a partially modified state that caused the permission issue.

## Likely Root Cause

The first `uv sync` failure (caused by the invalid package name `"360 Plugin"`) may have left the venv in a half-modified state. `uv sync` was partway through removing/reinstalling packages when it hit the name validation error. On subsequent attempts, the partially-modified directories now have broken permissions or locks.

## Possible Fixes (Not Yet Attempted)

1. **Reset ACLs on the venv** — `icacls .venv /reset /T /Q` to recursively reset all permissions to parent defaults.

2. **Delete and recreate the pycache** — Remove the specific locked directory, then re-run `uv sync`:
   ```
   rmdir /s /q ".venv\Lib\site-packages\h11\__pycache__"
   uv sync
   ```

3. **Nuke and rebuild the venv** — Delete `.venv` entirely, then `uv sync` to recreate from scratch. This is the nuclear option but guaranteed to work. All installed packages (torch, sam2, ultralytics, etc.) will be re-downloaded.
   ```
   rmdir /s /q .venv
   uv sync
   ```

4. **Take ownership first** — If the directory is owned by SYSTEM or another user:
   ```
   takeown /f .venv /r /d y
   icacls .venv /reset /T /Q
   uv sync
   ```

## Context

This happened during a session where `pyproject.toml` was edited multiple times:
- `name` changed from `"panosplat"` to `"360 Plugin"` (invalid — caused first uv sync failure)
- `name` changed to `"360-plugin"` (loaded but displayed wrong)
- `name` changed to `"360_Plugin"`
- `name` reverted to `"panosplat"`
- `description` changed twice

The first invalid name change is the most likely culprit for corrupting the venv state. All subsequent `uv sync` attempts have failed with the same access denied error regardless of what the pyproject.toml content actually says.
