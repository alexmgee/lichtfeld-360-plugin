# SAM v2 _C Extension — Bundling & Auto-Install

**Date:** 2026-04-04
**Supersedes:** Build-only workflow from `2026-04-04-sam2-c-extension-build-report.md`
**Status:** Implemented for current Windows matrix; runtime validation ongoing
**Revised:** Incorporates feedback from `2026-04-04-sam2-c-extension-bundling-response.md`

---

## Problem

The `sam2` PyPI package (v1.1.0) ships as a source tarball. The CUDA extension (`sam2._C`) that provides `get_connected_componnets()` for mask hole-filling is never compiled during `uv add sam2`. Without it, every SAM v2 propagation call emits:

```
UserWarning: cannot import name '_C' from 'sam2'
Skipping the post-processing step due to the error above.
```

SAM v2 still works — it just skips `fill_holes_in_mask_scores()`, which means small holes in tracked masks are not cleaned up. The previous session built `_C.pyd` successfully using a PowerShell build script (`dev/build_sam2_c_extension.ps1`), but the resulting file sat in `.venv/Lib/site-packages/sam2/` with no story for how other users would get it.

## Decision

Bundle a pre-built `_C.pyd` for the currently validated Windows support matrix and auto-install it into the installed `sam2` package. Treat runtime import verification as part of the install story, not an afterthought.

Note: `_C.pyd` is a CPython extension module tied to a specific Python ABI, torch binary layout, and CUDA runtime. It is **not** a generic dependency DLL like `lib/python3.dll`. Bundling it is closer to shipping a tightly versioned plugin-native binary.

## What Was Done

### 1. Bundled `_C.pyd` in `lib/`

Copied the built extension from `.venv/Lib/site-packages/sam2/_C.pyd` (449 KB) into the plugin's `lib/` directory:

```
lib/
├── python3.dll    (68 KB — pycolmap DLL dependency)
└── _C.pyd         (449 KB — sam2 connected-components extension)
```

### 2. Added install and verification helpers in `core/setup_checks.py`

**`_try_import_sam2_c()`** — Imports `torch` first (seeds Windows DLL search paths for CUDA runtime), then attempts `from sam2 import _C`. Returns True/False. This is the single source of truth for "does `_C` actually load."

**`_normalize_acl(path)`** — After copying `_C.pyd`, runs `icacls /reset` on the destination to clear inherited ACEs and reapply parent directory defaults. Addresses the `DLL load failed: Access is denied` error observed when copied files inherit restrictive permissions.

**`_install_sam2_c_extension(on_output)`** — The main install function:
1. Locates `lib/_C.pyd` and the installed `sam2` package
2. Compares file sizes to skip redundant copies (provisional heuristic)
3. Copies the file and normalizes ACLs
4. **Verifies the import actually succeeds** — file presence alone is not enough
5. If verification fails on an existing file, attempts ACL repair before giving up
6. Logs a clear warning on permanent failure (not just the upstream `_C` warning)

**`ensure_sam2_c_extension()`** — Runtime safety net called from `Sam2VideoBackend.initialize()`. Calls `_try_import_sam2_c()` first; only runs the install path if the import fails.

### 3. Install-time path: `install_video_tracking()`

After `uv add sam2` succeeds, `install_video_tracking()` calls `_install_sam2_c_extension()`. Users who install SAM v2 through the plugin UI get the extension automatically.

### 4. Runtime safety net: `Sam2VideoBackend.initialize()`

Calls `ensure_sam2_c_extension()` before loading the SAM v2 model. This is a best-effort repair path that covers:
- Users who installed `sam2` manually outside the plugin
- Cases where a `sam2` upgrade overwrote the extension
- Any scenario where the install-time path was bypassed

It does **not** guarantee success in all runtime environments — the LichtFeld application loader may impose constraints beyond what the plugin can control.

## Files Changed

| File | Change |
|------|--------|
| `lib/_C.pyd` | New — bundled pre-built extension |
| `core/setup_checks.py` | Added `_try_import_sam2_c()`, `_normalize_acl()`, `_install_sam2_c_extension()`, `ensure_sam2_c_extension()`; wired into `install_video_tracking()` |
| `core/backends.py` | `Sam2VideoBackend.initialize()` calls `ensure_sam2_c_extension()` before model load |

## Build Provenance

The bundled `_C.pyd` was compiled using `dev/build_sam2_c_extension.ps1` with:
- Source: `github.com/facebookresearch/sam2` HEAD (cloned 2026-04-04, no tagged release used)
- CUDA file: `sam2/csrc/connected_components.cu` (with a Windows-specific source workaround applied by the build script)
- Toolchain: MSVC (VS 2025 Community, **v14.44 toolset** — v14.29 was attempted first but missing `msvcprt.lib` for x64), CUDA 12.9
- torch: 2.11.0+cu128
- Target: Python 3.12 (cp312-win_amd64)
- Import validation: confirmed successful in plugin venv with `torch` imported first

The artifact is currently validated **only for this exact matrix**. If the plugin's torch, Python, or CUDA version changes, the extension must be rebuilt using the build script and checklist (`docs/2026-04-04-sam2-c-extension-windows-build-checklist.md`).

## Current Limitations

- **Windows only.** Linux would need a separate `.so` build — not currently needed since the plugin targets Windows.
- **Narrow support matrix.** Validated only for Python 3.12 + torch 2.11.0+cu128 + CUDA 12.9.
- **Runtime load not yet fully proven in LichtFeld.** The extension loads in the plugin venv from a shell, but the LichtFeld application runtime previously showed `DLL load failed: Access is denied` before ACL repair. The `_normalize_acl()` fix addresses this, but has not yet been confirmed inside a full LichtFeld masking run.
- **File-size skip heuristic is provisional.** A rebuilt binary with changed content but identical size would not be overwritten. Acceptable for now since rebuilds only happen when the toolchain changes, but a hash or post-copy import check would be stronger. (The current code does verify import after copy, which partially compensates.)
- **CUDA minor version mismatch.** Built with CUDA 12.9, torch expects 12.8. Has not caused issues — the extension only uses basic CUDA primitives (2D connected components on a grid).
