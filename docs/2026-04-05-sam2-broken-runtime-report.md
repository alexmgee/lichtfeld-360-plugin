# SAM2 Broken Runtime Report

**Date:** 2026-04-05  
**Status:** Observed in the live plugin `.venv` before repair  
**Purpose:** Record the exact broken `sam2` package state that was discovered after the earlier stabilization work, so the repair and follow-up verification are tied to concrete evidence.

---

## Executive Summary

The live plugin environment was found in a **half-installed / half-left-behind**
`sam2` state.

The important result is:

- `import sam2` succeeded only as a namespace package
- `sam2.build_sam` was missing
- there was no installed `sam2` distribution metadata
- the `site-packages/sam2` directory contained only copied extension artifacts

That means the environment could present the illusion that "`sam2` exists"
while still missing the actual Python modules required for the real
`Sam2VideoBackend`.

This is not a valid or trustworthy SAM2 install state.

---

## Observed Evidence

### Import state

Observed in the live plugin `.venv`:

- `sam2.__file__ == None`
- `sam2.__path__ == ['...\\.venv\\Lib\\site-packages\\sam2']`
- `importlib.util.find_spec('sam2.build_sam') == None`
- `importlib.util.find_spec('sam2._C')` resolved successfully

Interpretation:

- Python saw a `sam2` directory and treated it as a namespace package
- the compiled `_C.pyd` extension was present
- the actual Python modules, including `build_sam.py`, were not present

### Package metadata state

Observed:

- `importlib.metadata.version('sam2')` raised `PackageNotFoundError`

Interpretation:

- the environment did not contain a valid installed `sam2` distribution

### On-disk contents of `site-packages/sam2`

Observed files in the live environment:

- `_C.pyd`
- `_C.pyd.bak_20260404_050709`
- `_C.pyd.disabled`

Observed missing content:

- no `build_sam.py`
- no `sam2_video_predictor.py`
- no regular package Python modules
- no `sam2-*.dist-info` directory

Interpretation:

- the `sam2` directory survived only as an artifact container, not as a real
  package install

---

## Why This Can Happen

The most likely sequence is:

1. A real `sam2` package install existed at some earlier point.
2. Later package operations removed the real package files or replaced the env.
3. The copied `_C.pyd` artifact and its backup/disabled variants remained in
   `site-packages/sam2`.
4. Because that directory still existed, Python imported `sam2` as a namespace
   package.
5. But because the real Python modules were gone, imports like
   `sam2.build_sam` failed.

This is exactly the kind of broken state that can happen when:

- package manager operations change the venv
- custom runtime code copies files into `site-packages`
- uninstall/reinstall behavior only removes files it knows it owns

The copied `_C.pyd` files can keep a package directory alive even after the
real package install is gone.

---

## Why This Matters For The Plugin

The plugin's real SAM2 video backend is enabled by:

```python
from sam2.build_sam import build_sam2_video_predictor_hf
```

If `sam2.build_sam` is missing, then:

- `HAS_SAM2` is false
- the plugin cannot use the real `Sam2VideoBackend`
- the runtime falls back to the image-only `FallbackVideoBackend`

So this broken package state is not cosmetic.

It directly changes the algorithm path the plugin can use.

---

## Immediate Next Step

The environment should be repaired by:

1. moving the broken `site-packages/sam2` directory out of the way
2. reinstalling `sam2` cleanly from the locked `video-tracking` environment
3. reapplying `_C.pyd` only after the real package is back
4. verifying:
   - `sam2.build_sam` importable
   - `sam2` distribution metadata present
   - the plugin is once again capable of real `Sam2VideoBackend` use
