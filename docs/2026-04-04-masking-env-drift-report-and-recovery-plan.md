# Masking Environment Drift — Report And Recovery Plan

**Date:** 2026-04-04  
**Status:** Active investigation / planning document  
**Scope:** Masking environment drift, not masking algorithm changes  
**Purpose:** Document how the plugin venv drifted, what is known for certain, and the exact recovery sequence to use before further masking debugging.

---

## Executive Summary

The masking investigation is currently confounded by **environment drift**.

The important current facts are:

1. The earlier ownership/ACL problem appears to be repaired and is no longer the main blocker.
2. The plugin venv has been mutated by multiple mechanisms during this session:
   - `uv sync`
   - `uv pip install`
   - plugin setup/runtime helper behavior
   - manual `_C.pyd` experiments
3. The venv briefly lost the normal `sam2` package state entirely.
4. The `sam2` Python package has now been restored, but `torch` / `torchvision` are currently on **CPU** builds instead of the previously known-good **CUDA** builds.
5. That means current masking results are not trustworthy as a comparison against the earlier known-good session.

So the next step should **not** be more masking-code thrash.

The next step should be:

> restore the environment to one explicit known-good masking state, validate only that state, then rerun one controlled masking test.

---

## Why This Matters

Right now there are two different questions mixed together:

1. **Did masking code regress?**
2. **Is the environment still the same environment that produced the earlier good masks?**

At the moment, question 2 is unresolved enough that it can invalidate question 1.

If the runtime package state changed underneath the plugin, then:

- reverted code may still behave differently
- SAM2-specific fixes may appear not to matter
- masking quality can shift even if the code is nearly identical

---

## Confirmed Findings

## 1. Ownership/ACL repair likely succeeded

The earlier ACL/ownership issue was real and caused:

- `uv` cache access failures
- `.venv` access problems
- mixed-owner files

That repair appears to have succeeded well enough that it is no longer the main explanation for the current masking failures.

So this document does **not** treat ACLs as the current primary cause.

---

## 2. The project metadata already encodes CUDA torch for Windows

[pyproject.toml](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/pyproject.toml) currently says:

- `torch>=2.11.0`
- `torchvision>=0.26.0`
- Windows source for both should come from the `pytorch-cu128` index

It also defines:

- `[[tool.uv.index]]`
- `name = "pytorch-cu128"`
- `url = "https://download.pytorch.org/whl/cu128"`

The lockfile also contains both variants:

- CPU `torch 2.11.0` / `torchvision 0.26.0`
- Windows CUDA `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128`

So the repo configuration is **trying** to preserve CUDA torch for Windows.

That means the current CPU state is not best explained by:

- "the repo only supports CPU torch"

Instead, it is best explained by:

- environment mutation outside the clean locked sync path

---

## 3. The plugin’s install flows and ad hoc recovery flows do not behave the same way

[core/setup_checks.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/setup_checks.py) currently does two important things:

### `install_default_tier()`

Uses:

```text
uv sync --locked --no-dev
```

This is the clean project-aligned install path.

### `install_video_tracking()`

Uses:

```text
uv add sam2
```

Then installs the bundled `_C.pyd`.

That is a narrower mutation, but it still changes the venv.

### Why this matters

These paths are not the same as:

```text
uv pip install ...
```

`uv pip install` directly mutates the venv by resolving packages immediately, rather than syncing to the project’s locked environment model.

That difference is central to what happened here.

---

## 4. The venv temporarily lost the real `sam2` package state

At one point during investigation, the venv looked like this:

- `sam2` resolved only as a namespace package
- `sam2.__file__` was `None`
- `sam2.build_sam` was missing
- there was no `sam2*.dist-info`
- `site-packages/sam2/` contained only stray `_C.pyd` artifacts

That is not a healthy `sam2` installation.

This matters because the masking backend only enables the real SAM2 path if:

```python
from sam2.build_sam import build_sam2_video_predictor_hf
```

succeeds.

While the venv was in that broken state:

- `Sam2VideoBackend` would not have been selected
- the system would have fallen back to `FallbackVideoBackend`

That means some SAM2-targeted masking fixes would naturally appear to do nothing.

---

## 5. `sam2` is now installed again, but torch is currently wrong

The venv has since been partially repaired.

Current confirmed state:

- `sam2.build_sam` is present again
- `sam2 1.1.0` is installed again

But the current core runtime is now:

- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`
- `torch.version.cuda == None`
- `torch.cuda.is_available() == False`

That is **not** the previously known-good masking runtime.

The last known-good state from this session family was:

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- CUDA available
- `sam2.build_sam` importable

So the environment is still drifted.

---

## 6. The immediate cause of the latest drift is understood

The clearest recent mutation chain was:

1. The venv lost the normal `sam2` package state.
2. A repair install reintroduced `sam2` and its support packages.
3. That same repair also replaced:
   - `torch 2.11.0+cu128` with `torch 2.11.0`
   - `torchvision 0.26.0+cu128` with `torchvision 0.26.0`

The reason is simple:

- the repair used a direct `uv pip install` path
- it did **not** preserve the CUDA torch backend during that install

So the result was:

- `sam2` back
- CUDA torch gone

That explains the current state without invoking any new mystery.

---

## Root Cause Summary

The venv drift is best explained by **mixed package-management paths**, not by one single bug.

### Main contributors

1. `uv sync` can reconcile the env to the locked project state.
2. `uv pip install` can directly mutate the env outside that locked state.
3. Plugin runtime/setup helpers can mutate the env or `site-packages` after install.
4. Manual experiments with `_C.pyd` and package recovery can leave half-repaired states.

### Practical interpretation

The venv has not been managed by one single disciplined path.

It has been managed by:

- sync-style project reconciliation
- direct pip-style mutation
- runtime helper patching
- manual debugging interventions

That combination is why the environment has become hard to reason about.

---

## Why Further Masking Debugging Is Not Yet Trustworthy

Until the environment is restored to a single known-good state:

- code-level masking conclusions are shaky
- “revert didn’t help” is not definitive
- SAM2-vs-fallback comparisons are not clean

In other words:

> current mask outputs cannot be treated as strong evidence about the masking code alone
> because the runtime has not been held constant.

---

## Known-Good Target State

The environment target to restore to is:

### Core runtime

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torch.version.cuda == 12.8`
- `torch.cuda.is_available() == True`

### SAM2 stack

- `sam2 1.1.0`
- `sam2.build_sam` importable
- related support packages present:
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

### Existing image masking stack preserved

- `ultralytics`
- `segment-anything`
- `pycolmap`

---

## Recovery Plan

This is the recommended plan before any more masking code experiments.

## Phase 1 — Freeze

Do not:

- run plain `uv sync`
- run additional `uv pip install` commands casually
- change masking code
- change plugin setup logic

Goal:

- stop adding more variables

## Phase 2 — Restore only the drifted runtime pieces

The current evidence suggests the minimal restore target is:

- `torch`
- `torchvision`

Nothing else should be broadly reinstalled unless validation proves it is still missing.

Goal:

- restore CUDA torch without re-randomizing the rest of the venv

## Phase 3 — Validate environment only

Before rerunning masking, validate only:

- torch version
- torch CUDA version
- CUDA availability
- torchvision version
- `sam2.build_sam` importability

Goal:

- prove the runtime is back to the known-good masking state

## Phase 4 — Run one controlled masking test

Then run:

- one clip
- one output directory
- one reproducible command

No other changes.

Goal:

- test masking quality from a stable environment baseline

## Phase 5 — Resume algorithm debugging only if needed

If masks are still bad after the env is restored:

- return to the masking regression investigation
- now with a trustworthy runtime baseline

Only then should further code-level diagnosis resume.

---

## Workflow Rules Going Forward

To avoid repeating this:

1. Choose one primary environment-management path for masking debugging.
2. Avoid mixing `uv sync` and `uv pip install` unless the reason is explicit.
3. If direct venv mutation is necessary, write down:
   - command
   - expected effect
   - validation step
4. Validate environment state before interpreting masking regressions.
5. Treat `_C.pyd` experiments as separate from core package-state recovery.

---

## Bottom Line

The current masking investigation is blocked less by uncertainty in the masking code than by uncertainty in the runtime environment.

The important concrete facts are:

- ownership issues are no longer the main blocker
- `sam2` was broken and then partially restored
- the current env is still drifted because `torch` / `torchvision` are on CPU
- the repo itself already encodes CUDA torch for Windows
- the drift came from mixed environment mutation paths, especially direct `uv pip install`

So the disciplined next move is:

> restore one known-good CUDA + SAM2 environment state, validate it, then rerun masking once before touching the masking code again.
