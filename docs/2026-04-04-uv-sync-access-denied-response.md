# uv sync Access Denied — Response

**Date:** 2026-04-04  
**Reviewed doc:** `docs/2026-04-04-uv-sync-access-denied.md`

---

## Executive Read

The original report correctly identified that the failure is **not really about
`pyproject.toml` content**. But after inspection, the most likely root cause is
not "the invalid package name left the venv half-modified."

The strongest current evidence points to a **Windows ownership / ACL mismatch
inside `.venv`**, specifically in:

`C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv\Lib\site-packages\h11\__pycache__`

That blocked directory is currently owned by:

- `BILL_GATES\CodexSandboxOffline`

while the surrounding `.venv` tree is owned by:

- `BILL_GATES\alexm`

So the document should be updated from "likely venv corruption after invalid
package name" to:

- `uv sync` is hitting a subtree inside `.venv` that was created or modified by
  a different Windows security principal, and cannot safely remove it during
  sync

---

## What Was Verified

### 1. The blocked directory really exists

Confirmed:

- `.venv\Lib\site-packages\h11\__pycache__`

It contains normal `.pyc` files from `h11`, dated `2026-04-03 10:12 PM`.

### 2. The ownership mismatch is real

Confirmed on:

- `.venv`
- `.venv\Lib`
- `.venv\Lib\site-packages`
- `.venv\Lib\site-packages\h11`
- `.venv\Lib\site-packages\h11\__pycache__`
- `.venv\Lib\site-packages\h11\__pycache__\__init__.cpython-312.pyc`

Result:

- the `.venv` root and parent package dirs are owned by `BILL_GATES\alexm`
- the blocked `__pycache__` subtree is owned by `BILL_GATES\CodexSandboxOffline`

That is a concrete, directly observed mismatch.

### 3. The current code path is compatible with repeated `uv` writes into `.venv`

The plugin uses `uv`-based dependency management in `core/setup_checks.py`,
including:

- `install_default_tier()` -> `uv add ultralytics segment-anything torch torchvision`
- `install_video_tracking()` -> `uv add sam2`
- `install_premium_tier()` -> `uv add sam3`

So `.venv` is not just a passive cache. It is an active mutable install target.
That makes ownership consistency especially important.

---

## What This Means

### The current "likely root cause" section is too speculative

The original document says:

> The first `uv sync` failure (caused by the invalid package name `"360 Plugin"`)
> may have left the venv in a half-modified state.

That is possible in theory, but it is not the strongest explanation anymore.

The stronger explanation is simpler:

1. A subtree inside `.venv` was created or modified by another Windows
   principal
2. `uv sync` later tried to remove or replace that subtree
3. Windows denied the operation

The `pyproject.toml` edits only **triggered** `uv sync`; they were not the real
cause of the permission problem.

### This also fits the broader pattern seen in this repo lately

This repo has already hit multiple Windows file-access problems involving
artifacts created under a Codex/sandbox identity and then later used by the
normal app/user context.

So this is not an isolated mystery. It fits an observed pattern:

- text files usually survive this fine
- binary artifacts and runtime-managed directories are more fragile
- `.venv` is especially sensitive because tooling needs to delete and replace
  files there

---

## Recommended Changes To The Original Report

### 1. Rewrite "Likely Root Cause"

Suggested replacement:

> The most likely root cause is an ownership/ACL mismatch inside `.venv`. The
> blocked `h11/__pycache__` subtree is owned by a different Windows security
> principal than the surrounding venv directories. `uv sync` can read the
> environment well enough to begin work, but fails when it attempts to remove
> or replace that subtree during sync.

### 2. Move the invalid-name theory into a weaker "possible trigger" section

The invalid `name = "360 Plugin"` attempt may still matter as a timeline clue,
but it should no longer be presented as the primary explanation.

Suggested reframing:

> The invalid package name may have been the first event that forced a new
> `uv sync`, which exposed the pre-existing ownership issue. It is no longer
> the leading root-cause theory.

### 3. Add a verified evidence section

The report should explicitly record:

- owner of `.venv`: `BILL_GATES\alexm`
- owner of `h11/__pycache__`: `BILL_GATES\CodexSandboxOffline`

That makes the diagnosis much stronger than the current speculative version.

---

## Recovery Strategy

I would recommend a tiered recovery sequence.

### Option 1: Repair only the blocked subtree first

This is the least disruptive path.

Suggested sequence:

```powershell
$bad = 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv\Lib\site-packages\h11\__pycache__'
$user = "$env:USERDOMAIN\$env:USERNAME"

takeown.exe /F $bad /R /D Y
icacls.exe $bad /inheritance:e /T /C
icacls.exe $bad /grant "$user:(OI)(CI)F" /T /C
Remove-Item -LiteralPath $bad -Recurse -Force
```

Then retry:

```powershell
uv sync
```

### Option 2: Normalize the entire `.venv`

This is the better choice if more foreign-owned subtrees may exist.

```powershell
$venv = 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv'
$user = "$env:USERDOMAIN\$env:USERNAME"

takeown.exe /F $venv /R /D Y
icacls.exe $venv /inheritance:e /T /C
icacls.exe $venv /grant "$user:(OI)(CI)F" /T /C
```

Then retry:

```powershell
uv sync
```

### Option 3: Rebuild `.venv` from scratch

If Option 2 still fails, this becomes the cleanest and most predictable route.

```powershell
Remove-Item -LiteralPath 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin\.venv' -Recurse -Force
uv sync
```

This is slower and redownloads packages, but it avoids chasing multiple hidden
ownership problems one-by-one.

---

## Recommended Preference Order

For this specific incident, I would choose:

1. Repair the whole `.venv` ownership/ACL state
2. Retry `uv sync`
3. If it still fails, delete and rebuild `.venv`

I would **not** spend much more time on the invalid-name corruption theory
unless the ACL repair fails. The ownership mismatch is already enough to
explain the observed behavior.

---

## Longer-Term Recommendation

The important longer-term takeaway is:

- avoid letting mixed Windows principals write into `.venv`

For this project, `.venv` is part of the plugin's runtime-managed dependency
surface. That makes it a poor place for ad hoc writes by tools running under a
different identity.

If future helper scripts or automation touch `.venv`, they should either:

- run under the same user context that LichtFeld uses, or
- explicitly normalize ownership/ACLs after writing

Otherwise this class of failure is likely to recur.

---

## Bottom Line

The issue is now much more concrete than the original report suggests.

This does **not** look primarily like a `pyproject.toml` problem or even a
`uv` bug. It looks like `uv sync` is trying to mutate a `.venv` subtree that is
owned by the wrong Windows principal.

That gives a clear path forward:

- repair ownership / ACLs on `.venv`
- retry `uv sync`
- rebuild `.venv` if needed

If the original report is updated to reflect that evidence, it becomes a much
stronger incident record.
