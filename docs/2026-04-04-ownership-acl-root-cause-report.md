# Windows Ownership / ACL Root Cause Report

Date: 2026-04-04
Repo: `C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin`

## Summary

The recurring ownership problem is real, and the root cause is now clear:

- interactive user work runs as `BILL_GATES\alexm`
- Codex shell/tool work runs as `BILL_GATES\CodexSandboxOffline`
- new files created by the Codex-run process are therefore owned by `CodexSandboxOffline`
- edits to existing user-owned files often preserve the original `alexm` owner

That is why the issue feels intermittent instead of universal.

The result is a mixed-owner workspace and cache environment. Most of the time that is only untidy. It becomes operationally dangerous when the sandbox-owned process writes to:

- `.venv`
- copied binary artifacts such as `.pyd`
- user-profile caches such as `%LOCALAPPDATA%\\uv\\cache`
- user-profile config paths such as `%USERPROFILE%\\.config\\git`

## Evidence Collected

### 1. The Codex shell is not running as the interactive user

Observed from the active shell process:

- `whoami` -> `bill_gates\\codexsandboxoffline`
- `[System.Security.Principal.WindowsIdentity]::GetCurrent().Name` -> `BILL_GATES\\CodexSandboxOffline`

This is the key fact. Once that was confirmed, the ownership pattern matched the observed failures.

### 2. The main workspace root is user-owned

Observed owners:

- repo root -> `BILL_GATES\\alexm`
- `.venv` -> `BILL_GATES\\alexm`
- `%LOCALAPPDATA%\\uv\\cache` -> `BILL_GATES\\alexm`

So the parent containers are not fundamentally broken.

### 3. Recently created Codex-generated files are sandbox-owned

Examples observed:

- `docs/2026-04-04-marketplace-name-and-env-regression-report.md` -> `BILL_GATES\\CodexSandboxOffline`
- repo `tmp/` directory -> `BILL_GATES\\CodexSandboxOffline`
- many earlier inspection/response docs created during this session family are also `CodexSandboxOffline` owned

By contrast:

- `pyproject.toml` -> `BILL_GATES\\alexm`
- `uv.lock` -> `BILL_GATES\\alexm`
- `.venv\\Lib\\site-packages\\sam2\\_C.pyd` -> `BILL_GATES\\alexm` after manual ACL repair / user-side copy flow

This matches the rule:

- new file created by sandbox process -> sandbox owner
- existing file edited in place -> usually keeps original owner

### 4. Snapshot of repo ownership distribution

Recursive repo scan result at inspection time:

- `23163` entries owned by `BILL_GATES\\alexm`
- `160` entries owned by `BILL_GATES\\CodexSandboxOffline`
- `3705` entries owned by `BUILTIN\\Administrators`

This means the workspace is mostly fine, but there is a non-trivial tail of sandbox-owned paths.

### 5. The failures line up with mixed-owner paths

Observed incidents already seen in this repo:

- `uv sync` failed until `.venv` ownership/ACLs were repaired
- git emitted warnings about access to `%USERPROFILE%\\.config\\git\\ignore`
- copied `sam2\\_C.pyd` initially produced `DLL load failed ... Access is denied`
- `%LOCALAPPDATA%\\uv\\cache` previously caused problems when touched from the sandbox-side workflow

These are all paths where ownership and inherited ACLs matter much more than they do for a plain text markdown file inside the repo.

## Why It Has Felt Worse Recently

This issue has likely become more visible because recent work involved more operations that create fresh files instead of only editing existing ones:

- lots of newly created docs and reports
- temp/cache directories
- `uv` cache writes
- compiled and copied binary artifacts like `_C.pyd`

That combination increases the number of objects created directly by the sandbox identity.

## What Is Actually Dangerous vs Mostly Harmless

### Mostly harmless

Sandbox-owned markdown/docs inside the repo are usually still usable because inherited ACLs still grant `alexm` access.

### Risky

These are the paths most likely to break:

- `.venv\\Lib\\site-packages\\...`
- binary extension outputs (`.pyd`, `.dll`)
- `%LOCALAPPDATA%\\uv\\cache`
- `%USERPROFILE%\\.config\\git`
- copied artifacts that preserve restrictive ACLs from a source path

## Root Cause Statement

The ownership problem is not primarily a plugin bug.

It is a Windows mixed-principal problem caused by two different identities touching the same workspace and user-profile paths:

1. `alexm` creates or owns the workspace
2. `CodexSandboxOffline` creates new files during shell/tool operations
3. Windows records the creating identity as the file owner
4. Some copied/generated files inherit or preserve ACLs that are acceptable for one identity but awkward for the other

## Mitigations

### Already in progress

- `_C.pyd` copy/install flow now normalizes ACLs after copy
- plugin `uv` subprocesses now use a repo-local `UV_CACHE_DIR` under `tmp/uv-cache-runtime`

That second change is important because it avoids future plugin-driven `uv` writes into `%LOCALAPPDATA%\\uv\\cache`.

### Recommended workflow rules

1. Do not let Codex-driven commands write to user-profile cache/config paths unless necessary.
2. Prefer repo-local temp/cache roots for tool-driven package operations.
3. After copying binaries into `.venv`, normalize ACLs against a known-good neighboring file.
4. If odd access issues reappear, repair ownership on the whole repo / `.venv` instead of only the single failing file.

### Operational repair tool

Added helper script:

- `dev/fix_workspace_acl.ps1`

Purpose:

- repair ownership on the repo root
- repair `.venv`
- repair repo `tmp`
- optionally repair `%LOCALAPPDATA%\\uv\\cache`
- optionally repair `%USERPROFILE%\\.config\\git`

## Practical Conclusion

The issue is understood now.

The recurring failures are explained by sandbox-created files being owned by `CodexSandboxOffline`, especially when those files live in sensitive locations such as `.venv`, `uv` cache directories, or copied binary-extension paths.

The most important long-term mitigation is not "repair ACLs forever by hand." It is:

- keep automated cache/temp writes inside the repo when possible
- normalize ACLs after binary copies
- use the repair script when mixed-owner paths have already accumulated

## Next Steps

1. Keep using the repo-local `UV_CACHE_DIR` change.
2. Use `dev/fix_workspace_acl.ps1` when access problems reappear.
3. Avoid Codex-created writes to user-profile config/cache paths unless absolutely necessary.
4. If a future failure is specific to one binary, repair ACLs on that binary and its parent package directory first.
