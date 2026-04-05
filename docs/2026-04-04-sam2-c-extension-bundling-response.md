# SAM2 `_C` Bundling Doc — Response

> Date: 2026-04-04
> Responds to: `docs/2026-04-04-sam2-c-extension-bundling.md`
> Scope: Accuracy review, implementation status, and recommended revisions

## Executive Summary

The bundling document is directionally useful, but it currently overstates how complete and how broadly safe the `_C.pyd` story is.

What is true:

- a SAM2 `_C.pyd` binary has now been built successfully on this machine
- it can be copied into `site-packages\sam2`
- it loads in the plugin venv when imported through the same general path the runtime uses
- the plugin now has install-time and runtime hooks intended to keep `_C.pyd` present

What is not yet proven strongly enough to write as settled fact:

- that the current bundled binary is a stable broad-user solution
- that the runtime safety net fully covers real-world failure cases
- that the current implementation is robust against Windows ACL / ownership issues
- that the current provenance description matches the artifact that was actually produced

So the core conclusion is:

- the feature is implemented in an experimental / narrow-matrix sense
- the document should not yet read as though the bundling problem is fully solved for all plugin users

## What The Doc Gets Right

The document is right about the most important baseline facts:

- the `sam2` install path does not compile `_C` automatically
- missing `_C` does not break SAM2 tracking completely
- the plugin now bundles a prebuilt `_C.pyd`
- `core/setup_checks.py` and `core/backends.py` were updated to install and/or ensure the extension

Those claims line up with the current implementation in:

- `core/setup_checks.py`
- `core/backends.py`
- `lib/_C.pyd`

So as a "what changed in the repo" document, it is broadly correct.

## Main Findings

## 1. The status is too strong

The document says:

- `Status: Implemented`

That is stronger than the current evidence supports.

The better current status is something like:

- `Implemented, runtime validation ongoing`
- `Implemented experimentally`
- `Implemented for current Windows support matrix, still validating LichtFeld runtime behavior`

Why:

- the latest real masking run still emitted the SAM2 `_C` warning inside the actual LichtFeld flow
- the warning text changed from "cannot import `_C`" to `DLL load failed while importing _C: Access is denied`
- that means the remaining problem is no longer compile/link, but real runtime loading inside the app environment

That is progress, but it is not a finished broad-user story yet.

## 2. The runtime safety-net section overclaims

The bundling doc says `ensure_sam2_c_extension()` covers:

- manual installs
- upgrades that overwrite the extension
- any bypassed install path

That is too strong for the current implementation.

Current behavior:

- try `from sam2 import _C`
- if import fails, copy the bundled `_C.pyd`

What it does **not** currently guarantee by itself:

- importing `torch` first on Windows so DLL search paths are initialized
- repairing the ACL / ownership on the copied `_C.pyd`
- re-verifying that `_C` imports successfully after copy
- distinguishing "missing file" from "file present but access denied"
- recovering from a same-size but stale/bad `_C.pyd`

So the safety-net concept is good, but the doc should describe it as:

- a best-effort repair hook

not:

- a complete runtime guarantee

## 3. The provenance block is now stale

The provenance section no longer matches the actual build path that got the artifact to a working state.

The doc currently says the binary was built with:

- source: `HEAD`
- toolchain: `v14.29`

But the actual working path evolved beyond that:

- the effective working toolset became `14.44`, because the local `14.29` install was missing `msvcprt.lib` for x64
- the build helper now applies a Windows-specific source workaround to `sam2/csrc/connected_components.cu`
- the binary was validated with import-order sensitivity on Windows (`torch` first)
- the runtime story also encountered an ACL/ownership issue after copy

This matters because provenance is not just historical trivia here; it defines whether the binary is reproducible and supportable.

A better provenance section should mention:

- exact SAM2 source version or commit actually used
- that a Windows-only source workaround was applied during build
- the final working MSVC toolset version
- the Python / torch / CUDA matrix
- that the artifact is currently validated only for that matrix

## 4. The decision statement is too broad for the current evidence

The document’s decision is:

- bundle `_C.pyd` in `lib/`
- auto-install it into `sam2`
- same precedent as `lib/python3.dll`

The first two parts are fine as an implementation strategy.

The third part is misleading.

`python3.dll` is not the same kind of artifact as `_C.pyd`.

`_C.pyd` is:

- a CPython extension module
- tied to a particular Python ABI
- tied to a particular torch binary layout
- tied to CUDA/runtime behavior
- influenced by the Windows loader and file ACLs

So bundling `_C.pyd` is much closer to shipping a tightly versioned plugin-native binary than to shipping a generic dependency DLL.

The decision statement should reflect that narrower support model explicitly.

## 5. The file-size identity check is too weak for the doc’s confidence level

The current implementation uses file size as the skip-if-identical heuristic.

That is a practical shortcut, but the doc currently describes it too casually relative to the risk.

Why this matters:

- a rebuilt binary with changed content can still have the same size
- a copied binary can have the right size but the wrong ACL
- a same-size file can still fail to import in the real app environment

So this logic is acceptable for a quick local workflow, but the doc should frame it as:

- a provisional optimization

not:

- a sufficiently strong correctness check

If this stays in the product path, a hash or explicit import verification is safer.

## 6. The current implementation is not yet broad-user ready

This is the biggest product conclusion.

The bundling doc reads like the plugin now has a deployable solution for ordinary users. That is premature.

What we have really proven:

- a developer can build `_C.pyd` locally on this machine
- the plugin can bundle that binary
- the plugin can attempt to install it automatically
- the extension loads in the plugin venv under the right conditions

What we have **not** yet proven strongly enough:

- every normal LichtFeld runtime path loads it cleanly
- copied binaries always inherit app-loadable permissions
- upgrades and reinstalls remain clean without manual repair
- the current binary remains valid across future Python / torch / CUDA changes

That means the correct product framing is still:

- narrow Windows matrix
- experimental / provisional runtime support
- not yet a settled broad-user distribution story

## Current Runtime Interpretation

The most important new signal is the recent warning:

- `DLL load failed while importing _C: Access is denied`

That message changes the diagnosis materially.

It suggests:

- `_C.pyd` exists
- the loader is reaching the file
- the failure is about access or load context, not missing compilation

And that lines up with what was observed:

- shell import worked after the file’s owner / ACL was repaired
- the app still reported `Access is denied` before that fix

So the remaining risk is not "can we build `_C`?" anymore.

The remaining risk is:

- can the plugin install and load the bundled binary reliably in the actual app environment without manual repair?

That is a different and narrower problem, and the doc should say so.

## Recommended Revisions To The Bundling Doc

## 1. Change the status line

Recommended replacement:

```md
**Status:** Implemented experimentally; runtime validation ongoing
```

or:

```md
**Status:** Implemented for current Windows matrix; still validating LichtFeld runtime behavior
```

## 2. Narrow the decision language

Recommended change:

- keep the bundling decision
- remove the implication that this is already equivalent to a fully supported general-user solution

Suggested framing:

- bundle a prebuilt `_C.pyd` for the currently validated Windows support matrix
- auto-install it into the installed `sam2` package
- treat runtime verification as part of the install story, not an afterthought

## 3. Rewrite the runtime safety-net section

Recommended change:

- describe it as a best-effort repair path
- state clearly that successful copy does not by itself prove successful runtime load

Suggested additions:

- import `torch` before verifying `_C` on Windows
- re-verify import after copy
- normalize ACLs / ownership after copy
- fall back gracefully if runtime load still fails

## 4. Update the provenance section

Recommended additions:

- exact SAM2 source revision or tagged release
- actual final MSVC toolset used
- note that the helper applies a Windows-specific source workaround
- note that the artifact is currently validated only for the plugin’s current Python / torch matrix

## 5. Add an explicit limitations section

Suggested section:

### Current Limitations

- validated only on Windows
- validated only for the current plugin Python / torch stack
- runtime load behavior in LichtFeld is still being verified
- copied binaries may require ACL normalization on Windows

That would make the document much more truthful and much easier to maintain.

## Recommended Implementation Follow-Ups

The document should mention the following follow-up work explicitly if they are part of the intended final solution:

1. `ensure_sam2_c_extension()` should import `torch` before testing `_C` on Windows.
2. `_install_sam2_c_extension()` should normalize the destination file ACL to match a known-good file in the `sam2` package.
3. The runtime path should verify import success after copy, not just assume success.
4. The install path should not rely only on file size as the identity check forever.
5. The plugin should log a single clear plugin-level status if `_C` still cannot load, rather than letting the raw upstream warning remain the only signal.

## Bottom Line

The bundling document is close to being a useful implementation record, but it currently reads more finished than the evidence supports.

The strongest accurate version of the story is:

- bundling is now implemented
- local build + install are proven on this machine
- the artifact can load in the plugin venv
- the real remaining risk is reliable runtime loading in LichtFeld, especially on Windows
- broad-user support should only be claimed after that last part is demonstrated cleanly

So I would keep the document, but revise it to be more precise, more matrix-aware, and more honest about what is still under validation.
