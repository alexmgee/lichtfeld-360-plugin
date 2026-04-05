# SAM2 `_C` Extension Windows Build Checklist

> Date: 2026-04-04
> Machine-specific checklist for: `C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin`
> Goal: build `sam2._C` in a controlled way on this Windows machine

## Important Framing

This checklist is for:

- local validation
- developer use
- power-user experimentation

It is **not** the recommended default workflow for ordinary plugin users.

The plugin already works without `_C`. Use this checklist only if you want to:

- remove the warning
- test whether the hole-filling post-process materially improves masks

## What This Machine Already Has

Verified on this machine:

- plugin Python: `3.12`
- plugin torch: `2.11.0+cu128`
- `torch.version.cuda`: `12.8`
- installed CUDA toolkit: `12.9`
- Visual Studio path:
  - `C:\Program Files\Microsoft Visual Studio\18\Community`
- vcvars script:
  - `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`
- older MSVC toolset available:
  - `14.29`
- preferred complete x64 MSVC toolset on this machine:
  - `14.44`
- newer default MSVC toolset also available:
  - `14.50`

Also verified on this machine:

- the plugin venv's base interpreter comes from:
  - `C:\Users\alexm\LichtFeld-Studio\build\vcpkg_installed\x64-windows\tools\python3`
- that interpreter reports a non-existent include path via `sysconfig`
- the real Python headers live at:
  - `C:\Users\alexm\LichtFeld-Studio\build\vcpkg_installed\x64-windows\include\python3.12`
- the matching import libs live at:
  - `C:\Users\alexm\LichtFeld-Studio\build\vcpkg_installed\x64-windows\lib`

Important implication:

- do **not** use the default VS toolset first
- this machine's `14.29` install is missing `msvcprt.lib` for x64, so it cannot finish the link step
- prefer `14.44` on this machine
- ideally use a CUDA toolkit matching PyTorch's CUDA version, i.e. `12.8`

## Recommended Build Strategy

### Preferred path

1. Obtain a matching `sam2==1.1.0` source tree
2. Build `_C` in place from that source tree
3. Copy `_C.pyd` into the installed package
4. Verify import and runtime behavior

If you want to automate most of this on this machine, the repo now includes:

- `dev/build_sam2_c_extension.ps1`

That helper now also applies a local Windows-only source workaround before build:

- it backs up `sam2\csrc\connected_components.cu`
- it rewrites the CUDA translation unit to avoid `torch/extension.h` / `torch/script.h`
- it keeps the kernel logic the same, but reduces the torch/JIT header surface that `nvcc` has to compile on Windows
- this is a developer-build workaround for the current torch toolchain situation, not something ordinary plugin users should ever do manually

Example usage:

```powershell
Set-Location 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin'
.\dev\build_sam2_c_extension.ps1 -Sam2Source 'C:\Users\alexm\src\sam2-1.1.0'
```

If CUDA `12.8` is not installed yet and you intentionally want to try the riskier local `12.9` toolkit:

```powershell
.\dev\build_sam2_c_extension.ps1 -Sam2Source 'C:\Users\alexm\src\sam2-1.1.0' -AllowCudaMismatch
```

### Not recommended as the first path

- default VS toolset `19.50`
- ad-hoc JIT `torch.utils.cpp_extension.load()` as the permanent fix
- dynamic build inside the plugin's normal setup flow

## Pre-Flight Checks

Run these in PowerShell from:

```powershell
Set-Location 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin'
```

### 1. Verify the plugin Python / torch / CUDA state

```powershell
& '.\.venv\Scripts\python.exe' -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print('python ok'); print('torch', torch.__version__); print('torch.cuda', torch.version.cuda); print('CUDA_HOME', CUDA_HOME)"
```

Expected shape:

- torch should report `2.11.0+cu128`
- `torch.cuda` should report `12.8`
- `CUDA_HOME` will currently point at the installed toolkit path

### 2. Verify the preferred complete MSVC toolset can be selected

```powershell
& cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul && cl 2>&1 | findstr /C:"Version"'
```

Expected output:

- `Microsoft (R) C/C++ Optimizing Compiler Version 19.44...`

### 3. Verify `nvcc`

```powershell
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe' --version
```

This confirms the local toolkit exists, but remember:

- `12.9` does not match torch's `12.8`

## Preferred Prerequisite: Install CUDA 12.8 Toolkit

Because torch is `cu128`, the cleaner path is to install a CUDA `12.8` toolkit alongside `12.9`.

Recommended target path:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

If you install it, use that path for `CUDA_HOME` in the build commands below.

If you do **not** install it, you can still experiment with `12.9`, but treat that as a second-choice path.

## Step 1: Prepare a Matching SAM2 Source Tree

You need a source tree that matches the installed package version:

- `sam2==1.1.0`

Recommended location:

```text
C:\Users\alexm\src\sam2-1.1.0
```

Before building, verify the source tree contains:

- `setup.py`
- `sam2\csrc\connected_components.cu`

Verification command:

```powershell
$src = 'C:\Users\alexm\src\sam2-1.1.0'
Test-Path "$src\setup.py"
Test-Path "$src\sam2\csrc\connected_components.cu"
```

Both should return `True`.

## Step 2: Choose the CUDA Toolkit Path

### Preferred

```powershell
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
```

### Fallback / experiment only

```powershell
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
```

Then verify:

```powershell
Test-Path "$cuda\bin\nvcc.exe"
```

## Step 3: Set Common Paths

```powershell
$repo   = 'C:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin'
$py     = "$repo\.venv\Scripts\python.exe"
$pkg    = "$repo\.venv\Lib\site-packages\sam2"
$src    = 'C:\Users\alexm\src\sam2-1.1.0'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
```

## Step 4: Build `_C` In Place

This is the main build command.

It does four important things:

- selects the preferred complete `14.44` MSVC toolset
- sets `CUDA_HOME`
- disables SAM2's silent "build failed but continue anyway" behavior
- runs an in-place build from source

```powershell
$pyInclude = 'C:\Users\alexm\LichtFeld-Studio\build\vcpkg_installed\x64-windows\include\python3.12'
$pyLibDir  = 'C:\Users\alexm\LichtFeld-Studio\build\vcpkg_installed\x64-windows\lib'
& cmd /v:on /c "setlocal EnableDelayedExpansion && set CUDA_HOME=$cuda && set PATH=$cuda\bin;!PATH! && set SAM2_BUILD_ALLOW_ERRORS=0 && set DISTUTILS_USE_SDK=1 && call `"$vcvars`" -vcvars_ver=14.44 && set INCLUDE=$pyInclude;!INCLUDE! && set LIB=$pyLibDir;!LIB! && cd /d `"$src`" && `"$py`" setup.py build_ext --inplace"
```

Why this order matters:

- in `cmd.exe`, `%PATH%` is expanded before the chained command runs
- if you put `set PATH=...;%PATH%` after `vcvars64.bat` on the same line, that expansion can restore the old pre-`vcvars` path and make `cl.exe` disappear again
- setting CUDA first and then calling `vcvars64.bat` preserves both the CUDA path and the VC toolchain path
- this machine's embedded Python also needs an explicit header/lib override because its `sysconfig` include path does not exist
- `cmd /v:on` plus `!INCLUDE!` / `!LIB!` ensures those Python overrides are added after `vcvars64.bat` without losing the VS defaults

What success should produce:

- a built file at:
  - `C:\Users\alexm\src\sam2-1.1.0\sam2\_C.pyd`

Verify:

```powershell
Get-ChildItem "$src\sam2" -Filter '_C*.pyd'
```

Expected:

- ideally `_C.pyd`

## Step 5: Copy The Built Extension Into The Installed Package

```powershell
Copy-Item "$src\sam2\_C.pyd" "$pkg\_C.pyd" -Force
```

Verify:

```powershell
Get-ChildItem "$pkg" -Filter '_C*.pyd'
```

## Step 6: Verify Import

```powershell
& $py -c "import torch; from sam2 import _C; print(_C)"
```

Success means:

- the import no longer fails
- on Windows, importing `torch` first is important because PyTorch sets up its DLL search paths during import

## Step 7: Verify The Hole-Filling Path

This is a minimal smoke test that exercises the code path that previously emitted the warning.

```powershell
& $py -c "import torch; from sam2.utils.misc import fill_holes_in_mask_scores; x=torch.zeros((1,1,16,16), device='cuda'); y=fill_holes_in_mask_scores(x, 10); print(y.shape)"
```

Expected:

- no `_C` import warning
- a tensor shape printed

## Step 8: Run A Real Plugin Smoke Test

After import verification, run a small SAM2-backed masking job and confirm:

- the warning is gone
- tracking still works
- outputs look at least as good as before

This matters because a successful build is not enough; the extension must also load and behave correctly in the actual plugin runtime.

## If The Build Fails

## Failure Case A: `CUDA_HOME` / nvcc not found

Actions:

1. Verify the toolkit path exists
2. Make sure `CUDA_HOME` points to the chosen toolkit
3. Retry from the same command

## Failure Case B: unsupported Visual Studio / host compiler

If nvcc complains about an unsupported MSVC version:

1. Confirm you really invoked:
   - `vcvars64.bat -vcvars_ver=14.44`
2. Retry the build

Only if that still fails should you consider the upstream workaround:

- adding `-allow-unsupported-compiler` to SAM2's `setup.py`

That should be treated as a fallback experiment, not the first plan.

## Failure Case C: toolkit / torch binary mismatch

Symptoms may include:

- link errors
- runtime import failure after a "successful" build
- undefined symbols / DLL load issues

Actions:

1. Stop using CUDA `12.9`
2. Install CUDA `12.8`
3. Rebuild from scratch against the matching toolkit

## Failure Case D: build "succeeds" but `_C` still missing

Possible cause:

- SAM2 swallowed a build failure

Actions:

1. Make sure `SAM2_BUILD_ALLOW_ERRORS=0` was set
2. rerun the build
3. inspect the full console output

## Experimental Fallback: Use CUDA 12.9 Anyway

Only use this if you cannot install CUDA `12.8` yet.

Set:

```powershell
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
```

Then rerun the same in-place build command.

Treat any successful result as provisional until all three pass:

- `_C` imports
- smoke test passes
- real masking run succeeds without regressions

## What Not To Do

- Do not make this build part of the normal plugin install flow.
- Do not assume a successful local build is portable to all users.
- Do not ship a random `_C.pyd` without tying it to a precise support matrix.
- Do not use the default VS `19.50` toolset first if `14.44` is already available and complete on this machine.

## If You Want This For Broad Plugin Users Later

After local validation, the better long-term path is:

1. decide whether `_C` visibly improves masks enough to matter
2. define a narrow support matrix
3. build `_C.pyd` in CI for that matrix
4. distribute it as a prebuilt plugin artifact

That is much more realistic than asking every user to compile CUDA code on Windows.

## Success Criteria

Call the effort successful only if all of these are true:

1. `from sam2 import _C` works in the plugin venv
2. the warning disappears from actual SAM2 propagation
3. masking output is measurably better on your test scenes
4. the change is stable enough that you would consider supporting it in the plugin

If #3 is weak, it is reasonable to stop and leave `_C` optional.
