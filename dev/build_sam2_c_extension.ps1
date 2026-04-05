[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Sam2Source = "$env:USERPROFILE\src\sam2-1.1.0",
    [string]$ToolsetVersion = "14.44",
    [string]$CudaHome,
    [switch]$AllowCudaMismatch,
    [switch]$SkipCopy,
    [switch]$SkipSmokeTest,
    [string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Title
    Write-Host ("=" * 72)
}

function Fail {
    param([string]$Message)
    throw $Message
}

function Resolve-RequiredPath {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        Fail("$Label not found: $Path")
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-VisualStudioInstallPath {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        "C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    $vswhere = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $vswhere) {
        Fail("Could not find vswhere.exe. Install Visual Studio Build Tools or Visual Studio with C++ tools.")
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installPath) {
        Fail("No Visual Studio installation with VC tools was found by vswhere.")
    }

    return $installPath.Trim()
}

function Resolve-MsvcToolsetDirectory {
    param(
        [string]$VcToolsRoot,
        [string]$VersionPrefix
    )

    $matches = Get-ChildItem -LiteralPath $VcToolsRoot -Directory |
        Where-Object { $_.Name -like "$VersionPrefix*" } |
        Sort-Object Name -Descending
    return $matches | Select-Object -First 1
}

function Get-MsvcRuntimeLibPath {
    param([string]$ToolsetDirectory)
    return Join-Path $ToolsetDirectory "lib\x64\msvcprt.lib"
}

function Get-PythonValue {
    param(
        [string]$PythonExe,
        [string]$Code
    )
    $output = & $PythonExe -c $Code
    if ($LASTEXITCODE -ne 0) {
        Fail("Python command failed: $Code")
    }
    return ($output | Out-String).Trim()
}

function Run-CmdOrFail {
    param([string]$Command)
    Write-Host $Command
    & cmd.exe /v:on /c $Command
    if ($LASTEXITCODE -ne 0) {
        Fail("Command failed with exit code $LASTEXITCODE")
    }
}

function Apply-Sam2WindowsNvccWorkaround {
    param([string]$CudaSourcePath)

    $sourceText = Get-Content -LiteralPath $CudaSourcePath -Raw
    $marker = "// Codex Windows nvcc workaround: avoid torch/extension.h in CUDA translation unit."
    $patched = $sourceText
    if (-not $sourceText.Contains($marker)) {
        $backupPath = "$CudaSourcePath.codex_original.bak"
        if (-not (Test-Path -LiteralPath $backupPath)) {
            Copy-Item -LiteralPath $CudaSourcePath -Destination $backupPath -Force
            Write-Host "Backed up original CUDA source to $backupPath"
        }

        $includePattern = '#include <ATen/cuda/CUDAContext.h>\r?\n#include <cuda.h>\r?\n#include <cuda_runtime.h>\r?\n#include <torch/extension.h>\r?\n#include <torch/script.h>\r?\n#include <vector>'
        $includeReplacement = @"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <vector>

"$marker
"@.Trim()
        $patched = [regex]::Replace($patched, $includePattern, $includeReplacement, 1)
        if ($patched -eq $sourceText) {
            Fail("Could not apply the Windows nvcc workaround to $CudaSourcePath because the expected include block was not found.")
        }
    } else {
        Write-Host "SAM2 CUDA source already has the Windows nvcc workaround applied; refreshing replacement set."
    }

    $replacements = [ordered]@{
        "std::vector<torch::Tensor>" = "std::vector<at::Tensor>"
        "const torch::Tensor& inputs" = "const at::Tensor& inputs"
        "torch::kUInt8" = "c10::kByte"
        "torch::TensorOptions()" = "at::TensorOptions()"
        "torch::kInt32" = "c10::kInt"
        "torch::Tensor" = "at::Tensor"
        "torch::zeros" = "at::zeros"
        "at::kUInt8" = "c10::kByte"
        "at::kInt32" = "c10::kInt"
    }
    foreach ($entry in $replacements.GetEnumerator()) {
        $patched = $patched.Replace($entry.Key, $entry.Value)
    }

    if ($patched -ne $sourceText) {
        Set-Content -LiteralPath $CudaSourcePath -Value $patched -NoNewline
        Write-Host "Applied Windows nvcc workaround to $CudaSourcePath"
    } else {
        Write-Host "SAM2 CUDA source already matches the current Windows nvcc workaround."
    }
}

$transcriptStarted = $false
try {
    $RepoRoot = Resolve-RequiredPath -Path $RepoRoot -Label "Repo root"
    $pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $sitePackagesSam2 = Join-Path $RepoRoot ".venv\Lib\site-packages\sam2"
    $pythonExe = Resolve-RequiredPath -Path $pythonExe -Label "Plugin venv Python"
    $sitePackagesSam2 = Resolve-RequiredPath -Path $sitePackagesSam2 -Label "Installed sam2 package"
    $Sam2Source = Resolve-RequiredPath -Path $Sam2Source -Label "SAM2 source tree"

    $setupPy = Join-Path $Sam2Source "setup.py"
    $cudaSource = Join-Path $Sam2Source "sam2\csrc\connected_components.cu"
    $null = Resolve-RequiredPath -Path $setupPy -Label "sam2 setup.py"
    $null = Resolve-RequiredPath -Path $cudaSource -Label "sam2 connected_components.cu"

    if (-not $LogPath) {
        $logDir = Join-Path $RepoRoot "tmp"
        if (-not (Test-Path -LiteralPath $logDir)) {
            New-Item -ItemType Directory -Path $logDir | Out-Null
        }
        $LogPath = Join-Path $logDir ("sam2_c_build_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
    }

    Start-Transcript -Path $LogPath -Force | Out-Null
    $transcriptStarted = $true

    Write-Section "Environment"

    $torchVersion = Get-PythonValue -PythonExe $pythonExe -Code "import torch; print(torch.__version__)"
    $torchCuda = Get-PythonValue -PythonExe $pythonExe -Code "import torch; print(torch.version.cuda or '')"
    $pythonVersion = Get-PythonValue -PythonExe $pythonExe -Code "import sys; print(sys.version.split()[0])"
    $installedSam2Version = Get-PythonValue -PythonExe $pythonExe -Code "import importlib.metadata as m; print(m.version('sam2'))"
    $pythonBasePrefix = Get-PythonValue -PythonExe $pythonExe -Code "import sys; print(sys.base_prefix)"
    $pythonIncludeDir = Get-PythonValue -PythonExe $pythonExe -Code "import sysconfig; print(sysconfig.get_path('include') or '')"
    $pythonLibDir = Get-PythonValue -PythonExe $pythonExe -Code "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"
    $pythonMajorMinor = Get-PythonValue -PythonExe $pythonExe -Code "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $pythonMajorMinorCompact = Get-PythonValue -PythonExe $pythonExe -Code "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"

    $pythonRootCandidate = [System.IO.Path]::GetFullPath((Join-Path $pythonBasePrefix "..\.."))
    $pythonIncludeDerived = $false
    if (-not $pythonIncludeDir -or -not (Test-Path -LiteralPath $pythonIncludeDir)) {
        $includeCandidate = Join-Path $pythonRootCandidate "include\python$pythonMajorMinor"
        if (Test-Path -LiteralPath (Join-Path $includeCandidate "Python.h")) {
            $pythonIncludeDir = $includeCandidate
            $pythonIncludeDerived = $true
        } else {
            Fail("Python include directory could not be resolved. sysconfig reported '$pythonIncludeDir', and no Python.h was found under $includeCandidate")
        }
    }

    $pythonLibDerived = $false
    if (-not $pythonLibDir -or -not (Test-Path -LiteralPath $pythonLibDir)) {
        $libCandidate = Join-Path $pythonRootCandidate "lib"
        $versionedImportLib = Join-Path $libCandidate ("python{0}.lib" -f $pythonMajorMinorCompact)
        $genericImportLib = Join-Path $libCandidate "python3.lib"
        if ((Test-Path -LiteralPath $versionedImportLib) -or (Test-Path -LiteralPath $genericImportLib)) {
            $pythonLibDir = $libCandidate
            $pythonLibDerived = $true
        } else {
            Fail("Python library directory could not be resolved. sysconfig reported '$pythonLibDir', and no python import library was found under $libCandidate")
        }
    }

    Write-Host "RepoRoot:        $RepoRoot"
    Write-Host "Python:          $pythonExe"
    Write-Host "Python version:  $pythonVersion"
    Write-Host "sam2 version:    $installedSam2Version"
    Write-Host "torch version:   $torchVersion"
    Write-Host "torch CUDA:      $torchCuda"
    Write-Host "Python base:     $pythonBasePrefix"
    Write-Host "Python include:  $pythonIncludeDir"
    Write-Host "Python lib dir:  $pythonLibDir"
    Write-Host "sam2 source:     $Sam2Source"
    Write-Host "Log:             $LogPath"
    if ($pythonIncludeDerived) {
        Write-Warning "sysconfig include path did not exist; using derived Python include path."
    }
    if ($pythonLibDerived) {
        Write-Warning "sysconfig library path did not exist; using derived Python library path."
    }

    $vsInstallPath = Get-VisualStudioInstallPath
    $vcvars = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvars64.bat"
    $vcToolsRoot = Resolve-RequiredPath -Path (Join-Path $vsInstallPath "VC\Tools\MSVC") -Label "VC tools root"
    $vcvars = Resolve-RequiredPath -Path $vcvars -Label "vcvars64.bat"
    $resolvedToolsetDir = Resolve-MsvcToolsetDirectory -VcToolsRoot $vcToolsRoot -VersionPrefix $ToolsetVersion
    if (-not $resolvedToolsetDir) {
        Fail("Requested MSVC toolset '$ToolsetVersion' was not found under $vcToolsRoot")
    }
    $msvcRuntimeLib = Get-MsvcRuntimeLibPath -ToolsetDirectory $resolvedToolsetDir.FullName
    if (-not (Test-Path -LiteralPath $msvcRuntimeLib)) {
        $fallbackPrefixes = @("14.44", "14.50", "14.29", "14.16") | Where-Object { $_ -ne $ToolsetVersion }
        $fallbackDir = $null
        foreach ($prefix in $fallbackPrefixes) {
            $candidate = Resolve-MsvcToolsetDirectory -VcToolsRoot $vcToolsRoot -VersionPrefix $prefix
            if ($candidate -and (Test-Path -LiteralPath (Get-MsvcRuntimeLibPath -ToolsetDirectory $candidate.FullName))) {
                $fallbackDir = $candidate
                $ToolsetVersion = $prefix
                break
            }
        }
        if ($fallbackDir) {
            $resolvedToolsetDir = $fallbackDir
            $msvcRuntimeLib = Get-MsvcRuntimeLibPath -ToolsetDirectory $resolvedToolsetDir.FullName
            Write-Warning "Requested toolset is missing x64 msvcprt.lib; falling back to toolset $ToolsetVersion ($($resolvedToolsetDir.Name))."
        } else {
            Fail("Requested toolset '$ToolsetVersion' is missing x64 msvcprt.lib, and no fallback toolset with that runtime library was found.")
        }
    }
    Write-Host "Visual Studio:   $vsInstallPath"
    Write-Host "vcvars64.bat:    $vcvars"
    Write-Host "Toolset:         $ToolsetVersion"
    Write-Host "Toolset dir:     $($resolvedToolsetDir.FullName)"
    Write-Host "msvcprt.lib:     $msvcRuntimeLib"

    if (-not $CudaHome) {
        if (-not $torchCuda) {
            Fail("torch.version.cuda is empty; cannot infer a matching CUDA toolkit.")
        }
        $preferredCuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$torchCuda"
        if (Test-Path -LiteralPath (Join-Path $preferredCuda "bin\nvcc.exe")) {
            $CudaHome = $preferredCuda
        } elseif ($AllowCudaMismatch) {
            $fallbackCandidates = @(
                "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
                "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
            )
            $CudaHome = $fallbackCandidates |
                Where-Object { Test-Path -LiteralPath (Join-Path $_ "bin\nvcc.exe") } |
                Select-Object -First 1
            if (-not $CudaHome) {
                Fail("No CUDA toolkit with nvcc.exe was found in the fallback locations.")
            }
        } else {
            Fail("Matching CUDA toolkit not found at $preferredCuda. Install CUDA $torchCuda or rerun with -AllowCudaMismatch / -CudaHome.")
        }
    }

    $CudaHome = Resolve-RequiredPath -Path $CudaHome -Label "CUDA_HOME"
    $nvccExe = Resolve-RequiredPath -Path (Join-Path $CudaHome "bin\nvcc.exe") -Label "nvcc.exe"
    Write-Host "CUDA_HOME:       $CudaHome"
    if ($CudaHome -notmatch [regex]::Escape($torchCuda)) {
        Write-Warning "CUDA_HOME does not match torch.version.cuda ($torchCuda). Build/load compatibility is riskier in this mode."
    }

    Write-Section "Toolchain Probe"
    Run-CmdOrFail -Command ('call "{0}" -vcvars_ver={1} >nul && cl 2>&1 | findstr /C:"Version"' -f $vcvars, $ToolsetVersion)
    & $nvccExe --version
    if ($LASTEXITCODE -ne 0) {
        Fail("nvcc --version failed.")
    }

    Write-Section "Build"
    Apply-Sam2WindowsNvccWorkaround -CudaSourcePath $cudaSource
    # Use delayed expansion so the cmd.exe environment can be safely updated after
    # vcvars64.bat runs. This preserves both the VC toolchain settings and our CUDA /
    # Python header overrides in the same command chain.
    $buildCommand = 'setlocal EnableDelayedExpansion && set "CUDA_HOME={2}" && set "PATH={2}\bin;!PATH!" && set SAM2_BUILD_ALLOW_ERRORS=0 && set DISTUTILS_USE_SDK=1 && call "{0}" -vcvars_ver={1} && set "INCLUDE={5};!INCLUDE!" && set "LIB={6};!LIB!" && cd /d "{3}" && "{4}" setup.py build_ext --inplace' -f $vcvars, $ToolsetVersion, $CudaHome, $Sam2Source, $pythonExe, $pythonIncludeDir, $pythonLibDir
    Run-CmdOrFail -Command $buildCommand

    $builtExtension = Get-ChildItem -LiteralPath (Join-Path $Sam2Source "sam2") -Filter "_C*.pyd" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $builtExtension) {
        Fail("Build completed but no _C*.pyd was found under $Sam2Source\sam2")
    }
    Write-Host "Built extension: $($builtExtension.FullName)"

    if (-not $SkipCopy) {
        Write-Section "Install Built Extension"
        $targetExtension = Join-Path $sitePackagesSam2 "_C.pyd"
        $aclTemplatePath = Join-Path $sitePackagesSam2 "__init__.py"
        if (Test-Path -LiteralPath $targetExtension) {
            $backup = Join-Path $sitePackagesSam2 ("_C.pyd.bak_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
            Move-Item -LiteralPath $targetExtension -Destination $backup -Force
            Write-Host "Backed up existing _C.pyd to $backup"
        }
        Copy-Item -LiteralPath $builtExtension.FullName -Destination $targetExtension -Force
        Write-Host "Copied extension to $targetExtension"
        if (Test-Path -LiteralPath $aclTemplatePath) {
            try {
                $templateAcl = Get-Acl -LiteralPath $aclTemplatePath
                Set-Acl -LiteralPath $targetExtension -AclObject $templateAcl
                Write-Host "Applied ACL from $aclTemplatePath to $targetExtension"
            } catch {
                Write-Warning "Copied _C.pyd but failed to normalize its ACL from $aclTemplatePath. $_"
            }
        }
    } else {
        Write-Host "SkipCopy set; leaving built extension in source tree only."
    }

    Write-Section "Verify Import"
    & $pythonExe -c "import pathlib; import torch; from sam2 import _C; print(pathlib.Path(_C.__file__).resolve())"
    if ($LASTEXITCODE -ne 0) {
        Fail("Import verification failed for sam2._C")
    }

    if (-not $SkipSmokeTest) {
        Write-Section "Smoke Test"
        $cudaAvailable = Get-PythonValue -PythonExe $pythonExe -Code "import torch; print(torch.cuda.is_available())"
        if ($cudaAvailable -eq "True") {
            & $pythonExe -c "import torch; from sam2.utils.misc import fill_holes_in_mask_scores; x=torch.zeros((1,1,16,16), device='cuda'); y=fill_holes_in_mask_scores(x, 10); print(tuple(y.shape))"
            if ($LASTEXITCODE -ne 0) {
                Fail("Smoke test failed.")
            }
        } else {
            Write-Warning "torch.cuda.is_available() is False; skipping CUDA smoke test."
        }
    } else {
        Write-Host "SkipSmokeTest set; not running fill_holes_in_mask_scores smoke test."
    }

    Write-Section "Done"
    Write-Host "sam2._C build workflow completed successfully."
    Write-Host "Next step: run a real SAM2 masking job and verify the warning is gone."
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
