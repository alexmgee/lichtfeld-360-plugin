# COLMAP Cancellation Analysis

Date: 2026-03-29

## The Problem

When user clicks Cancel during COLMAP feature extraction, matching, or mapping,
nothing happens until the current C++ call finishes. The cancel button sets a flag
that only gets checked between steps — not during them.

## Why It Happens

pycolmap's `extract_features()`, `match_exhaustive()`, and `incremental_mapping()`
are blocking C++ calls. Once invoked, Python has no way to interrupt them. The
cancel flag is only checked between these calls via `_ensure_not_cancelled()`.

The COLMAP plugin (shadygm/Lichtfeld-COLMAP-Plugin) has the exact same limitation
with the exact same pattern: `_cancelled` flag, checked between stages.

## Options Evaluated

### Option 1: Subprocess COLMAP (rejected)

Run COLMAP as a separate process (`subprocess.Popen(["colmap", ...])`) instead of
calling pycolmap in-process. `process.terminate()` gives instant cancellation.

**Pros:**
- Real cancellation
- Process isolation (COLMAP crash doesn't crash LFS)
- Native stdout/stderr progress

**Cons:**
- Requires COLMAP binary (`colmap.exe`) as an additional dependency
- lyehe's Windows binary bundle is ~1.3 GB
- Users would need to download it separately or plugin auto-downloads on first run
- Breaks the zero-extra-steps installation that pycolmap provides
- Loses Python API for `apply_rig_config`, `Database.open`, `next_image_callback`

**Why rejected:** The self-contained pycolmap wheel installation is a major UX win.
"Plugin install → uv sync → done" with zero extra steps. Adding a 1.3 GB COLMAP
binary download would complicate setup significantly, especially for non-advanced
users. COLMAP installs are a known pain point in the photogrammetry community.

### Option 2: Hybrid (subprocess for compute, pycolmap for API) (not pursued)

Run heavy steps (`feature_extractor`, `exhaustive_matcher`, `mapper`) as subprocess
using the COLMAP binary, but keep pycolmap for `apply_rig_config` and database queries.

**Same cons as Option 1** — still requires the binary.

### Option 3: Accept the limitation (chosen)

Keep pycolmap in-process. Make the cancel UI honest about behavior.

**Decision:** Cancel sets the flag, COLMAP finishes its current step, then the
pipeline stops. The UI should communicate this clearly instead of pretending
cancellation is immediate.

## lyehe/build_gpu_colmap Release Assets (v4.0.2)

### pycolmap wheels (what we use):
- `+cuda` — links to system CUDA runtime (Windows, small wheel)
- `+cuda.cudss` — adds cuDSS sparse solver
- `+cu128.bundled` — bundles CUDA runtime (Linux only, no Windows variant)

### COLMAP binaries (standalone, not used):
- `COLMAP-windows-latest-CUDA.zip` — CLI only
- `COLMAP-windows-latest-CUDA-GUI.zip` — CLI + GUI
- cuDSS variants available

No alternate simpler install method exists. The `bundled` pycolmap variants
(which include CUDA runtime) are Linux-only.

## Future Reconsideration Triggers

- If pycolmap adds a cancel/abort API
- If COLMAP binary size drops significantly
- If a plugin auto-download mechanism becomes standard in LFS
- If users report cancel as a major pain point
