# Fisheye Pinhole Self-Contained COLMAP Output Plan

Date: 2026-05-28

## Goal

Make Fisheye (Pinhole) produce a single LFS dataset rooted at:

```text
output/colmap/
```

The root output directory is only a container for `colmap/`, optional `extracted/`, and `metadata/`.

`output/colmap/` must work through both LFS loading paths:

- Blender/NeRF loader via `colmap/transforms.json`
- COLMAP loader via `colmap/sparse/0`

## Target Structures

No checkboxes:

```text
output/
|-- colmap/
|   |-- transforms.json
|   |-- pointcloud.ply
|   |-- images/
|   |-- masks/
|   |-- sparse/
|   |   `-- 0/
|   |       |-- cameras.bin
|   |       |-- frames.bin
|   |       |-- images.bin
|   |       |-- points3D.bin
|   |       `-- rigs.bin
|   `-- database.db
`-- metadata/
    |-- timing.json
    `-- colmap_debug.log
```

Split streams kept:

```text
output/
|-- front.mp4
|-- back.mp4
|-- colmap/
|   |-- transforms.json
|   |-- pointcloud.ply
|   |-- images/
|   |-- masks/
|   |-- sparse/0/
|   `-- database.db
`-- metadata/
```

Split streams and raw fisheyes kept:

```text
output/
|-- front.mp4
|-- back.mp4
|-- colmap/
|   |-- transforms.json
|   |-- pointcloud.ply
|   |-- images/
|   |-- masks/
|   |-- sparse/0/
|   `-- database.db
|-- extracted/
|   |-- front/
|   |-- back/
|   `-- masks/
`-- metadata/
```

Split streams, raw fisheyes, and extracted data kept:

```text
output/
|-- front.mp4
|-- back.mp4
|-- colmap/
|   |-- transforms.json
|   |-- pointcloud.ply
|   |-- images/
|   |-- masks/
|   |-- sparse/0/
|   `-- database.db
|-- extracted/
|   |-- front/
|   |-- back/
|   |-- masks/
|   |-- pinhole_images/
|   `-- pinhole_masks/
`-- metadata/
```

## Required Invariants

- Auto-import target is always `output/colmap/` for Fisheye (Pinhole).
- `colmap/transforms.json` uses paths relative to `colmap/`:

```json
"file_path": "images/front_ctr_hi_000001.jpg",
"mask_path": "masks/front_ctr_hi_000001.png"
```

- `colmap/sparse/0` contains the same registered propagated view set as `transforms.json`, with flat image names:

```text
front_ctr_hi/000001.jpg -> front_ctr_hi_000001.jpg
front_ring_l_000001.jpg
back_ring_ur_000001.jpg
```

- Root `images/`, root `masks/`, and root `sparse/` should not remain after a successful Fisheye (Pinhole) run.
- `colmap/images/` and `colmap/masks/` are the final flattened image/mask folders.
- `extracted/pinhole_images/` and `extracted/pinhole_masks/` are only retained when `Keep extracted data` is enabled.

## Procedural Steps

### Phase 0: Recover and Reconcile

- [x] Recover tracked files to the last pushed baseline after the working-tree damage.
- [x] Recreate this plan document under `docs/`.
- [x] Reapply only the agreed `colmap/` dataset-base implementation.

### Phase 1: Config and UI

- [x] Replace `keep_fisheye_frames` with `keep_extracted_data`.
- [x] Rename the Fisheye (Pinhole) checkbox to `Keep extracted data`.
- [x] Pass `keep_extracted_data` into `PipelineConfig`.
- [x] Auto-import `fisheye_pinhole` via `output/colmap/`, not root and not the direct JSON file.

### Phase 2: Native Fisheye (Pinhole) Output

- [x] Copy unflattened pinhole images/masks to `extracted/` when `keep_extracted_data` is enabled.
- [x] Flatten final pinhole images/masks.
- [x] Move flattened images/masks into `colmap/images/` and `colmap/masks/`.
- [x] Write `colmap/transforms.json` and `colmap/pointcloud.ply`.
- [x] Move `database.db*` into `colmap/` when present.
- [x] Remove root `sparse/` after rewriting `colmap/sparse/0`.

### Phase 3: Sparse Rewrite

- [x] Add helper to flatten COLMAP image names.
- [x] Add helper to write a propagated all-view sparse model to `colmap/sparse/0`.
- [x] Verify helper on the real `test5_fisheyePinhole` sparse model: output sparse expanded from 378 registered reference images to 3024 registered images across all 16 views.

### Phase 4: Resume Mode

- [x] Prefer resuming from `colmap/images/`.
- [x] Fall back to old root `images/`.
- [x] When resuming from already-flattened images, stage reference views with stripped basenames so COLMAP still sees `front_ctr_hi/000001.jpg`, not doubled names.
- [x] Normalize output back into `colmap/images/` and `colmap/masks/`.
- [x] Write `colmap/transforms.json`.
- [x] Rewrite `colmap/sparse/0` as the propagated all-view sparse model.

### Phase 5: Validation

- [x] Syntax check modified Python files with system Python 3.12.
- [ ] Run focused tests.
- [ ] Run a real Fisheye (Pinhole) test.
- [ ] Confirm `output/colmap/` loads in LFS.
- [ ] Confirm training mask lookup uses `output/colmap/masks/`.
- [ ] Confirm `output/colmap/sparse/0` loads without image-not-found errors.

## Current Blocker

The local `.venv` was damaged during validation and currently lacks `pyvenv.cfg` / package contents. Tracked plugin files were recovered from Git, but validation that depends on the venv must wait until the environment is restored or run through another known-good Python.

The restored commit also does not contain a tracked `tests/` directory, so focused pytest validation cannot run from this checkout until tests are restored or recreated.
