# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""ERP Scaffold export — extract rig-origin poses from a COLMAP rig
reconstruction and write a transforms.json for equirectangular training.

The pinhole crops used for COLMAP alignment are scaffolding: this module
recovers the per-station rig pose, converts coordinates, and outputs the
original ERP frames with spherical camera metadata that LichtFeld Studio's
3DGUT trainer can consume directly.  After export, the pinhole artifacts
are deleted.

Coordinate conversion from COLMAP (OpenCV) to LFS (OpenGL + Y pre-comp):
  Cameras:     world flip diag(1,-1,-1) → col flip → Ry(180°) pre-comp
  Point cloud: world flip diag(1,-1,-1)
  applied_transform: diag(1,-1,-1)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# World transform: OpenCV (Y down, Z forward) → OpenGL (Y up, Z back).
_M = np.diag([1.0, -1.0, -1.0, 1.0])

# Y pre-compensation — LFS applies a 180° Y rotation on load.
_Ry180 = np.diag([-1.0, 1.0, -1.0, 1.0])

# applied_transform records the point cloud transform for LFS.
_APPLIED_TRANSFORM = [[1.0, 0.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 0.0]]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def _c2w_opencv_to_lfs(c2w: np.ndarray) -> np.ndarray:
    """Convert an OpenCV c2w matrix to LFS transforms.json convention.

    Steps:
      1. Left-multiply by diag(1,-1,-1,1) — world OpenCV → OpenGL
      2. Negate columns 1,2 — camera local axes OpenCV → OpenGL
      3. Left-multiply Ry(180°) — pre-compensate LFS loader Y rotation
    """
    t = _M @ c2w                   # world to OpenGL
    t[:3, 1:3] *= -1               # camera axes to OpenGL
    t = _Ry180 @ t                 # Y pre-compensation
    return t


# ---------------------------------------------------------------------------
# Point cloud export
# ---------------------------------------------------------------------------

def _write_pointcloud(recon, ply_path: Path) -> int:
    """Write a sparse point cloud in OpenGL world coordinates."""
    point_ids = sorted(recon.point3D_ids())
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    with ply_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(point_ids)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")

        for point_id in point_ids:
            point3d = recon.point3D(point_id)
            xyz = np.asarray(point3d.xyz, dtype=np.float64)
            color = np.asarray(point3d.color, dtype=np.uint8)
            # OpenCV → OpenGL world: negate Y and Z
            handle.write(
                f"{xyz[0]:.9f} {-xyz[1]:.9f} {-xyz[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    return len(point_ids)


# ---------------------------------------------------------------------------
# Frame-ID to ERP filename mapping
# ---------------------------------------------------------------------------

def _frame_stem_from_image_name(name: str) -> str:
    """Extract the shared frame stem from a pinhole image name.

    Pinhole images are stored as ``{view_folder}/{station}.jpg``,
    e.g. ``00_00/frame_0001.jpg``.  The stem (``frame_0001``) is
    shared with the source ERP file in ``extracted/frames/``.
    """
    return Path(name).stem


def _find_erp_file(frame_stem: str, erp_dir: Path) -> Optional[Path]:
    """Locate the ERP source file matching *frame_stem*."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = erp_dir / f"{frame_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_erp_scaffold(
    colmap_sparse_dir: Path,
    erp_frames_dir: Path,
    erp_masks_dir: Optional[Path],
    output_dir: Path,
    erp_width: int,
    erp_height: int,
    ref_pitch_deg: float = -35.0,
    reconstruction_obj=None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Read a COLMAP rig reconstruction and write an ERP transforms.json.

    Moves ERP frames from *erp_frames_dir* into ``output_dir/images/``
    (and masks into ``output_dir/masks/`` when masking is enabled) so
    that LichtFeld Studio finds the ``images/`` directory it requires.

    Args:
        colmap_sparse_dir: Path containing the COLMAP sparse model
            (cameras.txt / images.txt / points3D.txt or binary equivalents).
        erp_frames_dir: Directory with the extracted ERP source frames.
            Will be renamed to ``output_dir/images/``.
        erp_masks_dir: Directory with ERP masks, or *None* if masking is off.
            Will be renamed to ``output_dir/masks/`` when provided.
        output_dir: Root output directory.  The function writes
            ``pointcloud.ply`` and ``transforms.json`` here.
        erp_width: Width of ERP frames in pixels.
        erp_height: Height of ERP frames in pixels.
        ref_pitch_deg: Pitch of the reference sensor in degrees (used for
            rig-to-ERP orientation correction).
        reconstruction_obj: Optional pre-loaded ``pycolmap.Reconstruction``.
            If *None*, the reconstruction is loaded from *colmap_sparse_dir*.
        log_fn: Logging callback.

    Returns:
        Path to the written ``transforms.json``.
    """
    import pycolmap

    # ------------------------------------------------------------------
    # 1. Load reconstruction
    # ------------------------------------------------------------------
    if reconstruction_obj is not None:
        recon = reconstruction_obj
    else:
        log_fn("Loading COLMAP reconstruction from %s", colmap_sparse_dir)
        recon = pycolmap.Reconstruction(str(colmap_sparse_dir))

    # ------------------------------------------------------------------
    # 2. Extract rig-origin poses via the frame API
    # ------------------------------------------------------------------
    reg_frame_ids = sorted(recon.reg_frame_ids())
    if not reg_frame_ids:
        raise RuntimeError(
            "COLMAP reconstruction contains no registered frames — "
            "cannot export ERP scaffold"
        )

    # Pitch correction: the rig's forward direction equals the reference
    # sensor's forward, which points at ref_pitch_deg (e.g. -35°) below
    # the ERP image center.  Rotate the rig pose so that forward aligns
    # with the ERP center (pitch=0°).
    correction_rad = np.radians(-ref_pitch_deg)
    cos_p = np.cos(correction_rad)
    sin_p = np.sin(correction_rad)
    Rx_correction = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_p, -sin_p],
        [0.0, sin_p, cos_p],
    ])

    poses: dict[str, np.ndarray] = {}
    skipped_no_pose = 0
    skipped_no_erp = 0

    for fid in reg_frame_ids:
        frame = recon.frame(fid)
        rig_from_world = frame.rig_from_world
        if rig_from_world is None:
            skipped_no_pose += 1
            continue

        R_w2c = np.asarray(rig_from_world.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(rig_from_world.translation, dtype=np.float64)

        # Apply pitch correction: erp_from_world = Rx @ rig_from_world
        R_w2c = Rx_correction @ R_w2c
        t_w2c = Rx_correction @ t_w2c

        # Invert w2c → c2w (OpenCV convention)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        c2w_opencv = np.eye(4, dtype=np.float64)
        c2w_opencv[:3, :3] = R_c2w
        c2w_opencv[:3, 3] = t_c2w

        # Apply the same conversion chain as the working Metashape export
        c2w = _c2w_opencv_to_lfs(c2w_opencv)

        # Identify the frame stem from any image belonging to this frame.
        image_ids = frame.image_ids
        if not image_ids:
            skipped_no_pose += 1
            continue

        # Use the first image's name to derive the shared frame stem.
        first_image = recon.image(image_ids[0].id)
        frame_stem = _frame_stem_from_image_name(first_image.name)

        # Verify the ERP source file exists before recording.
        erp_file = _find_erp_file(frame_stem, erp_frames_dir)
        if erp_file is None:
            skipped_no_erp += 1
            continue

        poses[frame_stem] = c2w

    if not poses:
        raise RuntimeError(
            f"No ERP frames could be matched to registered COLMAP frames "
            f"(skipped_no_pose={skipped_no_pose}, skipped_no_erp={skipped_no_erp})"
        )

    log_fn(
        "Extracted %d rig-origin poses from %d registered frames "
        "(skipped: %d no pose, %d no ERP match)",
        len(poses), len(reg_frame_ids), skipped_no_pose, skipped_no_erp,
    )

    # ------------------------------------------------------------------
    # 3. Move ERP frames/masks to top-level images/ and masks/
    # ------------------------------------------------------------------
    images_dir = output_dir / "images"
    if not images_dir.exists():
        erp_frames_dir.rename(images_dir)
        log_fn("Moved %s → %s", erp_frames_dir, images_dir)

    masks_dir: Optional[Path] = None
    if erp_masks_dir is not None and erp_masks_dir.is_dir():
        masks_dir = output_dir / "masks"
        if not masks_dir.exists():
            erp_masks_dir.rename(masks_dir)
            log_fn("Moved %s → %s", erp_masks_dir, masks_dir)

    # Clean up empty extracted/ if both subdirs were moved out
    extracted_dir = output_dir / "extracted"
    if extracted_dir.is_dir() and not any(extracted_dir.iterdir()):
        extracted_dir.rmdir()

    # ------------------------------------------------------------------
    # 4. Export sparse point cloud
    # ------------------------------------------------------------------
    ply_path = output_dir / "pointcloud.ply"
    point_count = _write_pointcloud(recon, ply_path)
    log_fn("Exported ERP sparse point cloud (%d points): %s", point_count, ply_path)

    # ------------------------------------------------------------------
    # 5. Build frame list and write transforms.json
    # ------------------------------------------------------------------
    fl_x = erp_width / 2.0
    fl_y = erp_width / 2.0
    cx = erp_width / 2.0
    cy = erp_height / 2.0

    frames_list: list[dict] = []
    for frame_stem in sorted(poses):
        erp_file = _find_erp_file(frame_stem, images_dir)
        if erp_file is None:
            continue

        entry: dict = {
            "file_path": f"images/{erp_file.name}",
            "transform_matrix": poses[frame_stem].tolist(),
            "w": erp_width,
            "h": erp_height,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
        }

        if masks_dir is not None:
            mask_file = _find_erp_file(frame_stem, masks_dir)
            if mask_file is not None:
                entry["mask_path"] = f"masks/{mask_file.name}"

        frames_list.append(entry)

    transforms_path = output_dir / "transforms.json"
    transforms_data = {
        "camera_model": "EQUIRECTANGULAR",
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": "pointcloud.ply",
        "frames": frames_list,
    }

    with open(transforms_path, "w") as f:
        json.dump(transforms_data, f, indent=4)

    log_fn(
        "Wrote ERP scaffold transforms.json with %d frames (%dx%d)",
        len(frames_list), erp_width, erp_height,
    )

    return transforms_path


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def cleanup_pinhole_crops(output_dir: Path, log_fn: Callable[..., object] = logger.info) -> None:
    """Move pinhole image/mask directories out of the way before ERP export.

    Called before ``export_erp_scaffold`` so that ``extracted/frames/``
    can be renamed to ``images/``.  Pinhole crops are kept under
    ``pinhole_images/`` and ``pinhole_masks/`` for debugging.
    """
    for item_name in ("images", "masks"):
        src = output_dir / item_name
        dst = output_dir / f"pinhole_{item_name}"
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            src.rename(dst)
    log_fn("Moved pinhole crops to pinhole_images/ and pinhole_masks/ in %s", output_dir)


def cleanup_colmap_artifacts(output_dir: Path, log_fn: Callable[..., object] = logger.info) -> None:
    """Delete remaining COLMAP artifacts after ERP export.

    Called after ``export_erp_scaffold`` has read the reconstruction.
    Removes ``sparse/`` (which triggers LFS pinhole auto-detection),
    the feature database, and other intermediate files.
    """
    for item_name in ("database.db", "colmap_debug.log"):
        target = output_dir / item_name
        if not target.exists():
            continue
        target.unlink(missing_ok=True)
    log_fn("Removed COLMAP database from %s (kept sparse/ and rig_config.json)", output_dir)
