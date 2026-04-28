# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write transforms.json for LichtFeld Studio with COLMAP-derived poses.

Handles coordinate conversion from COLMAP (OpenCV, world-to-camera) to
LichtFeld's transforms.json format (OpenGL, camera-to-world with 180 deg Y
pre-compensation).

Two writers live here:
  - write_transforms_json: shared-intrinsics writer (used by the ERP path
    via scaffold.py's own inline writer; also kept for any future callers
    that need top-level fl_x/fl_y/cx/cy).
  - write_fisheye_transforms: per-frame intrinsics + applied_transform +
    pointcloud, for the dual fisheye native-output path. Each lens has
    different OPENCV_FISHEYE intrinsics so each frame entry carries its
    own fl_x/fl_y/cx/cy/k1-k4.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Coordinate conversion shared with scaffold.py — kept duplicated here
# rather than imported to avoid coupling fisheye output to the ERP module.
# If you change one, change the other.
_M = np.diag([1.0, -1.0, -1.0, 1.0])         # OpenCV → OpenGL world flip
_Ry180 = np.diag([-1.0, 1.0, -1.0, 1.0])     # cancels LFS loader 180° Y
_APPLIED_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
]


def _c2w_opencv_to_lfs(c2w: np.ndarray) -> np.ndarray:
    """Convert an OpenCV c2w 4x4 to LFS transforms.json convention.

    Steps (mirrors scaffold._c2w_opencv_to_lfs):
      1. Left-multiply by diag(1,-1,-1,1) — world OpenCV → OpenGL
      2. Negate columns 1,2 — camera local axes OpenCV → OpenGL
      3. Left-multiply Ry(180°) — pre-compensate LFS loader Y rotation
    """
    t = _M @ c2w
    t[:3, 1:3] *= -1
    t = _Ry180 @ t
    return t


def colmap_pose_to_c2w_opengl(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Convert COLMAP world-to-camera (R, t) to 4x4 camera-to-world in OpenGL convention.

    COLMAP stores poses as world-to-camera: p_cam = R @ p_world + t,
    in OpenCV convention (Y down, Z forward).

    LichtFeld's transforms.json expects camera-to-world matrices in OpenGL
    convention (Y up, Z back), with a 180 deg Y pre-compensation to cancel
    the rotation that the loader applies internally.

    Conversion steps:
        1. Invert w2c to c2w: R_c2w = R^T, t_c2w = -R^T @ t
        2. Build 4x4 matrix
        3. OpenCV -> OpenGL: flip Y and Z columns (c2w[:3, 1:3] *= -1)
        4. Pre-compensate for loader's 180 deg Y rotation: left-multiply by diag(-1, 1, -1, 1)

    Args:
        R_w2c: 3x3 rotation matrix (world-to-camera).
        t_w2c: 3-element translation vector (world-to-camera).

    Returns:
        4x4 camera-to-world matrix in OpenGL convention with Y pre-compensation.
    """
    # Step 1: Invert w2c to c2w
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    # Step 2: Build 4x4
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w

    # Step 3: OpenCV -> OpenGL: flip Y and Z columns
    c2w[:3, 1:3] *= -1

    # Step 4: 180 deg Y pre-compensation (cancels loader's internal rotation)
    y180 = np.diag([-1.0, 1.0, -1.0, 1.0])
    c2w = y180 @ c2w

    return c2w


def write_transforms_json(
    output_path: str | Path,
    camera_model: str,
    w: int,
    h: int,
    fl_x: float,
    fl_y: float,
    frames: list[dict],
    cx: float | None = None,
    cy: float | None = None,
    ply_file_path: str | None = None,
) -> None:
    """Write a transforms.json file in LichtFeld Studio format.

    Args:
        output_path: Destination file path.
        camera_model: Camera model string (e.g. "EQUIRECTANGULAR", "PINHOLE").
        w: Image width in pixels.
        h: Image height in pixels.
        fl_x: Focal length X. For ERP: w / 2.
        fl_y: Focal length Y. For ERP: w / 2 (= h for 2:1 aspect).
        frames: List of frame dicts, each with "file_path" and "transform_matrix".
        cx: Principal point X. For ERP: w / 2.
        cy: Principal point Y. For ERP: h / 2.
        ply_file_path: Optional relative path to a point cloud PLY file.
    """
    data = {
        "camera_model": camera_model,
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "frames": frames,
    }
    if cx is not None:
        data["cx"] = cx
    if cy is not None:
        data["cy"] = cy
    if ply_file_path is not None:
        data["ply_file_path"] = ply_file_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Dual fisheye: per-frame intrinsics + sparse pointcloud
# ---------------------------------------------------------------------------

def _write_sparse_pointcloud(recon, ply_path: Path) -> int:
    """Write the COLMAP sparse points as a coloured PLY in OpenGL world frame.

    Mirrors scaffold._write_pointcloud — duplicated to avoid importing from
    scaffold (which is ERP-specific in name). World-frame points get the
    same diag(1,-1,-1) flip as the c2w matrices.
    """
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
        for pid in point_ids:
            point3d = recon.point3D(pid)
            xyz = np.asarray(point3d.xyz, dtype=np.float64)
            color = np.asarray(point3d.color, dtype=np.uint8)
            # OpenCV → OpenGL: negate Y and Z
            handle.write(
                f"{xyz[0]:.9f} {-xyz[1]:.9f} {-xyz[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    return len(point_ids)


def write_fisheye_transforms(
    colmap_sparse_dir: str | Path,
    images_root: str | Path,
    output_dir: str | Path,
    *,
    transforms_filename: str = "transforms.json",
    ply_filename: str = "pointcloud.ply",
    reconstruction_obj=None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Write a fisheye-native transforms.json with per-frame intrinsics.

    Reads a COLMAP reconstruction produced by the dual fisheye pipeline
    (CameraMode.PER_FOLDER + camera_model="OPENCV_FISHEYE", typically two
    cameras: one for `images/front/` and one for `images/back/`). For each
    registered image, extracts:

        - camera-to-world pose (converted to LFS OpenGL + Y pre-comp)
        - per-frame OPENCV_FISHEYE intrinsics (fx, fy, cx, cy, k1-k4)
        - relative `images/<lens>/<frame>.jpg` path

    Then writes:
        - <output_dir>/<transforms_filename> with applied_transform + frames
        - <output_dir>/<ply_filename> with the sparse points

    The per-frame intrinsics pattern follows scaffold.py's existing approach
    — each frame entry carries its own w/h/fl_x/fl_y/cx/cy. Front and back
    lenses have different calibrations, so this is the right shape rather
    than top-level intrinsics.

    Args:
        colmap_sparse_dir: Directory containing the COLMAP sparse model
            (typically `<output>/sparse/0/`).
        images_root: Directory containing the staged images (typically
            `<output>/images/`). Used only as a sanity-check root —
            file_path entries in the JSON are recorded as
            `images/<lens>/<frame>.jpg` regardless of where this points.
        output_dir: Where to write transforms.json + pointcloud.ply.
        transforms_filename: Name of the transforms file (default
            "transforms.json").
        ply_filename: Name of the sparse PLY (default "pointcloud.ply").
        reconstruction_obj: Optional pre-loaded `pycolmap.Reconstruction`.
            If None, the reconstruction is loaded from `colmap_sparse_dir`.
        log_fn: Logging callback.

    Returns:
        Path to the written transforms.json.

    Raises:
        RuntimeError: If the reconstruction has no registered images, or
            if a non-OPENCV_FISHEYE camera model is encountered (we only
            know how to serialize OPENCV_FISHEYE's 8 params).
    """
    import pycolmap

    sparse_dir = Path(colmap_sparse_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if reconstruction_obj is not None:
        recon = reconstruction_obj
    else:
        log_fn("Loading COLMAP reconstruction from %s", sparse_dir)
        recon = pycolmap.Reconstruction(str(sparse_dir))

    reg_image_ids = sorted(recon.reg_image_ids())
    if not reg_image_ids:
        raise RuntimeError(
            "COLMAP reconstruction contains no registered images — "
            "cannot write fisheye transforms"
        )

    log_fn("Building fisheye transforms.json from %d registered images",
           len(reg_image_ids))

    # Cache per-camera intrinsics so we don't re-read them per frame.
    # Each entry: {"w","h","fl_x","fl_y","cx","cy","k1","k2","k3","k4"}.
    camera_intrinsics: dict[int, dict] = {}
    for cam_id, cam in recon.cameras.items():
        model_name = cam.model_name  # property in pycolmap 4.0
        if model_name != "OPENCV_FISHEYE":
            raise RuntimeError(
                f"Camera {cam_id} has model {model_name!r}; "
                "write_fisheye_transforms only supports OPENCV_FISHEYE"
            )
        params = list(cam.params)
        if len(params) != 8:
            raise RuntimeError(
                f"OPENCV_FISHEYE expects 8 params, got {len(params)} on "
                f"camera {cam_id}"
            )
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        camera_intrinsics[cam_id] = {
            "w": int(cam.width),
            "h": int(cam.height),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": float(k1),
            "k2": float(k2),
            "k3": float(k3),
            "k4": float(k4),
        }

    log_fn("Found %d unique cameras: %s",
           len(camera_intrinsics), sorted(camera_intrinsics.keys()))

    # Build per-frame entries.
    frames: list[dict] = []
    for img_id in reg_image_ids:
        image = recon.image(img_id)
        cam_from_world = image.cam_from_world()  # Rigid3d, OpenCV w2c
        R_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(cam_from_world.translation, dtype=np.float64)

        # Invert to c2w in OpenCV convention, then convert to LFS OpenGL.
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        c2w_opencv = np.eye(4, dtype=np.float64)
        c2w_opencv[:3, :3] = R_c2w
        c2w_opencv[:3, 3] = t_c2w
        c2w = _c2w_opencv_to_lfs(c2w_opencv)

        # COLMAP image names already include the subfolder prefix (e.g.
        # "front/frame_0001.jpg"); prepend "images/" for the JSON.
        # Normalise backslashes in case of Windows-style paths.
        rel_image_path = image.name.replace("\\", "/")
        if not rel_image_path.startswith("images/"):
            rel_image_path = f"images/{rel_image_path}"

        intr = camera_intrinsics[image.camera_id]
        entry = {
            "file_path": rel_image_path,
            "transform_matrix": c2w.tolist(),
            **intr,  # w, h, fl_x, fl_y, cx, cy, k1, k2, k3, k4
        }
        frames.append(entry)

    # Sort by file_path so front/ frames precede back/ deterministically and
    # within each lens the frame numbers run in order.
    frames.sort(key=lambda e: e["file_path"])

    # Sparse pointcloud
    ply_path = out_dir / ply_filename
    point_count = _write_sparse_pointcloud(recon, ply_path)
    log_fn("Wrote sparse pointcloud (%d points): %s", point_count, ply_path)

    # transforms.json
    transforms_path = out_dir / transforms_filename
    transforms_data = {
        "camera_model": "OPENCV_FISHEYE",
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": frames,
    }
    with transforms_path.open("w", encoding="utf-8") as fp:
        json.dump(transforms_data, fp, indent=4)
    log_fn(
        "Wrote fisheye transforms.json with %d frames across %d cameras: %s",
        len(frames), len(camera_intrinsics), transforms_path,
    )

    return transforms_path
