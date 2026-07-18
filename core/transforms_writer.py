# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write transforms.json for LichtFeld Studio with COLMAP-derived poses.

Handles coordinate conversion from COLMAP (OpenCV, world-to-camera) to
LichtFeld's transforms.json format (OpenGL, camera-to-world with 180 deg Y
pre-compensation).

Two writers live here:
  - write_transforms_json: shared-intrinsics writer for callers that need
    top-level fl_x/fl_y/cx/cy.
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

# Coordinate conversion kept self-contained here (no cross-module import)
# so fisheye output has no dependency on the ERP output path.
_M = np.diag([1.0, -1.0, -1.0, 1.0])         # OpenCV → OpenGL world flip
_Ry180 = np.diag([-1.0, 1.0, -1.0, 1.0])     # cancels LFS loader 180° Y
_APPLIED_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
]


def _c2w_opencv_to_lfs(c2w: np.ndarray) -> np.ndarray:
    """Convert an OpenCV c2w 4x4 to LFS transforms.json convention.

    Steps:
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


def _fisheye_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation matrix for fisheye virtual camera yaw/pitch."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float64)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float64)
    return ry @ rx


def _erp_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Pure-numpy copy of ``core.reframer.create_rotation_matrix``.

    Duplicated here (like ``_fisheye_rotation_matrix``) so this module stays
    free of the cv2 import that ``core.reframer`` carries. ``rows = [right,
    up, -forward]`` — an OpenGL world-to-camera. ``tests/test_erp_propagation``
    imports the REAL reframer function and asserts they agree, so any drift
    fails loudly.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    fwd = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw),
    ])
    r = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    rl = np.linalg.norm(r)
    r = np.array([1.0, 0.0, 0.0]) if rl < 1e-6 else r / rl
    u = np.cross(r, fwd)
    return np.array([r, u, -fwd])


def erp_view_rotation(
    yaw_deg: float, pitch_deg: float, flip_v: bool = False
) -> np.ndarray:
    """Camera-to-camera rotation for an ERP pinhole crop (propagation).

    Returns ``R_rel`` such that ``R_v = R_rel @ R_w2c`` is the crop's OpenCV
    world-to-camera rotation, given the native EQUIRECTANGULAR camera's
    world-to-camera ``R_w2c`` — the exact composition the proven fisheye twin
    ``write_native_propagated_transforms`` uses.

    Correct for the reframer's DEFAULT (fliplr'd) crop. The requirement, in
    COLMAP's world frame, is:

        R_rel.T @ opencv_ray(px,py) == F @ reframe_dir(px,py)

    where ``reframe_dir`` is the direction the crop pixel samples (in the
    reframer's own frame) and ``F = diag(1,-1,1)`` maps the reframer's y-up
    equirect frame to COLMAP's y-down EQUIRECTANGULAR frame. F was MEASURED
    against pycolmap on a real native solve (QA-C2 2026-07-15), exact to 1.7e-16.

    The reframer's always-on ``fliplr`` and COLMAP's y-down convention are two
    reflections that cancel, so the fliplr'd crop composes to a PROPER rotation:

        R_rel = create_rotation_matrix(yaw,pitch) @ diag(-1,1,-1)     (det = +1)

    Verified machine-exact in ``tests/test_erp_propagation`` (the intrinsics'
    principal point cx=cy=size/2 carries a documented <=1px offset from the
    fliplr, negligible for training; the rotation itself is exact).

    ``flip_v`` views (only reachable via a custom ``Ring(flip_vertical=True)``;
    no built-in preset uses them) add a second, uncancelled reflection and are
    NOT representable by a proper rotation, so this raises rather than emit a
    silently-mirrored pose.
    """
    if flip_v:
        raise NotImplementedError(
            "ERP pinhole propagation does not support flip_vertical (cubemap "
            "pole) views: the crop would be an improper reflection in COLMAP's "
            "frame. No built-in ERP preset uses flip_vertical."
        )
    R = _erp_rotation_matrix(yaw_deg, pitch_deg)
    return R @ np.diag([-1.0, 1.0, -1.0])


def _erp_crop_intrinsics(output_size: int, fov_deg: float) -> dict:
    """PINHOLE intrinsics for an ERP-propagated crop, half-integer-exact.

    LFS's rasterizer samples pixel centres at half-integer coordinates
    (RasterizeToPixelsFromWorld3DGSFwd.cu: ``px = j + 0.5f``), and COLMAP uses
    the same convention, so the principal point must be expressed in it. The
    reframe formula puts the optical axis through column/row INDEX ``W/2``;
    the reframer's always-on fliplr mirrors columns about ``(W-1)/2``, moving
    the axis to column index ``W/2 - 1``. Index ``i`` has its centre at
    continuous ``i + 0.5``, hence:

        cx = W/2 - 0.5   (flipped horizontal axis)
        cy = W/2 + 0.5   (unflipped vertical axis)

    (Identified in the 2026-07-15 adversarial review; ``W/2`` in both was a
    ~0.7px constant offset.)
    """
    fl = (output_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return {
        "w": int(output_size),
        "h": int(output_size),
        "fl_x": float(fl),
        "fl_y": float(fl),
        "cx": output_size / 2.0 - 0.5,
        "cy": output_size / 2.0 + 0.5,
    }


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
    # newline="\n": force LF endings. On Windows the default text mode writes
    # CRLF, and LFS's transforms loader reads in text mode then compares the
    # byte count to the on-disk stat size — CRLF stripping makes them differ
    # and it aborts with "Transforms JSON changed size while it was being read"
    # (transforms.cpp:288). LF-only keeps the two counts equal.
    with open(output_path, "w", newline="\n") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Dual fisheye: per-frame intrinsics + sparse pointcloud
# ---------------------------------------------------------------------------

def _write_sparse_pointcloud(recon, ply_path: Path) -> int:
    """Write the COLMAP sparse points as a coloured PLY in OpenGL world frame.

    Kept self-contained here (no cross-module import). World-frame points
    get the same diag(1,-1,-1) flip as the c2w matrices.
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
    masks_dir: str | Path | None = None,
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

    The per-frame intrinsics pattern gives each frame entry its own
    w/h/fl_x/fl_y/cx/cy. Front and back lenses have different calibrations,
    so this is the right shape rather than top-level intrinsics.

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
    detected_model: str = ""
    for cam_id, cam in recon.cameras.items():
        model_name = cam.model_name  # property in pycolmap 4.0
        params = list(cam.params)
        if model_name == "OPENCV_FISHEYE":
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
            detected_model = "OPENCV_FISHEYE"
        elif model_name == "PINHOLE":
            if len(params) != 4:
                raise RuntimeError(
                    f"PINHOLE expects 4 params, got {len(params)} on "
                    f"camera {cam_id}"
                )
            fx, fy, cx, cy = params
            camera_intrinsics[cam_id] = {
                "w": int(cam.width),
                "h": int(cam.height),
                "fl_x": float(fx),
                "fl_y": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }
            detected_model = "PINHOLE"
        else:
            raise RuntimeError(
                f"Camera {cam_id} has model {model_name!r}; "
                "write_fisheye_transforms supports OPENCV_FISHEYE and PINHOLE"
            )

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
        if masks_dir is not None:
            rel_no_prefix = rel_image_path.removeprefix("images/")
            mask_rel = Path(rel_no_prefix).with_suffix(".png")
            if (Path(masks_dir) / mask_rel).exists():
                entry["mask_path"] = f"masks/{mask_rel.as_posix()}"
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
        "camera_model": detected_model,
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": frames,
    }
    with transforms_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(transforms_data, fp, indent=4)
    log_fn(
        "Wrote fisheye transforms.json with %d frames across %d cameras: %s",
        len(frames), len(camera_intrinsics), transforms_path,
    )

    return transforms_path


def write_erp_native_transforms(
    colmap_sparse_dir: str | Path,
    output_dir: str | Path,
    *,
    masks_dir: str | Path | None = None,
    erp_width: int = 0,
    erp_height: int = 0,
    transforms_filename: str = "transforms.json",
    ply_filename: str = "pointcloud.ply",
    reconstruction_obj=None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Write a native ERP transforms.json from an EQUIRECTANGULAR COLMAP model.

    Per-image poses use ``image.cam_from_world()`` converted via
    ``_c2w_opencv_to_lfs``. No pitch correction or auto-orient.
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
            "cannot write native ERP transforms"
        )

    log_fn(
        "Building native ERP transforms.json from %d registered images",
        len(reg_image_ids),
    )

    camera_dims: dict[int, tuple[int, int]] = {}
    for cam_id, cam in recon.cameras.items():
        model_name = cam.model_name
        if model_name != "EQUIRECTANGULAR":
            raise RuntimeError(
                f"Camera {cam_id} has model {model_name!r}; "
                "write_erp_native_transforms expects EQUIRECTANGULAR"
            )
        params = list(cam.params)
        w = int(params[0]) if len(params) >= 1 else int(cam.width)
        h = int(params[1]) if len(params) >= 2 else int(cam.height)
        camera_dims[cam_id] = (w, h)

    frames: list[dict] = []
    for img_id in reg_image_ids:
        image = recon.image(img_id)
        cam_from_world = image.cam_from_world()
        R_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(cam_from_world.translation, dtype=np.float64)

        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        c2w_opencv = np.eye(4, dtype=np.float64)
        c2w_opencv[:3, :3] = R_c2w
        c2w_opencv[:3, 3] = t_c2w
        c2w = _c2w_opencv_to_lfs(c2w_opencv)

        w, h = camera_dims.get(image.camera_id, (erp_width, erp_height))
        if w <= 0 or h <= 0:
            raise RuntimeError(
                "ERP frame dimensions could not be determined from the "
                "EQUIRECTANGULAR camera or extraction metadata"
            )

        rel_image_path = image.name.replace("\\", "/")
        if not rel_image_path.startswith("images/"):
            rel_image_path = f"images/{rel_image_path}"

        fl_x = w / 2.0
        fl_y = w / 2.0
        cx = w / 2.0
        cy = h / 2.0

        entry: dict = {
            "file_path": rel_image_path,
            "transform_matrix": c2w.tolist(),
            "w": w,
            "h": h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
        }

        if masks_dir is not None:
            stem = Path(rel_image_path.removeprefix("images/")).stem
            mask_file = Path(masks_dir) / f"{stem}.png"
            if mask_file.exists():
                entry["mask_path"] = f"masks/{stem}.png"

        frames.append(entry)

    frames.sort(key=lambda e: e["file_path"])

    ply_path = out_dir / ply_filename
    point_count = _write_sparse_pointcloud(recon, ply_path)
    log_fn("Wrote sparse pointcloud (%d points): %s", point_count, ply_path)

    transforms_path = out_dir / transforms_filename
    transforms_data = {
        "camera_model": "EQUIRECTANGULAR",
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": frames,
    }
    with transforms_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(transforms_data, fp, indent=4)
    log_fn(
        "Wrote native ERP transforms.json with %d frames (%dx%d): %s",
        len(frames), w, h, transforms_path,
    )

    return transforms_path


def write_erp_propagated_transforms(
    colmap_sparse_dir: str | Path,
    output_dir: str | Path,
    view_config,
    *,
    output_size: int,
    images_prefix: str = "images",
    masks_dir: str | Path | None = None,
    transforms_filename: str = "transforms.json",
    ply_filename: str = "pointcloud.ply",
    propagated_sparse_output_dir: str | Path | None = None,
    reconstruction_obj=None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Export pinhole crops propagated from a native ERP (EQUIRECTANGULAR) solve.

    The ERP twin of ``write_native_propagated_transforms``. Every crop shares
    the native ERP camera's optical centre, so the pose is a pure rotation of
    the native pose: ``R_v = erp_view_rotation(view) @ R_w2c``,
    ``t_v = erp_view_rotation(view) @ t_w2c`` (see ``erp_view_rotation``).

    This writer emits POSES only. The crop images and per-view masks are
    produced by ``Reframer.reframe_batch`` and then FLATTENED to
    ``<output_dir>/<images_prefix>/<view>_<stem>.jpg`` and
    ``<output_dir>/masks/<view>_<stem>.png`` (``_flatten_view_folders``) —
    LFS's transforms loader resolves masks by bare image filename only, so
    basenames must be unique across the dataset.

    Args:
        colmap_sparse_dir: Native ERP COLMAP sparse model (EQUIRECTANGULAR).
        output_dir: Where to write transforms.json + pointcloud.ply.
        view_config: A ``presets.ViewConfig``; ``get_all_views()`` supplies
            (yaw, pitch, fov, name, flip_v) per view.
        output_size: Square crop side length (px) reframe_batch produced.
        images_prefix: Path prefix before ``<view>/<stem>.jpg`` (default
            ``images``).
        masks_dir: If given, a per-view masks root; a crop gets a
            ``mask_path`` only when ``masks_dir/<view>/<stem>.png`` exists.
        propagated_sparse_output_dir: Optional sparse/0 to write with all
            propagated pinhole images. Defaults to ``<output_dir>/sparse/0``.
        reconstruction_obj: Optional pre-loaded ``pycolmap.Reconstruction``.
        log_fn: Logging callback.

    Returns:
        Path to the written transforms.json.

    Raises:
        RuntimeError: no registered images, or a non-EQUIRECTANGULAR camera.
        NotImplementedError: a flip_vertical view (see ``erp_view_rotation``).
    """
    import pycolmap

    sparse_dir = Path(colmap_sparse_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if reconstruction_obj is not None:
        recon = reconstruction_obj
    else:
        log_fn("Loading native ERP reconstruction from %s", sparse_dir)
        recon = pycolmap.Reconstruction(str(sparse_dir))

    reg_image_ids = sorted(recon.reg_image_ids())
    if not reg_image_ids:
        raise RuntimeError(
            "Native ERP reconstruction contains no registered images — "
            "cannot export propagated pinhole crops"
        )

    for cam_id, cam in recon.cameras.items():
        if cam.model_name != "EQUIRECTANGULAR":
            raise RuntimeError(
                f"Camera {cam_id} has model {cam.model_name!r}; "
                "write_erp_propagated_transforms expects EQUIRECTANGULAR"
            )

    views = view_config.get_all_views()
    if not views:
        raise RuntimeError("View config produced no views for ERP propagation")

    masks_dir_path = Path(masks_dir) if masks_dir is not None else None

    def _intrinsics(fov_deg: float) -> dict:
        return _erp_crop_intrinsics(output_size, fov_deg)

    frames: list[dict] = []
    sparse_entries: list[dict] = []
    rendered_masks = 0

    for img_id in reg_image_ids:
        image = recon.image(img_id)
        cam_from_world = image.cam_from_world()
        R_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(cam_from_world.translation, dtype=np.float64)
        stem = Path(image.name.replace("\\", "/")).stem

        for yaw, pitch, fov, view_name, flip_v in views:
            R_rel = erp_view_rotation(yaw, pitch, flip_v)
            R_v = R_rel @ R_w2c
            t_v = R_rel @ t_w2c

            R_c2w = R_v.T
            t_c2w = -R_v.T @ t_v
            c2w_opencv = np.eye(4, dtype=np.float64)
            c2w_opencv[:3, :3] = R_c2w
            c2w_opencv[:3, 3] = t_c2w
            c2w_lfs = _c2w_opencv_to_lfs(c2w_opencv)

            # Flat, view-prefixed names (fisheye propagation convention).
            # LFS's transforms loader matches masks by bare image filename
            # only, so basenames must be unique across the whole dataset.
            flat_name = f"{view_name}_{stem}"
            intr = _intrinsics(fov)
            entry: dict = {
                "file_path": f"{images_prefix}/{flat_name}.jpg",
                "transform_matrix": c2w_lfs.tolist(),
                **intr,
            }
            if masks_dir_path is not None:
                mask_file = masks_dir_path / f"{flat_name}.png"
                if mask_file.exists():
                    entry["mask_path"] = f"masks/{flat_name}.png"
                    rendered_masks += 1
            frames.append(entry)

            sparse_entries.append({
                "name": f"{flat_name}.jpg",
                "view_name": view_name,
                "intrinsics": intr,
                "R_w2c": R_v.copy(),
                "t_w2c": t_v.copy(),
            })

    if not frames:
        raise RuntimeError("No ERP propagated pinhole frames were written")

    ply_path = out_dir / ply_filename
    point_count = _write_sparse_pointcloud(recon, ply_path)
    log_fn("Wrote sparse pointcloud (%d points): %s", point_count, ply_path)

    top_intr = _intrinsics(views[0][2])
    transforms_path = out_dir / transforms_filename
    transforms_data = {
        "camera_model": "PINHOLE",
        **top_intr,
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": sorted(frames, key=lambda e: e["file_path"]),
    }
    with transforms_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(transforms_data, fp, indent=4)
    log_fn(
        "Wrote ERP propagated pinhole export: %d crops from %d native images, "
        "%d masks, %s",
        len(frames), len(reg_image_ids), rendered_masks, transforms_path,
    )

    sparse_output_dir = (
        Path(propagated_sparse_output_dir)
        if propagated_sparse_output_dir is not None
        else out_dir / "sparse" / "0"
    )
    _write_native_pinhole_sparse_model(
        recon, sparse_entries, sparse_output_dir, log_fn=log_fn,
    )

    return transforms_path


def write_native_propagated_transforms(
    colmap_sparse_dir: str | Path,
    images_root: str | Path,
    output_dir: str | Path,
    view_config=None,
    *,
    masks_root: str | Path | None = None,
    transforms_filename: str = "transforms.json",
    ply_filename: str = "pointcloud.ply",
    file_path_prefix: str = "images",
    mask_path_prefix: str = "masks",
    propagated_sparse_output_dir: str | Path | None = None,
    reconstruction_obj=None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Export pinhole crops propagated from a native dual-fisheye solve.

    Source poses come from the registered native lens image itself:

        R_rel = fisheye_reframer._rotation_matrix(view.yaw, view.pitch)
        R_v   = R_rel @ R_w2c_lens
        t_v   = R_rel @ t_w2c_lens

    There is no reference crop term, no assumed back-lens flip, and no rig
    baseline offset. Only registered ``(frame, lens)`` native images emit
    crops, so native partial registration is preserved.
    """
    import cv2
    import pycolmap

    from .fisheye_calibration import DualFisheyeCalibration, FisheyeCalibration
    from .fisheye_reframer import (
        FISHEYE_PINHOLE_PRESET,
        FisheyeReframer,
        _rotation_matrix as _renderer_rotation_matrix,
    )

    sparse_dir = Path(colmap_sparse_dir)
    images_root = Path(images_root)
    out_dir = Path(output_dir)
    output_images_dir = out_dir / file_path_prefix
    output_masks_dir = out_dir / mask_path_prefix
    masks_root_path = Path(masks_root) if masks_root is not None else None

    if view_config is None:
        view_config = FISHEYE_PINHOLE_PRESET

    out_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    if masks_root_path is not None:
        output_masks_dir.mkdir(parents=True, exist_ok=True)

    if reconstruction_obj is not None:
        recon = reconstruction_obj
    else:
        log_fn("Loading native dual-fisheye reconstruction from %s", sparse_dir)
        recon = pycolmap.Reconstruction(str(sparse_dir))

    reg_image_ids = sorted(recon.reg_image_ids())
    if not reg_image_ids:
        raise RuntimeError(
            "Native reconstruction contains no registered images — "
            "cannot export propagated pinhole crops"
        )

    lens_camera_ids: dict[str, set[int]] = {"front": set(), "back": set()}
    registered_images: list[tuple[int, object, str, str]] = []
    for img_id in reg_image_ids:
        image = recon.image(img_id)
        rel_name = image.name.replace("\\", "/")
        parts = rel_name.split("/", 1)
        if len(parts) != 2 or parts[0] not in lens_camera_ids:
            log_fn("WARNING: skipping unexpected native image name %r", image.name)
            continue
        lens, frame_name = parts
        lens_camera_ids[lens].add(int(image.camera_id))
        registered_images.append((img_id, image, lens, frame_name))

    if not registered_images:
        raise RuntimeError("No front/back native images found in reconstruction")

    for lens, camera_ids in lens_camera_ids.items():
        if not camera_ids:
            raise RuntimeError(f"No registered {lens!r} images found")
        if len(camera_ids) != 1:
            raise RuntimeError(
                f"Expected one camera for {lens!r}, found {sorted(camera_ids)}"
            )

    lens_calibs: dict[str, FisheyeCalibration] = {}
    for lens, camera_ids in lens_camera_ids.items():
        camera_id = next(iter(camera_ids))
        cam = recon.cameras[camera_id]
        model_name = cam.model_name
        params = [float(x) for x in cam.params]
        if model_name != "OPENCV_FISHEYE" or len(params) != 8:
            raise RuntimeError(
                f"{lens} camera {camera_id} is {model_name!r} with "
                f"{len(params)} params; expected OPENCV_FISHEYE with 8 params"
            )
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        lens_calibs[lens] = FisheyeCalibration(
            camera_matrix=np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            dist_coeffs=np.array([[k1], [k2], [k3], [k4]], dtype=np.float64),
            image_size=(int(cam.width), int(cam.height)),
            rms_error=-1.0,
            num_images_used=0,
            fov_degrees=190.0,
        )
        log_fn(
            "Native %s camera %d: OPENCV_FISHEYE %dx%d params=%s",
            lens, camera_id, int(cam.width), int(cam.height), params,
        )

    calibration = DualFisheyeCalibration(
        front=lens_calibs["front"],
        back=lens_calibs["back"],
        camera_model="Native dual fisheye COLMAP refined",
        baseline_m=0.0,
    )
    reframer = FisheyeReframer(calibration)

    views_by_lens = {
        "front": list(view_config.views_for_lens("front")),
        "back": list(view_config.views_for_lens("back")),
    }
    if not views_by_lens["front"] or not views_by_lens["back"]:
        raise RuntimeError("View config must contain front and back views")

    def _pinhole_intrinsics(view) -> dict:
        crop = int(view_config.crop_size)
        fl = crop / (2.0 * np.tan(np.radians(view.fov_deg / 2.0)))
        return {
            "w": crop,
            "h": crop,
            "fl_x": float(fl),
            "fl_y": float(fl),
            "cx": crop / 2.0,
            "cy": crop / 2.0,
        }

    frames: list[dict] = []
    sparse_entries: list[dict] = []
    rendered_masks = 0
    missing_masks = 0

    for _img_id, image, lens, frame_name in sorted(
        registered_images, key=lambda item: (item[2], item[3])
    ):
        source_path = images_root / lens / frame_name
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"Could not read native image: {source_path}")

        native_mask = None
        if masks_root_path is not None:
            mask_path = masks_root_path / lens / Path(frame_name).with_suffix(".png")
            native_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if native_mask is None:
                missing_masks += 1

        cam_from_world = image.cam_from_world()
        R_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(cam_from_world.translation, dtype=np.float64)

        for view in views_by_lens[lens]:
            if native_mask is not None:
                crop, mask_crop = reframer.extract_view(
                    source, view, int(view_config.crop_size), native_mask,
                )
            else:
                crop = reframer.extract_view(
                    source, view, int(view_config.crop_size),
                )
                mask_crop = None

            flat_name = f"{view.name}_{frame_name}"
            crop_path = output_images_dir / flat_name
            if not cv2.imwrite(
                str(crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, int(view_config.quality)],
            ):
                raise RuntimeError(f"Could not write crop: {crop_path}")

            R_rel = _renderer_rotation_matrix(view.yaw_deg, view.pitch_deg)
            R_v = R_rel @ R_w2c
            t_v = R_rel @ t_w2c

            R_c2w = R_v.T
            t_c2w = -R_v.T @ t_v
            c2w_opencv = np.eye(4, dtype=np.float64)
            c2w_opencv[:3, :3] = R_c2w
            c2w_opencv[:3, 3] = t_c2w
            c2w_lfs = _c2w_opencv_to_lfs(c2w_opencv)

            intr = _pinhole_intrinsics(view)
            sparse_entries.append({
                "name": flat_name,
                "view_name": view.name,
                "intrinsics": intr,
                "R_w2c": R_v.copy(),
                "t_w2c": t_v.copy(),
            })
            entry = {
                "file_path": f"{file_path_prefix}/{flat_name}",
                "transform_matrix": c2w_lfs.tolist(),
                **intr,
            }

            if mask_crop is not None:
                mask_name = Path(flat_name).with_suffix(".png").name
                mask_out = output_masks_dir / mask_name
                if not cv2.imwrite(str(mask_out), mask_crop):
                    raise RuntimeError(f"Could not write mask crop: {mask_out}")
                entry["mask_path"] = f"{mask_path_prefix}/{mask_name}"
                rendered_masks += 1

            frames.append(entry)

    if not frames:
        raise RuntimeError("No native propagated pinhole frames were written")

    ply_path = out_dir / ply_filename
    point_count = _write_sparse_pointcloud(recon, ply_path)
    log_fn("Wrote sparse pointcloud (%d points): %s", point_count, ply_path)

    top_intr = _pinhole_intrinsics(view_config.views[0])
    transforms_path = out_dir / transforms_filename
    transforms_data = {
        "camera_model": "PINHOLE",
        **top_intr,
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": sorted(frames, key=lambda e: e["file_path"]),
    }
    with transforms_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(transforms_data, fp, indent=4)

    lens_counts = {
        lens: sum(1 for _i, _image, image_lens, _frame in registered_images
                  if image_lens == lens)
        for lens in ("front", "back")
    }
    log_fn(
        "Wrote native propagated pinhole export: %d crops from %d native images "
        "(front=%d, back=%d), %d masks, %s",
        len(frames), len(registered_images),
        lens_counts["front"], lens_counts["back"],
        rendered_masks, transforms_path,
    )
    if missing_masks:
        log_fn("WARNING: %d native masks were missing", missing_masks)

    sparse_output_dir = (
        Path(propagated_sparse_output_dir)
        if propagated_sparse_output_dir is not None
        else out_dir / "sparse" / "0"
    )
    _write_native_pinhole_sparse_model(
        recon,
        sparse_entries,
        sparse_output_dir,
        log_fn=log_fn,
    )

    return transforms_path


def _flat_colmap_image_name(image_name: str) -> str:
    """Convert COLMAP view-folder image names to final flat image names."""
    rel = Path(image_name.replace("\\", "/"))
    parent = rel.parent.as_posix().replace("/", "_")
    if not parent or parent == ".":
        return rel.name
    return f"{parent}_{rel.name}"


def _next_id(*maps) -> int:
    max_id = 0
    for mapping in maps:
        if mapping:
            max_id = max(max_id, max(int(k) for k in mapping.keys()))
    return max_id + 1


def _write_native_pinhole_sparse_model(
    native_recon,
    sparse_entries: list[dict],
    output_sparse_dir: Path,
    *,
    log_fn: Callable[..., object] = logger.info,
) -> None:
    """Write a clean pinhole-only sparse model from native propagated poses."""
    import shutil
    import pycolmap

    clean_recon = pycolmap.Reconstruction()
    camera_ids_by_view: dict[str, int] = {}
    image_names: set[str] = set()
    next_camera_id = 1
    next_image_id = 1

    for entry in sorted(sparse_entries, key=lambda e: e["name"]):
        name = entry["name"]
        if name in image_names:
            raise RuntimeError(f"Duplicate propagated image name: {name}")

        view_name = entry["view_name"]
        intr = entry["intrinsics"]
        camera_id = camera_ids_by_view.get(view_name)
        if camera_id is None:
            camera_id = next_camera_id
            next_camera_id += 1
            camera = pycolmap.Camera.create_from_model_name(
                camera_id,
                "PINHOLE",
                float(intr["fl_x"]),
                int(intr["w"]),
                int(intr["h"]),
            )
            camera.params = np.array(
                [
                    float(intr["fl_x"]),
                    float(intr["fl_y"]),
                    float(intr["cx"]),
                    float(intr["cy"]),
                ],
                dtype=np.float64,
            )
            clean_recon.add_camera_with_trivial_rig(camera)
            camera_ids_by_view[view_name] = camera_id

        image = pycolmap.Image(
            name=name,
            camera_id=camera_id,
            image_id=next_image_id,
        )
        next_image_id += 1
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(entry["R_w2c"]),
            entry["t_w2c"],
        )
        clean_recon.add_image_with_trivial_frame(image, cam_from_world)
        image_names.add(name)

    point_count = 0
    for point3D_id, point in native_recon.points3D.items():
        clean_recon.add_point3D_with_id(
            int(point3D_id),
            pycolmap.Point3D(
                xyz=np.asarray(point.xyz, dtype=np.float64),
                color=np.asarray(point.color, dtype=np.uint8),
            ),
        )
        point_count += 1

    if output_sparse_dir.exists():
        shutil.rmtree(str(output_sparse_dir))
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    clean_recon.write_binary(str(output_sparse_dir))
    log_fn(
        "Wrote native pinhole-only COLMAP sparse model to %s "
        "(%d PINHOLE cameras, %d images, %d points)",
        output_sparse_dir,
        len(camera_ids_by_view),
        len(image_names),
        point_count,
    )


def _write_propagated_sparse_model(
    recon,
    sparse_entries: list[dict],
    output_sparse_dir: Path,
    *,
    log_fn: Callable[..., object] = logger.info,
) -> None:
    """Write COLMAP sparse/0 with all propagated pinhole images registered."""
    import shutil
    import pycolmap

    changed_names = 0
    for _image_id, image in recon.images.items():
        flat_name = _flat_colmap_image_name(image.name)
        if flat_name != image.name:
            image.name = flat_name
            changed_names += 1

    existing_names = {image.name for image in recon.images.values()}
    next_camera_id = _next_id(recon.cameras, recon.rigs)
    next_image_id = _next_id(recon.images, recon.frames)
    camera_ids_by_view: dict[str, int] = {}
    added_images = 0

    for entry in sparse_entries:
        name = entry["name"]
        if name in existing_names:
            continue

        view_name = entry["view_name"]
        intr = entry["intrinsics"]
        camera_id = camera_ids_by_view.get(view_name)
        if camera_id is None:
            camera_id = next_camera_id
            next_camera_id += 1
            camera = pycolmap.Camera.create_from_model_name(
                camera_id,
                "PINHOLE",
                float(intr["fl_x"]),
                int(intr["w"]),
                int(intr["h"]),
            )
            camera.params = np.array(
                [
                    float(intr["fl_x"]),
                    float(intr["fl_y"]),
                    float(intr["cx"]),
                    float(intr["cy"]),
                ],
                dtype=np.float64,
            )
            recon.add_camera_with_trivial_rig(camera)
            camera_ids_by_view[view_name] = camera_id

        image = pycolmap.Image(
            name=name,
            camera_id=camera_id,
            image_id=next_image_id,
        )
        next_image_id += 1
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(entry["R_w2c"]),
            entry["t_w2c"],
        )
        recon.add_image_with_trivial_frame(image, cam_from_world)
        existing_names.add(name)
        added_images += 1

    if output_sparse_dir.exists():
        shutil.rmtree(str(output_sparse_dir))
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(output_sparse_dir))
    log_fn(
        "Wrote propagated COLMAP sparse model to %s (%d names flattened, %d images added)",
        output_sparse_dir,
        changed_names,
        added_images,
    )
