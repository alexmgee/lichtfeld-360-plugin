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
    with transforms_path.open("w", encoding="utf-8") as fp:
        json.dump(transforms_data, fp, indent=4)
    log_fn(
        "Wrote fisheye transforms.json with %d frames across %d cameras: %s",
        len(frames), len(camera_intrinsics), transforms_path,
    )

    return transforms_path


def write_rig_propagated_transforms(
    colmap_sparse_dir: str | Path,
    images_root: str | Path,
    output_dir: str | Path,
    view_config,
    baseline_m: float = 0.026,
    back_intrinsics: dict | None = None,
    transforms_filename: str = "transforms.json",
    ply_filename: str = "pointcloud.ply",
    file_path_prefix: str = "images",
    masks_root: str | Path | None = None,
    mask_path_prefix: str = "masks",
    propagated_sparse_output_dir: str | Path | None = None,
    log_fn: Callable[..., object] = logger.info,
) -> Path:
    """Write transforms.json by propagating a reference-view COLMAP reconstruction
    to all rig views via known geometry.

    COLMAP reconstructs only the reference sensor (``views[0]``, typically
    ``front_ctr_hi``). This function derives every other view's pose from the
    known ``cam_from_rig`` rotation (and baseline for back-lens views), then
    writes a single transforms.json covering all 16 views × N registered frames.

    Args:
        colmap_sparse_dir: COLMAP sparse model directory (contains the
            reference-view-only reconstruction).
        images_root: Parent directory of all view subfolders
            (e.g. ``<output>/images/``).
        output_dir: Where to write transforms.json + pointcloud.ply.
        view_config: ``FisheyeViewConfig`` with all 16 view definitions.
        baseline_m: Inter-lens baseline in metres (default 0.026 for DJI Osmo).
        back_intrinsics: Optional dict with w/h/fl_x/fl_y/cx/cy for back views.
            If None, computed from ``view_config.crop_size`` at 90° FOV.
        transforms_filename: Output transforms filename.
        ply_filename: Output pointcloud filename.
        file_path_prefix: Prefix to write before each flat image filename.
        masks_root: Optional directory containing flat mask PNGs matching the
            propagated flat image names.
        mask_path_prefix: Prefix to write before each flat mask filename.
        propagated_sparse_output_dir: Optional COLMAP sparse model directory to
            write with all propagated flat-name image entries. When omitted,
            only transforms.json and pointcloud.ply are written.
        log_fn: Logging callback.

    Returns:
        Path to the written transforms.json.
    """
    import pycolmap

    sparse_dir = Path(colmap_sparse_dir)
    images_root = Path(images_root)
    masks_root_path = Path(masks_root) if masks_root is not None else None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load reconstruction
    log_fn("Loading reference-view reconstruction from %s", sparse_dir)
    recon = pycolmap.Reconstruction(str(sparse_dir))

    reg_image_ids = sorted(recon.reg_image_ids())
    if not reg_image_ids:
        raise RuntimeError(
            "Reference-view reconstruction has no registered images — "
            "cannot propagate poses"
        )
    log_fn("Reference reconstruction: %d registered images", len(reg_image_ids))

    views = view_config.views
    ref_view = views[0]   # front_ctr_hi (rig reference)
    lo_view = views[7]    # front_ctr_lo (second reference)

    def _match_reference_view_name(image_name: str) -> tuple[str | None, str | None]:
        normalized = image_name.replace("\\", "/")
        basename = Path(normalized).name
        for view in (ref_view, lo_view):
            folder_prefix = view.name + "/"
            flat_prefix = view.name + "_"
            if normalized.startswith(folder_prefix):
                return view.name, normalized[len(folder_prefix):]
            if basename.startswith(flat_prefix):
                return view.name, basename[len(flat_prefix):]
        return None, None

    # Step 2: Find reference camera intrinsics from the reconstruction.
    # With PER_FOLDER, each subfolder gets its own camera. Find the one
    # belonging to the ref view by checking registered image names.
    ref_cam_id = None
    for img_id in reg_image_ids:
        image = recon.image(img_id)
        view_name, _bare = _match_reference_view_name(image.name)
        if view_name == ref_view.name:
            ref_cam_id = image.camera_id
            break
    if ref_cam_id is None:
        # All registered images are from the lo view — use that camera
        ref_cam_id = recon.image(reg_image_ids[0]).camera_id

    cam = recon.cameras[ref_cam_id]
    params = list(cam.params)
    front_intrinsics = {
        "w": int(cam.width),
        "h": int(cam.height),
        "fl_x": float(params[0]),
        "fl_y": float(params[1]),
        "cx": float(params[2]),
        "cy": float(params[3]),
    }
    log_fn("Refined front intrinsics: fl=%.2f,%.2f cx=%.2f cy=%.2f",
           front_intrinsics["fl_x"], front_intrinsics["fl_y"],
           front_intrinsics["cx"], front_intrinsics["cy"])

    # Step 3: Default back intrinsics from crop geometry
    if back_intrinsics is None:
        crop = view_config.crop_size
        fl = crop / 2.0  # 90° FOV
        back_intrinsics = {
            "w": crop, "h": crop,
            "fl_x": fl, "fl_y": fl,
            "cx": crop / 2.0, "cy": crop / 2.0,
        }

    # Step 4: Precompute cam_from_rig for all 16 views
    R_back = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    R_ref = _fisheye_rotation_matrix(ref_view.yaw_deg, ref_view.pitch_deg)

    view_transforms = []  # (R_rel, d_rig, is_back) per view
    for view in views:
        R_view = _fisheye_rotation_matrix(view.yaw_deg, view.pitch_deg)
        if view.source_lens == "back":
            R_view = R_view @ R_back
        R_rel = R_view @ R_ref.T
        # d_rig: position of this view's optical center in the rig frame.
        # Front views share the rig origin (front lens); back views are offset
        # by baseline_m along rig -Z (behind the front lens).
        d_rig = (np.array([0.0, 0.0, -baseline_m], dtype=np.float64)
                 if view.source_lens == "back"
                 else np.zeros(3, dtype=np.float64))
        view_transforms.append((R_rel, d_rig, view.source_lens == "back"))

    # Precompute lo→ref derivation rotation (for frames where only lo registered)
    R_lo_from_ref = _fisheye_rotation_matrix(lo_view.yaw_deg, lo_view.pitch_deg) @ R_ref.T

    # Step 5: Build per-frame pose map from both reference views.
    # Prefer front_ctr_hi; derive ref pose from front_ctr_lo when needed.
    frame_poses: dict[str, tuple] = {}  # bare_name → (R_w2c, t_w2c, source)
    for img_id in reg_image_ids:
        image = recon.image(img_id)
        view_name, bare = _match_reference_view_name(image.name)
        if view_name == ref_view.name:
            source = "ref"
        elif view_name == lo_view.name:
            source = "lo"
        else:
            log_fn("WARNING: unexpected image name %r in reconstruction", image.name)
            continue
        # Prefer ref; only use lo if ref isn't registered for this frame
        if bare not in frame_poses or source == "ref":
            cam_from_world = image.cam_from_world()
            R_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
            t_w2c = np.asarray(cam_from_world.translation, dtype=np.float64)
            frame_poses[bare] = (R_w2c, t_w2c, source)

    # Derive ref pose from lo pose for frames where only lo registered
    ref_count = sum(1 for _, _, s in frame_poses.values() if s == "ref")
    lo_count = sum(1 for _, _, s in frame_poses.values() if s == "lo")
    for bare in frame_poses:
        R_w2c, t_w2c, source = frame_poses[bare]
        if source == "lo":
            R_w2c = R_lo_from_ref.T @ R_w2c
            t_w2c = R_lo_from_ref.T @ t_w2c
            frame_poses[bare] = (R_w2c, t_w2c, "derived_from_lo")
    log_fn("Frame poses: %d from ref, %d derived from lo, %d total",
           ref_count, lo_count, len(frame_poses))

    # Step 6: Propagate each frame's ref pose to all 16 views
    frames = []
    sparse_entries = []
    missing_count = 0

    for bare_name, (R_w2c, t_w2c, _source) in sorted(frame_poses.items()):
        for i, view in enumerate(views):
            R_rel, d_rig, is_back = view_transforms[i]

            # Propagate w2c: cam_from_world = cam_from_rig @ rig_from_world
            # The rig-frame offset d_rig is subtracted BEFORE rotating into
            # the view's coordinate frame, so all views on the same lens
            # share one optical center in world space.
            R_v = R_rel @ R_w2c
            t_v = R_rel @ (t_w2c - d_rig)

            # Invert to c2w (OpenCV convention)
            R_c2w = R_v.T
            t_c2w = -R_v.T @ t_v
            c2w_opencv = np.eye(4, dtype=np.float64)
            c2w_opencv[:3, :3] = R_c2w
            c2w_opencv[:3, 3] = t_c2w

            # Convert to LFS coordinate system
            c2w_lfs = _c2w_opencv_to_lfs(c2w_opencv)

            # Flat naming: images/{view_name}_{frame_id}.jpg
            flat_name = f"{view.name}_{bare_name}"
            rel_path = f"{file_path_prefix}/{flat_name}"
            abs_path = images_root / flat_name
            if not abs_path.exists():
                # Pre-flatten layout: try subfolder path
                abs_path_sub = images_root / view.name / bare_name
                if not abs_path_sub.exists():
                    missing_count += 1
                    continue

            intr = back_intrinsics if is_back else front_intrinsics
            sparse_entries.append({
                "name": flat_name,
                "view_name": view.name,
                "intrinsics": intr,
                "R_w2c": R_v.copy(),
                "t_w2c": t_v.copy(),
            })
            entry = {
                "file_path": rel_path,
                "transform_matrix": c2w_lfs.tolist(),
                **intr,
            }
            if masks_root_path is not None:
                mask_name = Path(flat_name).with_suffix(".png")
                if (masks_root_path / mask_name).exists():
                    entry["mask_path"] = f"{mask_path_prefix}/{mask_name.as_posix()}"
            frames.append(entry)

    if missing_count:
        log_fn("WARNING: %d view images missing on disk (skipped)", missing_count)

    if not frames:
        raise RuntimeError("No valid frames after pose propagation")

    n_frames = len(frame_poses)
    log_fn("Propagated %d frame poses → %d total entries (%d views × %d frames)",
           n_frames, len(frames), len(views), n_frames)

    # Step 7: Write pointcloud + transforms.json
    ply_path = out_dir / ply_filename
    point_count = _write_sparse_pointcloud(recon, ply_path)
    log_fn("Wrote sparse pointcloud (%d points): %s", point_count, ply_path)

    transforms_path = out_dir / transforms_filename
    transforms_data = {
        "camera_model": "PINHOLE",
        # Top-level intrinsics required by LFS — it ignores per-frame values.
        # Use the front-lens (refined by COLMAP BA) as the shared intrinsics.
        **front_intrinsics,
        "applied_transform": _APPLIED_TRANSFORM,
        "ply_file_path": ply_filename,
        "frames": sorted(frames, key=lambda e: e["file_path"]),
    }
    with transforms_path.open("w", encoding="utf-8") as fp:
        json.dump(transforms_data, fp, indent=4)

    log_fn("Wrote propagated transforms.json with %d frames: %s",
           len(frames), transforms_path)

    if propagated_sparse_output_dir is not None:
        _write_propagated_sparse_model(
            recon,
            sparse_entries,
            Path(propagated_sparse_output_dir),
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
