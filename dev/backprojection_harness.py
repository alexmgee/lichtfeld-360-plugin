#!/usr/bin/env python
"""Backprojection validation harness.

Compares a candidate backprojection implementation against the current
production implementation on controlled test cases.

Usage:
    .venv/Scripts/python.exe dev/backprojection_harness.py

Outputs comparison metrics (IoU, changed pixels, runtime) and saves
visual comparison masks to dev/bp_harness_output/.
"""
import sys
import os
import time
from pathlib import Path

# Bootstrap DLL search BEFORE importing cv2 or any extension module
_plugin_root = str(Path(__file__).resolve().parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

sys.path.insert(0, str(Path(_plugin_root) / "tests"))
import conftest  # noqa: F401 — triggers _configure_windows_dll_search()
sys.path.pop(0)

import cv2
import numpy as np

import pycolmap
from core.masker import (
    _create_synthetic_camera,
    _look_at_rotation,
    _backproject_fisheye_mask_to_erp,
)


# ── Candidate implementation: pure numpy equidistant fisheye ─────


def _backproject_numpy(
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
) -> np.ndarray:
    """Candidate backprojection using inline numpy equidistant fisheye math.

    Replaces pycolmap.img_from_cam() with direct vectorized math for
    the zero-distortion OPENCV_FISHEYE model (r = f * theta).
    """
    erp_w, erp_h = erp_size
    fish_size = camera.width
    focal = camera.params[0]  # fx == fy for our symmetric camera
    cx_cam = camera.params[2]
    cy_cam = camera.params[3]

    # ERP pixel grid → world ray directions (same as production)
    u = np.arange(erp_w, dtype=np.float64) + 0.5
    v = np.arange(erp_h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)

    lon = ((uu / erp_w) * 2 - 1) * np.pi
    lat = (0.5 - vv / erp_h) * np.pi

    x_w = np.cos(lat) * np.sin(lon)
    y_w = np.sin(lat)
    z_w = np.cos(lat) * np.cos(lon)
    dirs_world = np.stack([x_w.ravel(), y_w.ravel(), z_w.ravel()], axis=1)

    # Rotate to camera space (same as production)
    R_cam_from_world = R_world_from_cam.T
    dirs_cam = (R_cam_from_world @ dirs_world.T).T

    # Forward hemisphere filter (same as production)
    forward = dirs_cam[:, 2] > 1e-8

    erp_mask = np.zeros(erp_h * erp_w, dtype=np.uint8)

    if not np.any(forward):
        return erp_mask.reshape(erp_h, erp_w)

    pts = dirs_cam[forward]

    # ── Pure numpy equidistant fisheye projection ──
    # For OPENCV_FISHEYE with k1-k4=0: r = f * theta
    # theta = angle from optical axis (+Z)
    # phi = azimuthal angle in the XY plane
    cam_x, cam_y, cam_z = pts[:, 0], pts[:, 1], pts[:, 2]
    r_xy = np.sqrt(cam_x ** 2 + cam_y ** 2)
    theta = np.arctan2(r_xy, cam_z)          # angle from +Z axis
    phi = np.arctan2(cam_y, cam_x)           # azimuthal angle

    r_px = focal * theta                     # equidistant: r = f * theta
    px = cx_cam + r_px * np.cos(phi)
    py = cy_cam + r_px * np.sin(phi)

    # Radial + bounds validity (same logic as production)
    radius = fish_size / 2.0
    r_from_center = np.sqrt((px - cx_cam) ** 2 + (py - cy_cam) ** 2)
    in_lens = r_from_center < (radius - 0.5)
    in_bounds = (
        in_lens
        & (px >= 0) & (px < fish_size)
        & (py >= 0) & (py < fish_size)
    )

    if np.any(in_bounds):
        forward_idx = np.where(forward)[0]
        valid_idx = forward_idx[in_bounds]
        px_int = np.clip(np.round(px[in_bounds]).astype(int), 0, fish_size - 1)
        py_int = np.clip(np.round(py[in_bounds]).astype(int), 0, fish_size - 1)
        erp_mask[valid_idx] = mask[py_int, px_int]

    return erp_mask.reshape(erp_h, erp_w)


# ── Candidate: downsampled backprojection ────────────────────────


def _backproject_downsampled(
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R_world_from_cam: np.ndarray,
    scale: float = 0.5,
) -> np.ndarray:
    """Candidate: run production backprojection at reduced ERP resolution,
    then upscale the binary result with nearest-neighbor.

    Uses the same production pycolmap projection math — only the grid
    density changes.
    """
    erp_w, erp_h = erp_size
    reduced_w = max(1, int(erp_w * scale))
    reduced_h = max(1, int(erp_h * scale))

    reduced_result = _backproject_fisheye_mask_to_erp(
        mask, (reduced_w, reduced_h), camera, R_world_from_cam,
    )

    if reduced_result.shape == (erp_h, erp_w):
        return reduced_result

    return cv2.resize(
        reduced_result, (erp_w, erp_h),
        interpolation=cv2.INTER_NEAREST,
    )


# ── Test case generators ─────────────────────────────────────────


def make_centered_blob_mask(size: int = 2048, blob_radius: int = 200) -> np.ndarray:
    """Circular blob at the center of the fisheye — stable case."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), blob_radius, 1, -1)
    return mask


def make_offcenter_blob_mask(
    size: int = 2048, offset_x: int = 400, offset_y: int = 300, blob_radius: int = 150,
) -> np.ndarray:
    """Blob offset from center — tests edge sampling behavior."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2 + offset_x, size // 2 + offset_y), blob_radius, 1, -1)
    return mask


def make_person_shape_mask(size: int = 2048) -> np.ndarray:
    """Rough person-shaped mask — ellipse body + circle head."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    # Body ellipse
    cv2.ellipse(mask, (cx, cy + 80), (100, 200), 0, 0, 360, 1, -1)
    # Head circle
    cv2.circle(mask, (cx, cy - 150), 60, 1, -1)
    return mask


def load_real_mask(path: str) -> np.ndarray | None:
    """Load a real SAM2-tracked fisheye mask if available."""
    p = Path(path)
    if not p.exists():
        return None
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


# ── Comparison metrics ───────────────────────────────────────────


def compute_metrics(ref: np.ndarray, candidate: np.ndarray) -> dict:
    """Compute IoU, changed pixel count, and percentage."""
    ref_bool = ref > 0
    cand_bool = candidate > 0

    intersection = np.sum(ref_bool & cand_bool)
    union = np.sum(ref_bool | cand_bool)
    iou = float(intersection / union) if union > 0 else 1.0

    changed = int(np.sum(ref_bool != cand_bool))
    total = ref.size
    changed_pct = 100.0 * changed / total

    return {
        "iou": iou,
        "changed_pixels": changed,
        "changed_pct": changed_pct,
        "ref_area": int(np.sum(ref_bool)),
        "cand_area": int(np.sum(cand_bool)),
    }


# ── Main harness ─────────────────────────────────────────────────


def run_test_case(
    name: str,
    mask: np.ndarray,
    erp_size: tuple[int, int],
    camera: pycolmap.Camera,
    R: np.ndarray,
    candidate_fn,
    candidate_label: str,
    output_dir: Path,
):
    """Run one test case: reference vs candidate, print metrics, save outputs."""
    print(f"\n{'─' * 50}")
    print(f"Test: {name}  [{candidate_label}]")
    print(f"  Mask shape: {mask.shape}, white pixels: {mask.sum()}")
    print(f"  ERP size: {erp_size[0]}x{erp_size[1]}")

    # Reference (production pycolmap path at full res)
    t0 = time.perf_counter()
    ref_result = _backproject_fisheye_mask_to_erp(mask, erp_size, camera, R)
    t_ref = time.perf_counter() - t0

    # Candidate
    t0 = time.perf_counter()
    cand_result = candidate_fn(mask, erp_size, camera, R)
    t_cand = time.perf_counter() - t0

    metrics = compute_metrics(ref_result, cand_result)

    # Area drift
    area_drift_pct = 0.0
    if metrics['ref_area'] > 0:
        area_drift_pct = 100.0 * abs(metrics['cand_area'] - metrics['ref_area']) / metrics['ref_area']

    print(f"  Reference:  {t_ref:.3f}s  (area={metrics['ref_area']})")
    print(f"  Candidate:  {t_cand:.3f}s  (area={metrics['cand_area']})")
    print(f"  Speedup:    {t_ref / t_cand:.1f}x" if t_cand > 0 else "  Speedup:    inf")
    print(f"  IoU:        {metrics['iou']:.6f}")
    print(f"  Changed:    {metrics['changed_pixels']} px ({metrics['changed_pct']:.4f}%)")
    print(f"  Area drift: {area_drift_pct:.2f}%")

    # Save visual outputs
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(case_dir / "input_mask.png"), mask * 255)
    cv2.imwrite(str(case_dir / "ref_erp.png"), ref_result * 255)
    cv2.imwrite(str(case_dir / "candidate_erp.png"), cand_result * 255)

    # Diff image: green = ref only, red = candidate only, white = both
    diff = np.zeros((*ref_result.shape, 3), dtype=np.uint8)
    both = (ref_result > 0) & (cand_result > 0)
    ref_only = (ref_result > 0) & (cand_result == 0)
    cand_only = (ref_result == 0) & (cand_result > 0)
    diff[both] = [255, 255, 255]
    diff[ref_only] = [0, 255, 0]    # green = in ref but not candidate
    diff[cand_only] = [0, 0, 255]   # red = in candidate but not ref
    cv2.imwrite(str(case_dir / "diff.png"), diff)

    return metrics, t_ref, t_cand, area_drift_pct


def run_candidate_suite(
    label: str,
    candidate_fn,
    masks: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    camera: pycolmap.Camera,
    output_dir: Path,
):
    """Run a full test suite for one candidate across all test cases."""
    print(f"\n{'=' * 60}")
    print(f"Candidate: {label}")
    print(f"{'=' * 60}")

    all_results = []

    # Half-res iteration cases (3840×1920)
    erp_half = (3840, 1920)
    for mask_name, mask in masks.items():
        for dir_name, R in rotations.items():
            name = f"{mask_name}_{dir_name}"
            r = run_test_case(
                name, mask, erp_half, camera, R,
                candidate_fn, label, output_dir / label,
            )
            all_results.append((name, *r))

    # Full-res signoff case (7680×3840)
    erp_full = (7680, 3840)
    print(f"\n{'─' * 50}")
    print("Full-res signoff (7680×3840):")
    for dir_name, R in rotations.items():
        name = f"person_fullres_{dir_name}"
        r = run_test_case(
            name, masks["person"], erp_full, camera, R,
            candidate_fn, label, output_dir / label,
        )
        all_results.append((name, *r))

    # Summary
    print(f"\n{'─' * 50}")
    print(f"Summary: {label}")
    print(f"{'Name':35s} {'IoU':>8s} {'Changed':>10s} {'AreaDrift':>10s} {'Ref(s)':>8s} {'Cand(s)':>8s} {'Speedup':>8s}")
    for name, metrics, t_ref, t_cand, area_drift in all_results:
        speedup = f"{t_ref / t_cand:.1f}x" if t_cand > 0 else "inf"
        print(f"{name:35s} {metrics['iou']:8.4f} {metrics['changed_pixels']:10d} "
              f"{area_drift:9.2f}% {t_ref:8.3f} {t_cand:8.3f} {speedup:>8s}")

    all_iou = [m['iou'] for _, m, _, _, _ in all_results]
    min_iou = min(all_iou)
    all_speedups = [t_ref / t_cand for _, _, t_ref, t_cand, _ in all_results if t_cand > 0]
    avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    max_area_drift = max(ad for _, _, _, _, ad in all_results)

    print(f"\n  Min IoU:        {min_iou:.6f}")
    print(f"  Avg speedup:    {avg_speedup:.1f}x")
    print(f"  Max area drift: {max_area_drift:.2f}%")

    if min_iou > 0.999:
        print(f"  Verdict: EQUIVALENT — safe to replace")
    elif min_iou > 0.99:
        print(f"  Verdict: CLOSE — review diff images")
    else:
        print(f"  Verdict: DIVERGES — do not replace")

    return all_results


def main():
    output_dir = Path(__file__).parent / "bp_harness_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = _create_synthetic_camera(2048)

    # Test directions
    dir_front = np.array([0.0, -0.9, 0.4])  # looking down and forward
    dir_front = dir_front / np.linalg.norm(dir_front)

    dir_side = np.array([0.8, -0.5, 0.3])  # off to the side
    dir_side = dir_side / np.linalg.norm(dir_side)

    rotations = {
        "front": _look_at_rotation(dir_front),
        "side": _look_at_rotation(dir_side),
    }

    masks = {
        "centered": make_centered_blob_mask(),
        "offcenter": make_offcenter_blob_mask(),
        "person": make_person_shape_mask(),
    }

    print("=" * 60)
    print("Backprojection Validation Harness")
    print(f"Camera: {camera.width}x{camera.width}, focal={camera.params[0]:.1f}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Candidate 1: downsampled at 0.5x
    def downsample_50(mask, erp_size, camera, R):
        return _backproject_downsampled(mask, erp_size, camera, R, scale=0.5)

    run_candidate_suite(
        "downsample_0.5", downsample_50,
        masks, rotations, camera, output_dir,
    )

    # Candidate 2: downsampled at 0.25x
    def downsample_25(mask, erp_size, camera, R):
        return _backproject_downsampled(mask, erp_size, camera, R, scale=0.25)

    run_candidate_suite(
        "downsample_0.25", downsample_25,
        masks, rotations, camera, output_dir,
    )


if __name__ == "__main__":
    main()
