# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owned image-folder dual-fisheye pipeline."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .colmap_runner import ColmapConfig, ColmapRunner
from .fisheye_reframer import FISHEYE_PINHOLE_PRESET
from .frame_source import (
    assert_source_reads_safe,
    graceful_delete_source,
    list_images_natural,
    pair_fisheye_one_folder,
    pair_fisheye_two_folders,
    stage_fisheye_frames,
)
from .transforms_writer import (
    write_fisheye_transforms,
    write_native_propagated_transforms,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _absorb_source(folder, output_root) -> None:
    """Graceful-delete a source folder whose frames are preserved in the kept
    dataset. A kept folder (leftover sidecar/subdir, or the guard refusing a
    genuine self-deletion case) is a WARNING, never a run failure -- matching
    ERP Native's rmdir-warning precedent. The reconstruction is already
    complete when this runs."""
    try:
        _removed, folder_removed = graceful_delete_source(folder, output_root)
    except ValueError as exc:
        logger.warning("Source folder kept: %s", exc)
        return
    if not folder_removed:
        logger.warning(
            "Source folder %s not empty after removing its images; "
            "left in place.", folder)


def _export_lens_mask_deliverable(
    pairs, staged_masks_root, dest_root, *, two_folder: bool,
) -> int:
    """Copy staged-name lens masks (front/000001.png) to the kept root
    ``<output>/masks/`` deliverable, keyed to the USER's original filenames
    and mirroring the source folder structure (QA-C2 run 8): one-folder
    source -> flat ``masks/<orig_stem>.png``; two-folder -> ``masks/front/``
    + ``masks/back/``. Masks stay PNG (lossless binary). Skips pairs whose
    staged mask is missing. Returns the copied count."""
    staged = Path(staged_masks_root)
    dest = Path(dest_root)
    copied = 0
    for i, (front_path, back_path) in enumerate(pairs, start=1):
        for lens, orig in (("front", front_path), ("back", back_path)):
            src = staged / lens / f"{i:06d}.png"
            if not src.is_file():
                continue
            if two_folder:
                target = dest / lens / f"{Path(orig).stem}.png"
            else:
                target = dest / f"{Path(orig).stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
    return copied


def _remove_emptied_parent(front_src, back_src, output_root) -> None:
    """After absorbing two-folder sources (e.g. frames/front + frames/back),
    drop their now-empty shared parent so no husk folder is left behind
    (QA-C2 run 6). rmdir-only -- anything still inside keeps it -- and never
    the output root itself."""
    try:
        fp = Path(front_src).resolve().parent
        bp = Path(back_src).resolve().parent
        out = Path(output_root).resolve()
        if fp == bp and fp != out and not out.is_relative_to(fp):
            fp.rmdir()
    except OSError:
        pass


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _result(
    *,
    success: bool,
    dataset_path: Path | None,
    num_output_images: int,
    num_registered_frames: int,
    views_per_frame: int,
    preset_signature: str,
    error: str = "",
    num_source_frames: int = 0,
    colmap_stats=None,
) -> dict:
    result = {
        "success": success,
        "dataset_path": str(dataset_path) if dataset_path is not None else "",
        "num_source_frames": int(num_source_frames),
        "num_output_images": int(num_output_images),
        "num_registered_frames": int(num_registered_frames),
        "views_per_frame": int(views_per_frame),
        "preset_signature": preset_signature,
        "gpu_extraction": False,
    }
    if colmap_stats is not None:
        # Registration diagnostics for the panel summary (Registered images /
        # Complete rig frames / per-view breakdown), mirroring the ERP result.
        result["num_aligned_cameras"] = int(
            getattr(colmap_stats, "num_registered_images", 0))
        result["num_complete_frames"] = int(
            getattr(colmap_stats, "num_complete_frames", 0))
        result["num_partial_frames"] = int(
            getattr(colmap_stats, "num_partial_frames", 0))
        result["expected_images_by_view"] = dict(
            getattr(colmap_stats, "expected_images_by_view", {}) or {})
        result["registered_images_by_view"] = dict(
            getattr(colmap_stats, "registered_images_by_view", {}) or {})
        result["partial_frame_examples"] = list(
            getattr(colmap_stats, "partial_frame_examples", []) or [])
        result["dropped_frame_examples"] = list(
            getattr(colmap_stats, "dropped_frame_examples", []) or [])
    if error:
        result["error"] = error
    return result


def run_image_folder_fisheye(cfg, update, cancel_check) -> dict:
    """Run image-folder dual-fisheye native solve and optional propagation."""
    # Imported lazily from .pipeline (where it currently lives) to avoid a
    # circular import: pipeline imports this module via its thin wrapper.
    from .pipeline import resolve_image_folder_masking

    resolve_image_folder_masking(cfg)

    from .fisheye_priors import infer_fisheye_camera_params

    output_mode = (cfg.output_mode or "fisheye").lower()
    training = (cfg.training_output or "native").lower()
    if output_mode == "fisheye_pinhole":
        track = "pinhole"
    elif output_mode == "fisheye":
        if training not in ("native", "pinhole", "both"):
            raise ValueError(f"Unknown training_output: {cfg.training_output!r}")
        track = training
    else:
        raise ValueError(f"Unsupported fisheye output_mode: {cfg.output_mode!r}")

    if cfg.image_front_dir:
        front_src = cfg.image_front_dir
        back_src = cfg.image_back_dir
        assert_source_reads_safe(front_src, cfg.output_dir)
        assert_source_reads_safe(back_src, cfg.output_dir)
        pairs = pair_fisheye_two_folders(front_src, back_src)
    else:
        front_src = back_src = cfg.image_source_dir
        assert_source_reads_safe(front_src, cfg.output_dir)
        files, _warning = list_images_natural(cfg.image_source_dir)
        pairs = pair_fisheye_one_folder(files)

    out = Path(cfg.output_dir)
    colmap_root = out / "colmap"
    if track == "native":
        dataset_dir = colmap_root
    elif track == "both":
        dataset_dir = colmap_root / "native"
    else:
        dataset_dir = colmap_root / "_native_tmp"

    images_root = dataset_dir / "images"
    delivered_path = (
        dataset_dir / "transforms.json"
        if track in ("native", "both")
        else colmap_root / "transforms.json"
    )

    def _stage_progress(current: int, total: int, message: str) -> None:
        pct = (current / max(total, 1)) * 20.0
        update("staging", pct, message)

    update("staging", 0.0, "Staging paired fisheye images...")
    num_pairs = stage_fisheye_frames(
        pairs,
        images_root,
        front_source=front_src,
        back_source=back_src,
        cancel_check=cancel_check,
        on_progress=_stage_progress,
    )

    mask_enabled = cfg.enable_masking
    colmap_mask_path = None
    if mask_enabled:
        update("masking", 20.0, "Initializing SAM 3 for fisheye masking...")

        import cv2
        import numpy as np

        from .backends import Sam3Backend
        from .fisheye_circle_mask import generate_fisheye_circle_mask

        backend = Sam3Backend(confidence_threshold=0.3)
        backend.initialize()

        try:
            # Masks are generated against the STAGED images (000001.jpg names)
            # so COLMAP can consume them during the solve. For the pinhole
            # track this lives in the throwaway temp; the kept root deliverable
            # is exported afterwards keyed to the user's ORIGINAL filenames
            # (_export_lens_mask_deliverable, QA-C2 run 8).
            masks_dir = dataset_dir / "masks"
            lens_names = ("front", "back")
            total_frames = sum(
                len(list((images_root / lens).glob("*.jpg")))
                + len(list((images_root / lens).glob("*.png")))
                for lens in lens_names
            )
            frame_idx = 0
            circle_cache: dict[tuple[int, int], np.ndarray] = {}

            for lens in lens_names:
                lens_frames_dir = images_root / lens
                lens_masks_dir = masks_dir / lens
                lens_masks_dir.mkdir(parents=True, exist_ok=True)

                frame_files = sorted(
                    frame
                    for frame in lens_frames_dir.iterdir()
                    if frame.suffix.lower() in _IMAGE_EXTENSIONS
                )

                for frame_file in frame_files:
                    if cancel_check():
                        raise RuntimeError("Cancelled")

                    frame_idx += 1
                    pct = 20.0 + (frame_idx / max(total_frames, 1)) * 25.0
                    update(
                        "masking",
                        pct,
                        f"SAM 3 masking {lens}/{frame_file.name} "
                        f"({frame_idx}/{total_frames})",
                    )

                    image = cv2.imread(str(frame_file))
                    if image is None:
                        logger.warning("Could not read %s, skipping", frame_file)
                        continue

                    height, width = image.shape[:2]
                    detection = backend.detect_and_segment(
                        image, cfg.mask_prompts,
                    )
                    keep_mask = ((detection == 0).astype(np.uint8)) * 255

                    cache_key = (width, height)
                    if cache_key not in circle_cache:
                        circle = generate_fisheye_circle_mask(
                            width,
                            height,
                            margin_percent=cfg.fisheye_circle_margin,
                        )
                        circle_cache[cache_key] = (
                            (1 - circle).astype(np.uint8) * 255
                        )
                    circle_keep = circle_cache[cache_key]
                    final_mask = cv2.bitwise_and(keep_mask, circle_keep)

                    mask_out = lens_masks_dir / f"{frame_file.stem}.png"
                    cv2.imwrite(str(mask_out), final_mask)
        finally:
            backend.cleanup()

        colmap_mask_path = masks_dir
        if cancel_check():
            return _result(
                success=False,
                dataset_path=delivered_path,
                num_source_frames=num_pairs,
                num_output_images=0,
                num_registered_frames=0,
                views_per_frame=0,
                preset_signature="",
                error="Cancelled",
            )

    rig_config_path = None
    if cfg.use_rig:
        from .rig_config import write_dual_fisheye_rig_config

        rig_config_path = dataset_dir / "rig_config.json"
        write_dual_fisheye_rig_config(str(rig_config_path))

    camera_params = infer_fisheye_camera_params(cfg.camera_family)
    has_prior = camera_params is not None
    if not has_prior:
        logger.info(
            "No calibrated prior for camera_family=%r; using fisheye "
            "default_focal_length_factor=0.30",
            cfg.camera_family,
        )

    colmap_config = ColmapConfig(
        camera_model="OPENCV_FISHEYE",
        camera_mode="PER_FOLDER",
        camera_params=camera_params,
        default_focal_length_factor=None if has_prior else 0.30,
        matcher=cfg.colmap_matcher,
        match_budget_tier=cfg.colmap_match_budget_tier,
        max_num_matches_override=cfg.colmap_max_num_matches,
        refine_focal_length=True,
        refine_principal_point=has_prior,
        refine_extra_params=has_prior,
        refine_sensor_from_rig=True,
        sift_max_num_features_override=cfg.sift_max_features,
        sift_max_image_size_override=cfg.sift_max_image_size,
        feature_type=cfg.colmap_feature_type,
        matcher_type=cfg.colmap_matcher_type,
        mapper=cfg.colmap_mapper,
        ba_solver=cfg.colmap_ba_solver,
        vocab_tree_path=cfg.vocab_tree_path or None,
        loop_detection=cfg.loop_detection,
        sequential_overlap=cfg.colmap_sequential_overlap,
        guided_matching=cfg.colmap_guided_matching,
        sift_estimate_affine_shape=cfg.colmap_sift_affine_dsp,
        sift_domain_size_pooling=cfg.colmap_sift_affine_dsp,
    )

    def _colmap_progress(_stage: str, pct: float, message: str) -> None:
        update("colmap", 50.0 + pct * 40.0, message)

    update("colmap", 50.0, "Running COLMAP (OPENCV_FISHEYE, PER_FOLDER)...")
    runner = ColmapRunner(
        images_dir=images_root,
        output_dir=dataset_dir,
        rig_config_path=rig_config_path,
        mask_path=colmap_mask_path,
        config=colmap_config,
        on_progress=_colmap_progress,
        cancel_check=cancel_check,
    )
    colmap_result = runner.run()
    if not colmap_result.success:
        return _result(
            success=False,
            dataset_path=delivered_path,
            num_source_frames=num_pairs,
            num_output_images=0,
            num_registered_frames=colmap_result.num_registered_frames,
            views_per_frame=colmap_result.views_per_frame,
            preset_signature="",
            error=f"COLMAP failed: {colmap_result.error}",
            colmap_stats=colmap_result,
        )

    sparse_dir = dataset_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = dataset_dir / "sparse"

    update("output", 92.0, "Writing dataset output...")
    # The lens masks live where the masking block actually wrote them:
    # dataset_dir/masks for native/both, but root <output>/masks for the
    # pinhole tracks (a kept deliverable). Use that real location so the
    # propagation writer renders per-crop masks (not the empty temp/masks).
    native_masks_arg = (
        colmap_mask_path
        if colmap_mask_path is not None and colmap_mask_path.is_dir()
        else None
    )

    if track == "native":
        dataset_path = write_fisheye_transforms(
            sparse_dir,
            images_root,
            dataset_dir,
            masks_dir=native_masks_arg,
        )
        if cancel_check():
            return _result(
                success=False,
                dataset_path=dataset_path,
                num_source_frames=num_pairs,
                num_output_images=_count_images(images_root),
                num_registered_frames=colmap_result.num_registered_frames,
                colmap_stats=colmap_result,
                views_per_frame=2,
                preset_signature="OPENCV_FISHEYE native",
                error="Cancelled",
            )
        _absorb_source(front_src, cfg.output_dir)
        if back_src != front_src:
            if cancel_check():
                return _result(
                    success=False,
                    dataset_path=dataset_path,
                    num_source_frames=num_pairs,
                    num_output_images=_count_images(images_root),
                    num_registered_frames=colmap_result.num_registered_frames,
                    colmap_stats=colmap_result,
                    views_per_frame=2,
                    preset_signature="OPENCV_FISHEYE native",
                    error="Cancelled",
                )
            _absorb_source(back_src, cfg.output_dir)
            _remove_emptied_parent(front_src, back_src, cfg.output_dir)
        num_output_images = _count_images(images_root)
        views_per_frame = 2
        preset_signature = "OPENCV_FISHEYE native"

    elif track == "both":
        dataset_path = write_fisheye_transforms(
            sparse_dir,
            images_root,
            dataset_dir,
            masks_dir=native_masks_arg,
        )
        pinhole_dir = colmap_root / "pinhole"
        write_native_propagated_transforms(
            sparse_dir,
            images_root,
            pinhole_dir,
            FISHEYE_PINHOLE_PRESET,
            masks_root=native_masks_arg,
            propagated_sparse_output_dir=pinhole_dir / "sparse" / "0",
        )
        if cancel_check():
            return _result(
                success=False,
                dataset_path=dataset_path,
                num_source_frames=num_pairs,
                num_output_images=_count_images(images_root),
                num_registered_frames=colmap_result.num_registered_frames,
                colmap_stats=colmap_result,
                views_per_frame=2,
                preset_signature="OPENCV_FISHEYE native + propagated pinhole",
                error="Cancelled",
            )
        _absorb_source(front_src, cfg.output_dir)
        if back_src != front_src:
            if cancel_check():
                return _result(
                    success=False,
                    dataset_path=dataset_path,
                    num_source_frames=num_pairs,
                    num_output_images=_count_images(images_root),
                    num_registered_frames=colmap_result.num_registered_frames,
                    colmap_stats=colmap_result,
                    views_per_frame=2,
                    preset_signature=(
                        "OPENCV_FISHEYE native + propagated pinhole"
                    ),
                    error="Cancelled",
                )
            _absorb_source(back_src, cfg.output_dir)
            _remove_emptied_parent(front_src, back_src, cfg.output_dir)
        num_output_images = _count_images(images_root)
        views_per_frame = 2
        preset_signature = "OPENCV_FISHEYE native + propagated pinhole"

    else:
        dataset_path = write_native_propagated_transforms(
            sparse_dir,
            images_root,
            colmap_root,
            FISHEYE_PINHOLE_PRESET,
            masks_root=native_masks_arg,
            propagated_sparse_output_dir=colmap_root / "sparse" / "0",
        )
        if native_masks_arg is not None:
            exported = _export_lens_mask_deliverable(
                pairs, native_masks_arg, out / "masks",
                two_folder=bool(cfg.image_front_dir),
            )
            logger.info(
                "Kept %d lens masks at %s, named after the source images",
                exported, out / "masks",
            )
        if cancel_check():
            return _result(
                success=False,
                dataset_path=dataset_path,
                num_source_frames=num_pairs,
                num_output_images=_count_images(colmap_root / "images"),
                num_registered_frames=colmap_result.num_registered_frames,
                colmap_stats=colmap_result,
                views_per_frame=FISHEYE_PINHOLE_PRESET.total_views(),
                preset_signature="Native-propagated fisheye pinhole",
                error="Cancelled",
            )
        shutil.rmtree(dataset_dir, ignore_errors=True)
        num_output_images = _count_images(colmap_root / "images")
        views_per_frame = FISHEYE_PINHOLE_PRESET.total_views()
        preset_signature = "Native-propagated fisheye pinhole"

    update("complete", 100.0, "Image-folder fisheye pipeline complete.")
    return _result(
        success=True,
        dataset_path=dataset_path,
        num_source_frames=num_pairs,
        num_output_images=num_output_images,
        num_registered_frames=colmap_result.num_registered_frames,
        colmap_stats=colmap_result,
        views_per_frame=views_per_frame,
        preset_signature=preset_signature,
    )
