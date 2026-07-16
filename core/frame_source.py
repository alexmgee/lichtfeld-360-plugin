# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure helpers for the "Select Image Folder" frame source.

Covers listing a folder's images in natural order, guarding against a
source directory that overlaps the pipeline's output directory, pairing
dual-fisheye front/back images (one shared folder or two separate
folders), validating that every image has a matching mask, and staging
user-supplied images into the pipeline's owned working directories
(mirroring the ERP and dual-fisheye extractor output layouts).

No cv2, no torch, no pycolmap, no panel imports -- path/name/copy logic
only.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _natural_key(name: str):
    """Sort key so "img2" precedes "img10": digit runs sort as ints."""
    parts = re.split(r'(\d+)', name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def list_images_natural(directory) -> tuple[list[Path], Optional[str]]:
    """List images directly inside `directory`, naturally sorted.

    Returns `(sorted_files, warning)`. `warning` is None unless the
    files' frame-number widths (the last digit run in each stem) are
    inconsistent, e.g. a mix of "2", "002", and "010" -- a signal that
    sequential/index-based pairing could misorder frames.
    """
    directory = Path(directory)
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: _natural_key(p.name))

    widths = set()
    for p in files:
        digit_runs = re.findall(r'\d+', p.stem)
        if digit_runs:
            widths.add(len(digit_runs[-1]))

    warning = None
    if len(widths) > 1:
        warning = (
            f"Inconsistent frame-number padding in {directory} "
            f"(widths: {sorted(widths)}); sequential matching may misorder frames."
        )
    return files, warning


def assert_source_outside_output(src, out) -> None:
    """Raise ValueError if `src` and `out` are the same dir, or nested.

    Guards against the image-folder source overlapping the pipeline's
    owned output directory -- staging would then delete or overwrite the
    user's own source images.
    """
    s = Path(src).resolve()
    o = Path(out).resolve()

    if s == o:
        raise ValueError(f"Source and output are the same directory: {s}")
    if o.is_relative_to(s):
        raise ValueError(
            f"Output directory {o} is inside source directory {s}; "
            f"choose an output directory outside the source."
        )
    if s.is_relative_to(o):
        raise ValueError(
            f"Source directory {s} is inside output directory {o}; "
            f"choose a source directory outside the output."
        )
    return None


# Directories every image-folder track writes under the unified layout. The
# dataset (native and/or pinhole) lives under ``<output>/colmap/**``, plugin
# masks under ``<output>/masks/**``, and the panel writes ``<output>/metadata/
# timing.json`` on completion/failure. No image-folder track writes root
# ``images/``, ``sparse/``, ``extracted/``, ``native/`` or ``pinhole/`` any
# more, so those are no longer collision targets and the guard is mode-agnostic.
_WRITE_DIRS_ALWAYS = ("colmap", "masks", "metadata")


def assert_source_reads_safe(source, output) -> None:
    """Guard for a read-in-place image source: the run READS the source and
    must never write into or delete it.

    Unlike ``assert_source_outside_output``, this permits the source to live
    inside the output directory (e.g. ``<output>/images`` or ``<output>/source``)
    as long as it does not overlap a directory the run writes to
    (``_WRITE_DIRS_ALWAYS``). Raises ValueError (with a panel-friendly message)
    on a real collision.
    """
    s = Path(source).resolve()
    o = Path(output).resolve()
    if s == o:
        raise ValueError(
            f"Your image folder and the Output Path are the same folder "
            f"({s}). Point Output somewhere else."
        )
    if o.is_relative_to(s):
        raise ValueError(
            f"The Output Path {o} is inside your image folder. "
            "Point Output to a folder outside your images."
        )
    for name in _WRITE_DIRS_ALWAYS:
        wr = (o / name).resolve()
        if s == wr or s.is_relative_to(wr) or wr.is_relative_to(s):
            raise ValueError(
                f"Your image folder sits where this run writes its '{name}' "
                f"folder. Put your frames in a subfolder (e.g. {o / 'source'}) "
                f"and point Output at the parent."
            )
    return None


def pair_fisheye_one_folder(files) -> list[tuple[Path, Path]]:
    """Pair front_/back_ dual-fisheye images living in one shared folder.

    Classifies each file by its (case-insensitive) "front"/"back" name
    prefix, strips a single leading separator (`_`, `-`, or space) from
    the remainder, and pairs files whose remainder matches
    case-insensitively. Raises ValueError naming the offending file for
    anything unclassifiable or left unpaired.
    """
    fronts: dict[str, tuple[str, Path]] = {}
    backs: dict[str, tuple[str, Path]] = {}

    for f in files:
        p = Path(f)
        lower_name = p.name.lower()
        if lower_name.startswith("front"):
            remainder = p.name[len("front"):]
            if remainder[:1] in ("_", "-", " "):
                remainder = remainder[1:]
            fronts[remainder.lower()] = (remainder, p)
        elif lower_name.startswith("back"):
            remainder = p.name[len("back"):]
            if remainder[:1] in ("_", "-", " "):
                remainder = remainder[1:]
            backs[remainder.lower()] = (remainder, p)
        else:
            raise ValueError(
                f"{p.name} is not a front_/back_ file "
                f"(name must start with 'front' or 'back')"
            )

    for key, (remainder, front_path) in fronts.items():
        if key not in backs:
            raise ValueError(
                f"Unpaired fisheye file {front_path.name}: no back_ file "
                f"matches remainder {remainder!r}"
            )

    for key, (remainder, back_path) in backs.items():
        if key not in fronts:
            raise ValueError(
                f"Unpaired fisheye file {back_path.name}: no front_ file "
                f"matches remainder {remainder!r}"
            )

    pairs = [
        (remainder, front_path, backs[key][1])
        for key, (remainder, front_path) in fronts.items()
    ]
    pairs.sort(key=lambda item: _natural_key(item[0]))
    return [(front_path, back_path) for _remainder, front_path, back_path in pairs]


def pair_fisheye_two_folders(front_dir, back_dir) -> list[tuple[Path, Path]]:
    """Pair dual-fisheye images living in two separate front/back folders.

    Pairs by natural-sorted index (no filename matching): the Nth
    front-folder image pairs with the Nth back-folder image. Raises
    ValueError if the two folders don't hold the same number of images.
    """
    front_files, _ = list_images_natural(front_dir)
    back_files, _ = list_images_natural(back_dir)

    if len(front_files) != len(back_files):
        shorter_len = min(len(front_files), len(back_files))
        longer = front_files if len(front_files) > len(back_files) else back_files
        first_extra = longer[shorter_len]
        raise ValueError(
            f"Front/back image counts differ: {len(front_files)} front vs "
            f"{len(back_files)} back; first extra file: {first_extra.name}"
        )

    return list(zip(front_files, back_files))


def validate_mask_pairing(images, masks) -> None:
    """Raise ValueError naming the first image with no matching mask.

    A mask matches an image when it shares the same stem (extension may
    differ, e.g. `frame_0001.jpg` <-> `frame_0001.png`).
    """
    mask_stems = {Path(m).stem for m in masks}
    for image in sorted(images, key=lambda p: _natural_key(Path(p).name)):
        image_path = Path(image)
        if image_path.stem not in mask_stems:
            raise ValueError(
                f"No mask found for image {image_path.name} "
                f"(expected a mask with stem {image_path.stem!r})"
            )
    return None


def stage_erp_frames(
    source_dir,
    output_dir,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Copy user-supplied ERP images into the pipeline's owned working dir.

    Destination is `<output_dir>/extracted/frames`, idempotently
    recreated (any stale contents are removed first). Original filenames
    are preserved. A cancellation raises RuntimeError before copying the
    next file, so exactly the files already copied remain on disk.
    """
    dest = Path(output_dir) / "extracted" / "frames"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    files, _warning = list_images_natural(source_dir)
    total = len(files)

    staged = 0
    for i, f in enumerate(files, start=1):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")
        shutil.copy2(f, dest / f.name)
        staged += 1
        if on_progress is not None:
            on_progress(i, total, f.name)

    return staged


def stage_fisheye_frames(
    pairs,
    images_root,
    front_source=None,
    back_source=None,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Copy paired front/back images into the dataset's images root.

    Destination is `<images_root>/{front,back}`, idempotently recreated, using
    extract_dual_fisheye's zero-padded 6-digit naming (front/000001.jpg,
    back/000001.jpg, ...) rather than the original filenames. The
    `paired_extraction_manifest.json` is written at `<images_root>.parent`
    (beside the images tree, never inside it -- COLMAP scans `images_root`
    recursively and must not see the JSON). Any stale manifest is overwritten;
    unlike extract_dual_fisheye, this helper is meant to re-run safely.
    """
    staged_root = Path(images_root)
    staged_root.mkdir(parents=True, exist_ok=True)

    front_dest = staged_root / "front"
    if front_dest.exists():
        shutil.rmtree(front_dest)
    front_dest.mkdir(parents=True)

    back_dest = staged_root / "back"
    if back_dest.exists():
        shutil.rmtree(back_dest)
    back_dest.mkdir(parents=True)

    manifest_path = staged_root.parent / "paired_extraction_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    manifest = {
        "schema_version": 1,
        "dataset_type": "paired_split_frames",
        "front_video": str(Path(front_source).resolve()) if front_source else "",
        "back_video": str(Path(back_source).resolve()) if back_source else "",
        "mode": "image_folder",
        "scoring_method": None,
        "scene_detection": False,
        "interval_sec": None,
        "fps": None,
        "selection_method": "image_folder",
        "gpu_accelerated": False,
        "pairs": [],
    }

    total = len(pairs)
    for i, (front_path, back_path) in enumerate(pairs, start=1):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        name_front = f"{i:06d}{Path(front_path).suffix.lower()}"
        name_back = f"{i:06d}{Path(back_path).suffix.lower()}"
        shutil.copy2(front_path, front_dest / name_front)
        shutil.copy2(back_path, back_dest / name_back)

        manifest["pairs"].append({
            "pair_index": i,
            "frame_id": f"{i:06d}",
            "front_image": f"front/{name_front}",
            "back_image": f"back/{name_back}",
            # Staged names are index-based, giving each front/back pair a
            # shared basename; these map back to the user's original files.
            # (The shared basename also lets the dual-fisheye rig group the
            # pair into one frame, but only when the opt-in rig is enabled.)
            "original_front": Path(front_path).name,
            "original_back": Path(back_path).name,
        })

        if on_progress is not None:
            on_progress(i, total, name_front)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest["pairs"])


def relocate_erp_frames_to_colmap(source_dir, colmap_dir) -> tuple[int, bool]:
    """Move a read-in-place ERP source folder's frames into ``<colmap_dir>/images``.

    The ERP Native image-folder path lets COLMAP read the user's frames where
    they sit; only *after* COLMAP succeeds are the frames relocated here so the
    transforms.json ``images/<name>`` paths resolve under ``colmap/``. Filenames
    are preserved exactly -- COLMAP recorded ``image.name`` relative to the flat
    source root, so those names are bare filenames that map straight to
    ``images/<name>``. Only image files are moved; the emptied source folder is
    then removed with ``rmdir``, which refuses -- safely -- if anything
    unexpected (a stray non-image file or a subdirectory) remains, so
    unrecognised content is never force-deleted.

    Returns ``(moved_count, source_removed)``.
    """
    source = Path(source_dir)
    dest = Path(colmap_dir) / "images"
    dest.mkdir(parents=True, exist_ok=True)

    files, _warning = list_images_natural(source)
    moved = 0
    for f in files:
        shutil.move(str(f), str(dest / f.name))
        moved += 1

    source_removed = False
    try:
        source.rmdir()
        source_removed = True
    except OSError:
        pass

    return moved, source_removed


def graceful_delete_source(folder, output_root) -> tuple[int, bool]:
    """Delete a copy-staged fisheye source folder AFTER its frames are safely
    preserved in a kept native dataset (the fisheye Native/Both delete-on-success
    step).

    Removes only files whose suffix is in ``IMAGE_EXTENSIONS``, then ``rmdir``.
    Any leftover sidecar, subdirectory, or unsupported/duplicate-name file blocks
    the ``rmdir`` (``OSError`` caught) and the folder is KEPT -- this is never an
    ``rmtree``, so unrecognised content is never force-deleted.

    A source in a plain subfolder of the Output Path (e.g. ``<output>/frames``)
    is absorbed normally -- the same supported pattern ERP Native absorbs via
    ``relocate_erp_frames_to_colmap``. Refuses outright (``ValueError``, before
    deleting anything) only on genuine self-deletion: ``folder`` IS the output
    root (or contains it), or sits inside one of the run's own write dirs
    (``colmap/``, ``masks/``, ``metadata/``) -- deleting there would eat the
    dataset just built.

    Returns ``(removed_count, folder_removed)``.
    """
    src = Path(folder).resolve()
    out = Path(output_root).resolve()
    if src == out or out.is_relative_to(src):
        raise ValueError(
            f"Refusing to delete {src}: it is the Output directory."
        )
    for name in _WRITE_DIRS_ALWAYS:
        wr = (out / name).resolve()
        if src == wr or src.is_relative_to(wr):
            raise ValueError(
                f"Refusing to delete {src}: it is inside this run's '{name}' "
                f"output folder."
            )
    if not src.exists():
        return 0, False

    removed = 0
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            p.unlink()
            removed += 1

    folder_removed = False
    try:
        src.rmdir()
        folder_removed = True
    except OSError:
        pass

    return removed, folder_removed


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) from JPEG bytes by scanning to its SOF marker."""
    i = 2  # skip the SOI marker (FF D8)
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1  # fill byte; keep scanning
            continue
        i += 2
        if marker == 0x00 or marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue  # stuffing / standalone markers carry no length payload
        seg_len = int.from_bytes(data[i:i + 2], "big")
        # SOF0..SOF15 hold the frame dimensions (excluding DHT/JPG/DAC).
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height = int.from_bytes(data[i + 3:i + 5], "big")
            width = int.from_bytes(data[i + 5:i + 7], "big")
            return width, height
        i += seg_len
    raise ValueError("No SOF marker found; not a decodable JPEG.")


def _image_dimensions(path) -> tuple[int, int]:
    """Return (width, height) of a PNG or JPEG by parsing its header only.

    Pure-Python (no cv2) so image-folder stats stay unit-testable without
    the CUDA/OpenCV DLL bootstrap that full image decode requires.
    """
    data = Path(path).read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    if data[:2] == b"\xff\xd8":
        return _jpeg_dimensions(data)
    raise ValueError(f"Unsupported image (not PNG or JPEG): {path}")


def staged_frame_stats(directory) -> tuple[int, int, int]:
    """Return (count, width, height) for a folder of images.

    Substitutes the values a video extractor would report (frame count and
    the ERP frame dimensions) when the pipeline skips Stage 1 and uses a
    user-supplied image folder instead. Dimensions come from the first
    image in natural order.
    """
    files, _warning = list_images_natural(directory)
    if not files:
        raise ValueError(f"No images found in {directory}")
    width, height = _image_dimensions(files[0])
    return len(files), width, height
