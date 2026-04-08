# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility helpers for SAM 3 image-only usage in the plugin.

The upstream ``sam3`` package eagerly imports several video/training modules at
module import time, even when callers only need the image-model API. On the
plugin's Windows Python 3.12 runtime, some of those transitive training/video
dependencies are not importable even though the image-model path itself works.

For the SAM3 cubemap path we only need:
- ``build_sam3_image_model``
- ``Sam3Processor``

So we install tiny placeholder modules for the unused video/training symbols
before importing ``sam3``. This keeps the image path available without claiming
that the video path is ready.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

_IMAGE_IMPORT_STUBS_ACTIVE = False
_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"


def _register_stub(module_name: str, **attrs: Any) -> None:
    """Register a lightweight module stub if it is not already loaded."""
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module


def enable_sam3_image_import_compat() -> None:
    """Install import stubs needed for image-only SAM3 usage.

    These placeholders satisfy eager imports from ``sam3.model_builder`` that
    are only required for video/training flows. They are intentionally minimal
    because the plugin's current SAM3 path is image-only.
    """
    global _IMAGE_IMPORT_STUBS_ACTIVE
    if _IMAGE_IMPORT_STUBS_ACTIVE:
        return

    _register_stub(
        "sam3.train.data.collator",
        BatchedDatapoint=object,
    )
    _register_stub(
        "sam3.model.sam3_video_inference",
        Sam3VideoInferenceWithInstanceInteractivity=type(
            "Sam3VideoInferenceWithInstanceInteractivity", (), {}
        ),
    )
    _register_stub(
        "sam3.model.sam3_video_predictor",
        Sam3VideoPredictorMultiGPU=type("Sam3VideoPredictorMultiGPU", (), {}),
    )
    _IMAGE_IMPORT_STUBS_ACTIVE = True


def import_sam3_image_api():
    """Import the SAM3 image-model API with plugin compatibility shims."""
    enable_sam3_image_import_compat()
    from sam3 import build_sam3_image_model  # type: ignore[import-untyped]
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]

    return build_sam3_image_model, Sam3Processor


def get_sam3_bpe_path() -> str:
    """Return a usable BPE vocabulary path for the SAM3 tokenizer.

    The editable clone used by Reconstruction Zone resolves this asset via
    package resources. The PyPI wheel currently installed in the plugin venv
    expects a relative ``../assets`` path that is not present on disk. We
    therefore fall back to a vendored copy inside this plugin when needed.
    """
    candidates: list[Path] = []
    try:
        import sam3  # type: ignore[import-untyped]

        package_dir = Path(sam3.__file__).resolve().parent
        candidates.append(package_dir / "assets" / _BPE_FILENAME)
        candidates.append(package_dir.parent / "assets" / _BPE_FILENAME)
    except ImportError:
        pass

    plugin_root = Path(__file__).resolve().parent.parent
    candidates.append(plugin_root / "core" / "assets" / _BPE_FILENAME)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"SAM3 tokenizer vocabulary not found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def sam3_image_api_available() -> bool:
    """Return True if the image-model API is importable in this runtime."""
    try:
        import_sam3_image_api()
        get_sam3_bpe_path()
        return True
    except (ImportError, FileNotFoundError):
        return False


def sam3_video_api_available() -> bool:
    """Return True if the real SAM3 video API is importable.

    Once image-only compatibility stubs are active in-process, we intentionally
    report the video API as unavailable so callers don't mistake the stubbed
    image-only import path for real video support.
    """
    if _IMAGE_IMPORT_STUBS_ACTIVE:
        return False
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False
