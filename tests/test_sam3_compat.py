# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for SAM3 image-only compatibility shims."""
from __future__ import annotations

import importlib
import sys

import core.sam3_compat as sam3_compat


def _restore_modules(saved: dict[str, object | None]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_enable_sam3_image_import_compat_registers_required_stubs():
    names = [
        "sam3.train.data.collator",
        "sam3.model.sam3_video_inference",
        "sam3.model.sam3_video_predictor",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)

    compat = importlib.reload(sam3_compat)
    try:
        compat.enable_sam3_image_import_compat()

        collator = sys.modules["sam3.train.data.collator"]
        video_inference = sys.modules["sam3.model.sam3_video_inference"]
        video_predictor = sys.modules["sam3.model.sam3_video_predictor"]

        assert compat._IMAGE_IMPORT_STUBS_ACTIVE
        assert getattr(collator, "BatchedDatapoint") is object
        assert hasattr(
            video_inference, "Sam3VideoInferenceWithInstanceInteractivity",
        )
        assert hasattr(video_predictor, "Sam3VideoPredictorMultiGPU")
    finally:
        _restore_modules(saved)
        importlib.reload(sam3_compat)


def test_sam3_video_api_unavailable_after_image_stubs():
    compat = importlib.reload(sam3_compat)
    try:
        compat.enable_sam3_image_import_compat()
        assert not compat.sam3_video_api_available()
    finally:
        importlib.reload(sam3_compat)
