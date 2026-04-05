# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for masking setup state detection (two-tier system)."""
from __future__ import annotations

from unittest.mock import patch

from core.setup_checks import MaskingSetupState, check_masking_setup


def _patch_all(**overrides):
    """Helper to patch all check functions with defaults (all False) + overrides."""
    defaults = dict(
        _check_hf_token=False, _check_hf_access=False,
        _check_torch_installed=False, _check_yolo_installed=False,
        _check_sam1_installed=False, _check_sam2_installed=False,
        _check_sam3_installed=False, _check_weights_downloaded=False,
    )
    defaults.update(overrides)
    import contextlib
    return contextlib.ExitStack()  # dummy; real usage below


def test_setup_state_all_missing():
    """When nothing is installed, no backend should be available."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_yolo_installed", return_value=False), \
         patch("core.setup_checks._check_sam1_installed", return_value=False), \
         patch("core.setup_checks._check_sam2_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert not state.is_ready
        assert state.active_backend is None
        assert state.first_incomplete_step == 1
        assert state.capability_level == 0


def test_default_tier_ready():
    """When YOLO + SAM v1 + torch are installed, default tier is ready."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert state.default_tier_ready
        assert not state.premium_tier_ready
        assert not state.video_tracking_ready
        assert state.active_backend == "yolo_sam1"
        assert state.is_ready
        assert state.first_incomplete_step == 0
        assert state.capability_level == 1


def test_video_tracking_ready():
    """When SAM v2 is also installed, video tracking is ready (level 2)."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert state.default_tier_ready
        assert state.video_tracking_ready
        assert state.capability_level == 2


def test_premium_tier_ready():
    """When SAM 3 is also installed with weights, premium tier takes priority."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=True):
        state = check_masking_setup()
        assert state.default_tier_ready
        assert state.premium_tier_ready
        assert state.active_backend == "sam3"
        assert state.is_ready
        assert state.capability_level == 3


def test_torch_missing_nothing_works():
    """Without torch, neither tier is ready."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=True):
        state = check_masking_setup()
        assert not state.default_tier_ready
        assert not state.premium_tier_ready
        assert state.active_backend is None
        assert state.capability_level == 0
