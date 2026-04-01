# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for masking setup state detection."""
from __future__ import annotations

from unittest.mock import patch

from core.setup_checks import MaskingSetupState, check_masking_setup


def test_setup_state_all_missing():
    """When nothing is installed, all steps should be incomplete."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert not state.has_token
        assert not state.has_access
        assert not state.has_torch
        assert not state.has_sam3
        assert not state.has_weights
        assert not state.is_ready


def test_setup_state_all_present():
    """When everything is installed, is_ready should be True."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=True):
        state = check_masking_setup()
        assert state.is_ready


def test_first_incomplete_step_no_token():
    """First incomplete step should be 1 when token is missing."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert state.first_incomplete_step == 1


def test_first_incomplete_step_has_token_no_deps():
    """First incomplete step should be 2 when token verified but deps missing."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert state.first_incomplete_step == 2


def test_first_incomplete_step_has_deps_no_weights():
    """First incomplete step should be 3 when deps installed but weights missing."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False):
        state = check_masking_setup()
        assert state.first_incomplete_step == 3


def test_first_incomplete_step_all_done():
    """First incomplete step should be 0 when everything is set up."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=True):
        state = check_masking_setup()
        assert state.first_incomplete_step == 0
