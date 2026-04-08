# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for masking setup state detection (two-tier system)."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.setup_checks import (
    MaskingSetupState,
    _install_sam2_runtime,
    check_masking_setup,
    check_sam3_setup,
    install_default_tier,
    install_video_tracking,
    make_sam3_install_failure_report,
    verify_hf_token_detailed,
)


def test_setup_state_all_missing():
    """When nothing is installed, no backend should be available."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=False), \
         patch("core.setup_checks._check_yolo_installed", return_value=False), \
         patch("core.setup_checks._check_sam1_installed", return_value=False), \
         patch("core.setup_checks._check_sam2_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert not state.is_ready
        assert not state.fullcircle_ready
        assert not state.sam3_ready
        assert state.active_backend is None
        assert state.first_incomplete_step == 1
        assert state.capability_level == 0


def test_default_tier_ready():
    """When YOLO + SAM v1 + torch are installed but SAM v2 is missing,
    default_tier_ready is True but no method is actually runnable."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert state.default_tier_ready
        assert not state.fullcircle_ready
        assert not state.sam3_ready
        assert not state.masking_ready
        assert not state.is_ready
        assert not state.premium_tier_ready
        assert not state.video_tracking_ready
        assert state.active_backend == "yolo_sam1"
        assert state.first_incomplete_step == 0
        assert state.capability_level == 1


def test_fullcircle_ready():
    """FullCircle requires YOLO+SAM v1+SAM v2. When all present, fullcircle_ready
    is True, sam3_ready is False, and is_ready is True."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert state.fullcircle_ready
        assert not state.sam3_ready
        assert state.masking_ready
        assert state.is_ready
        assert state.video_tracking_ready
        assert state.capability_level == 2


def test_sam3_ready_without_sam2():
    """SAM 3 cubemap masking requires only torch+sam3+weights. No SAM v2 needed.
    fullcircle_ready must remain False, sam3_ready True, is_ready True."""
    with patch("core.setup_checks._check_hf_token", return_value=True), \
         patch("core.setup_checks._check_hf_access", return_value=True), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=False), \
         patch("core.setup_checks._check_sam1_installed", return_value=False), \
         patch("core.setup_checks._check_sam2_installed", return_value=False), \
         patch("core.setup_checks._check_sam3_installed", return_value=True), \
         patch("core.setup_checks._check_weights_downloaded", return_value=True), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert not state.fullcircle_ready
        assert state.sam3_ready
        assert not state.masking_ready
        assert state.is_ready
        assert state.active_backend == "sam3"


def test_video_tracking_ready():
    """When SAM v2 is also installed, video tracking is ready (level 2)."""
    with patch("core.setup_checks._check_hf_token", return_value=False), \
         patch("core.setup_checks._check_hf_access", return_value=False), \
         patch("core.setup_checks._check_torch_installed", return_value=True), \
         patch("core.setup_checks._check_yolo_installed", return_value=True), \
         patch("core.setup_checks._check_sam1_installed", return_value=True), \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._check_sam3_installed", return_value=False), \
         patch("core.setup_checks._check_weights_downloaded", return_value=False), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
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
         patch("core.setup_checks._check_weights_downloaded", return_value=True), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert state.default_tier_ready
        assert state.premium_tier_ready
        assert state.fullcircle_ready
        assert state.sam3_ready
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
         patch("core.setup_checks._check_weights_downloaded", return_value=True), \
         patch("core.setup_checks._check_sam3_video_installed", return_value=False):
        state = check_masking_setup()
        assert not state.default_tier_ready
        assert not state.premium_tier_ready
        assert not state.fullcircle_ready
        assert not state.sam3_ready
        assert state.active_backend is None
        assert state.capability_level == 0


def test_install_sam2_runtime_repairs_shipped_fullcircle_runtime():
    progress = Mock()

    with patch("core.setup_checks._quarantine_broken_sam2_namespace"), \
         patch("core.setup_checks._run_uv_command", return_value=True) as mock_sync, \
         patch("core.setup_checks._check_sam2_installed", return_value=True), \
         patch("core.setup_checks._install_sam2_c_extension"):
        assert _install_sam2_runtime(on_output=progress)

    mock_sync.assert_called_once_with(
        ["sync", "--locked", "--no-dev"],
        on_output=progress,
    )


def test_install_default_tier_reports_fullcircle_runtime_ready():
    progress = Mock()

    with patch("core.setup_checks._install_sam2_runtime", return_value=True) as mock_repair, \
         patch("core.setup_checks._download_sam1_weights") as mock_weights:
        assert install_default_tier(on_output=progress)

    mock_repair.assert_called_once_with(on_output=progress)
    mock_weights.assert_called_once_with(on_output=progress)
    progress.assert_called_with(
        "FullCircle runtime ready. SAM v2 model weights download on first use."
    )


def test_install_video_tracking_is_legacy_repair_alias():
    progress = Mock()

    with patch("core.setup_checks._install_sam2_runtime", return_value=True) as mock_repair:
        assert install_video_tracking(on_output=progress)

    mock_repair.assert_called_once_with(on_output=progress)
    progress.assert_called_with("SAM v2 runtime ready. Model weights download on first use.")


def test_check_sam3_setup_needs_token():
    report = check_sam3_setup(setup_state=MaskingSetupState())

    assert report.token_status == "missing"
    assert report.overall_stage == "needs_token"
    assert "token" in report.next_action.lower()


def test_check_sam3_setup_ready_to_install():
    report = check_sam3_setup(
        setup_state=MaskingSetupState(
            has_torch=True,
            has_token=True,
            has_access=True,
        )
    )

    assert report.access_status == "granted"
    assert report.runtime_status == "missing"
    assert report.overall_stage == "ready_to_install"


def test_check_sam3_setup_needs_weights():
    report = check_sam3_setup(
        setup_state=MaskingSetupState(
            has_torch=True,
            has_token=True,
            has_access=True,
            has_sam3=True,
            has_weights=False,
        )
    )

    assert report.runtime_status == "installed"
    assert report.weights_status == "missing"
    assert report.overall_stage == "needs_weights"


def test_verify_hf_token_detailed_invalid_token():
    fake_hf = SimpleNamespace(
        login=lambda token: None,
        model_info=Mock(side_effect=Exception("401 Unauthorized")),
    )

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), \
         patch("core.setup_checks.check_masking_setup", return_value=MaskingSetupState()):
        report = verify_hf_token_detailed("hf_bad_token")

    assert report.token_status == "invalid"
    assert report.access_status == "unknown"
    assert report.overall_stage == "needs_token"


def test_verify_hf_token_detailed_access_pending():
    fake_hf = SimpleNamespace(
        login=lambda token: None,
        model_info=Mock(side_effect=Exception("403 gated repo")),
    )

    with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), \
         patch("core.setup_checks.check_masking_setup", return_value=MaskingSetupState(has_torch=True)):
        report = verify_hf_token_detailed("hf_pending_token")

    assert report.token_status == "verified"
    assert report.access_status == "pending"
    assert report.overall_stage == "needs_access"


def test_make_sam3_install_failure_report_marks_runtime_broken():
    report = make_sam3_install_failure_report(
        "sam3 import failed",
        setup_state=MaskingSetupState(has_torch=True, has_token=True, has_access=True),
    )

    assert report.runtime_status == "broken"
    assert report.overall_stage == "error"
    assert "Retry" in report.next_action
