# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Camera core processing modules."""

from .analyzer import VideoAnalyzer, VideoInfo
from .colmap_runner import ColmapConfig, ColmapResult, ColmapRunner
from .masker import Masker, MaskConfig, MaskResult, is_masking_available
from .pipeline import PipelineConfig, PipelineJob, PipelineResult
from .presets import Ring, ViewConfig, VIEW_PRESETS
from .reframer import Reframer, ReframeResult, reframe_view, compute_pinhole_intrinsics
from .rig_config import generate_rig_config, write_rig_config
from .sharpest_extractor import SharpestConfig, SharpestExtractor, SharpestResult
from .transforms_writer import colmap_pose_to_c2w_opengl, write_transforms_json

__all__ = [
    # analyzer
    "VideoAnalyzer",
    "VideoInfo",
    # colmap_runner
    "ColmapConfig",
    "ColmapResult",
    "ColmapRunner",
    # masker
    "Masker",
    "MaskConfig",
    "MaskResult",
    "is_masking_available",
    # pipeline
    "PipelineConfig",
    "PipelineJob",
    "PipelineResult",
    # presets
    "Ring",
    "ViewConfig",
    "VIEW_PRESETS",
    # reframer
    "Reframer",
    "ReframeResult",
    "reframe_view",
    "compute_pinhole_intrinsics",
    # rig_config
    "generate_rig_config",
    "write_rig_config",
    # sharpest_extractor
    "SharpestConfig",
    "SharpestExtractor",
    "SharpestResult",
    # transforms_writer
    "colmap_pose_to_c2w_opengl",
    "write_transforms_json",
]
