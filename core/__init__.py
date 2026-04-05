# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""PanoSplat core processing modules."""

from .analyzer import VideoAnalyzer, VideoInfo
from .backends import (
    MaskingBackend, YoloSamBackend, Sam3Backend,
    VideoTrackingBackend, FallbackVideoBackend, Sam2VideoBackend,
    get_backend, get_backend_name, get_video_backend,
)
from .colmap_runner import ColmapConfig, ColmapResult, ColmapRunner
from .cubemap_projection import CubemapProjection
from .masker import Masker, MaskConfig, MaskResult, is_masking_available
from .overlap_mask import compute_overlap_masks
from .pipeline import PipelineConfig, PipelineJob, PipelineResult
from .presets import FreeView, Ring, ViewConfig, VIEW_PRESETS
from .reframer import Reframer, ReframeResult, reframe_view, compute_pinhole_intrinsics
from .rig_config import generate_rig_config, write_rig_config
from .sharpest_extractor import SharpestConfig, SharpestExtractor, SharpestResult
from .transforms_writer import colmap_pose_to_c2w_opengl, write_transforms_json

__all__ = [
    # analyzer
    "VideoAnalyzer",
    "VideoInfo",
    # backends
    "MaskingBackend",
    "YoloSamBackend",
    "Sam3Backend",
    "VideoTrackingBackend",
    "FallbackVideoBackend",
    "Sam2VideoBackend",
    "get_backend",
    "get_backend_name",
    "get_video_backend",
    # colmap_runner
    "ColmapConfig",
    "ColmapResult",
    "ColmapRunner",
    # cubemap_projection
    "CubemapProjection",
    # masker
    "Masker",
    "MaskConfig",
    "MaskResult",
    "is_masking_available",
    # overlap_mask
    "compute_overlap_masks",
    # pipeline
    "PipelineConfig",
    "PipelineJob",
    "PipelineResult",
    # presets
    "FreeView",
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
