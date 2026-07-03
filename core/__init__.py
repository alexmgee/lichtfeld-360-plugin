# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Plugin core processing modules.

This package keeps its public re-exports lazy so the plugin UI can load
without importing the full masking and COLMAP stack up front.
"""

from importlib import import_module

_EXPORTS = {
    "VideoAnalyzer": (".analyzer", "VideoAnalyzer"),
    "VideoInfo": (".analyzer", "VideoInfo"),
    "MaskingBackend": (".backends", "MaskingBackend"),
    "YoloSamBackend": (".backends", "YoloSamBackend"),
    "Sam3Backend": (".backends", "Sam3Backend"),
    "VideoTrackingBackend": (".backends", "VideoTrackingBackend"),
    "FallbackVideoBackend": (".backends", "FallbackVideoBackend"),
    "Sam2VideoBackend": (".backends", "Sam2VideoBackend"),
    "get_backend": (".backends", "get_backend"),
    "get_backend_name": (".backends", "get_backend_name"),
    "get_video_backend": (".backends", "get_video_backend"),
    "ColmapConfig": (".colmap_runner", "ColmapConfig"),
    "ColmapResult": (".colmap_runner", "ColmapResult"),
    "ColmapRunner": (".colmap_runner", "ColmapRunner"),
    "CubemapProjection": (".cubemap_projection", "CubemapProjection"),
    "Masker": (".masker", "Masker"),
    "MaskConfig": (".masker", "MaskConfig"),
    "MaskResult": (".masker", "MaskResult"),
    "is_masking_available": (".masker", "is_masking_available"),
    "compute_overlap_masks": (".overlap_mask", "compute_overlap_masks"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "PipelineJob": (".pipeline", "PipelineJob"),
    "PipelineResult": (".pipeline", "PipelineResult"),
    "FreeView": (".presets", "FreeView"),
    "Ring": (".presets", "Ring"),
    "ViewConfig": (".presets", "ViewConfig"),
    "VIEW_PRESETS": (".presets", "VIEW_PRESETS"),
    "Reframer": (".reframer", "Reframer"),
    "ReframeResult": (".reframer", "ReframeResult"),
    "reframe_view": (".reframer", "reframe_view"),
    "compute_pinhole_intrinsics": (".reframer", "compute_pinhole_intrinsics"),
    "generate_rig_config": (".rig_config", "generate_rig_config"),
    "write_rig_config": (".rig_config", "write_rig_config"),
    "SharpestConfig": (".sharpest_extractor", "SharpestConfig"),
    "SharpestExtractor": (".sharpest_extractor", "SharpestExtractor"),
    "SharpestResult": (".sharpest_extractor", "SharpestResult"),
    "colmap_pose_to_c2w_opengl": (".transforms_writer", "colmap_pose_to_c2w_opengl"),
    "write_transforms_json": (".transforms_writer", "write_transforms_json"),
    "write_erp_native_transforms": (".transforms_writer", "write_erp_native_transforms"),
    "export_erp_scaffold": (".scaffold", "export_erp_scaffold"),
    "cleanup_pinhole_crops": (".scaffold", "cleanup_pinhole_crops"),
    "cleanup_colmap_artifacts": (".scaffold", "cleanup_colmap_artifacts"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
