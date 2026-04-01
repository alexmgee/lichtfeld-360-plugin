# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the video analyzer module."""

from core.analyzer import VideoAnalyzer, VideoInfo


def test_ffprobe_path_exists():
    analyzer = VideoAnalyzer()
    assert analyzer.ffprobe_path is not None
    assert len(analyzer.ffprobe_path) > 0


def test_erp_detection_2_to_1():
    assert VideoInfo.is_equirectangular(7680, 3840) is True


def test_erp_detection_not_erp():
    assert VideoInfo.is_equirectangular(1920, 1080) is False


def test_erp_detection_4k_erp():
    assert VideoInfo.is_equirectangular(3840, 1920) is True


def test_erp_detection_close_but_not_quite():
    assert VideoInfo.is_equirectangular(1920, 1070) is False


def test_erp_detection_zero_height():
    assert VideoInfo.is_equirectangular(1920, 0) is False


def test_video_info_to_dict():
    info = VideoInfo(
        path="/tmp/test.mp4",
        filename="test.mp4",
        format="mp4",
        codec="hevc",
        width=7680,
        height=3840,
        fps=30.0,
        duration_seconds=120.0,
        frame_count=3600,
        is_erp=True,
    )
    d = info.to_dict()
    assert d["width"] == 7680
    assert d["is_equirectangular"] is True
    assert d["codec"] == "hevc"


def test_duration_formatted():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=0, height=0, fps=30, duration_seconds=3661, frame_count=0,
    )
    assert VideoAnalyzer.get_duration_formatted(info) == "1:01:01"


def test_duration_formatted_short():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=0, height=0, fps=30, duration_seconds=65, frame_count=0,
    )
    assert VideoAnalyzer.get_duration_formatted(info) == "1:05"


def test_estimate_frame_count():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=0, height=0, fps=30, duration_seconds=120, frame_count=0,
    )
    assert VideoAnalyzer.estimate_frame_count(info, 2.0) == 60


def test_estimate_frame_count_zero_interval():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=0, height=0, fps=30, duration_seconds=120, frame_count=0,
    )
    assert VideoAnalyzer.estimate_frame_count(info, 0) == 0


def test_recommendations_8k_erp():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=7680, height=3840, fps=30, duration_seconds=60, frame_count=1800,
        is_erp=True,
    )
    VideoAnalyzer._generate_recommendations(info)
    assert info.recommended_interval == 2.0


def test_recommendations_4k_erp():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=3840, height=1920, fps=30, duration_seconds=60, frame_count=1800,
        is_erp=True,
    )
    VideoAnalyzer._generate_recommendations(info)
    assert info.recommended_interval == 1.0


def test_recommendations_non_erp():
    info = VideoInfo(
        path="x", filename="x", format="mp4", codec="h264",
        width=1920, height=1080, fps=30, duration_seconds=60, frame_count=1800,
        is_erp=False,
    )
    VideoAnalyzer._generate_recommendations(info)
    assert info.recommended_interval == 0.5
