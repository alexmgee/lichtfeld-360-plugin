# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the sharpest frame extractor chunking algorithm."""

from core.sharpest_extractor import SharpestExtractor


def test_simple_chunking():
    """Each chunk should pick the frame with lowest blur score."""
    # (frame_number, blur_score, scene_score)
    metadata = [
        (0, 50.0, 0.0), (1, 30.0, 0.0), (2, 80.0, 0.0), (3, 20.0, 0.0), (4, 60.0, 0.0),
        (5, 40.0, 0.0), (6, 10.0, 0.0), (7, 70.0, 0.0), (8, 55.0, 0.0), (9, 45.0, 0.0),
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    assert len(best) == 2
    assert best[0] == 3   # lowest blur in chunk 0-4
    assert best[1] == 6   # lowest blur in chunk 5-9


def test_scene_change_splits_chunk():
    """A scene change mid-chunk should split it, producing an extra frame."""
    metadata = [
        (0, 50.0, 0.0), (1, 30.0, 0.0), (2, 20.0, 0.5),
        (3, 40.0, 0.0), (4, 35.0, 0.0),
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    # Scene change at frame 2 (score 0.5 >= 0.3) splits:
    #   sub-chunk 1: frames 0,1 -> winner is frame 1 (blur 30)
    #   sub-chunk 2: frames 2,3,4 -> winner is frame 2 (blur 20)
    assert len(best) == 2
    assert best[0] == 1   # lowest blur in sub-chunk before scene change
    assert best[1] == 2   # lowest blur in sub-chunk from scene change onward


def test_single_frame_chunk():
    """A single frame should be selected as the winner."""
    metadata = [(0, 50.0, 0.0)]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    assert len(best) == 1
    assert best[0] == 0


def test_empty_metadata():
    """Empty input should return an empty list."""
    best = SharpestExtractor._parse_best_frames([], chunk_size=5, scene_threshold=0.3)
    assert best == []


def test_exact_chunk_boundary():
    """Frames that exactly fill chunks should not produce extra selections."""
    metadata = [
        (0, 10.0, 0.0), (1, 20.0, 0.0), (2, 30.0, 0.0),
        (3, 15.0, 0.0), (4, 25.0, 0.0), (5, 35.0, 0.0),
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=3, scene_threshold=0.3)
    assert len(best) == 2
    assert best[0] == 0   # lowest blur in chunk 0-2
    assert best[1] == 3   # lowest blur in chunk 3-5


def test_partial_last_chunk():
    """A partial final chunk should still produce a winner."""
    metadata = [
        (0, 50.0, 0.0), (1, 30.0, 0.0), (2, 80.0, 0.0),
        (3, 20.0, 0.0), (4, 60.0, 0.0),
        (5, 40.0, 0.0), (6, 10.0, 0.0),  # partial chunk of 2
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    assert len(best) == 2
    assert best[0] == 3   # lowest blur in chunk 0-4
    assert best[1] == 6   # lowest blur in partial chunk 5-6


def test_multiple_scene_changes_in_chunk():
    """Multiple scene changes in one chunk should produce multiple sub-chunks."""
    metadata = [
        (0, 50.0, 0.0),
        (1, 30.0, 0.4),   # scene change
        (2, 80.0, 0.0),
        (3, 20.0, 0.5),   # scene change
        (4, 60.0, 0.0),
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    # sub-chunk 1: frame 0 -> winner 0
    # sub-chunk 2: frames 1,2 -> winner 1 (blur 30)
    # sub-chunk 3: frames 3,4 -> winner 3 (blur 20)
    assert len(best) == 3
    assert best[0] == 0
    assert best[1] == 1
    assert best[2] == 3


def test_scene_change_at_first_frame_no_split():
    """Scene change on the very first frame of a chunk should not create an empty sub-chunk."""
    metadata = [
        (0, 50.0, 0.5),   # scene change but nothing before it
        (1, 30.0, 0.0),
        (2, 80.0, 0.0),
    ]
    best = SharpestExtractor._parse_best_frames(metadata, chunk_size=5, scene_threshold=0.3)
    # No prior sub-chunk to flush, so all frames stay in one sub-chunk
    assert len(best) == 1
    assert best[0] == 1   # lowest blur


def test_split_at_scenes_empty():
    """_split_at_scenes with empty input returns empty list."""
    result = SharpestExtractor._split_at_scenes([], threshold=0.3)
    assert result == []
