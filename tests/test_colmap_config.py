"""Tests for ColmapConfig COLMAP 4.1 fields."""
from core.colmap_runner import ColmapConfig


def test_default_feature_type():
    c = ColmapConfig()
    assert c.feature_type == "sift"


def test_default_matcher_type():
    c = ColmapConfig()
    assert c.matcher_type == "bruteforce"


def test_default_mapper():
    c = ColmapConfig()
    assert c.mapper == "incremental"


def test_default_ba_solver():
    c = ColmapConfig()
    assert c.ba_solver == "auto"


def test_matcher_type_map():
    """Verify compound FeatureMatcherType dispatch logic."""
    from core.colmap_runner import _MATCHER_TYPE_MAP
    assert _MATCHER_TYPE_MAP[("sift", "bruteforce")] == "SIFT_BRUTEFORCE"
    assert _MATCHER_TYPE_MAP[("sift", "lightglue")] == "SIFT_LIGHTGLUE"
    assert _MATCHER_TYPE_MAP[("aliked_n16rot", "lightglue")] == "ALIKED_LIGHTGLUE"
    assert _MATCHER_TYPE_MAP[("aliked_n32", "bruteforce")] == "ALIKED_BRUTEFORCE"


def test_fisheye_config():
    c = ColmapConfig(
        camera_model="OPENCV_FISHEYE",
        refine_principal_point=True,
        refine_extra_params=True,
        feature_type="sift",
        mapper="incremental",
    )
    assert c.camera_model == "OPENCV_FISHEYE"
    assert c.refine_principal_point is True
    assert c.refine_extra_params is True
