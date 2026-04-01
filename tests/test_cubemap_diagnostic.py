# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic tests for cubemap reprojection correctness.

Creates a synthetic equirectangular image with known color-coded regions
and verifies each cubemap face captures the expected part of the sphere.
"""

import numpy as np
import cv2

from core.presets import VIEW_PRESETS
from core.reframer import reframe_view, create_rotation_matrix


def _make_test_equirect(h: int = 512, w: int = 1024) -> np.ndarray:
    """Create an equirect image with 6 color-coded sextants.

    Equirectangular layout (center = lon 0, lat 0):
        Horizontal: left edge = lon -180, center = lon 0, right edge = lon +180
        Vertical: top = lat +90 (north pole), bottom = lat -90 (south pole)

    We paint 4 longitude bands × 2 latitude halves so each cubemap face
    should see a dominant color:

        Lon band     | Columns      | Color (BGR)      | Faces seeing it
        -------------|------------- |------------------|------------------
        -180..-90    | 0..255       | Red (0,0,200)    | 00_02 right half
        -90..0       | 256..511     | Green (0,180,0)  | 00_00 left half, 00_03 right half
        0..+90       | 512..767     | Blue (200,0,0)   | 00_00 right half, 00_01 left half
        +90..+180    | 768..1023    | Yellow (0,200,200)| 00_01 right half, 00_02 left half

    Upper half (lat 0..+90) is full brightness, lower half (lat -90..0)
    is half brightness.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 4 longitude quadrants
    img[:, :w // 4] = [0, 0, 200]           # Red: lon -180..-90
    img[:, w // 4:w // 2] = [0, 180, 0]     # Green: lon -90..0
    img[:, w // 2:3 * w // 4] = [200, 0, 0] # Blue: lon 0..+90
    img[:, 3 * w // 4:] = [0, 200, 200]     # Yellow: lon +90..+180

    # Darken bottom half (south hemisphere)
    img[h // 2:] = (img[h // 2:].astype(np.float32) * 0.5).astype(np.uint8)

    return img


def _face_mean_bgr(equirect: np.ndarray, yaw: float, pitch: float,
                   fov: float, size: int = 256) -> tuple[float, float, float]:
    """Reframe one face and return its mean BGR."""
    face = reframe_view(equirect, fov_deg=fov, yaw_deg=yaw,
                        pitch_deg=pitch, out_size=size)
    b, g, r = face.mean(axis=(0, 1))
    return float(b), float(g), float(r)


def _dominant_channel(bgr: tuple[float, float, float]) -> str:
    """Return which channel dominates, or 'mixed'."""
    b, g, r = bgr
    mx = max(b, g, r)
    if mx < 10:
        return "black"
    channels = []
    if r > mx * 0.6:
        channels.append("R")
    if g > mx * 0.6:
        channels.append("G")
    if b > mx * 0.6:
        channels.append("B")
    return "+".join(channels) if channels else "mixed"


class TestCubemapFaceContent:
    """Verify each cubemap face captures the correct part of the sphere."""

    def setup_method(self):
        self.equirect = _make_test_equirect()
        self.config = VIEW_PRESETS["cubemap"]
        self.views = {
            name: (yaw, pitch, fov)
            for yaw, pitch, fov, name, _flip in self.config.get_all_views()
        }

    def _get_face(self, name: str) -> np.ndarray:
        yaw, pitch, fov = self.views[name]
        return reframe_view(self.equirect, fov_deg=fov, yaw_deg=yaw,
                            pitch_deg=pitch, out_size=256)

    def _face_bgr(self, name: str) -> tuple[float, float, float]:
        yaw, pitch, fov = self.views[name]
        return _face_mean_bgr(self.equirect, yaw, pitch, fov)

    # -- Horizon faces --

    def test_front_face_00_00_sees_green_blue_boundary(self):
        """yaw=0 looks at lon=0: left half Green, right half Blue."""
        face = self._get_face("00_00")
        left_half = face[:, :128]
        right_half = face[:, 128:]

        # Left half should be predominantly green
        lb, lg, lr = left_half.mean(axis=(0, 1))
        assert lg > lb and lg > lr, (
            f"00_00 left half should be green, got BGR=({lb:.0f},{lg:.0f},{lr:.0f})"
        )

        # Right half should be predominantly blue
        rb, rg, rr = right_half.mean(axis=(0, 1))
        assert rb > rg and rb > rr, (
            f"00_00 right half should be blue, got BGR=({rb:.0f},{rg:.0f},{rr:.0f})"
        )

    def test_right_face_00_01_sees_blue_yellow_boundary(self):
        """yaw=90 looks at lon=+90: left half Blue, right half Yellow."""
        face = self._get_face("00_01")
        left_half = face[:, :128]
        right_half = face[:, 128:]

        lb, lg, lr = left_half.mean(axis=(0, 1))
        assert lb > lg and lb > lr, (
            f"00_01 left half should be blue, got BGR=({lb:.0f},{lg:.0f},{lr:.0f})"
        )

        rb, rg, rr = right_half.mean(axis=(0, 1))
        assert rg > 50 and rr > 50, (
            f"00_01 right half should be yellow (R+G), got BGR=({rb:.0f},{rg:.0f},{rr:.0f})"
        )

    def test_back_face_00_02_sees_yellow_red_boundary(self):
        """yaw=180 looks at lon=180/-180: left half Yellow, right half Red."""
        face = self._get_face("00_02")
        left_half = face[:, :128]
        right_half = face[:, 128:]

        lb, lg, lr = left_half.mean(axis=(0, 1))
        assert lg > 50 and lr > 50, (
            f"00_02 left half should be yellow, got BGR=({lb:.0f},{lg:.0f},{lr:.0f})"
        )

        rb, rg, rr = right_half.mean(axis=(0, 1))
        assert rr > rg and rr > rb, (
            f"00_02 right half should be red, got BGR=({rb:.0f},{rg:.0f},{rr:.0f})"
        )

    def test_left_face_00_03_sees_red_green_boundary(self):
        """yaw=270 looks at lon=-90: left half Red, right half Green."""
        face = self._get_face("00_03")
        left_half = face[:, :128]
        right_half = face[:, 128:]

        lb, lg, lr = left_half.mean(axis=(0, 1))
        assert lr > lg and lr > lb, (
            f"00_03 left half should be red, got BGR=({lb:.0f},{lg:.0f},{lr:.0f})"
        )

        rb, rg, rr = right_half.mean(axis=(0, 1))
        assert rg > rr and rg > rb, (
            f"00_03 right half should be green, got BGR=({rb:.0f},{rg:.0f},{rr:.0f})"
        )

    # -- Pole faces --

    def test_zenith_02_00_is_brighter_than_nadir_01_00(self):
        """Zenith face (02_00, pitch=90) should be brighter (upper hemisphere)."""
        zenith_bgr = self._face_bgr("02_00")
        nadir_bgr = self._face_bgr("01_00")
        zenith_lum = sum(zenith_bgr) / 3
        nadir_lum = sum(nadir_bgr) / 3
        assert zenith_lum > nadir_lum * 1.3, (
            f"Zenith should be notably brighter: zenith_lum={zenith_lum:.0f}, nadir_lum={nadir_lum:.0f}"
        )

    def test_zenith_02_00_has_all_four_colors(self):
        """Zenith face (02_00) should see parts of all 4 longitude quadrants."""
        face = self._get_face("02_00")
        b, g, r = face.mean(axis=(0, 1))
        assert b > 20 and g > 20 and r > 20, (
            f"Zenith face should have all color channels present, got BGR=({b:.0f},{g:.0f},{r:.0f})"
        )

    def test_nadir_01_00_has_all_four_colors(self):
        """Nadir face (01_00) should see parts of all 4 longitude quadrants."""
        face = self._get_face("01_00")
        b, g, r = face.mean(axis=(0, 1))
        assert b > 10 and g > 10 and r > 10, (
            f"Nadir face should have all color channels present, got BGR=({b:.0f},{g:.0f},{r:.0f})"
        )

    # -- Horizon faces: upper/lower brightness --

    def test_horizon_faces_brighter_on_top(self):
        """All horizon faces should be brighter in their top half (north hemisphere)."""
        for name in ["00_00", "00_01", "00_02", "00_03"]:
            face = self._get_face(name)
            top_lum = face[:128].mean()
            bot_lum = face[128:].mean()
            assert top_lum > bot_lum * 1.3, (
                f"{name} top half should be brighter: top={top_lum:.0f}, bot={bot_lum:.0f}"
            )

    # -- Rotation matrix sanity for cubemap angles --

    def test_cubemap_forward_vectors(self):
        """Every cubemap face's forward vector should point where yaw/pitch say."""
        expected_forwards = {
            "00_00": [0, 0, 1],       # yaw=0: forward
            "00_01": [1, 0, 0],       # yaw=90: right
            "00_02": [0, 0, -1],      # yaw=180: back
            "00_03": [-1, 0, 0],      # yaw=270: left
            "01_00": [0, -1, 0],      # pitch=-90: down (nadir)
            "02_00": [0, 1, 0],       # pitch=90: up (zenith)
        }
        for name, expected in expected_forwards.items():
            yaw, pitch, fov = self.views[name]
            R = create_rotation_matrix(yaw, pitch)
            forward = R @ np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(
                forward, expected, atol=1e-6,
                err_msg=f"{name} (yaw={yaw}, pitch={pitch}) forward vector wrong"
            )

    # -- Coverage: adjacent faces should share edge content --

    def test_front_and_right_share_boundary(self):
        """The right edge of 00_00 and left edge of 00_01 should see similar content."""
        front = self._get_face("00_00")
        right = self._get_face("00_01")

        # Right edge of front face
        front_edge = front[:, -16:].mean(axis=(0, 1))
        # Left edge of right face
        right_edge = right[:, :16].mean(axis=(0, 1))

        # Both should be predominantly blue (lon ~ +45..+90)
        assert front_edge[0] > 50, f"Front right edge should be blue, got BGR={front_edge}"
        assert right_edge[0] > 50, f"Right left edge should be blue, got BGR={right_edge}"


class TestCubemapDiagnosticReport:
    """Print a full diagnostic report (always passes, for visual inspection)."""

    def test_print_diagnostic(self):
        equirect = _make_test_equirect()
        config = VIEW_PRESETS["cubemap"]

        print("\n=== CUBEMAP REPROJECTION DIAGNOSTIC ===\n")
        print("Equirect: 1024x512, 4 color bands (R/G/B/Y), upper bright / lower dark\n")

        for yaw, pitch, fov, name, _flip in config.get_all_views():
            face = reframe_view(equirect, fov_deg=fov, yaw_deg=yaw,
                                pitch_deg=pitch, out_size=256)
            b, g, r = face.mean(axis=(0, 1))
            R = create_rotation_matrix(yaw, pitch)
            fwd = R @ np.array([0.0, 0.0, 1.0])
            dom = _dominant_channel((b, g, r))

            print(f"  {name}: yaw={yaw:6.1f} pitch={pitch:6.1f} "
                  f"fwd=[{fwd[0]:+.2f},{fwd[1]:+.2f},{fwd[2]:+.2f}] "
                  f"BGR=({b:5.1f},{g:5.1f},{r:5.1f}) dominant={dom}")

        print("\n=== END DIAGNOSTIC ===")
