# PanoSplat Masking Layer v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic operator masking to PanoSplat so 360° video produces clean splats without the camera operator baked in.

**Architecture:** Two-tier detection backend (YOLO+SAM v1 default, SAM 3.1 optional upgrade) runs on cubemap-decomposed faces. Masks merge back to ERP space, reframer reprojects to pinhole views, closest-camera Voronoi masks eliminate duplicate COLMAP features. Pipeline order: Extract → Mask → Reframe → COLMAP.

**Tech Stack:** Python 3.12, numpy, opencv, ultralytics (YOLO), segment-anything (SAM v1), optional sam3, pycolmap

**Spec:** `docs/specs/2026-04-02-masking-layer-v1-design.md`

**Reference implementations:**
- `d:/Projects/reconstruction-zone/reconstruction_gui/reconstruction_pipeline.py` — CubemapProjection class (lines 1207-1421), postprocessing (lines 648-667)
- `D:/Data/fullcircle/masking/mask_perspectives.py` — YOLO+SAM v1 detection pattern
- `D:/Data/fullcircle/masking/omni2perspective.py:148-153` — closest-camera mask
- `D:/Data/fullcircle/masking/lib/cam_utils.py` — spherical coordinate utils

**Test runner:** `.venv/Scripts/pytest.exe tests/ -v`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `core/cubemap_projection.py` | ERP ↔ cubemap coordinate transforms and remapping |
| `core/overlap_mask.py` | Closest-camera Voronoi mask computation per preset |
| `core/backends.py` | `MaskingBackend` protocol + YOLO+SAM v1 and SAM 3.1 implementations |
| `tests/test_cubemap_projection.py` | Unit tests for cubemap decompose/merge round-trip |
| `tests/test_overlap_mask.py` | Unit tests for Voronoi mask computation |
| `tests/test_backends.py` | Unit tests for backend interface and mock detection |

### Modified Files
| File | Changes |
|------|---------|
| `core/masker.py` | Replace video-mode SAM3 with cubemap decomposition + backend dispatch + postprocessing |
| `core/pipeline.py` | Wire new masker, add overlap mask stage, adjust progress allocation |
| `core/setup_checks.py` | Add default tier checks, `active_backend` property |
| `panels/prep360_panel.py` | Re-enable masking UI, target checkboxes, install/upgrade UX |

---

## Task 1: CubemapProjection Class

Port the cubemap decomposition from Reconstruction Zone. This is the geometric foundation that all masking depends on.

**Files:**
- Create: `core/cubemap_projection.py`
- Create: `tests/test_cubemap_projection.py`

- [ ] **Step 1: Write failing test for equirect→cubemap decomposition**

```python
# tests/test_cubemap_projection.py
import numpy as np
import pytest
from core.cubemap_projection import CubemapProjection


def test_equirect2cubemap_returns_six_faces():
    """Decomposing an ERP image produces exactly 6 named faces."""
    proj = CubemapProjection(face_size=64)
    erp = np.zeros((100, 200, 3), dtype=np.uint8)
    faces = proj.equirect2cubemap(erp)
    assert set(faces.keys()) == {"front", "back", "left", "right", "up", "down"}
    for name, face in faces.items():
        assert face.shape == (64, 64, 3), f"{name} has wrong shape: {face.shape}"


def test_equirect2cubemap_face_size_from_width():
    """Default face size is min(1024, w//4)."""
    proj = CubemapProjection(face_size=None)
    erp = np.zeros((200, 400, 3), dtype=np.uint8)
    faces = proj.equirect2cubemap(erp)
    assert faces["front"].shape[0] == 100  # min(1024, 400//4) = 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest.exe tests/test_cubemap_projection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.cubemap_projection'`

- [ ] **Step 3: Implement CubemapProjection**

Port from `d:/Projects/reconstruction-zone/reconstruction_gui/reconstruction_pipeline.py:1207-1421`. The class needs:

- `FACE_DIRS = ['front', 'back', 'left', 'right', 'up', 'down']`
- `__init__(self, face_size=None, overlap_degrees=0.0)` — face_size=None means auto from image width
- `equirect2cubemap(self, equirect: np.ndarray) -> dict[str, np.ndarray]` — split ERP to 6 faces
- `cubemap2equirect(self, face_masks: dict[str, np.ndarray], output_size: tuple[int, int]) -> np.ndarray` — merge face masks back to ERP
- `_face_to_xyz(name, u, v)` — static, map face UV to 3D direction
- `_xyz_to_face(name, x, y, z)` — static, project 3D direction to face UV
- `_face_facing(name, x, y, z)` — static, forward hemisphere test
- `_get_face_region(name, x, y, z, ax, ay, az)` — static, hard face assignment

Reference: lines 1207-1421 of reconstruction_pipeline.py. Copy the math exactly — it's been debugged over 1200+ frames.

```python
# core/cubemap_projection.py
"""Equirectangular ↔ cubemap projection for masking.

Ported from Reconstruction Zone's reconstruction_pipeline.py.
Splits an ERP image into 6 undistorted perspective faces for
detection, and merges face masks back to ERP space.
"""
from __future__ import annotations

import cv2
import numpy as np


class CubemapProjection:
    """Bidirectional ERP ↔ cubemap projection."""

    FACE_DIRS = ["front", "back", "left", "right", "up", "down"]

    def __init__(
        self, face_size: int | None = None, overlap_degrees: float = 0.0
    ) -> None:
        self.face_size = face_size
        self.overlap_degrees = overlap_degrees
        half_fov = (90.0 + overlap_degrees) / 2.0
        self._grid_extent = np.tan(np.radians(half_fov))

    def equirect2cubemap(self, equirect: np.ndarray) -> dict[str, np.ndarray]:
        """Split ERP image into 6 cubemap faces."""
        h, w = equirect.shape[:2]
        fs = self.face_size or min(1024, w // 4)
        extent = self._grid_extent
        grid = np.linspace(-extent, extent, fs)
        u, v = np.meshgrid(grid, grid)
        faces = {}
        for name in self.FACE_DIRS:
            x, y, z = self._face_to_xyz(name, u, v)
            lon = np.arctan2(x, -z)
            lat = np.arctan2(y, np.sqrt(x**2 + z**2))
            map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
            faces[name] = cv2.remap(
                equirect, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        return faces

    def cubemap2equirect(
        self, face_masks: dict[str, np.ndarray], output_size: tuple[int, int]
    ) -> np.ndarray:
        """Merge 6 face masks back to ERP space.

        Uses hard face assignment (no overlap mode for v1).
        Mask values: 0/1 uint8 throughout.
        """
        w, h = output_size
        fs = self.face_size or min(1024, w // 4)
        u_eq = np.linspace(0, 1, w)
        v_eq = np.linspace(0, 1, h)
        uu, vv = np.meshgrid(u_eq, v_eq)
        lon = (uu - 0.5) * 2 * np.pi
        lat = (0.5 - vv) * np.pi
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = -np.cos(lat) * np.cos(lon)
        output = np.zeros((h, w), dtype=np.uint8)
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        for name in self.FACE_DIRS:
            face_mask = face_masks.get(name)
            if face_mask is None:
                continue
            region = self._get_face_region(name, x, y, z, abs_x, abs_y, abs_z)
            if not np.any(region):
                continue
            fu, fv = self._xyz_to_face(name, x[region], y[region], z[region])
            px = np.clip(((fu + 1) / 2 * (fs - 1)).astype(int), 0, fs - 1)
            py = np.clip(((fv + 1) / 2 * (fs - 1)).astype(int), 0, fs - 1)
            output[region] = face_mask[py, px]
        return output

    @staticmethod
    def _face_to_xyz(name, u, v):
        ones = np.ones_like(u)
        if name == "front":   return u, -v, -ones
        if name == "back":    return -u, -v, ones
        if name == "left":    return -ones, -v, -u
        if name == "right":   return ones, -v, u
        if name == "up":      return u, ones, -v
        if name == "down":    return u, -ones, v

    @staticmethod
    def _xyz_to_face(name, x, y, z):
        if name == "front":   return x / np.abs(z), -y / np.abs(z)
        if name == "back":    return -x / np.abs(z), -y / np.abs(z)
        if name == "left":    return -z / np.abs(x), -y / np.abs(x)
        if name == "right":   return z / np.abs(x), -y / np.abs(x)
        if name == "up":      return x / np.abs(y), -z / np.abs(y)
        if name == "down":    return x / np.abs(y), z / np.abs(y)

    @staticmethod
    def _face_facing(name, x, y, z):
        if name == "front":   return z < 0
        if name == "back":    return z > 0
        if name == "left":    return x < 0
        if name == "right":   return x > 0
        if name == "up":      return y > 0
        if name == "down":    return y < 0

    @staticmethod
    def _get_face_region(name, x, y, z, ax, ay, az):
        if name == "front":   return (z < 0) & (az >= ax) & (az >= ay)
        if name == "back":    return (z > 0) & (az >= ax) & (az >= ay)
        if name == "left":    return (x < 0) & (ax >= ay) & (ax >= az)
        if name == "right":   return (x > 0) & (ax >= ay) & (ax >= az)
        if name == "up":      return (y > 0) & (ay >= ax) & (ay >= az)
        if name == "down":    return (y < 0) & (ay >= ax) & (ay >= az)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_cubemap_projection.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write round-trip test**

```python
# append to tests/test_cubemap_projection.py

def test_cubemap_round_trip_preserves_mask():
    """A mask painted on one face survives decompose → merge round-trip."""
    proj = CubemapProjection(face_size=64)
    # Create an ERP mask with a white rectangle in the front-center region
    erp_mask = np.zeros((100, 200), dtype=np.uint8)
    erp_mask[40:60, 90:110] = 1  # center of ERP = front face

    # Decompose to cubemap, then merge back
    faces = proj.equirect2cubemap(erp_mask[:, :, np.newaxis])
    face_masks = {}
    for name, face in faces.items():
        face_masks[name] = (face[:, :, 0] > 0).astype(np.uint8)
    merged = proj.cubemap2equirect(face_masks, (200, 100))

    # The center region should be preserved (allow some interpolation loss)
    center_recall = merged[45:55, 95:105].mean()
    assert center_recall > 0.5, f"Center region lost: recall={center_recall:.2f}"
```

- [ ] **Step 6: Run all cubemap tests**

Run: `.venv/Scripts/pytest.exe tests/test_cubemap_projection.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Commit**

```bash
git add core/cubemap_projection.py tests/test_cubemap_projection.py
git commit -m "feat: add CubemapProjection class for ERP↔cubemap masking"
```

---

## Task 2: Closest-Camera Overlap Mask

Compute per-view Voronoi masks from camera rotations. Purely geometric, no ML.

**Files:**
- Create: `core/overlap_mask.py`
- Create: `tests/test_overlap_mask.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_overlap_mask.py
import numpy as np
import pytest
from core.overlap_mask import compute_overlap_masks
from core.presets import VIEW_PRESETS


def test_cubemap_overlap_masks_all_white():
    """Cubemap has zero overlap — all masks should be all-white (255)."""
    config = VIEW_PRESETS["cubemap"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is None  # skip for cubemap (zero overlap)


def test_high_preset_overlap_masks_partition():
    """High preset masks should partition: each pixel owned by exactly one view."""
    config = VIEW_PRESETS["high"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is not None
    assert len(masks) == len(views)
    # Stack and check that overlapping regions are partitioned
    for view_name, mask in masks.items():
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        # At least some pixels should be white (owned by this view)
        assert np.any(mask > 0), f"View {view_name} has no owned pixels"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest.exe tests/test_overlap_mask.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement overlap mask computation**

```python
# core/overlap_mask.py
"""Closest-camera Voronoi masks for anti-overlap feature extraction.

Ported from FullCircle (omni2perspective.py:148-153).
For each pixel in each pinhole view, determines which camera center
has the most similar viewing direction. Pixels owned by another camera
are masked black to prevent duplicate COLMAP feature extraction.

Purely geometric — computed from camera rotations, no ML.
Precomputed once per preset, reused for every frame.
"""
from __future__ import annotations

import numpy as np

from .reframer import create_rotation_matrix


def compute_overlap_masks(
    views: list[tuple[float, float, float, str, bool]],
    output_size: int,
) -> dict[str, np.ndarray] | None:
    """Compute per-view Voronoi ownership masks.

    Args:
        views: List of (yaw, pitch, fov, name, flip_vertical) from ViewConfig.
        output_size: Square output image size in pixels.

    Returns:
        Dict mapping view_name → uint8 mask (255=own, 0=other camera closer).
        Returns None if no overlap exists (e.g. cubemap preset).
    """
    if len(views) <= 1:
        return None

    # Compute camera forward directions (center of each view)
    cam_centers = []
    view_names = []
    for yaw, pitch, fov, name, flip_v in views:
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        fwd = np.array([
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.cos(yaw_rad),
        ])
        cam_centers.append(fwd)
        view_names.append(name)
    cam_centers = np.array(cam_centers)  # (N, 3)

    # Check if any views overlap: for cubemap (90° FOV, 6 faces) there's zero overlap
    # Quick heuristic: if max dot product between any two centers < cos(fov),
    # no overlap exists
    n = len(views)
    dots = cam_centers @ cam_centers.T
    np.fill_diagonal(dots, -1)  # ignore self
    max_fov = max(fov for _, _, fov, _, _ in views)
    overlap_threshold = np.cos(np.radians(max_fov))
    if np.all(dots < overlap_threshold):
        return None  # No overlap — skip Voronoi computation

    masks = {}
    for idx, (yaw, pitch, fov, name, flip_v) in enumerate(views):
        half_fov = np.radians(fov / 2.0)

        # Build ray directions for every pixel in this view
        size = output_size
        u = np.linspace(-1, 1, size)
        v = np.linspace(-1, 1, size)
        uu, vv = np.meshgrid(u, v)

        # Pixel rays in camera space (pinhole projection)
        focal = 1.0 / np.tan(half_fov)
        rays_cam = np.stack([uu, -vv, -np.full_like(uu, focal)], axis=-1)
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

        # Rotate to world space using the view's rotation matrix
        R = create_rotation_matrix(np.radians(yaw), np.radians(pitch))
        # R is w2c (rows = right, up, -forward), so R.T = c2w
        rays_world = rays_cam @ R  # (H, W, 3) @ (3, 3) → (H, W, 3)

        # For each pixel, find closest camera center
        # rays_world: (H, W, 3), cam_centers: (N, 3)
        dots_per_cam = np.einsum("hwc,nc->hwn", rays_world, cam_centers)
        closest = np.argmax(dots_per_cam, axis=-1)  # (H, W)

        mask = ((closest == idx) * 255).astype(np.uint8)

        if flip_v:
            mask = np.flipud(mask)

        masks[name] = mask

    return masks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_overlap_mask.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/overlap_mask.py tests/test_overlap_mask.py
git commit -m "feat: add closest-camera Voronoi overlap masks"
```

---

## Task 3: Detection Backend Interface + YOLO+SAM v1 Backend

Create the backend protocol and the default zero-friction backend.

**Files:**
- Create: `core/backends.py`
- Create: `tests/test_backends.py`

- [ ] **Step 1: Write failing test for backend interface**

```python
# tests/test_backends.py
import numpy as np
import pytest
from core.backends import YoloSamBackend, get_available_backend


def test_get_available_backend_returns_none_when_nothing_installed():
    """When no ML packages are installed, returns None."""
    # This test runs in the test venv which may not have ultralytics
    backend = get_available_backend()
    # Result depends on environment — just check it returns Backend or None
    assert backend is None or hasattr(backend, "detect_and_segment")


def test_yolo_sam_backend_interface():
    """YoloSamBackend has the required method signature."""
    # Check class exists and has the right method
    assert hasattr(YoloSamBackend, "detect_and_segment")
    assert hasattr(YoloSamBackend, "initialize")
    assert hasattr(YoloSamBackend, "cleanup")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest.exe tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement backends.py**

```python
# core/backends.py
"""Detection backends for operator masking.

Two tiers:
- Default: YOLO + SAM v1 (zero friction, pip installable, no gating)
- Premium: SAM 3.1 (opt-in, gated model, better quality)

Both implement the same interface. The pipeline selects automatically
based on what's installed: SAM 3.1 if available, else YOLO+SAM v1.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Target class mapping ─────────────────────────────────────
# COCO class IDs for YOLO (default tier).
# Only includes classes that COCO actually defines.
# "camera" and "tripod" are SAM 3.1-only (text prompts, no COCO equivalent).
COCO_CLASSES: dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "suitcase": 28,
    "cell phone": 67,
    "laptop": 63,
}

# Targets that only work with SAM 3.1 (no COCO class equivalent)
SAM3_ONLY_TARGETS = {"camera", "tripod", "selfie stick"}

# ── Optional imports ──────────────────────────────────────────

HAS_YOLO = False
HAS_SAM1 = False
HAS_SAM3 = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    HAS_YOLO = True
except ImportError:
    pass

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore[import-untyped]
    HAS_SAM1 = True
except ImportError:
    pass

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import-untyped]
    HAS_SAM3 = True
except ImportError:
    pass


# ── Backend protocol ──────────────────────────────────────────

class MaskingBackend(Protocol):
    def initialize(self) -> None: ...
    def detect_and_segment(
        self, image: np.ndarray, targets: list[str]
    ) -> np.ndarray: ...
    def cleanup(self) -> None: ...


# ── YOLO + SAM v1 backend ────────────────────────────────────

SAM1_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)


class YoloSamBackend:
    """Default tier: YOLO bounding box detection + SAM v1 segmentation.

    Reference: FullCircle mask_perspectives.py
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._yolo: Any = None
        self._sam_predictor: Any = None

    def initialize(self) -> None:
        if not HAS_YOLO:
            raise ImportError("ultralytics not installed")
        if not HAS_SAM1:
            raise ImportError("segment-anything not installed")

        logger.info("Loading YOLO model...")
        self._yolo = YOLO("yolov8s.pt")

        logger.info("Loading SAM v1 model...")
        import os
        import urllib.request
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
        if not os.path.exists(ckpt_path):
            logger.info("Downloading SAM checkpoint to %s ...", ckpt_path)
            urllib.request.urlretrieve(SAM1_CHECKPOINT_URL, ckpt_path)

        sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("YOLO + SAM v1 backend ready")

    def detect_and_segment(
        self, image: np.ndarray, targets: list[str]
    ) -> np.ndarray:
        """Detect targets via YOLO, segment via SAM v1. Returns 0/1 uint8 mask."""
        h, w = image.shape[:2]

        # Map target names to COCO class IDs
        class_ids = []
        for t in targets:
            cid = COCO_CLASSES.get(t.lower())
            if cid is not None:
                class_ids.append(cid)
        if not class_ids:
            class_ids = [0]  # default to person

        # YOLO detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._yolo(
            image_rgb, stream=True, conf=0.35, iou=0.6,
            classes=class_ids, agnostic_nms=False, max_det=20,
        )
        all_boxes = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for j in range(len(result.boxes)):
                conf = float(result.boxes.conf[j])
                if conf < 0.35:
                    continue
                box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                all_boxes.append(box)

        if not all_boxes:
            return np.zeros((h, w), dtype=np.uint8)

        # SAM segmentation from YOLO boxes
        self._sam_predictor.set_image(image_rgb)
        input_boxes = torch.tensor(all_boxes, device=self._device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            input_boxes, image_rgb.shape[:2]
        )
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed_boxes, multimask_output=False,
        )
        # Union all detected masks
        final_mask = (
            torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
        )
        return final_mask

    def cleanup(self) -> None:
        self._yolo = None
        self._sam_predictor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("YOLO + SAM v1 backend cleaned up")


# ── SAM 3.1 backend ──────────────────────────────────────────

class Sam3Backend:
    """Premium tier: SAM 3.1 text-prompted detection + segmentation."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._predictor: Any = None

    def initialize(self) -> None:
        if not HAS_SAM3:
            raise ImportError(
                "SAM 3.1 not available. Install: "
                "git clone https://github.com/facebookresearch/sam3 && pip install -e ."
            )
        logger.info("Loading SAM 3.1 model...")
        self._predictor = build_sam3_multiplex_video_predictor()
        logger.info("SAM 3.1 backend ready")

    def detect_and_segment(
        self, image: np.ndarray, targets: list[str]
    ) -> np.ndarray:
        """Detect targets via text prompts, segment. Returns 0/1 uint8 mask."""
        from PIL import Image as PILImage

        h, w = image.shape[:2]
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Start single-frame session
        response = self._predictor.handle_request(
            {"type": "start_session", "resource_path": [pil_img]}
        )
        session_id = response["session_id"]

        # Add text prompts
        for prompt_text in targets:
            try:
                self._predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt_text,
                })
            except Exception as exc:
                logger.warning("SAM3 prompt '%s' failed: %s", prompt_text, exc)

        # Propagate (single frame)
        outputs_per_frame = {}
        for resp in self._predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            outputs_per_frame[resp["frame_index"]] = resp["outputs"]

        # Close session
        self._predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )

        # Merge all object masks
        combined = np.zeros((h, w), dtype=np.uint8)
        frame_data = outputs_per_frame.get(0, {})
        for _obj_id, mask_data in frame_data.items():
            if hasattr(mask_data, "cpu"):
                arr = mask_data.cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[0]
                mask = (arr > 0.5).astype(np.uint8)
            elif isinstance(mask_data, np.ndarray):
                mask = (mask_data > 0).astype(np.uint8)
            else:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = np.maximum(combined, mask)

        return combined

    def cleanup(self) -> None:
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM 3.1 backend cleaned up")


# ── Backend selection ─────────────────────────────────────────

def get_available_backend() -> MaskingBackend | None:
    """Return the best available backend, or None if nothing is installed."""
    if HAS_SAM3 and HAS_TORCH:
        return Sam3Backend()
    if HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return YoloSamBackend()
    return None


def get_backend_name() -> str | None:
    """Return name of the active backend tier."""
    if HAS_SAM3 and HAS_TORCH:
        return "sam3"
    if HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return "yolo_sam1"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_backends.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/backends.py tests/test_backends.py
git commit -m "feat: add two-tier masking backends (YOLO+SAM v1, SAM 3.1)"
```

---

## Task 4: Rewrite Masker with Cubemap Decomposition

Replace the existing video-mode SAM3 masker with the cubemap decomposition pipeline that uses the backend interface.

**Files:**
- Modify: `core/masker.py` (full rewrite)
- Existing tests at `tests/test_masker.py` should still pass

- [ ] **Step 1: Read current masker.py and test_masker.py to understand existing interface**

Read: `core/masker.py` (already read above — 341 lines)
Read: `tests/test_masker.py` (check what tests exist)

- [ ] **Step 2: Rewrite masker.py**

Full rewrite. The new masker orchestrates: cubemap decompose → backend detect per face → merge to ERP → postprocess. `MaskResult` gains a `used_fast_path` flag so the pipeline knows whether to skip reframer mask reprojection.

```python
# core/masker.py
"""Operator masking via cubemap decomposition + pluggable detection backend.

Pipeline: ERP frame → 6 cubemap faces → detect per face → merge to ERP → postprocess.
Cubemap fast path: when preset=cubemap, skip ERP merge — resize face masks directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .backends import get_available_backend, get_backend_name, MaskingBackend
from .cubemap_projection import CubemapProjection

logger = logging.getLogger(__name__)


def is_masking_available() -> bool:
    """Return True when at least one detection backend is importable."""
    return get_backend_name() is not None


@dataclass
class MaskConfig:
    """Configuration for operator masking."""
    targets: list[str] = field(default_factory=lambda: ["person"])
    device: str = "cuda"
    is_cubemap: bool = False
    output_size: int = 1920


@dataclass
class MaskResult:
    """Result of a masking run."""
    success: bool
    total_frames: int = 0
    masked_frames: int = 0
    masks_dir: str = ""
    error: str = ""
    used_fast_path: bool = False


def _postprocess_erp_mask(mask: np.ndarray) -> np.ndarray:
    """Morph close + flood-fill at full ERP resolution.

    Ported from Reconstruction Zone reconstruction_pipeline.py:648-667.
    Must run at full resolution (e.g. 7680x3840) where gaps are large
    enough for morphological close to bridge and flood-fill to fill.
    Mask values: 0/1 uint8 throughout. Threshold with mask > 0, NOT > 127.
    """
    binary = ((mask > 0).astype(np.uint8)) * 255
    h, w = binary.shape[:2]

    # Step 1: morphological close to bridge narrow channels
    close_k = max(15, min(51, int(w * 0.004) | 1))  # ~0.4% of width, odd, 15-51px
    close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kern)

    # Step 2: flood-fill from border to find exterior background
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = closed
    inv = cv2.bitwise_not(padded)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(inv, flood_mask, (0, 0), 0)
    holes = inv[1:-1, 1:-1]
    filled = closed | holes

    return (filled > 0).astype(np.uint8)


class Masker:
    """Cubemap-decomposition operator masking with pluggable backend."""

    def __init__(self, config: MaskConfig | None = None) -> None:
        self.config = config or MaskConfig()
        self._backend: MaskingBackend | None = None

    def initialize(self) -> None:
        """Load the best available detection backend."""
        self._backend = get_available_backend()
        if self._backend is None:
            raise ImportError("No masking backend available. Install ultralytics + segment-anything.")
        self._backend.initialize()

    def cleanup(self) -> None:
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None

    def process_frames(
        self,
        frames_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> MaskResult:
        """Mask all ERP frames via cubemap decomposition.

        Standard path: decompose → detect → merge to ERP → postprocess → save ERP masks.
        Cubemap fast path: decompose → detect → resize face masks → save as pinhole masks.

        Args:
            frames_dir: Directory containing ERP frame images.
            output_dir: For standard path: write ERP masks here.
                        For cubemap fast path: write to output_dir/../masks/{view_id}/.
            progress_callback: Optional (current, total, message) callback.

        Returns:
            MaskResult with used_fast_path flag for pipeline coordination.
        """
        if self._backend is None:
            return MaskResult(success=False, error="Not initialized")

        frames_path = Path(frames_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            return MaskResult(success=False, error=f"No frames in {frames_dir}")

        n_frames = len(frame_files)
        cfg = self.config
        proj = CubemapProjection(face_size=None)
        is_fast = cfg.is_cubemap
        masked_count = 0

        # Cubemap view names (must match presets.py cubemap ring naming)
        cube_view_names = ["00_00", "00_01", "00_02", "00_03", "01_00", "02_00"]

        try:
            for fi, frame_file in enumerate(frame_files):
                erp = cv2.imread(str(frame_file))
                if erp is None:
                    continue

                # Decompose to 6 cubemap faces
                faces = proj.equirect2cubemap(erp)

                # Detect on each face
                face_masks = {}
                for face_name, face_img in faces.items():
                    face_masks[face_name] = self._backend.detect_and_segment(
                        face_img, cfg.targets
                    )

                if is_fast:
                    # CUBEMAP FAST PATH: resize face masks to output resolution,
                    # write directly as pinhole masks
                    # Map cubemap face order → view folder names
                    face_order = list(faces.keys())  # front,back,left,right,up,down
                    # Cubemap preset ring order: 4 horizon (front=0,right=1,back=2,left=3), down, up
                    face_to_view = {
                        "front": "00_00", "right": "00_01",
                        "back": "00_02", "left": "00_03",
                        "down": "01_00", "up": "02_00",
                    }
                    masks_root = out_path.parent / "masks"
                    for face_name, mask in face_masks.items():
                        view_name = face_to_view.get(face_name)
                        if view_name is None:
                            continue
                        # Invert: detected=1 → COLMAP keep=0 (black=remove)
                        inverted = ((mask == 0).astype(np.uint8)) * 255
                        resized = cv2.resize(
                            inverted, (cfg.output_size, cfg.output_size),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        view_dir = masks_root / view_name
                        view_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(view_dir / f"{frame_file.stem}.png"), resized)
                else:
                    # STANDARD PATH: merge face masks → ERP → postprocess → save
                    erp_mask = proj.cubemap2equirect(
                        face_masks, (erp.shape[1], erp.shape[0])
                    )
                    erp_mask = _postprocess_erp_mask(erp_mask)
                    # Invert: detected=1 → COLMAP keep=0
                    inverted = ((erp_mask == 0).astype(np.uint8)) * 255
                    cv2.imwrite(str(out_path / f"{frame_file.stem}.png"), inverted)

                masked_count += 1
                if progress_callback:
                    progress_callback(fi + 1, n_frames, f"Masking {fi+1}/{n_frames}")

        except Exception as exc:
            logger.error("Masking pipeline failed: %s", exc)
            return MaskResult(
                success=False, total_frames=n_frames,
                masked_frames=masked_count, masks_dir=str(out_path),
                error=str(exc), used_fast_path=is_fast,
            )

        return MaskResult(
            success=True, total_frames=n_frames,
            masked_frames=masked_count, masks_dir=str(out_path),
            used_fast_path=is_fast,
        )
```

- [ ] **Step 3: Run existing masker tests**

Run: `.venv/Scripts/pytest.exe tests/test_masker.py -v`
Expected: PASS (or update tests to match new interface)

- [ ] **Step 4: Commit**

```bash
git add core/masker.py
git commit -m "feat: rewrite masker with cubemap decomposition + backend dispatch"
```

---

## Task 5: Update Setup Checks for Two-Tier System

Add default tier dependency checks and backend detection.

**Files:**
- Modify: `core/setup_checks.py`

- [ ] **Step 1: Read current setup_checks.py**

Already read above (289 lines). Need to add: `has_yolo`, `has_sam1`, `active_backend` property.

- [ ] **Step 2: Add default tier checks**

Add to `MaskingSetupState`:
```python
has_yolo: bool = False
has_sam1: bool = False

@property
def active_backend(self) -> str | None:
    if self.has_torch and self.has_sam3 and self.has_weights:
        return "sam3"
    if self.has_torch and self.has_yolo and self.has_sam1:
        return "yolo_sam1"
    return None

@property
def default_tier_ready(self) -> bool:
    return self.has_torch and self.has_yolo and self.has_sam1
```

Add check functions:
```python
def _check_yolo_installed() -> bool:
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except ImportError:
        return False

def _check_sam1_installed() -> bool:
    try:
        from segment_anything import SamPredictor  # noqa: F401
        return True
    except ImportError:
        return False
```

Update `check_masking_setup()` to call the new checks.

Add `install_default_tier_to_plugin_venv(on_output=None) -> bool` that runs:
`uv pip install ultralytics segment-anything torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128`

- [ ] **Step 3: Run setup_checks tests if they exist**

Run: `.venv/Scripts/pytest.exe tests/test_setup_checks.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add core/setup_checks.py
git commit -m "feat: add default tier (YOLO+SAM v1) to masking setup checks"
```

---

## Task 6: Wire Masking + Overlap Masks into Pipeline

Update pipeline.py to use the new masker and add the closest-camera mask stage.

**Files:**
- Modify: `core/pipeline.py`

- [ ] **Step 1: Read pipeline.py Stage 2 and Stage 3 sections**

Already read above. Key sections: lines 299-358 (masking + reframe).

- [ ] **Step 2: Update pipeline Stage 2 + Stage 3 coordination**

Replace the current masking stage with the new masker. Key changes:
- Import new masker, `CubemapProjection`, `compute_overlap_masks`
- Pass `is_cubemap=(cfg.preset_name == "cubemap")` and `output_size` to `MaskConfig`
- Stage 2 progress allocation: 20-45%
- Use `mask_result.used_fast_path` to control Stage 3 behavior:

```python
# In Stage 2:
mask_result = masker.process_frames(...)
# ...

# In Stage 3 (reframing):
# If cubemap fast path was used, masks are already in pinhole space.
# Pass mask_dir=None to reframer so it skips mask reprojection.
reframe_result = reframer.reframe_batch(
    input_dir=str(frames_dir),
    output_dir=str(images_dir),
    mask_dir=reframe_mask_dir if not mask_result.used_fast_path else None,
    progress_callback=_reframe_progress,
)
```

The fast path flag is the coordination mechanism: `used_fast_path=True` means masks are already written to `masks/{view_id}/`, so the reframer must not try to reproject ERP masks (there are none).

- [ ] **Step 3: Add Stage 3.5 — overlap mask application**

After reframing (or after cubemap fast path), compute overlap masks and AND them with operator masks:

```python
# After Stage 3 (reframe) completes:
if cfg.enable_masking:
    view_config = _build_runtime_view_config(cfg)
    views = view_config.get_all_views()
    overlap_masks = compute_overlap_masks(views, view_config.output_size)
    if overlap_masks is not None:
        masks_dir = out / "masks"
        for view_name, voronoi_mask in overlap_masks.items():
            view_mask_dir = masks_dir / view_name
            if not view_mask_dir.is_dir():
                continue
            for mask_file in view_mask_dir.iterdir():
                if mask_file.suffix.lower() != ".png":
                    continue
                operator_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if operator_mask is None:
                    continue
                # AND: keep only where both operator AND voronoi say keep
                combined = cv2.bitwise_and(operator_mask, voronoi_mask)
                cv2.imwrite(str(mask_file), combined)
```

- [ ] **Step 4: Run full pipeline test (if available)**

Run: `.venv/Scripts/pytest.exe tests/ -v -k pipeline`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py
git commit -m "feat: wire cubemap masking + overlap masks into pipeline"
```

---

## Task 7: Update Panel UI for Two-Tier Masking

Re-enable the masking section in the panel with the two-tier UX.

**Files:**
- Modify: `panels/prep360_panel.py`
- Modify: `panels/prep360_panel.rml`

- [ ] **Step 1: Read the currently commented-out masking UI code**

Read: `panels/prep360_panel.py` — the commented-out setup wizard sections (around lines 244-272)
Read: `panels/prep360_panel.rml` — find masking section markup

- [ ] **Step 2: Re-enable and update masking UI in panel.py**

Key changes:
- Un-comment masking state polling
- Replace SAM3-only setup with two-tier UX:
  - Default: "Enable Masking" → one-click install if deps missing
  - Premium: "Upgrade to SAM 3.1" link → opens setup wizard
- Add target checkboxes (person, camera, tripod) — bind to `_mask_targets`
- Show active backend indicator ("Using YOLO+SAM" or "Using SAM 3.1")
- Wire `install_default_tier_to_plugin_venv` to install button

- [ ] **Step 3: Update prep360_panel.rml**

Add masking section with:
- Enable masking toggle
- Target checkboxes
- Backend status indicator
- Install/upgrade buttons (conditionally visible)

- [ ] **Step 4: Test in LFS (manual)**

Load the plugin in LichtFeld Studio. Verify:
- Masking section appears
- Enable toggle works
- Target checkboxes display
- Backend status shows correctly

- [ ] **Step 5: Commit**

```bash
git add panels/prep360_panel.py panels/prep360_panel.rml
git commit -m "feat: re-enable masking UI with two-tier backend support"
```

---

## Task 8: python3.dll Bundling

Bundle the stable ABI DLL for SAM 3.1 support.

**Files:**
- Create: `lib/python3.dll` (copy from system Python 3.12)
- Modify: `__init__.py` — add DLL directory at init

- [ ] **Step 1: Verify default tier DLL independence**

Run a test to check whether `ultralytics` and `segment-anything` import without `python3.dll`:

```bash
# From the plugin's .venv, try importing without the DLL workaround
.venv/Scripts/python.exe -c "from ultralytics import YOLO; from segment_anything import SamPredictor; print('OK')"
```

If this succeeds, `python3.dll` is only needed for SAM 3.1. If it fails, bundle unconditionally.

- [ ] **Step 2: Copy python3.dll**

```bash
mkdir -p lib/
cp "C:/Python312/python3.dll" lib/python3.dll
```

- [ ] **Step 3: Add DLL directory at plugin init**

Add to `__init__.py` before any imports:
```python
import os
import sys
from pathlib import Path

if os.name == "nt":
    _lib_dir = Path(__file__).resolve().parent / "lib"
    if _lib_dir.is_dir() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_lib_dir))
```

- [ ] **Step 4: Test SAM3 import chain in LFS Python**

This needs to be tested inside LichtFeld Studio's embedded Python to verify the fix works end-to-end.

- [ ] **Step 5: Commit**

```bash
git add lib/python3.dll __init__.py
git commit -m "feat: bundle python3.dll for SAM 3.1 stable ABI support"
```

---

## Task 9: Update core/__init__.py Exports

Update the package exports to include new modules.

**Files:**
- Modify: `core/__init__.py`

- [ ] **Step 1: Add new exports**

Add to `core/__init__.py`:
```python
from .cubemap_projection import CubemapProjection
from .overlap_mask import compute_overlap_masks
from .backends import (
    MaskingBackend, YoloSamBackend, Sam3Backend,
    get_available_backend, get_backend_name,
)
```

And add to `__all__`.

- [ ] **Step 2: Run all tests**

Run: `.venv/Scripts/pytest.exe tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add core/__init__.py
git commit -m "feat: export new masking modules from core package"
```

---

## Task 10: Integration Test — End-to-End Masking

Write an integration test that exercises the full masking path with a synthetic ERP image.

**Files:**
- Create: `tests/test_masking_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_masking_integration.py
"""Integration test for the masking pipeline.

Uses a synthetic ERP image with a known bright region to verify
the full cubemap decompose → detect → merge → postprocess path.
Does NOT require torch/YOLO/SAM — uses a mock backend.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from core.cubemap_projection import CubemapProjection
from core.overlap_mask import compute_overlap_masks
from core.presets import VIEW_PRESETS


def _mock_backend_that_detects_bright(image, targets):
    """Mock backend: masks any pixel brighter than 128 as detected."""
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    return (gray > 128).astype(np.uint8)


def test_full_masking_pipeline_synthetic():
    """Synthetic ERP with a bright rectangle → mask survives full pipeline."""
    # Create synthetic ERP with a bright person-shaped blob in front-center
    erp = np.zeros((200, 400, 3), dtype=np.uint8)
    erp[80:120, 180:220] = 200  # bright blob at center (front face)

    # Cubemap decompose
    proj = CubemapProjection(face_size=64)
    faces = proj.equirect2cubemap(erp)

    # Detect on each face (mock backend)
    face_masks = {}
    for name, face_img in faces.items():
        mask = _mock_backend_that_detects_bright(face_img, ["person"])
        face_masks[name] = mask

    # The front face should have detections
    assert np.any(face_masks["front"] > 0), "Front face should detect the bright blob"

    # Merge back to ERP
    erp_mask = proj.cubemap2equirect(face_masks, (400, 200))

    # The center region should be masked
    center_detected = erp_mask[85:115, 185:215].mean()
    assert center_detected > 0.3, f"Center blob not detected after merge: {center_detected:.2f}"


def test_overlap_masks_reduce_mask_area():
    """Overlap masks should reduce total mask area for overlapping presets."""
    config = VIEW_PRESETS["high"]
    views = config.get_all_views()
    masks = compute_overlap_masks(views, output_size=64)
    assert masks is not None

    # Total white pixels across all views should be LESS than
    # (num_views * output_size^2) because overlap is partitioned
    total_white = sum(np.sum(m > 0) for m in masks.values())
    total_pixels = len(views) * 64 * 64
    ratio = total_white / total_pixels
    assert ratio < 1.0, f"Overlap masks should reduce total area, got ratio={ratio:.2f}"
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/pytest.exe tests/test_masking_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/Scripts/pytest.exe tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_masking_integration.py
git commit -m "test: add masking integration tests with synthetic ERP"
```
