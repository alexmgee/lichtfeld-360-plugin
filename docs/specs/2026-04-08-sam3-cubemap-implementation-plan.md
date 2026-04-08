# SAM 3 Cubemap Masking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SAM 3 text-prompted cubemap masking as an independent alternative to the FullCircle pipeline, with UI method selection and SAM 3.1 video tracking groundwork.

**Architecture:** New `Sam3CubemapMasker` class in `core/sam3_masker.py` runs independently of the existing `Masker`. Pipeline selects which masker to use based on `masking_method` config field. Panel UI adds a Method dropdown that routes between FullCircle and SAM 3 install/control flows.

**Tech Stack:** Python 3.12, sam3 (PyPI), PyTorch, OpenCV, numpy, CubemapProjection (existing), LichtFeld RML/RCSS UI

**Spec:** `docs/specs/2026-04-08-sam3-cubemap-masking-design.md`
**Restore point:** `ebfc024`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `pyproject.toml` | Add `sam3-masking` optional extra |
| Modify | `core/backends.py` | Fix Sam3Backend API, add Sam3VideoBackend stub, HAS_SAM3_VIDEO flag |
| Modify | `core/setup_checks.py` | Change install_premium_tier to uv sync, add has_sam3_video field |
| Create | `core/sam3_masker.py` | New cubemap masking pipeline: ERP→cubemap→SAM3→reassemble→reframe |
| Modify | `core/pipeline.py` | Add masking_method to PipelineConfig, route to Sam3CubemapMasker |
| Modify | `panels/prep360_panel.py` | Method dropdown, SAM 3 conditional UI states, masking_ready bypass |
| Modify | `panels/prep360_panel.rml` | Method dropdown element, SAM 3 setup/ready conditional blocks |
| Create | `tests/test_sam3_masker.py` | Tests for Sam3CubemapMasker (mocked SAM 3 backend) |

---

### Task 1: Add `sam3-masking` optional dependency

**Files:**
- Modify: `pyproject.toml:20-24`

- [ ] **Step 1: Add the optional extra**

Add `sam3-masking` extra after the existing `video-tracking` block in `pyproject.toml`:

```toml
[project.optional-dependencies]
video-tracking = [
    "sam2==1.1.0",
    "huggingface-hub==1.9.0",
]
sam3-masking = [
    "sam3>=0.1.3",
    "huggingface-hub>=1.9.0",
]
```

- [ ] **Step 2: Regenerate the lock file**

Run: `cd c:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin && .venv\Scripts\python.exe -m uv lock`

If this fails due to torch version conflicts, check sam3's torch requirement against the existing `torch>=2.11.0` pin and adjust.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add sam3-masking optional dependency extra"
```

---

### Task 2: Fix Sam3Backend API mismatches

**Files:**
- Modify: `core/backends.py:47-52` (imports), `core/backends.py:395-453` (Sam3Backend class), `core/backends.py:455-475` (batch_detect_boxes)

- [ ] **Step 1: Fix the import path**

Change `core/backends.py` line 48 from:

```python
from sam3.model_builder import build_sam3_image_model  # type: ignore[import-untyped]
```

to:

```python
from sam3 import build_sam3_image_model  # type: ignore[import-untyped]
```

This matches the working `reconstruction_gui/test_cubemap_sam3.py` and is confirmed identical via `sam3/__init__.py` re-export.

- [ ] **Step 2: Fix initialize() — device handling and confidence threshold**

Replace `Sam3Backend.__init__` and `initialize` (lines 395-408):

```python
def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3) -> None:
    self._device = device
    self._confidence_threshold = confidence_threshold
    self._model: Any = None
    self._processor: Any = None

def initialize(self) -> None:
    if not HAS_SAM3:
        raise ImportError(
            "SAM 3 not available. Install via plugin settings."
        )

    # Flash Attention 3 detection + fallback
    fa3_available = False
    try:
        from flash_attn_interface import flash_attn_func  # noqa: F401
        fa3_available = True
    except ImportError:
        pass

    if not fa3_available:
        try:
            from sam3.model import decoder as _dec
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _orig_sdpa_kernel = sdpa_kernel
            _dec.sdpa_kernel = lambda *a, **kw: _orig_sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
            )
            logger.info("Patched SAM 3 decoder to allow MATH attention fallback")
        except Exception as exc:
            logger.warning("Could not patch SAM 3 decoder attention: %s", exc)

    logger.info("Loading SAM 3 image model (FA3=%s)...", fa3_available)
    self._model = build_sam3_image_model()
    self._model = self._model.to(self._device)
    self._model.eval()
    self._processor = Sam3Processor(self._model, confidence_threshold=self._confidence_threshold)
    logger.info("SAM 3 backend ready")
```

- [ ] **Step 3: Fix detect_and_segment() — add reset_all_prompts**

In `detect_and_segment` (line 429), add `reset_all_prompts` before each prompt. Replace the for loop:

```python
combined = np.zeros((h, w), dtype=np.uint8)
for prompt_text in targets:
    try:
        self._processor.reset_all_prompts(state)
        output = self._processor.set_text_prompt(
            state=state, prompt=prompt_text
        )
        masks = output.get("masks")
        if masks is None:
            continue
        for mask in masks:
            if hasattr(mask, "cpu"):
                arr = mask.cpu().numpy()
            else:
                arr = np.array(mask)
            if arr.ndim == 3:
                arr = arr[0]
            binary = (arr > 0.5).astype(np.uint8)
            if binary.shape[:2] != (h, w):
                binary = cv2.resize(
                    binary, (w, h), interpolation=cv2.INTER_NEAREST
                )
            combined = np.maximum(combined, binary)
    except Exception as exc:
        logger.warning("SAM 3 prompt '%s' failed: %s", prompt_text, exc)
```

- [ ] **Step 4: Fix batch_detect_boxes() — use native scores**

Replace `batch_detect_boxes` (lines 455-475):

```python
def batch_detect_boxes(
    self,
    images: list[np.ndarray],
    detection_confidence: float = 0.35,
) -> list[list[tuple[np.ndarray, float]]]:
    """Per-image detection via SAM 3. Returns bounding boxes from mask contours."""
    from PIL import Image as PILImage

    all_detections: list[list[tuple[np.ndarray, float]]] = []
    for image in images:
        h, w = image.shape[:2]
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        state = self._processor.set_image(pil_img)

        detections: list[tuple[np.ndarray, float]] = []
        self._processor.reset_all_prompts(state)
        output = self._processor.set_text_prompt(state=state, prompt="person")
        masks = output.get("masks")
        scores = output.get("scores")

        if masks is not None and len(masks) > 0:
            for i, mask in enumerate(masks):
                score = float(scores[i].cpu()) if scores is not None and i < len(scores) else 1.0
                if hasattr(mask, "cpu"):
                    arr = mask.cpu().numpy()
                else:
                    arr = np.array(mask)
                if arr.ndim == 3:
                    arr = arr[0]
                binary = (arr > 0.5).astype(np.uint8)
                if binary.shape[:2] != (h, w):
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw * ch > 100:
                        box = np.array([x, y, x + cw, y + ch])
                        detections.append((box, score))

        all_detections.append(detections)
    return all_detections
```

- [ ] **Step 5: Commit**

```bash
git add core/backends.py
git commit -m "fix: align Sam3Backend with real SAM 3 API

Fix device handling (no device arg, use .to()), add confidence_threshold
to Sam3Processor constructor, add reset_all_prompts between prompts,
use native output scores in batch_detect_boxes, add Flash Attention 3
detection with MATH fallback."
```

---

### Task 3: Update setup_checks.py — install path and SAM 3.1 groundwork

**Files:**
- Modify: `core/setup_checks.py:17-88` (MaskingSetupState), `core/setup_checks.py:122-127` (_check_sam3_installed), `core/setup_checks.py:509-531` (install_premium_tier)

- [ ] **Step 1: Add has_sam3_video field to MaskingSetupState**

Add field at line 33 (after `has_weights`):

```python
# SAM 3.1 video tracking (future)
has_sam3_video: bool = False
```

- [ ] **Step 2: Add _check_sam3_video_installed function**

After `_check_sam3_installed` (line 127), add:

```python
def _check_sam3_video_installed() -> bool:
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: F401
        return True
    except ImportError:
        return False
```

- [ ] **Step 3: Update check_masking_setup to populate has_sam3_video**

In `check_masking_setup()` (line 172-184), add `has_sam3_video` to the return:

```python
def check_masking_setup() -> MaskingSetupState:
    torch_ok = _check_torch_installed()
    return MaskingSetupState(
        has_torch=torch_ok,
        has_yolo=_check_yolo_installed(),
        has_sam1=_check_sam1_installed(),
        has_sam2=_check_sam2_installed(),
        has_token=_check_hf_token(),
        has_access=_check_hf_access(),
        has_sam3=_check_sam3_installed(),
        has_weights=_check_weights_downloaded(),
        has_sam3_video=_check_sam3_video_installed(),
    )
```

- [ ] **Step 4: Change install_premium_tier to use uv sync**

Replace `install_premium_tier` (lines 509-531):

```python
def install_premium_tier(on_output=None) -> bool:
    """Install SAM 3 into the plugin venv.

    Requires torch already installed (from default tier).
    Downloads SAM 3 weights eagerly after install.
    """
    ok = _run_uv_command([
        "sync",
        "--locked",
        "--no-dev",
        "--extra", "sam3-masking",
    ], on_output=on_output)
    if not ok:
        return False

    # Eagerly download SAM 3 weights
    try:
        if on_output:
            on_output("Downloading SAM 3 weights (~3.5 GB)...")
        download_model_weights()
        if on_output:
            on_output("SAM 3 weights downloaded.")
    except Exception as exc:
        logger.warning("SAM 3 weight download failed: %s", exc)

    return True
```

- [ ] **Step 5: Commit**

```bash
git add core/setup_checks.py
git commit -m "feat: update setup_checks for sam3-masking extra and SAM 3.1 groundwork

Change install_premium_tier from 'uv add sam3' to 'uv sync --extra
sam3-masking'. Add has_sam3_video field and detection check for future
SAM 3.1 video tracking."
```

---

### Task 4: Add Sam3VideoBackend stub to backends.py

**Files:**
- Modify: `core/backends.py` (after Sam2VideoBackend, before get_video_backend)

- [ ] **Step 1: Add HAS_SAM3_VIDEO flag and conditional import**

After the existing `HAS_SAM2` block (lines 505-510), add:

```python
HAS_SAM3_VIDEO = False
try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import-untyped]
    HAS_SAM3_VIDEO = True
except ImportError:
    pass
```

- [ ] **Step 2: Add Sam3VideoBackend stub**

Before `get_video_backend` (line 1551), add:

```python
class Sam3VideoBackend:
    """Future: SAM 3.1 multiplex video tracking.

    Stub only — not yet implemented. Implements VideoTrackingBackend
    protocol so the interface is defined for future work.
    """

    def initialize(self) -> None:
        raise NotImplementedError("SAM 3.1 video tracking not yet implemented")

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
        initial_box: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        raise NotImplementedError("SAM 3.1 video tracking not yet implemented")

    def cleanup(self) -> None:
        pass
```

- [ ] **Step 3: Commit**

```bash
git add core/backends.py
git commit -m "feat: add SAM 3.1 video tracking groundwork stub

HAS_SAM3_VIDEO flag, conditional import of multiplex video predictor,
Sam3VideoBackend stub implementing VideoTrackingBackend protocol.
get_video_backend selection unchanged — SAM v2 stays active."
```

---

### Task 5: Create Sam3CubemapMasker

**Files:**
- Create: `core/sam3_masker.py`
- Create: `tests/test_sam3_masker.py`

- [ ] **Step 1: Write failing test for Sam3CubemapMasker**

Create `tests/test_sam3_masker.py`:

```python
# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for SAM 3 cubemap masker — mocked SAM 3 backend."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_erp_with_blob(w=2048, h=1024):
    """Create a synthetic ERP image with a bright blob (simulates a person)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Place a bright rectangle in the front-facing region
    cx, cy = w // 2, h // 2
    img[cy - 50:cy + 50, cx - 30:cx + 30] = (200, 200, 200)
    return img


def _mock_sam3_mask(h, w, has_detection=True):
    """Return a mock SAM 3 output dict."""
    if has_detection:
        mask = np.zeros((1, h, w), dtype=np.float32)
        mask[0, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1.0
    else:
        mask = np.zeros((1, h, w), dtype=np.float32)

    import torch
    return {
        "masks": torch.tensor(mask),
        "scores": torch.tensor([0.95]) if has_detection else torch.tensor([0.0]),
    }


class TestSam3CubemapMasker:

    def test_import(self):
        """Module is importable."""
        from core.sam3_masker import Sam3CubemapMasker
        assert Sam3CubemapMasker is not None

    def test_process_single_frame(self, tmp_path):
        """Single ERP frame produces masks in the correct output layout."""
        from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig
        from core.presets import get_preset

        # Write a synthetic ERP frame
        import cv2
        erp = _make_erp_with_blob()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

        view_config = get_preset("cubemap")

        config = Sam3MaskerConfig(
            prompts=["person"],
            confidence_threshold=0.3,
        )
        masker = Sam3CubemapMasker(config)

        # Mock the SAM 3 backend so we don't need real weights
        mock_backend = MagicMock()
        mock_backend.detect_and_segment.return_value = np.ones((512, 512), dtype=np.uint8)
        masker._backend = mock_backend
        masker._initialized = True

        output_dir = tmp_path / "output"
        result = masker.process_frames(
            frames_dir=str(frames_dir),
            output_dir=str(output_dir),
            view_config=view_config,
        )

        # Check that mask files were created
        masks_dir = output_dir / "masks"
        assert masks_dir.exists()

        # Should have subdirectories for each view
        view_dirs = sorted(masks_dir.iterdir())
        assert len(view_dirs) > 0

        # Each view dir should have a mask file
        for vd in view_dirs:
            mask_files = list(vd.glob("*.png"))
            assert len(mask_files) == 1

    def test_mask_polarity(self, tmp_path):
        """Output masks should be COLMAP polarity: white=keep, black=remove."""
        from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig
        from core.presets import get_preset
        import cv2

        erp = _make_erp_with_blob()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        cv2.imwrite(str(frames_dir / "frame_00001.jpg"), erp)

        view_config = get_preset("cubemap")
        config = Sam3MaskerConfig(prompts=["person"])
        masker = Sam3CubemapMasker(config)

        # Mock backend returns detection in center of face
        def mock_detect(image, targets, **kwargs):
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
            return mask

        mock_backend = MagicMock()
        mock_backend.detect_and_segment.side_effect = mock_detect
        masker._backend = mock_backend
        masker._initialized = True

        output_dir = tmp_path / "output"
        masker.process_frames(
            frames_dir=str(frames_dir),
            output_dir=str(output_dir),
            view_config=view_config,
        )

        # Read any output mask and check polarity
        masks_dir = output_dir / "masks"
        any_mask_file = next(masks_dir.rglob("*.png"))
        mask = cv2.imread(str(any_mask_file), cv2.IMREAD_GRAYSCALE)

        # COLMAP polarity: white (255) = keep, black (0) = remove
        # Since we detected something, some pixels should be 0 (removed)
        # and most should be 255 (kept)
        assert mask is not None
        assert 0 in mask, "Mask should have black (removed) pixels where person was detected"
        assert 255 in mask, "Mask should have white (kept) pixels for background"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_sam3_masker.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'core.sam3_masker'`

- [ ] **Step 3: Write Sam3CubemapMasker implementation**

Create `core/sam3_masker.py`:

```python
# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""SAM 3 cubemap masking — independent path.

Pipeline per ERP frame:
1. CubemapProjection.equirect2cubemap() → 6 cube faces
2. Sam3Backend.detect_and_segment() on each face (text-prompted)
3. CubemapProjection.cubemap2equirect() → merged ERP mask
4. Invert polarity (SAM white=detected → COLMAP white=keep)
5. Optional dilation
6. Reframe ERP mask into pinhole views via Reframer
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .cubemap_projection import CubemapProjection

logger = logging.getLogger(__name__)


@dataclass
class Sam3MaskerConfig:
    """Configuration for SAM 3 cubemap masking."""

    prompts: list[str] = field(default_factory=lambda: ["person", "tripod"])
    confidence_threshold: float = 0.3
    dilation_px: int = 3
    output_size: int = 1920  # Pinhole view output resolution
    face_size: int | None = None  # None = auto (min(1024, w // 4))


@dataclass
class Sam3MaskerResult:
    """Result from Sam3CubemapMasker.process_frames()."""

    success: bool = False
    total_frames: int = 0
    masked_frames: int = 0
    mask_dir: str = ""


class Sam3CubemapMasker:
    """Independent SAM 3 masking path via cubemap decomposition.

    Does NOT use direction estimation, synthetic views, or video tracking.
    Each ERP frame is decomposed into 6 cubemap faces, SAM 3 runs
    text-prompted detection+segmentation on each face, face masks are
    merged back to ERP, inverted to COLMAP polarity, and reframed
    into the active preset's pinhole views.
    """

    def __init__(self, config: Sam3MaskerConfig | None = None) -> None:
        self.config = config or Sam3MaskerConfig()
        self._backend: Any = None
        self._initialized = False

    def initialize(self) -> None:
        """Load Sam3Backend."""
        from .backends import Sam3Backend

        self._backend = Sam3Backend(
            confidence_threshold=self.config.confidence_threshold,
        )
        self._backend.initialize()
        self._initialized = True
        logger.info("Sam3CubemapMasker initialized")

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str,
        view_config: Any,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Sam3MaskerResult:
        """Process extracted ERP frames through cubemap → SAM 3 → reframe.

        Args:
            frames_dir: Directory of extracted ERP frames (jpg/png).
            output_dir: Root output directory. Masks written to output_dir/masks/{view_id}/.
            view_config: ViewConfig with preset geometry.
            progress_callback: Optional (current, total, message) callback.

        Returns:
            Sam3MaskerResult with statistics.
        """
        if not self._initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        frames_path = Path(frames_dir)
        out_path = Path(output_dir)
        masks_root = out_path / "masks"

        # Find frames
        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            logger.warning("No frames found in %s", frames_dir)
            return Sam3MaskerResult(success=True, total_frames=0)

        result = Sam3MaskerResult(total_frames=len(frame_files))

        # Output size from view config — use first view's FOV to determine
        # a reasonable default if not explicitly configured
        output_size = self.config.output_size
        dilation_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.dilation_px * 2 + 1,) * 2)
            if self.config.dilation_px > 0 else None
        )

        for i, frame_file in enumerate(frame_files):
            if progress_callback:
                progress_callback(i, len(frame_files), f"SAM 3 masking {frame_file.name}")

            erp = cv2.imread(str(frame_file))
            if erp is None:
                logger.warning("Could not read %s, skipping", frame_file)
                continue

            erp_mask = self._process_single_erp(erp, dilation_kernel)

            if erp_mask is not None and np.any(erp_mask < 255):
                result.masked_frames += 1

            # Reframe ERP mask into pinhole views
            self._write_reframed_masks(
                erp_mask, view_config, self.config.output_size, masks_root, frame_file.stem,
            )

        result.success = True
        result.mask_dir = str(masks_root)

        if progress_callback:
            progress_callback(len(frame_files), len(frame_files), "SAM 3 masking complete")

        logger.info(
            "Sam3CubemapMasker: %d/%d frames masked",
            result.masked_frames, result.total_frames,
        )
        return result

    def _process_single_erp(
        self, erp: np.ndarray, dilation_kernel: np.ndarray | None,
    ) -> np.ndarray:
        """Process one ERP frame. Returns COLMAP-polarity mask (255=keep, 0=remove)."""
        h, w = erp.shape[:2]
        face_size = self.config.face_size or min(1024, w // 4)
        cubemap = CubemapProjection(face_size)

        # 1. Decompose ERP → 6 cubemap faces
        faces = cubemap.equirect2cubemap(erp)

        # 2. SAM 3 per-face detection
        face_masks: dict[str, np.ndarray] = {}
        for face_name, face_img in faces.items():
            detection = self._backend.detect_and_segment(
                face_img,
                targets=self.config.prompts,
                detection_confidence=self.config.confidence_threshold,
            )
            face_masks[face_name] = detection

        # 3. Reassemble face masks → ERP mask
        erp_detection = cubemap.cubemap2equirect(face_masks, (w, h))

        # 4. Invert polarity: SAM white=detected → COLMAP white=keep
        erp_mask = ((erp_detection == 0).astype(np.uint8)) * 255

        # 5. Dilation (erode keep region = dilate masked-out region)
        if dilation_kernel is not None:
            erp_mask = cv2.erode(erp_mask, dilation_kernel, iterations=1)

        return erp_mask

    def _write_reframed_masks(
        self,
        erp_mask: np.ndarray,
        view_config: Any,
        output_size: int,
        masks_root: Path,
        frame_stem: str,
    ) -> None:
        """Reframe a single ERP mask into per-view pinhole masks and write to disk.

        Uses the standalone reframe_view() with mode="nearest" for binary masks.
        ViewConfig.get_all_views() returns (yaw, pitch, fov, name, flip_vertical).
        """
        from .reframer import reframe_view

        views = view_config.get_all_views()

        for yaw, pitch, fov, view_name, flip_v in views:
            view_dir = masks_root / view_name
            view_dir.mkdir(parents=True, exist_ok=True)

            pinhole_mask = reframe_view(
                erp_mask, fov, yaw, pitch, output_size, mode="nearest",
            )
            if flip_v:
                pinhole_mask = np.flipud(pinhole_mask)

            mask_path = view_dir / f"{frame_stem}.png"
            cv2.imwrite(str(mask_path), pinhole_mask)

    def cleanup(self) -> None:
        """Release SAM 3 backend resources."""
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None
        self._initialized = False
        logger.info("Sam3CubemapMasker cleaned up")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_sam3_masker.py -v`

Expected: PASS (at least `test_import`). If tests fail, debug based on import issues or mock setup.

Note: The masker uses the standalone `reframe_view()` function from `core/reframer.py` with `mode="nearest"` for binary masks. This function accepts `(equirect, fov_deg, yaw_deg, pitch_deg, out_size, mode)` — verified at reframer.py:106-187.

- [ ] **Step 5: Verify tests pass with the reframe_view dependency**

The tests mock `_backend.detect_and_segment` but let `reframe_view` and `CubemapProjection` run for real (they're pure numpy, no model loading). If tests fail on import, check that `core/reframer.py` is importable from the test environment.

- [ ] **Step 6: Commit**

```bash
git add core/sam3_masker.py tests/test_sam3_masker.py
git commit -m "feat: add Sam3CubemapMasker — independent cubemap masking path

ERP → cubemap decomposition → SAM 3 per-face detection → reassemble →
COLMAP polarity inversion → dilation → reframe to pinhole views.
No direction estimation, no video tracking."
```

---

### Task 6: Wire Sam3CubemapMasker into pipeline.py

**Files:**
- Modify: `core/pipeline.py:63-68` (PipelineConfig), `core/pipeline.py:392-428` (masking stage)

- [ ] **Step 1: Add masking_method to PipelineConfig**

At `core/pipeline.py` line 66, after `mask_backend`, add:

```python
masking_method: str = "fullcircle"  # "fullcircle" or "sam3_cubemap"
```

- [ ] **Step 2: Add SAM 3 cubemap branch in the default masking path**

In the default pipeline path (line 392+), before the existing masking block, add the SAM 3 branch:

```python
        else:
            # ── Default path: mask ERP first, then reframe images + masks ──

            erp_masks_dir = extracted_dir / "masks"
            reframe_mask_dir: Optional[str] = None

            if cfg.enable_masking and cfg.masking_method == "sam3_cubemap":
                # SAM 3 cubemap path — independent of FullCircle pipeline
                from .sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig

                self._update("masking", 20.0, "Initializing SAM 3 cubemap masker...")

                sam3_cfg = Sam3MaskerConfig(
                    prompts=cfg.mask_prompts,
                    confidence_threshold=0.3,
                    output_size=cfg.output_size,
                )
                sam3_masker = Sam3CubemapMasker(sam3_cfg)
                sam3_masker.initialize()

                try:
                    def _sam3_progress(cur: int, total: int, msg: str) -> None:
                        pct = 20 + (cur / max(total, 1)) * 25
                        self._update("masking", pct, msg)

                    sam3_result = sam3_masker.process_frames(
                        frames_dir=str(frames_dir),
                        output_dir=str(out),
                        view_config=view_config,
                        progress_callback=_sam3_progress,
                    )

                    if sam3_result.success and sam3_result.masked_frames > 0:
                        reframe_mask_dir = sam3_result.mask_dir
                finally:
                    sam3_masker.cleanup()

                if self._check_cancel():
                    raise RuntimeError("Cancelled")

            elif cfg.enable_masking and is_masking_available():
                # Existing FullCircle path — unchanged
                self._update("masking", 20.0, "Initializing masking backend...")
                # ... (existing code continues unchanged)
```

The key change: wrap the existing masking block in `elif` so the SAM 3 path and FullCircle path are mutually exclusive.

- [ ] **Step 3: Handle SAM 3 mask directory for reframing**

The SAM 3 masker writes masks directly to `output_dir/masks/{view_id}/` — already in pinhole view layout. When the SAM 3 path is active, the reframe stage should NOT try to reframe ERP masks (they've already been reframed by the masker). Instead, set `reframe_mask_dir = None` so the reframer only handles images.

Wait — the SAM 3 masker handles reframing internally. So the pipeline's reframe stage just needs to reframe images without masks. The `reframe_mask_dir` stays `None` for the SAM 3 path.

Verify this is the case by checking that `reframe_mask_dir` defaults to `None` and is only set when FullCircle masking produces ERP masks.

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `.venv\Scripts\pytest.exe tests/test_pipeline.py -v`

Expected: PASS — existing pipeline tests should not be affected since `masking_method` defaults to `"fullcircle"`.

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py
git commit -m "feat: wire Sam3CubemapMasker into pipeline

Add masking_method field to PipelineConfig. When 'sam3_cubemap',
pipeline uses Sam3CubemapMasker instead of Masker. FullCircle
path unchanged."
```

---

### Task 7: Update panel UI — Method dropdown and SAM 3 states

**Files:**
- Modify: `panels/prep360_panel.py:205-214` (bindings), `panels/prep360_panel.py:770-798` (install handler), `panels/prep360_panel.py:1039-1058` (config builder)
- Modify: `panels/prep360_panel.rml:100-127` (masking section)

- [ ] **Step 1: Add masking_method_idx state and bindings**

In `prep360_panel.py`, add instance variable in `__init__`:

```python
self._masking_method_idx = 0  # 0=FullCircle, 1=SAM 3 Cubemap
```

Add bindings after the existing masking bindings (around line 214):

```python
model.bind("masking_method_idx", lambda: str(self._masking_method_idx), self._set_masking_method)

# SAM 3 conditional states
model.bind_func("show_masking_fullcircle", lambda: self._masking_method_idx == 0)
model.bind_func("show_masking_sam3", lambda: self._masking_method_idx == 1)
model.bind_func("show_masking_sam3_setup", lambda: self._masking_method_idx == 1 and not self._setup_state.premium_tier_ready)
model.bind_func("show_masking_sam3_ready", lambda: self._masking_method_idx == 1 and self._setup_state.premium_tier_ready)
```

Add the setter:

```python
def _set_masking_method(self, val):
    try:
        self._masking_method_idx = int(val)
    except (ValueError, TypeError):
        self._masking_method_idx = 0
```

- [ ] **Step 2: Update install handler to handle SAM 3 path (line 788)**

In `_on_install_default_tier` at line 788, change:

```python
self._masking_available = self._setup_state.masking_ready
```

to:

```python
if self._masking_method_idx == 1:
    # SAM 3 path — doesn't require SAM v2
    self._masking_available = self._setup_state.premium_tier_ready
else:
    self._masking_available = self._setup_state.masking_ready
```

- [ ] **Step 3: Update PipelineConfig builder (line 1039+)**

In the config builder, add `masking_method`:

```python
masking_method = "sam3_cubemap" if self._masking_method_idx == 1 else "fullcircle"

config = PipelineConfig(
    ...
    masking_method=masking_method,
    ...
)
```

- [ ] **Step 4: Update prep360_panel.rml — add Method dropdown and conditional blocks**

Replace the masking section (lines 100-127) with:

```xml
    <div class="section-header text-accent" id="hdr-masking" data-event-click="toggle_section('masking')">
      <span class="section-arrow text-accent is-expanded" id="arrow-masking">&#x25B6;</span>
      <span class="text-accent">Operator Masking</span>
    </div>
    <div class="section-content" id="sec-masking">
      <!-- Method dropdown — always visible -->
      <div class="setting-row setting-row--dropdown setting-row--dropdown-high">
        <span class="prop-label">Method</span>
        <select data-value="masking_method_idx">
          <option value="0">FullCircle</option>
          <option value="1">SAM 3 Cubemap</option>
        </select>
      </div>

      <!-- FullCircle path -->
      <div data-if="show_masking_fullcircle">
        <div data-if="show_masking_install">
          <div class="setting-row">
            <span class="status-text status-muted">Masking requires the full stack, including SAM v2 video tracking.</span>
          </div>
          <div class="setting-row">
            <button class="btn btn-primary" data-event-click="install_masking_deps" data-attr-disabled="install_busy">{{install_button_text}}</button>
          </div>
        </div>
        <div data-if="show_masking_controls">
          <div class="setting-row">
            <span class="prop-label">Enable</span>
            <input type="checkbox" data-checked="enable_masking" />
          </div>
          <div class="setting-row">
            <span class="prop-label">Diagnostics</span>
            <input type="checkbox" data-checked="enable_diagnostics" />
          </div>
          <div class="setting-row">
            <span class="prop-label">Backend</span>
            <span class="status-text status-muted">{{masking_backend_text}}</span>
          </div>
        </div>
      </div>

      <!-- SAM 3 path — setup needed -->
      <div data-if="show_masking_sam3_setup">
        <div class="setting-row">
          <span class="status-text status-muted">SAM 3 requires a HuggingFace account with model access.</span>
        </div>
        <div class="setting-row">
          <button class="btn btn--secondary btn--compact" data-event-click="open_hf_signup">1. Create HuggingFace Account</button>
        </div>
        <div class="setting-row">
          <button class="btn btn--secondary btn--compact" data-event-click="open_hf_model">2. Request SAM 3 Access</button>
        </div>
        <div class="setting-row">
          <span class="prop-label">3. Token</span>
          <input type="text" data-value="hf_token_input" />
          <button class="btn btn--secondary btn--compact" data-event-click="verify_hf_token">Verify</button>
        </div>
        <div class="setting-row">
          <span class="status-text status-muted">{{hf_verify_text}}</span>
        </div>
        <div class="setting-row">
          <button class="btn btn--secondary btn--compact" data-event-click="open_hf_tokens">Open Tokens Page</button>
        </div>
        <div class="setting-row">
          <button class="btn btn-primary" data-event-click="install_premium_tier" data-attr-disabled="install_busy">{{install_button_text}}</button>
        </div>
      </div>

      <!-- SAM 3 path — ready -->
      <div data-if="show_masking_sam3_ready">
        <div class="setting-row">
          <span class="prop-label">Enable</span>
          <input type="checkbox" data-checked="enable_masking" />
        </div>
        <div class="setting-row">
          <span class="prop-label">Diagnostics</span>
          <input type="checkbox" data-checked="enable_diagnostics" />
        </div>
        <div class="setting-row">
          <span class="prop-label">Backend</span>
          <span class="status-text status-muted">SAM 3 (0.9B params)</span>
        </div>
        <div class="setting-row">
          <span class="prop-label">Prompts</span>
          <input type="text" data-value="mask_prompts_str" />
        </div>
      </div>
    </div>
```

- [ ] **Step 5: Add install_premium_tier event binding**

In `prep360_panel.py`, add the event binding (near line 254):

```python
model.bind_event("install_premium_tier", self._on_install_premium_tier)
model.bind_event("open_hf_signup", self._on_open_hf_signup)
model.bind_event("open_hf_model", self._on_open_hf_model)
model.bind_event("open_hf_tokens", self._on_open_hf_tokens)
model.bind_event("verify_hf_token", self._on_verify_hf_token)
```

Check which of these already exist (some do from the existing scaffolding). Only add missing ones.

Add the premium tier install handler if it doesn't exist:

```python
def _on_install_premium_tier(self, handle, event, args):
    del handle, event, args
    if self._install_busy:
        return
    self._install_busy = True
    self._install_button_text = "Installing SAM 3..."
    if self._handle:
        self._handle.dirty_all()

    def _install():
        def _progress(msg):
            self._install_button_text = msg
            if self._handle:
                self._handle.dirty_all()

        ok = install_premium_tier(on_output=_progress)
        self._install_busy = False
        self._setup_state = check_masking_setup()
        self._masking_available = self._setup_state.premium_tier_ready
        if ok:
            self._install_button_text = "SAM 3 installed"
            self._enable_masking = self._masking_available
        else:
            self._install_button_text = "Install failed — retry"
            self._enable_masking = False
        if self._handle:
            self._handle.dirty_all()

    threading.Thread(target=_install, daemon=True).start()
```

- [ ] **Step 6: Commit**

```bash
git add panels/prep360_panel.py panels/prep360_panel.rml
git commit -m "feat: add masking method selector and SAM 3 UI flow

Method dropdown at top of masking section routes between FullCircle
and SAM 3 paths. SAM 3 path shows guided HF setup when not installed,
masking controls when ready. masking_ready bypass for SAM 3 path."
```

---

### Task 8: Install sam3 and end-to-end validation

**Files:**
- No file changes — runtime verification

- [ ] **Step 1: Install sam3 into the plugin venv**

Run:
```bash
cd c:\Users\alexm\.lichtfeld\plugins\lichtfeld-360-plugin
.venv\Scripts\python.exe -m uv sync --extra sam3-masking
```

If this fails, check error output for version conflicts and adjust `pyproject.toml` accordingly.

- [ ] **Step 2: Verify sam3 is importable**

Run:
```bash
.venv\Scripts\python.exe -c "from sam3 import build_sam3_image_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Verify import path equivalence**

Run:
```bash
.venv\Scripts\python.exe -c "
from sam3 import build_sam3_image_model as a
from sam3.model_builder import build_sam3_image_model as b
print('Same function:', a is b)
"
```

Expected: `Same function: True`

- [ ] **Step 4: Run all tests**

Run: `.venv\Scripts\pytest.exe tests/ -v`

Expected: All existing tests PASS, new `test_sam3_masker.py` tests PASS.

- [ ] **Step 5: Test with real ERP frame (manual)**

If a test ERP frame is available (e.g., from `D:\Capture\deskTest\`), run the SAM 3 cubemap masker manually to verify mask quality:

```python
from core.sam3_masker import Sam3CubemapMasker, Sam3MaskerConfig
from core.presets import get_preset

config = Sam3MaskerConfig(prompts=["person"])
masker = Sam3CubemapMasker(config)
masker.initialize()

view_config = get_preset("cubemap")
result = masker.process_frames(
    frames_dir="D:/Capture/deskTest/extracted/frames",
    output_dir="D:/Capture/deskTest/sam3_test",
    view_config=view_config,
)
print(result)
masker.cleanup()
```

Inspect output masks visually.

- [ ] **Step 6: Commit any fixes from validation**

```bash
git add -A
git commit -m "fix: adjustments from end-to-end SAM 3 validation"
```

---

## Task Order Summary

1. **pyproject.toml** — add sam3-masking extra, regenerate lock
2. **backends.py** — fix Sam3Backend API (device, threshold, reset_all_prompts, scores, FA3)
3. **setup_checks.py** — install_premium_tier to uv sync, has_sam3_video field
4. **backends.py** — add Sam3VideoBackend stub + HAS_SAM3_VIDEO
5. **sam3_masker.py** — create Sam3CubemapMasker + tests
6. **pipeline.py** — wire masking_method routing
7. **panel UI** — method dropdown, conditional states, masking_ready bypass
8. **Install + validate** — end-to-end with real sam3 package
