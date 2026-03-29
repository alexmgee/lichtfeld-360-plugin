# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""SAM 3.1 operator masking with graceful degradation.

Wraps SAM 3.1's multiplex video predictor for text-prompted detection,
segmentation, and tracking of objects (e.g. photographers, tripods) across
equirectangular frames.

Frames are passed directly as PIL images — no temp directories needed.
SAM 3.1's io_utils.load_resource_as_video_frames (verified in
sam3/model/io_utils.py:42-67) accepts a list of PIL.Image.Image.

Masks are saved with COLMAP polarity: white (255) = keep, black (0) = remove.
This requires inverting SAM's output (white = detected object).

This module imports cleanly without torch or sam3. Check
``is_masking_available()`` before constructing a ``Masker``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- must not blow up when torch/sam3 are absent
# ---------------------------------------------------------------------------

HAS_TORCH = False
HAS_SAM3 = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import-untyped]

    HAS_SAM3 = True
except ImportError:
    pass


def is_masking_available() -> bool:
    """Return True when both torch and SAM 3.1 are importable."""
    return HAS_TORCH and HAS_SAM3


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _decode_mask(mask_data: Any, shape: tuple[int, int]) -> np.ndarray:
    """Decode mask data to a binary uint8 array (0 or 255).

    Handles numpy arrays, torch tensors, and RLE dicts (integer counts or
    COCO-style string counts) as returned by SAM 3.1's propagate_in_video.
    """
    if isinstance(mask_data, np.ndarray):
        return (mask_data > 0).astype(np.uint8) * 255

    # Torch tensor
    if hasattr(mask_data, "cpu"):
        arr = mask_data.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]  # (1, H, W) -> (H, W)
        return (arr > 0.5).astype(np.uint8) * 255

    if isinstance(mask_data, dict):
        counts = mask_data.get("counts", [])
        size = mask_data.get("size", list(shape))

        if isinstance(counts, str):
            try:
                from pycocotools import mask as mask_utils  # type: ignore[import-untyped]

                decoded = mask_utils.decode(mask_data)
                return (decoded > 0).astype(np.uint8) * 255
            except ImportError:
                logger.warning("pycocotools not available for COCO RLE decoding")
                return np.zeros(shape, dtype=np.uint8)

        # Integer run-length counts
        h, w = size[0], size[1]
        flat = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        val = 0
        for count in counts:
            flat[pos : pos + count] = val
            pos += count
            val = 1 - val
        mask = flat.reshape((h, w), order="F")
        return mask.astype(np.uint8) * 255

    return np.zeros(shape, dtype=np.uint8)


def _invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert mask for COLMAP convention: white=keep, black=remove.

    SAM outputs white=detected. COLMAP expects white=keep.
    Verified from reconstruction_pipeline.py:2446: (1 - result.mask) * 255
    """
    return 255 - mask


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MaskConfig:
    """Configuration for SAM 3.1 masking."""

    prompts: list[str] = field(default_factory=lambda: ["person"])
    device: str = "cuda"
    prompt_frame_index: int = 0
    output_format: str = "png"
    confidence_threshold: float = 0.5


@dataclass
class MaskResult:
    """Result of a masking run."""

    success: bool
    total_frames: int = 0
    masked_frames: int = 0
    masks_dir: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Masker
# ---------------------------------------------------------------------------


class Masker:
    """SAM 3.1 text-prompted operator masking for ERP frames.

    Usage::

        masker = Masker(MaskConfig(prompts=["person", "tripod"]))
        masker.initialize()
        result = masker.process_frames("extracted/frames/", "extracted/masks/")
        masker.cleanup()
    """

    def __init__(self, config: MaskConfig | None = None) -> None:
        if not HAS_SAM3:
            raise ImportError(
                "SAM 3.1 not available. Install: "
                "git clone https://github.com/facebookresearch/sam3 && pip install -e ."
            )
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")

        self.config = config or MaskConfig()
        self._predictor: Any = None
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load SAM 3.1 model. Call once before processing."""
        logger.info("Loading SAM 3.1 multiplex video predictor")
        self._predictor = build_sam3_multiplex_video_predictor()
        logger.info("SAM 3.1 multiplex video predictor ready")

    def cleanup(self) -> None:
        """Release model resources and free GPU memory."""
        if self._session_id is not None and self._predictor is not None:
            try:
                self._predictor.handle_request(
                    {"type": "close_session", "session_id": self._session_id}
                )
            except Exception:
                pass
            self._session_id = None
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        if HAS_TORCH and torch.cuda.is_available():  # type: ignore[possibly-undefined]
            torch.cuda.empty_cache()
        logger.info("Masker cleaned up")

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process_frames(
        self,
        frames_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> MaskResult:
        """Detect and mask objects in all frames.

        Loads frames as PIL images and passes them directly to SAM 3.1's
        video predictor. Outputs inverted binary masks (white=keep,
        black=remove) to *output_dir*.

        Args:
            frames_dir: Directory containing ERP frame images.
            output_dir: Directory to write mask PNGs.
            progress_callback: Optional ``(current, total, message)`` callback.

        Returns:
            MaskResult with statistics.
        """
        if self._predictor is None:
            return MaskResult(
                success=False, error="Not initialized. Call initialize() first."
            )

        from PIL import Image

        frames_path = Path(frames_dir)
        masks_dir = Path(output_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Discover frame files
        frame_files = sorted(
            f
            for f in frames_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_files:
            return MaskResult(
                success=False,
                error=f"No frame images found in {frames_dir}",
                masks_dir=str(masks_dir),
            )

        n_frames = len(frame_files)
        logger.info("Processing %d frames with SAM 3.1 video pipeline", n_frames)

        # Load frames as PIL images
        pil_images = [Image.open(f).convert("RGB") for f in frame_files]

        try:
            # Start session with PIL image list
            # Verified: sam3/model/io_utils.py:42-67 accepts list[PIL.Image.Image]
            response = self._predictor.handle_request(
                {"type": "start_session", "resource_path": pil_images}
            )
            self._session_id = response["session_id"]
            logger.info("SAM 3.1 session started: %s", self._session_id)

            # Add text prompts on the designated frame
            prompt_idx = min(self.config.prompt_frame_index, n_frames - 1)
            for prompt_text in self.config.prompts:
                try:
                    self._predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": self._session_id,
                            "frame_index": prompt_idx,
                            "text": prompt_text,
                        }
                    )
                    logger.info("Added prompt '%s' on frame %d", prompt_text, prompt_idx)
                except Exception as exc:
                    logger.warning("Prompt '%s' failed: %s", prompt_text, exc)

            if progress_callback:
                progress_callback(0, n_frames, "Prompts added, propagating...")

            # Propagate through video (streaming API yields per-frame results)
            outputs_per_frame: dict[int, dict[int, Any]] = {}
            for response in self._predictor.handle_stream_request(
                {"type": "propagate_in_video", "session_id": self._session_id}
            ):
                frame_idx = response["frame_index"]
                outputs_per_frame[frame_idx] = response["outputs"]

            # Write inverted masks (white=keep, black=remove)
            masked_frames = 0
            for idx in range(n_frames):
                frame_data = outputs_per_frame.get(idx, {})
                h, w = pil_images[idx].size[1], pil_images[idx].size[0]

                # Merge all object masks (union)
                combined = np.zeros((h, w), dtype=np.uint8)
                for _obj_id, mask_data in frame_data.items():
                    mask = _decode_mask(mask_data, (h, w))
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined = np.maximum(combined, mask)

                # Invert: SAM white=detected -> COLMAP white=keep
                inverted = _invert_mask(combined)
                mask_path = masks_dir / f"{frame_files[idx].stem}.png"
                cv2.imwrite(str(mask_path), inverted)
                masked_frames += 1

                if progress_callback:
                    progress_callback(
                        masked_frames, n_frames, f"Tracking {masked_frames}/{n_frames}"
                    )

            # Close session
            self._predictor.handle_request(
                {"type": "close_session", "session_id": self._session_id}
            )
            self._session_id = None

        except Exception as exc:
            logger.error("SAM 3.1 pipeline failed: %s", exc)
            return MaskResult(
                success=False,
                total_frames=n_frames,
                masked_frames=0,
                masks_dir=str(masks_dir),
                error=str(exc),
            )

        logger.info(
            "SAM 3.1 pipeline complete: %d/%d frames masked", masked_frames, n_frames
        )
        return MaskResult(
            success=True,
            total_frames=n_frames,
            masked_frames=masked_frames,
            masks_dir=str(masks_dir),
        )
