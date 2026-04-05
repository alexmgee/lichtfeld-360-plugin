# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detection backends for operator masking.

Two tiers:
- Default: YOLO + SAM v1 (zero friction, pip installable, no gating)
- Premium: SAM 3 (opt-in, gated model, better quality)

Both implement the same interface. The pipeline uses whichever
backend the user has chosen and installed.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
    from sam3.model_builder import build_sam3_image_model  # type: ignore[import-untyped]
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]
    HAS_SAM3 = True
except ImportError:
    pass


# ── Backend protocol ──────────────────────────────────────────

class MaskingBackend(Protocol):
    def initialize(self) -> None: ...
    def detect_and_segment(
        self,
        image: np.ndarray,
        targets: list[str],
        detection_confidence: float = 0.35,
        single_primary_box: bool = False,
    ) -> np.ndarray: ...
    def batch_detect_boxes(
        self,
        images: list[np.ndarray],
        detection_confidence: float = 0.35,
    ) -> list[list[tuple[np.ndarray, float]]]: ...
    def cleanup(self) -> None: ...


# ── YOLO + SAM v1 backend ────────────────────────────────────

SAM1_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)


class YoloSamBackend:
    """Default tier: YOLO bounding box detection + SAM v1 ViT-H segmentation.

    Reference: FullCircle mask_perspectives.py
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._yolo: Any = None
        self._sam_predictor: Any = None

    @staticmethod
    def _select_person_boxes(
        boxes: list[np.ndarray],
        confidences: list[float],
        single_primary_box: bool,
    ) -> list[np.ndarray]:
        """Select person boxes to pass to SAM.

        For cubemap direct masking we assume a single operator and keep only
        the strongest detected person box. This avoids feeding a weaker,
        disconnected false-positive box into SAM and masking extra furniture
        or edges. Default masking keeps the legacy all-box behavior.
        """
        if not boxes:
            return []
        if not single_primary_box:
            return boxes

        best_idx = max(range(len(boxes)), key=lambda idx: confidences[idx])
        return [boxes[best_idx]]

    def initialize(self) -> None:
        if not HAS_YOLO:
            raise ImportError("ultralytics not installed")
        if not HAS_SAM1:
            raise ImportError("segment-anything not installed")

        logger.info("Loading YOLO model...")
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ultralytics")
        os.makedirs(cache_dir, exist_ok=True)
        yolo_path = os.path.join(cache_dir, "yolov8s.pt")
        self._yolo = YOLO(yolo_path)

        logger.info("Loading SAM v1 ViT-H model...")
        import os
        import urllib.request
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sam")
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
        if not os.path.exists(ckpt_path):
            logger.info("Downloading SAM ViT-H checkpoint to %s ...", ckpt_path)
            urllib.request.urlretrieve(SAM1_CHECKPOINT_URL, ckpt_path)

        sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("YOLO + SAM v1 backend ready")

    def detect_and_segment(
        self,
        image: np.ndarray,
        targets: list[str],
        detection_confidence: float = 0.35,
        single_primary_box: bool = False,
    ) -> np.ndarray:
        """Detect person via YOLO, segment via SAM v1. Returns 0/1 uint8 mask."""
        h, w = image.shape[:2]

        # YOLO detection — person = COCO class 0
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._yolo(
            image_rgb, stream=True, verbose=False, conf=detection_confidence,
            iou=0.6, classes=[0], agnostic_nms=False, max_det=20,
        )
        all_boxes = []
        confidences = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for j in range(len(result.boxes)):
                conf = float(result.boxes.conf[j])
                if conf < detection_confidence:
                    continue
                box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                all_boxes.append(box)
                confidences.append(conf)

        selected_boxes = self._select_person_boxes(
            all_boxes, confidences, single_primary_box,
        )

        if not selected_boxes:
            return np.zeros((h, w), dtype=np.uint8)

        # SAM segmentation from YOLO boxes
        self._sam_predictor.set_image(image_rgb)
        input_boxes = torch.as_tensor(
            np.array(selected_boxes), device=self._device,
        )
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

    def batch_detect_boxes(
        self,
        images: list[np.ndarray],
        detection_confidence: float = 0.35,
    ) -> list[list[tuple[np.ndarray, float]]]:
        """Run batched YOLO person detection on BGR images.

        Returns per-image list of (xyxy_box, confidence) tuples.
        COCO class 0 (person) only. RGB conversion handled internally.
        """
        images_rgb = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images
        ]
        batch_results = list(self._yolo(
            images_rgb, stream=True, verbose=False, conf=detection_confidence,
            iou=0.6, classes=[0], agnostic_nms=False, max_det=20,
        ))
        all_detections: list[list[tuple[np.ndarray, float]]] = []
        for result in batch_results:
            detections: list[tuple[np.ndarray, float]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    conf = float(result.boxes.conf[j])
                    if conf < detection_confidence:
                        continue
                    box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                    detections.append((box, conf))
            all_detections.append(detections)
        return all_detections

    def cleanup(self) -> None:
        self._yolo = None
        self._sam_predictor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("YOLO + SAM v1 backend cleaned up")


# ── SAM 3 backend ────────────────────────────────────────────

class Sam3Backend:
    """Premium tier: SAM 3 text-prompted detection + segmentation.

    Uses the image API (build_sam3_image_model + Sam3Processor),
    NOT the video/multiplex API. Each pinhole view is an independent
    image, not a video sequence.
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    def initialize(self) -> None:
        if not HAS_SAM3:
            raise ImportError(
                "SAM 3 not available. Install: uv add sam3"
            )
        logger.info("Loading SAM 3 image model...")
        self._model = build_sam3_image_model(device=self._device)
        self._processor = Sam3Processor(self._model)
        logger.info("SAM 3 backend ready")

    def detect_and_segment(
        self,
        image: np.ndarray,
        targets: list[str],
        detection_confidence: float = 0.35,
        single_primary_box: bool = False,
    ) -> np.ndarray:
        """Detect targets via text prompts, segment. Returns 0/1 uint8 mask."""
        from PIL import Image as PILImage

        h, w = image.shape[:2]
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        state = self._processor.set_image(pil_img)

        combined = np.zeros((h, w), dtype=np.uint8)
        for prompt_text in targets:
            try:
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

        return combined

    def batch_detect_boxes(
        self,
        images: list[np.ndarray],
        detection_confidence: float = 0.35,
    ) -> list[list[tuple[np.ndarray, float]]]:
        """Per-image detection via SAM 3. Returns bounding boxes from mask contours."""
        all_detections: list[list[tuple[np.ndarray, float]]] = []
        for image in images:
            mask = self.detect_and_segment(image, ["person"], detection_confidence)
            detections: list[tuple[np.ndarray, float]] = []
            if mask.sum() > 0:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 100:
                        box = np.array([x, y, x + w, y + h])
                        detections.append((box, 1.0))
            all_detections.append(detections)
        return all_detections

    def cleanup(self) -> None:
        self._model = None
        self._processor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM 3 backend cleaned up")


# ── Backend selection ─────────────────────────────────────────

def get_backend(preference: str | None = None) -> MaskingBackend | None:
    """Return the requested backend, or None if it's not available.

    Args:
        preference: "sam3" for premium tier, "yolo_sam1" for default tier,
                    or None to return the default tier if available.
    """
    if preference == "sam3" and HAS_SAM3 and HAS_TORCH:
        return Sam3Backend()
    if preference == "yolo_sam1" and HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return YoloSamBackend()
    if preference is None and HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return YoloSamBackend()
    return None


# ── Video tracking backends ──────────────────────────────────────

HAS_SAM2 = False
try:
    from sam2.build_sam import build_sam2_video_predictor_hf  # type: ignore[import-untyped]
    HAS_SAM2 = True
except ImportError:
    pass


class VideoTrackingBackend(Protocol):
    """Interface for sequence-level video tracking on synthetic views."""
    def initialize(self) -> None: ...
    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]: ...
    def cleanup(self) -> None: ...


class FallbackVideoBackend:
    """Wraps a MaskingBackend. Calls detect_and_segment per-frame.

    No temporal consistency, but works with any image backend.
    Used when SAM v2 is not available or as a runtime recovery path.

    Ownership rule: this class does NOT own the wrapped image backend
    in the normal Track A path (Masker owns it). In the runtime recovery
    path (fresh backend created after video backend failure),
    process_frames() owns both the fresh backend and this wrapper.
    """

    def __init__(
        self, image_backend: MaskingBackend, targets: list[str]
    ) -> None:
        self._backend = image_backend
        self._targets = targets

    def initialize(self) -> None:
        # No-op: the wrapped backend is already initialized
        pass

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]:
        """Run per-frame detection on each frame independently."""
        results = []
        for frame in frames:
            mask = self._backend.detect_and_segment(frame, self._targets)
            results.append(mask)
        return results

    def cleanup(self) -> None:
        # No-op: we don't own the wrapped backend in the normal path.
        # In the recovery path, process_frames() handles cleanup.
        pass


SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"


class Sam2VideoBackend:
    """SAM v2 video tracking on synthetic views.

    API confirmed from D:/Data/fullcircle/thirdparty/sam-ui/samui/sam.py.
    Uses the standard sam2 predictor API (not the sam-ui helper).

    Frames are resized to 512px min dimension, written as numbered JPEGs
    to a tempdir. SAM v2's init_state() reads from disk. Masks are
    collected by out_frame_idx, then resized back to the caller's
    expected resolution before returning.
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._predictor: Any = None

    def initialize(self) -> None:
        if not HAS_SAM2:
            raise ImportError("sam2 not installed")

        import os
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        # Ensure the bundled _C.pyd extension is in the sam2 package
        from .setup_checks import ensure_sam2_c_extension
        ensure_sam2_c_extension()

        logger.info("Loading SAM v2 video predictor (%s)...", SAM2_MODEL_ID)
        self._predictor = build_sam2_video_predictor_hf(
            SAM2_MODEL_ID, device=self._device,
        )
        logger.info("SAM v2 video predictor ready")

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]:
        """Track a person across synthetic fisheye frames.

        Args:
            frames: List of BGR images (synthetic fisheye views).
            initial_mask: Ignored — uses center-point click instead.
            initial_frame_idx: Frame to prompt on (caller-selected).

        Returns:
            List of binary masks (0/1 uint8) at original frame resolution,
            one per input frame. None entries become zero masks.
        """
        import tempfile
        from PIL import Image

        if self._predictor is None:
            raise RuntimeError("Sam2VideoBackend not initialized")

        n_frames = len(frames)
        if n_frames == 0:
            return []

        orig_h, orig_w = frames[0].shape[:2]

        # Preallocate results by frame index
        results: list[np.ndarray | None] = [None] * n_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Resize and write numbered JPEGs
            resized_w, resized_h = 0, 0
            for i, frame in enumerate(frames):
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                min_dim = min(pil.size)
                if min_dim > 512:
                    scale = 512 / min_dim
                    new_w = int(pil.size[0] * scale)
                    new_h = int(pil.size[1] * scale)
                    pil = pil.resize((new_w, new_h))
                resized_w, resized_h = pil.size
                pil.save(f"{tmpdir}/{i:04d}.jpg", quality=100)

            # Step 2: Init state
            with torch.inference_mode():
                state = self._predictor.init_state(tmpdir)

                # Step 3: Prompt at center of resized frame
                cx = resized_w / 2.0
                cy = resized_h / 2.0
                points = np.array([[cx, cy]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=initial_frame_idx,
                    obj_id=0,
                    points=points,
                    labels=labels,
                )

                # Step 4+5: Propagate forward, then reverse
                for out_idx, out_obj_ids, out_logits in self._predictor.propagate_in_video(
                    state,
                    start_frame_idx=initial_frame_idx,
                    reverse=False,
                ):
                    mask = (torch.sigmoid(out_logits[:, 0]) > 0.5).cpu().numpy().astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]
                    results[out_idx] = mask

                if initial_frame_idx > 0:
                    for out_idx, out_obj_ids, out_logits in self._predictor.propagate_in_video(
                        state,
                        start_frame_idx=initial_frame_idx,
                        reverse=True,
                    ):
                        mask = (torch.sigmoid(out_logits[:, 0]) > 0.5).cpu().numpy().astype(np.uint8)
                        if mask.ndim == 3:
                            mask = mask[0]
                        results[out_idx] = mask

                self._predictor.reset_state(state)

        # Step 6+7: Resize masks back to original frame resolution
        final: list[np.ndarray] = []
        for mask in results:
            if mask is None:
                final.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
            elif mask.shape[:2] != (orig_h, orig_w):
                final.append(cv2.resize(
                    mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ))
            else:
                final.append(mask)

        return final

    def cleanup(self) -> None:
        self._predictor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM v2 video backend cleaned up")


def get_video_backend(
    preference: str | None = None,
    fallback_image_backend: MaskingBackend | None = None,
    targets: list[str] | None = None,
) -> VideoTrackingBackend | None:
    """Return best available video tracking backend.

    Priority: Sam2VideoBackend (if sam2 available + torch)
            > FallbackVideoBackend (wraps image backend)
            > None (if no image backend provided)
    """
    # Track B5 will add: if preference == "sam3" and HAS_SAM3 ...
    if HAS_SAM2 and HAS_TORCH and preference != "fallback":
        return Sam2VideoBackend()
    if fallback_image_backend is not None:
        return FallbackVideoBackend(
            fallback_image_backend, targets or ["person"]
        )
    return None


def get_backend_name(preference: str | None = None) -> str | None:
    """Return name of the backend that would be selected."""
    if preference == "sam3" and HAS_SAM3 and HAS_TORCH:
        return "sam3"
    if preference == "yolo_sam1" and HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return "yolo_sam1"
    if preference is None and HAS_YOLO and HAS_SAM1 and HAS_TORCH:
        return "yolo_sam1"
    return None
