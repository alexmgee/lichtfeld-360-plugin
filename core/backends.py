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
        primary_box_mode: str = "confidence",
        constrain_to_primary_box: bool = False,
        primary_box_padding: float = 0.35,
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


def _box_center_distance(box: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> float:
    """Distance from a box center to the image center in pixels."""
    h, w = image_shape[:2]
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return float(np.hypot(cx - (w / 2.0), cy - (h / 2.0)))


def _box_center(box: np.ndarray) -> tuple[float, float]:
    """Center of an xyxy box."""
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def _box_to_box_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Distance between the centers of two boxes."""
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    return float(np.hypot(ax - bx, ay - by))


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _expand_box(
    box: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    """Expand a box by a fractional padding and clamp it to the image."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(padding)
    pad_y = bh * float(padding)
    ex1 = max(0, int(np.floor(x1 - pad_x)))
    ey1 = max(0, int(np.floor(y1 - pad_y)))
    ex2 = min(w, int(np.ceil(x2 + pad_x)))
    ey2 = min(h, int(np.ceil(y2 + pad_y)))
    return ex1, ey1, ex2, ey2


def select_primary_person_box(
    boxes: list[np.ndarray],
    confidences: list[float],
    image_shape: tuple[int, int, int] | tuple[int, int],
    mode: str = "confidence",
) -> tuple[np.ndarray, float] | None:
    """Pick a single person box for SAM prompting.

    Modes:
    - ``confidence``: strongest detection wins, center/area only break ties
    - ``center``: closest-to-center detection wins, confidence/area break ties

    ``center`` is useful for synthetic fisheye views where the intended
    operator is expected near the view center.
    """
    if not boxes:
        return None

    candidates: list[tuple[np.ndarray, float, float, float]] = []
    for box, conf in zip(boxes, confidences):
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        dist = _box_center_distance(box, image_shape)
        candidates.append((box, float(conf), dist, area))

    if mode == "center":
        best_box, best_conf, _dist, _area = min(
            candidates,
            key=lambda item: (item[2], -item[1], -item[3]),
        )
    else:
        best_box, best_conf, _dist, _area = min(
            candidates,
            key=lambda item: (-item[1], item[2], -item[3]),
        )
    return best_box, best_conf


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
        image_shape: tuple[int, int, int] | tuple[int, int],
        primary_box_mode: str,
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

        selected = select_primary_person_box(
            boxes, confidences, image_shape, mode=primary_box_mode,
        )
        if selected is None:
            return []
        best_box, _best_conf = selected
        return [best_box]

    def _segment_boxes(
        self,
        image: np.ndarray,
        boxes: list[np.ndarray],
        constrain_to_primary_box: bool = False,
        primary_box_padding: float = 0.35,
    ) -> np.ndarray:
        """Segment one or more boxes on an image via SAM v1."""
        h, w = image.shape[:2]
        if not boxes:
            return np.zeros((h, w), dtype=np.uint8)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(image_rgb)
        input_boxes = torch.as_tensor(np.array(boxes), device=self._device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            input_boxes, image_rgb.shape[:2]
        )
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        final_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
        if constrain_to_primary_box and len(boxes) == 1:
            ex1, ey1, ex2, ey2 = _expand_box(
                boxes[0], image.shape, primary_box_padding,
            )
            clipped = np.zeros_like(final_mask)
            clipped[ey1:ey2, ex1:ex2] = final_mask[ey1:ey2, ex1:ex2]
            final_mask = clipped
        return final_mask

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
        primary_box_mode: str = "confidence",
        constrain_to_primary_box: bool = False,
        primary_box_padding: float = 0.35,
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
            all_boxes, confidences, single_primary_box, image.shape, primary_box_mode,
        )

        if not selected_boxes:
            return np.zeros((h, w), dtype=np.uint8)

        # SAM segmentation from YOLO boxes
        return self._segment_boxes(
            image,
            selected_boxes,
            constrain_to_primary_box=(
                single_primary_box and constrain_to_primary_box
            ),
            primary_box_padding=primary_box_padding,
        )

    def segment_boxes(
        self,
        image: np.ndarray,
        boxes: list[np.ndarray],
        constrain_to_primary_box: bool = False,
        primary_box_padding: float = 0.35,
    ) -> np.ndarray:
        """Segment explicit boxes via SAM v1 without rerunning YOLO."""
        return self._segment_boxes(
            image,
            boxes,
            constrain_to_primary_box=constrain_to_primary_box,
            primary_box_padding=primary_box_padding,
        )

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
        primary_box_mode: str = "confidence",
        constrain_to_primary_box: bool = False,
        primary_box_padding: float = 0.35,
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
        initial_box: np.ndarray | None = None,
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
        self,
        image_backend: MaskingBackend,
        targets: list[str],
        detection_confidence: float = 0.25,
        single_primary_box: bool = True,
        primary_box_mode: str = "center",
        constrain_to_primary_box: bool = True,
        primary_box_padding: float = 0.35,
    ) -> None:
        self._backend = image_backend
        self._targets = targets
        self._detection_confidence = detection_confidence
        self._single_primary_box = single_primary_box
        self._primary_box_mode = primary_box_mode
        self._constrain_to_primary_box = constrain_to_primary_box
        self._primary_box_padding = primary_box_padding
        self._detected_box_padding = min(primary_box_padding, 0.25)
        self._low_conf_box_padding = min(primary_box_padding, 0.18)
        self._propagated_box_padding = min(primary_box_padding, 0.12)
        self._local_redetect_box_padding = min(primary_box_padding, 0.30)
        self._trusted_direct_box_padding = min(primary_box_padding, 0.30)
        self._trusted_local_redetect_box_padding = min(primary_box_padding, 0.34)
        self._uncertain_local_redetect_box_padding = min(primary_box_padding, 0.24)
        self._center_redetect_box_padding = min(primary_box_padding, 0.32)
        self._recovery_dilate_kernel = 3
        self._completeness_kernel = 3
        self._completeness_min_confidence = 0.38
        self._completeness_max_center_dist = 520.0
        self._completeness_bottom_exclusion = 0.16
        self._completeness_top_fraction = 0.48
        self._completeness_mid_start_fraction = 0.16
        self._completeness_mid_end_fraction = 0.80
        self._completeness_top_kernel = (5, 7)
        self._completeness_side_kernel = (7, 3)
        self._reprompt_min_confidence = 0.42
        self._reprompt_max_center_dist = 520.0
        self._reprompt_box_expand = 0.12
        # Synthetic fallback should strongly prefer detections near the
        # expected operator-centered region. Far-off boxes are more likely
        # to be the wrong subject or a poor fallback anchor.
        self._max_center_dist = 450.0
        self._local_redetect_confidence = min(self._detection_confidence, 0.20)
        self._center_redetect_confidence = min(self._detection_confidence, 0.18)
        self._local_search_scale = 1.8
        self._local_search_scale_max = 2.2
        self._local_search_min_pad = 64.0
        self._center_search_half_width = self._max_center_dist + 192.0
        self._center_search_half_height = self._max_center_dist + 256.0
        self._local_redetect_iou_floor = 0.15
        self._local_redetect_min_center_shift = 72.0
        self._local_center_dist_slack = 96.0
        self._local_center_dist_worsen_limit = 64.0
        self._local_redetect_iou_override = 0.25
        # Allow short continuity gaps, but reuse the last *detected* box
        # rather than chaining propagated boxes forward indefinitely.
        self._max_box_reuse_gap = 2
        self._last_track_meta: list[dict[str, Any]] = []

    @property
    def last_track_meta(self) -> list[dict[str, Any]]:
        """Per-frame metadata from the most recent track_sequence call."""
        return self._last_track_meta

    def _select_temporal_box(
        self,
        detections: list[tuple[np.ndarray, float]],
        image_shape: tuple[int, int, int] | tuple[int, int],
        continuity_box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """Select a primary box, preferring continuity when available."""
        if not detections:
            return None

        boxes = [box for box, _ in detections]
        confs = [float(conf) for _, conf in detections]
        if continuity_box is None:
            selected = select_primary_person_box(
                boxes, confs, image_shape, mode=self._primary_box_mode,
            )
            if selected is None:
                return None
            best_box, best_conf = selected
            if _box_center_distance(best_box, image_shape) > self._max_center_dist:
                return None
            return best_box, best_conf

        candidates: list[tuple[np.ndarray, float, float, float, float]] = []
        for box, conf in zip(boxes, confs):
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            continuity_dist = _box_to_box_center_distance(box, continuity_box)
            center_dist = _box_center_distance(box, image_shape)
            candidates.append((box, conf, continuity_dist, center_dist, area))

        best_box, best_conf, _cont, _center, _area = min(
            candidates,
            key=lambda item: (item[2], item[3], -item[1], -item[4]),
        )
        if _box_center_distance(best_box, image_shape) > self._max_center_dist:
            return None
        return best_box, best_conf

    def initialize(self) -> None:
        # No-op: the wrapped backend is already initialized
        pass

    def _padding_for_selection(
        self,
        selection_source: str,
        selection_confidence: float | None,
        selection_center_dist: float | None = None,
    ) -> float:
        """Choose a box clip padding based on how trustworthy the selection is."""
        if selection_source.startswith("local_redetect"):
            if selection_confidence is not None and selection_confidence < 0.36:
                return self._uncertain_local_redetect_box_padding
            if (
                selection_center_dist is not None
                and selection_center_dist <= 180.0
            ):
                return self._trusted_local_redetect_box_padding
            return self._local_redetect_box_padding
        if selection_source.startswith("center_redetect"):
            return self._center_redetect_box_padding
        if selection_source.startswith("propagated"):
            return self._propagated_box_padding
        if (
            selection_source == "detected_temporal"
            and selection_confidence is not None
            and selection_confidence >= 0.42
            and selection_center_dist is not None
            and selection_center_dist <= 90.0
        ):
            return self._trusted_direct_box_padding
        if selection_confidence is not None and selection_confidence < 0.4:
            return self._low_conf_box_padding
        return self._detected_box_padding

    def _should_apply_recovery_dilation(
        self,
        selection_source: str,
        selection_confidence: float | None,
        selection_center_dist: float | None,
    ) -> bool:
        """Apply a tiny dilation only to trusted centered current-frame recoveries."""
        if selection_source not in (
            "detected_temporal",
            "local_redetect_prev",
            "local_redetect_next",
        ):
            return False
        if selection_confidence is None or selection_confidence < 0.38:
            return False
        if selection_center_dist is None or selection_center_dist > 220.0:
            return False
        return True

    def _should_apply_completeness_recovery(
        self,
        selection_source: str,
        selection_confidence: float | None,
        selection_center_dist: float | None,
    ) -> bool:
        """Apply a small upper-biased completeness pass to trusted current-frame masks."""
        if selection_source not in (
            "detected_temporal",
            "local_redetect_prev",
            "local_redetect_next",
            "center_redetect_none",
            "center_redetect_prev",
            "center_redetect_next",
            "direction_search",
            "altview_prev",
            "altview_next",
        ):
            return False
        if selection_confidence is None or selection_confidence < self._completeness_min_confidence:
            return False
        if selection_center_dist is None or selection_center_dist > self._completeness_max_center_dist:
            return False
        return True

    def _apply_recovery_dilation(
        self,
        mask: np.ndarray,
        selected_box: np.ndarray,
        image_shape: tuple[int, int, int] | tuple[int, int],
        clip_padding: float,
    ) -> np.ndarray:
        """Dilate slightly, then re-clip to the trusted selection box region."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._recovery_dilate_kernel, self._recovery_dilate_kernel),
        )
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        ex1, ey1, ex2, ey2 = _expand_box(selected_box, image_shape, clip_padding)
        clipped = np.zeros_like(dilated)
        clipped[ey1:ey2, ex1:ex2] = dilated[ey1:ey2, ex1:ex2]
        return clipped

    def _apply_completeness_recovery(
        self,
        mask: np.ndarray,
        selected_box: np.ndarray,
        image_shape: tuple[int, int, int] | tuple[int, int],
        clip_padding: float,
    ) -> np.ndarray:
        """Recover small missing operator regions without expanding into foot-level shadow too aggressively."""
        ex1, ey1, ex2, ey2 = _expand_box(selected_box, image_shape, clip_padding)
        if ex2 <= ex1 or ey2 <= ey1:
            return mask

        region_h = ey2 - ey1
        if region_h < 8:
            return mask

        completeness_y2 = max(
            ey1 + 4,
            ey2 - int(round(region_h * self._completeness_bottom_exclusion)),
        )
        if completeness_y2 <= ey1:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._completeness_kernel, self._completeness_kernel),
        )
        recovered = mask.astype(np.uint8).copy()
        roi = recovered[ey1:completeness_y2, ex1:ex2]
        if roi.size == 0:
            return mask
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        expanded = cv2.dilate(closed, kernel, iterations=1)
        recovered[ey1:completeness_y2, ex1:ex2] = np.maximum(roi, expanded)

        # Add a small top-focused pass for head gaps without opening the lower shadow zone.
        top_y2 = min(
            completeness_y2,
            ey1 + max(4, int(round(region_h * self._completeness_top_fraction))),
        )
        if top_y2 > ey1:
            top_roi = recovered[ey1:top_y2, ex1:ex2]
            if top_roi.size > 0:
                top_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    self._completeness_top_kernel,
                )
                top_closed = cv2.morphologyEx(
                    top_roi,
                    cv2.MORPH_CLOSE,
                    top_kernel,
                    iterations=1,
                )
                top_expanded = cv2.dilate(top_closed, top_kernel, iterations=1)
                recovered[ey1:top_y2, ex1:ex2] = np.maximum(top_roi, top_expanded)

        # Add a narrow mid-band side recovery to catch small arm/forearm gaps.
        mid_y1 = ey1 + int(round(region_h * self._completeness_mid_start_fraction))
        mid_y2 = min(
            completeness_y2,
            ey1 + int(round(region_h * self._completeness_mid_end_fraction)),
        )
        if mid_y2 > mid_y1:
            mid_roi = recovered[mid_y1:mid_y2, ex1:ex2]
            if mid_roi.size > 0:
                side_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    self._completeness_side_kernel,
                )
                side_closed = cv2.morphologyEx(
                    mid_roi,
                    cv2.MORPH_CLOSE,
                    side_kernel,
                    iterations=1,
                )
                side_expanded = cv2.dilate(side_closed, side_kernel, iterations=1)
                recovered[mid_y1:mid_y2, ex1:ex2] = np.maximum(mid_roi, side_expanded)

        return recovered

    def _should_apply_reprompt_recovery(
        self,
        selection_source: str,
        selection_confidence: float | None,
        selection_center_dist: float | None,
    ) -> bool:
        """Run a second trusted segmentation prompt before relying on morphology alone."""
        if selection_source not in (
            "detected_temporal",
            "local_redetect_prev",
            "local_redetect_next",
            "center_redetect_none",
            "center_redetect_prev",
            "center_redetect_next",
            "direction_search",
            "altview_prev",
            "altview_next",
        ):
            return False
        if selection_confidence is None or selection_confidence < self._reprompt_min_confidence:
            return False
        if selection_center_dist is None or selection_center_dist > self._reprompt_max_center_dist:
            return False
        return True

    def _apply_reprompt_recovery(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        selected_box: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Re-prompt SAM with a slightly larger trusted box and merge only upper/mid support."""
        ex1, ey1, ex2, ey2 = _expand_box(
            selected_box,
            frame.shape,
            self._reprompt_box_expand,
        )
        if ex2 <= ex1 or ey2 <= ey1:
            return mask, 0

        reprompt_box = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)
        reprompt_mask = self._backend.segment_boxes(
            frame,
            [reprompt_box],
            constrain_to_primary_box=True,
            primary_box_padding=0.0,
        )
        if reprompt_mask is None or reprompt_mask.shape != mask.shape:
            return mask, 0

        region_h = ey2 - ey1
        merge_y2 = max(
            ey1 + 4,
            ey2 - int(round(region_h * self._completeness_bottom_exclusion)),
        )
        if merge_y2 <= ey1:
            return mask, 0

        recovered = mask.astype(np.uint8).copy()
        current_roi = recovered[ey1:merge_y2, ex1:ex2]
        reprompt_roi = reprompt_mask[ey1:merge_y2, ex1:ex2]
        if current_roi.size == 0 or reprompt_roi.size == 0:
            return mask, 0

        merged = np.maximum(current_roi, reprompt_roi)
        gain_pixels = int(merged.sum() - current_roi.sum())
        if gain_pixels <= 0:
            return mask, 0
        recovered[ey1:merge_y2, ex1:ex2] = merged
        return recovered, gain_pixels

    def _build_local_search_window(
        self,
        propagated_box: np.ndarray,
        image_shape: tuple[int, int, int] | tuple[int, int],
        propagation_gap: int,
    ) -> tuple[int, int, int, int]:
        """Build a local re-detect crop around a propagated synthetic box."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in propagated_box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        scale = min(
            self._local_search_scale + (0.2 * max(0, int(propagation_gap) - 1)),
            self._local_search_scale_max,
        )
        pad_x = max(((scale - 1.0) * bw) / 2.0, self._local_search_min_pad)
        pad_y = max(((scale - 1.0) * bh) / 2.0, self._local_search_min_pad)
        wx1 = max(0, int(np.floor(x1 - pad_x)))
        wy1 = max(0, int(np.floor(y1 - pad_y)))
        wx2 = min(w, int(np.ceil(x2 + pad_x)))
        wy2 = min(h, int(np.ceil(y2 + pad_y)))
        return wx1, wy1, wx2, wy2

    def _build_center_search_window(
        self,
        image_shape: tuple[int, int, int] | tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Build a center-focused search crop for operator-centered synthetic views."""
        h, w = image_shape[:2]
        cx = w / 2.0
        cy = h / 2.0
        half_w = min(w / 2.0, self._center_search_half_width)
        half_h = min(h / 2.0, self._center_search_half_height)
        wx1 = max(0, int(np.floor(cx - half_w)))
        wy1 = max(0, int(np.floor(cy - half_h)))
        wx2 = min(w, int(np.ceil(cx + half_w)))
        wy2 = min(h, int(np.ceil(cy + half_h)))
        return wx1, wy1, wx2, wy2

    def _select_local_redetect_box(
        self,
        detections: list[tuple[np.ndarray, float]],
        propagated_box: np.ndarray,
        image_shape: tuple[int, int, int] | tuple[int, int],
    ) -> tuple[np.ndarray, float, float, float, float] | None:
        """Choose a plausible current-frame box near a propagated hint."""
        if not detections:
            return None

        prop_center_dist = _box_center_distance(propagated_box, image_shape)
        prop_w = max(1.0, float(propagated_box[2] - propagated_box[0]))
        prop_h = max(1.0, float(propagated_box[3] - propagated_box[1]))
        prop_diag = max(1.0, float(np.hypot(prop_w, prop_h)))
        max_shift = max(self._local_redetect_min_center_shift, 0.75 * prop_diag)
        center_dist_limit = min(
            self._max_center_dist,
            prop_center_dist + max(self._local_center_dist_slack, 0.5 * prop_diag),
        )

        candidates: list[tuple[np.ndarray, float, float, float, float, float]] = []
        for box, conf in detections:
            iou = _box_iou(box, propagated_box)
            center_shift = _box_to_box_center_distance(box, propagated_box)
            center_dist = _box_center_distance(box, image_shape)
            if center_dist > center_dist_limit:
                continue
            if iou < self._local_redetect_iou_floor and center_shift > max_shift:
                continue
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            agreement = max(iou, max(0.0, 1.0 - (center_shift / max(max_shift, 1.0))))
            candidates.append((box, float(conf), iou, center_shift, center_dist, agreement + (area * 1e-8)))

        if not candidates:
            return None

        best_box, best_conf, best_iou, best_shift, best_center_dist, _score = max(
            candidates,
            key=lambda item: (item[5], item[1], -item[4]),
        )
        if (
            best_center_dist > prop_center_dist + self._local_center_dist_worsen_limit
            and best_iou < self._local_redetect_iou_override
        ):
            return None
        return best_box, best_conf, best_iou, best_shift, best_center_dist

    def _select_center_redetect_box(
        self,
        detections: list[tuple[np.ndarray, float]],
        image_shape: tuple[int, int, int] | tuple[int, int],
        continuity_box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, float | None] | None:
        """Choose a center-biased current-frame box in a synthetic view."""
        if not detections:
            return None

        candidates: list[tuple[np.ndarray, float, float, float | None, float]] = []
        for box, conf in detections:
            center_dist = _box_center_distance(box, image_shape)
            if center_dist > self._max_center_dist:
                continue
            continuity_shift = (
                _box_to_box_center_distance(box, continuity_box)
                if continuity_box is not None else None
            )
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            candidates.append((box, float(conf), center_dist, continuity_shift, area))

        if not candidates:
            return None

        if continuity_box is None:
            best_box, best_conf, best_center_dist, _best_shift, _area = min(
                candidates,
                key=lambda item: (item[2], -item[1], -item[4]),
            )
            return best_box, best_conf, best_center_dist, None

        best_box, best_conf, best_center_dist, best_shift, _area = min(
            candidates,
            key=lambda item: (item[2], item[3] if item[3] is not None else float("inf"), -item[1], -item[4]),
        )
        return best_box, best_conf, best_center_dist, best_shift

    def _attempt_local_redetect(
        self,
        frame: np.ndarray,
        propagated_box: np.ndarray,
        propagation_gap: int,
    ) -> tuple[tuple[np.ndarray, float, float, float, float] | None, dict[str, Any]]:
        """Refine a propagated box using a local current-frame re-detect."""
        window = self._build_local_search_window(
            propagated_box,
            frame.shape,
            propagation_gap=propagation_gap,
        )
        wx1, wy1, wx2, wy2 = window
        meta: dict[str, Any] = {
            "local_redetect_attempted": True,
            "local_redetect_window": [wx1, wy1, wx2, wy2],
            "local_redetect_candidates": 0,
            "local_redetect_selected_confidence": None,
            "local_redetect_selected_iou": None,
            "local_redetect_selected_center_shift": None,
            "local_redetect_replaced_propagation": False,
            "propagation_gap": int(propagation_gap),
        }

        if wx2 <= wx1 or wy2 <= wy1:
            return None, meta

        crop = frame[wy1:wy2, wx1:wx2]
        if crop.size == 0:
            return None, meta

        crop_detections = self._backend.batch_detect_boxes(
            [crop],
            detection_confidence=self._local_redetect_confidence,
        )
        local_candidates: list[tuple[np.ndarray, float]] = []
        if crop_detections and crop_detections[0]:
            for box, conf in crop_detections[0]:
                mapped = np.array(
                    [
                        float(box[0]) + wx1,
                        float(box[1]) + wy1,
                        float(box[2]) + wx1,
                        float(box[3]) + wy1,
                    ],
                    dtype=np.float32,
                )
                local_candidates.append((mapped, float(conf)))

        meta["local_redetect_candidates"] = len(local_candidates)
        selected = self._select_local_redetect_box(
            local_candidates,
            propagated_box,
            frame.shape,
        )
        if selected is None:
            return None, meta

        best_box, best_conf, best_iou, best_shift, _best_center_dist = selected
        meta["local_redetect_selected_confidence"] = float(best_conf)
        meta["local_redetect_selected_iou"] = float(best_iou)
        meta["local_redetect_selected_center_shift"] = float(best_shift)
        meta["local_redetect_replaced_propagation"] = True
        return selected, meta

    def _attempt_center_redetect(
        self,
        frame: np.ndarray,
        continuity_box: np.ndarray | None = None,
    ) -> tuple[tuple[np.ndarray, float, float, float | None] | None, dict[str, Any]]:
        """Run a center-focused re-detect on the current synthetic frame."""
        window = self._build_center_search_window(frame.shape)
        wx1, wy1, wx2, wy2 = window
        meta: dict[str, Any] = {
            "center_redetect_attempted": True,
            "center_redetect_window": [wx1, wy1, wx2, wy2],
            "center_redetect_candidates": 0,
            "center_redetect_selected_confidence": None,
            "center_redetect_selected_center_dist": None,
            "center_redetect_selected_continuity_shift": None,
            "center_redetect_replaced_selection": False,
        }

        if wx2 <= wx1 or wy2 <= wy1:
            return None, meta

        crop = frame[wy1:wy2, wx1:wx2]
        if crop.size == 0:
            return None, meta

        crop_detections = self._backend.batch_detect_boxes(
            [crop],
            detection_confidence=self._center_redetect_confidence,
        )
        center_candidates: list[tuple[np.ndarray, float]] = []
        if crop_detections and crop_detections[0]:
            for box, conf in crop_detections[0]:
                mapped = np.array(
                    [
                        float(box[0]) + wx1,
                        float(box[1]) + wy1,
                        float(box[2]) + wx1,
                        float(box[3]) + wy1,
                    ],
                    dtype=np.float32,
                )
                center_candidates.append((mapped, float(conf)))

        meta["center_redetect_candidates"] = len(center_candidates)
        selected = self._select_center_redetect_box(
            center_candidates,
            frame.shape,
            continuity_box=continuity_box,
        )
        if selected is None:
            return None, meta

        best_box, best_conf, best_center_dist, best_shift = selected
        meta["center_redetect_selected_confidence"] = float(best_conf)
        meta["center_redetect_selected_center_dist"] = float(best_center_dist)
        meta["center_redetect_selected_continuity_shift"] = (
            float(best_shift) if best_shift is not None else None
        )
        meta["center_redetect_replaced_selection"] = True
        return selected, meta

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
        initial_box: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Run per-frame detection with lightweight temporal box continuity."""
        del initial_mask, initial_box
        if not frames:
            self._last_track_meta = []
            return []

        if not hasattr(self._backend, "batch_detect_boxes") or not hasattr(self._backend, "segment_boxes"):
            results = []
            self._last_track_meta = []
            for frame in frames:
                mask = self._backend.detect_and_segment(
                    frame,
                    self._targets,
                    detection_confidence=self._detection_confidence,
                    single_primary_box=self._single_primary_box,
                    primary_box_mode=self._primary_box_mode,
                    constrain_to_primary_box=self._constrain_to_primary_box,
                    primary_box_padding=self._primary_box_padding,
                )
                results.append(mask)
                self._last_track_meta.append({})
            return results

        detections_per_frame = self._backend.batch_detect_boxes(
            frames,
            detection_confidence=self._detection_confidence,
        )
        n_frames = len(frames)
        selected_boxes: list[np.ndarray | None] = [None] * n_frames
        selected_confidences: list[float | None] = [None] * n_frames
        selected_sources: list[str] = ["none"] * n_frames
        propagation_gaps: list[int | None] = [None] * n_frames

        # Anchor the sequence on the caller's prompt frame.
        anchor_selection = self._select_temporal_box(
            detections_per_frame[initial_frame_idx],
            frames[initial_frame_idx].shape,
            continuity_box=None,
        )
        if anchor_selection is not None:
            selected_boxes[initial_frame_idx], selected_confidences[initial_frame_idx] = anchor_selection
            selected_sources[initial_frame_idx] = "anchor_detected"
            propagation_gaps[initial_frame_idx] = 0

        # Forward pass
        last_detected_box = selected_boxes[initial_frame_idx]
        gap = 0
        for i in range(initial_frame_idx + 1, n_frames):
            selected = self._select_temporal_box(
                detections_per_frame[i], frames[i].shape, continuity_box=last_detected_box,
            )
            if selected is not None:
                selected_boxes[i], selected_confidences[i] = selected
                selected_sources[i] = "detected_temporal"
                propagation_gaps[i] = 0
                last_detected_box = selected_boxes[i]
                gap = 0
            elif last_detected_box is not None and gap < self._max_box_reuse_gap:
                gap += 1
                selected_boxes[i] = last_detected_box.copy()
                selected_confidences[i] = None
                selected_sources[i] = "propagated_prev"
                propagation_gaps[i] = gap
            else:
                last_detected_box = None
                gap = 0

        # Backward pass
        last_detected_box = selected_boxes[initial_frame_idx]
        gap = 0
        for i in range(initial_frame_idx - 1, -1, -1):
            selected = self._select_temporal_box(
                detections_per_frame[i], frames[i].shape, continuity_box=last_detected_box,
            )
            if selected is not None:
                selected_boxes[i], selected_confidences[i] = selected
                selected_sources[i] = "detected_temporal"
                propagation_gaps[i] = 0
                last_detected_box = selected_boxes[i]
                gap = 0
            elif last_detected_box is not None and gap < self._max_box_reuse_gap:
                gap += 1
                selected_boxes[i] = last_detected_box.copy()
                selected_confidences[i] = None
                selected_sources[i] = "propagated_next"
                propagation_gaps[i] = gap
            else:
                last_detected_box = None
                gap = 0

        results = []
        self._last_track_meta = []
        for frame, selected_box, selected_conf, selected_source, propagation_gap, detections in zip(
            frames,
            selected_boxes,
            selected_confidences,
            selected_sources,
            propagation_gaps,
            detections_per_frame,
        ):
            local_redetect_meta: dict[str, Any] = {
                "local_redetect_attempted": False,
                "local_redetect_window": None,
                "local_redetect_candidates": 0,
                "local_redetect_selected_confidence": None,
                "local_redetect_selected_iou": None,
                "local_redetect_selected_center_shift": None,
                "local_redetect_replaced_propagation": False,
                "center_redetect_attempted": False,
                "center_redetect_window": None,
                "center_redetect_candidates": 0,
                "center_redetect_selected_confidence": None,
                "center_redetect_selected_center_dist": None,
                "center_redetect_selected_continuity_shift": None,
                "center_redetect_replaced_selection": False,
                "propagation_gap": propagation_gap,
            }
            if selected_box is not None and selected_source.startswith("propagated"):
                redetect_result, local_redetect_meta = self._attempt_local_redetect(
                    frame,
                    selected_box,
                    propagation_gap=propagation_gap or 1,
                )
                if redetect_result is not None:
                    selected_box, selected_conf, _selected_iou, _selected_shift, _selected_center_dist = redetect_result
                    selected_source = (
                        "local_redetect_prev"
                        if selected_source.endswith("prev")
                        else "local_redetect_next"
                    )
                else:
                    center_result, center_meta = self._attempt_center_redetect(
                        frame,
                        continuity_box=selected_box,
                    )
                    local_redetect_meta.update(center_meta)
                    if center_result is not None:
                        selected_box, selected_conf, _selected_center_dist, _selected_shift = center_result
                        selected_source = (
                            "center_redetect_prev"
                            if selected_source.endswith("prev")
                            else "center_redetect_next"
                        )

            if selected_box is None:
                center_result, center_meta = self._attempt_center_redetect(frame)
                local_redetect_meta.update(center_meta)
                if center_result is not None:
                    selected_box, selected_conf, _selected_center_dist, _selected_shift = center_result
                    selected_source = "center_redetect_none"

            if selected_box is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                center_dist = None
                clip_padding = None
                dilation_applied = False
                reprompt_applied = False
                reprompt_gain_pixels = 0
                completeness_applied = False
            else:
                center_dist = _box_center_distance(selected_box, frame.shape)
                clip_padding = self._padding_for_selection(
                    selected_source, selected_conf, center_dist,
                )
                mask = self._backend.segment_boxes(
                    frame,
                    [selected_box],
                    constrain_to_primary_box=self._constrain_to_primary_box,
                    primary_box_padding=clip_padding,
                )
                dilation_applied = self._should_apply_recovery_dilation(
                    selected_source,
                    selected_conf,
                    center_dist,
                )
                if dilation_applied:
                    mask = self._apply_recovery_dilation(
                        mask,
                        selected_box,
                        frame.shape,
                        clip_padding,
                    )
                reprompt_applied = False
                reprompt_gain_pixels = 0
                if self._should_apply_reprompt_recovery(
                    selected_source,
                    selected_conf,
                    center_dist,
                ):
                    mask, reprompt_gain_pixels = self._apply_reprompt_recovery(
                        frame,
                        mask,
                        selected_box,
                    )
                    reprompt_applied = reprompt_gain_pixels > 0
                completeness_applied = self._should_apply_completeness_recovery(
                    selected_source,
                    selected_conf,
                    center_dist,
                )
                if completeness_applied:
                    mask = self._apply_completeness_recovery(
                        mask,
                        selected_box,
                        frame.shape,
                        clip_padding,
                    )
            results.append(mask)
            self._last_track_meta.append({
                "selected_confidence": selected_conf,
                "selected_center_dist": center_dist,
                "selection_source": selected_source,
                "clip_padding": clip_padding if selected_box is not None else None,
                "dilation_applied": bool(selected_box is not None and dilation_applied),
                "dilation_kernel": (
                    self._recovery_dilate_kernel
                    if selected_box is not None and dilation_applied
                    else None
                ),
                "reprompt_applied": bool(selected_box is not None and reprompt_applied),
                "reprompt_gain_pixels": (
                    int(reprompt_gain_pixels)
                    if selected_box is not None and reprompt_applied
                    else None
                ),
                "completeness_applied": bool(selected_box is not None and completeness_applied),
                "completeness_kernel": (
                    self._completeness_kernel
                    if selected_box is not None and completeness_applied
                    else None
                ),
                "detections": len(detections),
                **local_redetect_meta,
            })
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
        initial_box: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Track a person across synthetic fisheye frames.

        Args:
            frames: List of BGR images (synthetic fisheye views).
            initial_mask: Unused legacy parameter.
            initial_frame_idx: Frame to prompt on (caller-selected).
            initial_box: Optional xyxy box on the prompt frame. When present,
                SAM v2 is prompted with this box instead of a blind center click.

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

                # Step 3: Prompt on the selected frame. Prefer an explicit
                # operator box when available; otherwise fall back to the
                # historical center-point click.
                if initial_box is not None:
                    box = np.asarray(initial_box, dtype=np.float32).copy()
                    scale_x = resized_w / max(float(orig_w), 1.0)
                    scale_y = resized_h / max(float(orig_h), 1.0)
                    box[[0, 2]] *= scale_x
                    box[[1, 3]] *= scale_y
                    self._predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=initial_frame_idx,
                        obj_id=0,
                        box=box,
                    )
                else:
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
