from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from detectors.dart import DARTDetectorModel


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_pil_rgb(item: Any) -> Image.Image:
    """
    Accept common image inputs and return a PIL RGB image.
    Supported:
      - str / Path
      - PIL.Image
      - np.ndarray (HxW, HxWx3, HxWx4)
      - (image, context) tuples
    """
    if isinstance(item, tuple) and len(item) > 0:
        item = item[0]

    if isinstance(item, Image.Image):
        return item if item.mode == "RGB" else item.convert("RGB")

    if isinstance(item, (str, Path)):
        return Image.open(item).convert("RGB")

    if isinstance(item, np.ndarray):
        arr = item

        # grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[2] == 4:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGBA").convert("RGB")

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected image array of shape HxWx3 or HxWx4, got {arr.shape}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # NOTE:
        # If this came from OpenCV, convert BGR -> RGB BEFORE calling this function.
        return Image.fromarray(arr, mode="RGB")

    raise TypeError(
        f"Unsupported image input type: {type(item)!r}. "
        "Supported: Path/str, PIL.Image, np.ndarray, (image, context)."
    )


def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """
    Convert [N, 4] boxes from XYWH -> XYXY.
    """
    if boxes_xywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes_xywh.astype(np.float32, copy=True)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return boxes


def _pack_detection_rows_xyxy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> torch.Tensor:
    """
    Produce a YOLO-like Nx6 tensor:
        [x1, y1, x2, y2, conf, cls]
    """
    n = len(scores)
    if n == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    rows = np.zeros((n, 6), dtype=np.float32)
    rows[:, 0:4] = boxes_xyxy
    rows[:, 4] = scores
    rows[:, 5] = class_ids
    return torch.from_numpy(rows)


def _pack_detection_rows_xywh(
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> torch.Tensor:
    """
    Produce a YOLO-like Nx6 tensor:
        [x, y, w, h, conf, cls]
    """
    n = len(scores)
    if n == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    rows = np.zeros((n, 6), dtype=np.float32)
    rows[:, 0:4] = boxes_xywh
    rows[:, 4] = scores
    rows[:, 5] = class_ids
    return torch.from_numpy(rows)


# ---------------------------------------------------------------------
# YOLO-like result wrapper around DART output
# ---------------------------------------------------------------------

@dataclass
class DARTResults:
    """
    Lightweight YOLO-style result object for compatibility.

    Fields intentionally mirror the subset your old code uses:
      - names
      - ims
      - xyxy
      - xywh

    We also keep the raw arrays for convenience:
      - boxes_xyxy
      - boxes_xywh
      - scores
      - class_ids
    """
    names: dict[int, str]
    ims: list[np.ndarray]
    xyxy: list[torch.Tensor]
    xywh: list[torch.Tensor]
    boxes_xyxy: np.ndarray
    boxes_xywh: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray


# ---------------------------------------------------------------------
# Detector cache
# ---------------------------------------------------------------------

_DART_MODEL_CACHE: dict[tuple, "DARTDetectorModel"] = {}


def get_dart_detector(
    checkpoint: str = "sam3.pt",
    *,
    classes: Sequence[str] = ("animal",),
    device: str | None = None,
    imgsz: int = 1008,
    confidence: float = 0.30,
    nms: float = 0.70,
    compile_mode: str | None = None,
    max_detections: int | None = None,
    largest_only: bool = False,
    skip_blocks: set[int] | None = None,
    mask_blocks: list[str] | None = None,
) -> "DARTDetectorModel":
    """
    Build and cache a DARTDetectorModel so it is not reloaded every call.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (
        str(checkpoint),
        tuple(classes),
        device,
        int(imgsz),
        float(confidence),
        float(nms),
        compile_mode,
        max_detections,
        bool(largest_only),
        tuple(sorted(skip_blocks)) if skip_blocks else None,
        tuple(mask_blocks) if mask_blocks else None,
    )

    if cache_key not in _DART_MODEL_CACHE:
        detector = DARTDetectorModel(
            classes=list(classes),
            checkpoint=checkpoint,
            device=device,
            imgsz=imgsz,
            confidence=confidence,
            nms=nms,
            compile_mode=compile_mode,
            max_detections=max_detections,
            largest_only=largest_only,
            skip_blocks=skip_blocks,
            mask_blocks=mask_blocks,
        )
        detector.warmup()
        _DART_MODEL_CACHE[cache_key] = detector

    return _DART_MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------
# Drop-in replacement for the old MegaDetector/YOLO API
# ---------------------------------------------------------------------

def predict_md(
    im,
    megadetector_model: str | None = None,
    size: int = 1008,
    *,
    classes: Sequence[str] = ("animal",),
    confidence: float = 0.30,
    nms: float = 0.70,
    device: str | None = None,
    compile_mode: str | None = None,
    max_detections: int | None = None,
    largest_only: bool = False,
    skip_blocks: set[int] | None = None,
    mask_blocks: list[str] | None = None,
) -> DARTResults:
    """
    Drop-in replacement for the old YOLO/MegaDetector predict_md().

    Parameters
    ----------
    im
        Input image (PIL, ndarray, path, etc.)
    megadetector_model
        NOW USED AS THE SAM3 CHECKPOINT PATH.
        You can keep the old variable name to minimize code changes.
        Example: "sam3.pt"
    size
        DART/SAM3 resolution (must be divisible by 14)
    classes
        Class names to embed in the detector.
        IMPORTANT:
        This wrapper currently behaves as a single-class detector because
        your DARTDetectorModel.inference() currently returns only boxes/scores,
        not class IDs. For the animal-cropping workflow this is exactly what
        you want, so the default is ("animal",).

    Returns
    -------
    DARTResults
        A YOLO-like compatibility object with:
          - names
          - ims
          - xyxy
          - xywh
    """
    if len(classes) != 1:
        raise ValueError(
            "This drop-in wrapper currently supports a single class only "
            "because DARTDetectorModel.inference() returns boxes/scores but not class IDs. "
            "For animal cropping, use classes=('animal',)."
        )

    checkpoint = megadetector_model or "sam3.pt"
    image = _ensure_pil_rgb(im)

    detector = get_dart_detector(
        checkpoint=checkpoint,
        classes=classes,
        device=device,
        imgsz=size,
        confidence=confidence,
        nms=nms,
        compile_mode=compile_mode,
        max_detections=max_detections,
        largest_only=largest_only,
        skip_blocks=skip_blocks,
        mask_blocks=mask_blocks,
    )

    raw = detector.inference([image])[0]

    boxes_xywh = raw.get("bboxes", np.zeros((0, 4), dtype=np.float32))
    scores = raw.get("bbox_scores", np.zeros((0,), dtype=np.float32))

    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    # Single-class wrapper: class id 0 == "animal"
    class_ids = np.zeros((len(scores),), dtype=np.float32)

    names = {0: classes[0]}
    im_np = np.asarray(image)

    return DARTResults(
        names=names,
        ims=[im_np],
        xyxy=[_pack_detection_rows_xyxy(boxes_xyxy, scores, class_ids)],
        xywh=[_pack_detection_rows_xywh(boxes_xywh, scores, class_ids)],
        boxes_xyxy=boxes_xyxy,
        boxes_xywh=boxes_xywh,
        scores=scores,
        class_ids=class_ids,
    )


def crop_animal_detections(
    img_in,
    yolo_results: DARTResults | dict[str, np.ndarray],
    likelihood_th: float,
):
    """
    Drop-in replacement for the old crop_animal_detections().

    Works with:
      - DARTResults (returned by predict_md above)
      - raw DART dicts: {'bboxes': ..., 'bbox_scores': ...}

    Returns
    -------
    list[np.ndarray]
        Cropped animal images as numpy arrays.
    """
    img = _ensure_pil_rgb(img_in)
    img_w, img_h = img.size

    list_np_animal_crops: list[np.ndarray] = []

    # -----------------------------------------------------------------
    # Case 1: YOLO-like DARTResults wrapper
    # -----------------------------------------------------------------
    if isinstance(yolo_results, DARTResults):
        if len(yolo_results.xyxy) == 0:
            return list_np_animal_crops

        det_array = yolo_results.xyxy[0].detach().cpu().numpy()
        names = yolo_results.names
        animal_class_id = next((k for k, v in names.items() if v == "animal"), 0)

        for j in range(det_array.shape[0]):
            xmin = max(0, int(math.floor(det_array[j, 0])))
            ymin = max(0, int(math.floor(det_array[j, 1])))
            xmax = min(img_w, int(math.ceil(det_array[j, 2])))
            ymax = min(img_h, int(math.ceil(det_array[j, 3])))

            pred_llk = float(det_array[j, 4])
            pred_label = int(det_array[j, 5])

            if pred_label != animal_class_id:
                continue
            if pred_llk < likelihood_th:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))
            list_np_animal_crops.append(np.asarray(crop))

        return list_np_animal_crops

    # -----------------------------------------------------------------
    # Case 2: raw DART output dict {'bboxes': XYWH, 'bbox_scores': ...}
    # -----------------------------------------------------------------
    if isinstance(yolo_results, dict):
        boxes_xywh = np.asarray(
            yolo_results.get("bboxes", np.zeros((0, 4), dtype=np.float32)),
            dtype=np.float32,
        )
        scores = np.asarray(
            yolo_results.get("bbox_scores", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        )

        for (x, y, w, h), score in zip(boxes_xywh, scores):
            if float(score) < likelihood_th:
                continue

            xmin = max(0, int(math.floor(x)))
            ymin = max(0, int(math.floor(y)))
            xmax = min(img_w, int(math.ceil(x + w)))
            ymax = min(img_h, int(math.ceil(y + h)))

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))
            list_np_animal_crops.append(np.asarray(crop))

        return list_np_animal_crops

    raise TypeError(
        f"Unsupported results type: {type(yolo_results)!r}. "
        "Expected DARTResults or raw DART dict."
    )
