from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert [N, 4] boxes from XYXY -> XYWH.
    """
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes_xyxy.astype(np.float32, copy=True)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
    return boxes


def _coerce_to_pil_rgb(item: Any) -> Image.Image:
    """
    Accept common DLC/external-detector inputs:
      - str / Path
      - PIL.Image
      - np.ndarray
      - (image, context) tuples
    and return a PIL RGB image.
    """
    # Handle DLC-style tuples such as (image, context)
    if isinstance(item, tuple) and len(item) > 0:
        item = item[0]

    if isinstance(item, Image.Image):
        return item if item.mode == "RGB" else item.convert("RGB")

    if isinstance(item, (str, Path)):
        return Image.open(item).convert("RGB")

    if isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 2:
            # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 ndarray for image input, got shape={arr.shape}")

        # If coming from OpenCV, the caller should convert BGR->RGB before passing.
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(
        f"Unsupported image input type: {type(item)!r}. Supported: Path/str, PIL.Image, np.ndarray, (image, context)."
    )


class DARTDetectorModel:
    """
    PyTorch-only SAM3/DART detector adapter for DeepLabCut external-detector workflows.

    Contract:
        inference(images, shelf_writer=None) -> list[dict]
    Each dict contains:
        {
            "bboxes": np.ndarray[N, 4],      # XYWH in pixels
            "bbox_scores": np.ndarray[N],    # detection confidences
        }
    """

    def __init__(
        self,
        classes: list[str],
        checkpoint: str | None = r"C:\Users\Cyril A\Desktop\Code\DeepLabCut-CAch\scripts\detectors\sam3.pt",
        device: str = "cuda",
        imgsz: int = 1008,
        confidence: float = 0.30,
        nms: float = 0.70,
        compile_mode: str | None = None,
        max_detections: int | None = None,
        largest_only: bool = False,
        skip_blocks: set[int] | None = None,
        mask_blocks: list[str] | None = None,
    ) -> None:
        if imgsz % 14 != 0:
            raise ValueError(f"imgsz must be divisible by 14, got {imgsz}")
        if not Path(checkpoint).is_file():
            logger.warning(f"Checkpoint file not found: {checkpoint}. Attempting to download.")
            from sam3.model_builder import download_ckpt_from_hf
            checkpoint = download_ckpt_from_hf()

        self.classes = list(classes)
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.nms = nms
        self.compile_mode = compile_mode
        self.max_detections = max_detections
        self.largest_only = largest_only
        self.skip_blocks = skip_blocks
        self.mask_blocks = mask_blocks

        # Build SAM3 model (PyTorch only)
        pruned_config = load_pruned_config(checkpoint) if checkpoint else None
        if pruned_config is not None:
            model = build_pruned_sam3_image_model(
                checkpoint_path=checkpoint,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
                skip_blocks=skip_blocks,
            )
            # Matches DART demo behavior for distilled checkpoints
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint,
                eval_mode=True,
                skip_blocks=skip_blocks,
                mask_blocks=mask_blocks,
            )

        # Precompute position encodings if using a non-default resolution
        if imgsz != 1008:
            pos_enc = model.backbone.vision_backbone.position_encoding
            pos_enc.precompute_for_resolution(imgsz)

        # Fast predictor, PyTorch only: no TRT args passed here
        self.predictor = Sam3MultiClassPredictorFast(
            model,
            device=device,
            resolution=imgsz,
            compile_mode=compile_mode,
            use_fp16=(device.startswith("cuda")),
            presence_threshold=0.05,
            detection_only=True,
            trt_engine_path=None,
            trt_enc_dec_engine_path=None,
        )

        # Precompute text embeddings once
        self.predictor.set_classes(self.classes)

        # Optional warmup
        self._warmed_up = False

    def warmup(self) -> None:
        """
        Optional one-time warmup to pay compile/CUDA startup cost upfront.
        """
        if self._warmed_up:
            return

        dummy = Image.new("RGB", (self.imgsz, self.imgsz))
        with torch.inference_mode():
            state = self.predictor.set_image(dummy)
            _ = self.predictor.predict(
                state,
                confidence_threshold=self.confidence,
                nms_threshold=self.nms,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._warmed_up = True

    def _postprocess(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if boxes_xyxy.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        # Keep only largest detection if requested
        if self.largest_only:
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            keep = np.array([int(np.argmax(areas))], dtype=np.int64)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]

        # Keep top-k by confidence if requested
        if self.max_detections is not None and len(scores) > self.max_detections:
            order = np.argsort(-scores)[: self.max_detections]
            boxes_xyxy = boxes_xyxy[order]
            scores = scores[order]

        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

        # Clamp widths/heights to non-negative
        boxes_xywh[:, 2:] = np.maximum(boxes_xywh[:, 2:], 0.0)

        return boxes_xywh.astype(np.float32), scores.astype(np.float32)

    def inference(self, images, shelf_writer=None, profile: bool = False):
        outputs = []

        if not self._warmed_up:
            self.warmup()

        with torch.inference_mode():
            for item in tqdm(images):
                image = _coerce_to_pil_rgb(item)

                state = self.predictor.set_image(image)
                results = self.predictor.predict(
                    state,
                    confidence_threshold=self.confidence,
                    nms_threshold=self.nms,
                )

                boxes_xyxy = results["boxes"].detach().cpu().numpy()
                scores = results["scores"].detach().cpu().numpy()

                boxes_xywh, scores = self._postprocess(boxes_xyxy, scores)

                outputs.append(
                    {
                        "bboxes": boxes_xywh,
                        "bbox_scores": scores,
                    }
                )

        return outputs
