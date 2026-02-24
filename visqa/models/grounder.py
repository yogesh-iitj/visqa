"""
Grounding models: text query → bounding box proposals.

Supports:
- Grounding DINO (IDEA-Research)
- OWL-ViT (Google)
- Ensemble of both
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch


class BaseGrounder(ABC):
    """Abstract base class for text-conditioned object grounders."""

    def __init__(self, device: str = "cuda", box_threshold: float = 0.3,
                 text_threshold: float = 0.25):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @abstractmethod
    def predict(
        self, image: np.ndarray, query: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Predict bounding boxes for a text query in an image.

        Args:
            image: RGB image (H, W, 3) uint8
            query: text query string

        Returns:
            boxes:  (N, 4) float32 in xyxy format
            scores: (N,) float32 confidence scores
            labels: list of N string labels
        """
        ...


class GroundingDINOGrounder(BaseGrounder):
    """
    Grounding DINO wrapper.
    Install: pip install groundingdino-py
    Weights: https://github.com/IDEA-Research/GroundingDINO
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_size: str = "swinb",  # "swint" | "swinb"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.model = self._load_model(config_path, checkpoint_path)

    def _load_model(self, config_path, checkpoint_path):
        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            raise ImportError(
                "Grounding DINO not installed. Run: pip install groundingdino-py\n"
                "And download weights: python scripts/download_weights.py --models gdino"
            )

        # Default paths
        if config_path is None:
            _base = "GroundingDINO/groundingdino/config/"
            config_path = (
                _base + "GroundingDINO_SwinB_cfg.py"
                if self.model_size == "swinb"
                else _base + "GroundingDINO_SwinT_OGC.py"
            )
        if checkpoint_path is None:
            checkpoint_path = (
                "weights/groundingdino_swinb_cogcoor.pth"
                if self.model_size == "swinb"
                else "weights/groundingdino_swint_ogc.pth"
            )

        model = load_model(config_path, checkpoint_path, device=self.device)
        model.eval()
        return model

    def predict(
        self, image: np.ndarray, query: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        from groundingdino.util.inference import predict as gdino_predict
        from PIL import Image as PILImage
        import torchvision.transforms as T

        # Ensure query ends with period for GDINO
        caption = query.lower().strip()
        if not caption.endswith("."):
            caption += "."

        pil_image = PILImage.fromarray(image)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(pil_image, None)

        with torch.no_grad():
            boxes, logits, phrases = gdino_predict(
                model=self.model,
                image=img_tensor,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        H, W = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []

        # Convert from cx,cy,w,h normalized to xyxy absolute
        boxes_np = boxes.cpu().numpy()
        boxes_abs = np.zeros_like(boxes_np)
        boxes_abs[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * W
        boxes_abs[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * H
        boxes_abs[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * W
        boxes_abs[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * H

        scores = logits.cpu().numpy()
        return boxes_abs.astype(np.float32), scores.astype(np.float32), phrases


class OWLViTGrounder(BaseGrounder):
    """
    OWL-ViT (Google) zero-shot detection wrapper via HuggingFace Transformers.
    """

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.processor, self.model = self._load_model()

    def _load_model(self):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        processor = OwlViTProcessor.from_pretrained(self.model_name)
        model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        model.to(self.device).eval()
        return processor, model

    def predict(
        self, image: np.ndarray, query: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        texts = [[query]]  # OWL-ViT takes list of lists

        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold,
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = [query] * len(boxes)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []

        return boxes.astype(np.float32), scores.astype(np.float32), labels


class EnsembleGrounder(BaseGrounder):
    """
    Ensemble of Grounding DINO + OWL-ViT with NMS fusion.
    """

    def __init__(self, gdino_kwargs=None, owlvit_kwargs=None, nms_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        gdino_kwargs = gdino_kwargs or {}
        owlvit_kwargs = owlvit_kwargs or {}
        self.gdino = GroundingDINOGrounder(
            device=self.device,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            **gdino_kwargs
        )
        self.owlvit = OWLViTGrounder(
            device=self.device,
            box_threshold=self.box_threshold,
            **owlvit_kwargs
        )
        self.nms_threshold = nms_threshold

    def predict(
        self, image: np.ndarray, query: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        boxes1, scores1, labels1 = self.gdino.predict(image, query)
        boxes2, scores2, labels2 = self.owlvit.predict(image, query)

        if len(boxes1) == 0 and len(boxes2) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []

        all_boxes = np.concatenate([boxes1, boxes2], axis=0) if len(boxes1) > 0 and len(boxes2) > 0 else (boxes1 if len(boxes1) > 0 else boxes2)
        all_scores = np.concatenate([scores1, scores2]) if len(scores1) > 0 and len(scores2) > 0 else (scores1 if len(scores1) > 0 else scores2)
        all_labels = labels1 + labels2

        # NMS
        keep = _nms(all_boxes, all_scores, self.nms_threshold)
        return all_boxes[keep], all_scores[keep], [all_labels[i] for i in keep]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Simple CPU NMS."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)


class GrounderFactory:
    """Factory for building grounding models by name."""

    @staticmethod
    def build(name: str, **kwargs) -> BaseGrounder:
        builders = {
            "gdino": GroundingDINOGrounder,
            "grounding_dino": GroundingDINOGrounder,
            "owlvit": OWLViTGrounder,
            "owl_vit": OWLViTGrounder,
            "ensemble": EnsembleGrounder,
            "gdino_owlvit_ensemble": EnsembleGrounder,
        }
        if name not in builders:
            raise ValueError(f"Unknown grounder: {name}. Choose from: {list(builders.keys())}")
        return builders[name](**kwargs)
