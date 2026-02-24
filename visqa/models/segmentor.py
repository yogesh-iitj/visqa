"""
SAM2 segmentor wrapper for single-frame mask prediction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class SAM2Segmentor:
    """
    Wraps SAM2 image predictor for single-frame segmentation.

    Supports prompting with:
    - Bounding boxes
    - Point prompts
    - Combined box + point prompts
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_large.yaml",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        multimask_output: bool = True,
        stability_score_thresh: float = 0.95,
    ):
        self.device = device
        self.multimask_output = multimask_output
        self.stability_score_thresh = stability_score_thresh
        self.predictor = self._load_predictor(model_cfg, checkpoint)

    def _load_predictor(self, model_cfg: str, checkpoint: Optional[str]):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "And download weights: python scripts/download_weights.py --models sam2"
            )

        # Auto-detect checkpoint from model_cfg name
        if checkpoint is None:
            name_map = {
                "sam2_hiera_tiny": "weights/sam2_hiera_tiny.pt",
                "sam2_hiera_small": "weights/sam2_hiera_small.pt",
                "sam2_hiera_base_plus": "weights/sam2_hiera_base_plus.pt",
                "sam2_hiera_large": "weights/sam2_hiera_large.pt",
            }
            key = model_cfg.replace(".yaml", "")
            checkpoint = name_map.get(key, "weights/sam2_hiera_large.pt")

        sam2 = build_sam2(model_cfg, checkpoint, device=self.device)
        return SAM2ImagePredictor(sam2)

    def predict_from_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict a segmentation mask given a bounding box prompt.

        Args:
            image: RGB image (H, W, 3) uint8
            box:   (4,) xyxy bounding box

        Returns:
            mask:  (H, W) binary uint8 mask
            score: confidence score
        """
        self.predictor.set_image(image)
        box_tensor = box[None]  # (1, 4)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                box=box_tensor,
                multimask_output=self.multimask_output,
            )

        if self.multimask_output:
            best_idx = np.argmax(scores)
        else:
            best_idx = 0

        mask = masks[best_idx].astype(np.uint8)
        score = float(scores[best_idx])
        return mask, score

    def predict_from_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict mask from point prompts.

        Args:
            image:  RGB (H, W, 3)
            points: (N, 2) xy coordinates
            labels: (N,) 1=foreground, 0=background

        Returns:
            mask, score
        """
        self.predictor.set_image(image)

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=self.multimask_output,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8), float(scores[best_idx])

    def predict_from_box_and_points(
        self,
        image: np.ndarray,
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Predict mask using both box and point prompts."""
        self.predictor.set_image(image)

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box[None],
                multimask_output=self.multimask_output,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8), float(scores[best_idx])
