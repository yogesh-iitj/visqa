"""
SAM2 video tracker — propagates masks across entire video sequences.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class SAM2Tracker:
    """
    Wraps SAM2 VideoPredictor for temporal mask propagation.

    Given seed masks on key frames, propagates them forward (and backward)
    through all frames using SAM2's memory-based tracking.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_large.yaml",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.predictor = self._load_predictor(model_cfg, checkpoint)

    def _load_predictor(self, model_cfg: str, checkpoint: Optional[str]):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. See: https://github.com/facebookresearch/sam2"
            )

        if checkpoint is None:
            name_map = {
                "sam2_hiera_tiny": "weights/sam2_hiera_tiny.pt",
                "sam2_hiera_small": "weights/sam2_hiera_small.pt",
                "sam2_hiera_base_plus": "weights/sam2_hiera_base_plus.pt",
                "sam2_hiera_large": "weights/sam2_hiera_large.pt",
            }
            key = model_cfg.replace(".yaml", "")
            checkpoint = name_map.get(key, "weights/sam2_hiera_large.pt")

        return build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)

    def propagate(
        self,
        frames: List[np.ndarray],
        seed_masks: np.ndarray,
        seed_frame_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate seed masks across all video frames.

        Args:
            frames:             List of (H, W, 3) RGB frames
            seed_masks:         (T, H, W) array — non-zero frames are seeds
            seed_frame_indices: indices used as seeds (frames with valid masks)

        Returns:
            prop_masks: (T, H, W) uint8 binary masks for all frames
            prop_scores:(T,) float32 confidence scores
        """
        T, H, W = seed_masks.shape
        prop_masks = np.zeros((T, H, W), dtype=np.uint8)
        prop_scores = np.zeros(T, dtype=np.float32)

        # Save frames to temp directory (SAM2 video predictor needs image files)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for i, frame in enumerate(frames):
                frame_path = tmp_dir / f"{i:06d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            inference_state = self.predictor.init_state(video_path=str(tmp_dir))

            # Add seeds as prompts
            for seed_idx in seed_frame_indices:
                if seed_masks[seed_idx].sum() == 0:
                    continue
                mask_tensor = torch.from_numpy(seed_masks[seed_idx]).to(self.device)
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=seed_idx,
                    obj_id=1,
                    mask=mask_tensor,
                )

            # Propagate forward
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state
            ):
                if len(obj_ids) > 0:
                    mask = (mask_logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                    score = torch.sigmoid(mask_logits[0]).max().item()
                    prop_masks[frame_idx] = mask
                    prop_scores[frame_idx] = score

            self.predictor.reset_state(inference_state)

        return prop_masks, prop_scores

    def propagate_multi_object(
        self,
        frames: List[np.ndarray],
        seeds: Dict[int, Dict[int, np.ndarray]],  # {query_id: {frame_idx: mask}}
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Propagate multiple objects simultaneously.

        Args:
            frames: video frames
            seeds:  {obj_id: {frame_idx: mask}}

        Returns:
            {obj_id: (masks_T_H_W, scores_T)}
        """
        T = len(frames)
        H, W = frames[0].shape[:2]
        results = {obj_id: (np.zeros((T, H, W), dtype=np.uint8), np.zeros(T, dtype=np.float32))
                   for obj_id in seeds}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for i, frame in enumerate(frames):
                cv2.imwrite(str(tmp_dir / f"{i:06d}.jpg"),
                           cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            inference_state = self.predictor.init_state(video_path=str(tmp_dir))

            for obj_id, frame_masks in seeds.items():
                for frame_idx, mask in frame_masks.items():
                    if mask.sum() == 0:
                        continue
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=torch.from_numpy(mask).to(self.device),
                    )

            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state
            ):
                for i, obj_id in enumerate(obj_ids):
                    if obj_id in results:
                        mask = (mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                        score = torch.sigmoid(mask_logits[i]).max().item()
                        results[obj_id][0][frame_idx] = mask
                        results[obj_id][1][frame_idx] = score

            self.predictor.reset_state(inference_state)

        return results
