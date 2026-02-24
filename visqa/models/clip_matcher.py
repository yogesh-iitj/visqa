"""
CLIP-based query-frame matching.
Used to score frame relevance to a text query (frame selection, re-ranking).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


class CLIPMatcher:
    """
    Uses OpenCLIP to compute similarity between text queries and video frames.
    Useful for:
    - Selecting the best key frames for a query
    - Re-ranking bounding box proposals
    - Filtering out irrelevant frames before segmentation
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.device = device
        self.model, self.preprocess, self.tokenizer = self._load_model(model_name, pretrained)

    def _load_model(self, model_name: str, pretrained: str):
        try:
            import open_clip
        except ImportError:
            raise ImportError("Run: pip install open_clip_torch")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess, tokenizer

    @torch.no_grad()
    def encode_text(self, queries: List[str]) -> np.ndarray:
        """Encode text queries to normalized feature vectors. Returns (N, D)."""
        tokens = self.tokenizer(queries).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode video frames to normalized feature vectors. Returns (N, D)."""
        from PIL import Image as PILImage

        images = [self.preprocess(PILImage.fromarray(f)).unsqueeze(0) for f in frames]
        images = torch.cat(images, dim=0).to(self.device)
        features = self.model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def score_frames(self, frames: List[np.ndarray], query: str) -> np.ndarray:
        """
        Score each frame for relevance to a query.

        Returns:
            scores: (N,) float array of cosine similarities in [0, 1]
        """
        text_feat = self.encode_text([query])       # (1, D)
        frame_feats = self.encode_frames(frames)    # (N, D)
        scores = (frame_feats @ text_feat.T).squeeze()  # (N,)
        # Normalize to [0, 1]
        scores = (scores + 1) / 2
        return scores.astype(np.float32)

    def select_key_frames(
        self, frames: List[np.ndarray], query: str, top_k: int = 5
    ) -> List[int]:
        """Return indices of the top-k frames most relevant to query."""
        scores = self.score_frames(frames, query)
        top_k_indices = np.argsort(scores)[::-1][:top_k].tolist()
        return sorted(top_k_indices)  # sorted by time

    def score_crops(
        self, image: np.ndarray, boxes: np.ndarray, query: str
    ) -> np.ndarray:
        """
        Score each bounding box crop for relevance to query.

        Args:
            image: full frame (H, W, 3)
            boxes: (N, 4) xyxy boxes
            query: text query

        Returns:
            scores: (N,) relevance scores
        """
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = image  # fallback
            crops.append(crop)

        return self.score_frames(crops, query)
