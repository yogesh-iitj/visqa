"""
Visualization utilities for overlaying masks, boxes, and labels on frames.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Distinct colors for up to 10 objects
PALETTE = [
    (255, 56, 56),   # Red
    (56, 56, 255),   # Blue
    (56, 255, 56),   # Green
    (255, 168, 56),  # Orange
    (168, 56, 255),  # Purple
    (56, 255, 255),  # Cyan
    (255, 56, 255),  # Magenta
    (255, 255, 56),  # Yellow
    (56, 168, 255),  # Sky blue
    (255, 128, 0),   # Deep orange
]


class VisualizationRenderer:
    """Renders masks, bounding boxes, and labels onto image frames."""

    def __init__(self, mask_alpha: float = 0.4, font_scale: float = 0.6,
                 font_thickness: int = 2):
        self.mask_alpha = mask_alpha
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self._query_color_map = {}

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        if label not in self._query_color_map:
            idx = len(self._query_color_map) % len(PALETTE)
            self._query_color_map[label] = PALETTE[idx]
        return self._query_color_map[label]

    def draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        label: str = "",
        color: Optional[Tuple[int, int, int]] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Overlay a binary mask on an image with transparency.

        Args:
            image: RGB (H, W, 3) uint8
            mask:  (H, W) binary mask
            label: object label for color lookup
            color: explicit RGB color (overrides label-based color)
            alpha: opacity of mask overlay

        Returns:
            image with mask overlay
        """
        if mask.sum() == 0:
            return image

        alpha = alpha or self.mask_alpha
        color = color or self._get_color(label)

        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        result = image.copy()
        result = cv2.addWeighted(result, 1.0, colored_mask, alpha, 0)

        # Draw mask contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)

        return result

    def draw_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
        label: str = "",
        score: Optional[float] = None,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw a bounding box with optional label and score.

        Args:
            image: RGB (H, W, 3)
            box:   (4,) xyxy format
            label: text label
            score: optional confidence score to display
            color: explicit RGB color

        Returns:
            image with box drawn
        """
        if box.sum() == 0:
            return image

        color = color or self._get_color(label)
        x1, y1, x2, y2 = map(int, box)

        result = image.copy()

        # Draw thick bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        if label or score is not None:
            display_text = label
            if score is not None:
                display_text += f" {score:.2f}"

            # Truncate long labels
            if len(display_text) > 40:
                display_text = display_text[:37] + "..."

            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale, self.font_thickness
            )

            label_y = max(y1, text_h + 6)
            cv2.rectangle(
                result,
                (x1, label_y - text_h - 6),
                (x1 + text_w + 4, label_y + baseline - 2),
                color, -1
            )
            cv2.putText(
                result, display_text,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale, (255, 255, 255),
                self.font_thickness, cv2.LINE_AA,
            )

        return result

    def draw_frame_info(
        self,
        image: np.ndarray,
        frame_idx: int,
        total_frames: int,
        queries: List[str],
    ) -> np.ndarray:
        """Draw frame counter and active query list."""
        result = image.copy()
        H, W = result.shape[:2]

        # Frame counter
        cv2.putText(
            result, f"Frame {frame_idx + 1}/{total_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Query legend
        for i, q in enumerate(queries):
            color = self._get_color(q)
            y = H - 20 - i * 25
            cv2.circle(result, (15, y), 8, color, -1)
            cv2.putText(
                result, q[:50], (28, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        return result

    def render_side_by_side(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
    ) -> np.ndarray:
        """Create a side-by-side comparison of original vs annotated frame."""
        H, W = original.shape[:2]
        divider = np.full((H, 4, 3), 128, dtype=np.uint8)
        return np.concatenate([original, divider, annotated], axis=1)
