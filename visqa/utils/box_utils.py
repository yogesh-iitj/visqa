"""Bounding box manipulation utilities."""

import numpy as np


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Convert binary masks to tight bounding boxes.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        boxes: (N, 4) xyxy float bounding boxes
    """
    N = masks.shape[0]
    boxes = np.zeros((N, 4), dtype=np.float32)

    for i, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

    return boxes


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    result = boxes.copy()
    result[..., 2] = boxes[..., 0] + boxes[..., 2]
    result[..., 3] = boxes[..., 1] + boxes[..., 3]
    return result


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    result = boxes.copy()
    result[..., 2] = boxes[..., 2] - boxes[..., 0]
    result[..., 3] = boxes[..., 3] - boxes[..., 1]
    return result


def scale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale boxes by (sx, sy) factors."""
    result = boxes.copy()
    result[..., [0, 2]] *= sx
    result[..., [1, 3]] *= sy
    return result


def expand_boxes(boxes: np.ndarray, expand_ratio: float = 0.1) -> np.ndarray:
    """Expand boxes by a ratio of their size."""
    result = boxes.copy()
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    result[..., 0] -= w * expand_ratio / 2
    result[..., 1] -= h * expand_ratio / 2
    result[..., 2] += w * expand_ratio / 2
    result[..., 3] += h * expand_ratio / 2
    return result


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip boxes to image dimensions."""
    result = boxes.copy()
    result[..., [0, 2]] = result[..., [0, 2]].clip(0, width)
    result[..., [1, 3]] = result[..., [1, 3]].clip(0, height)
    return result


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + 1e-6)
