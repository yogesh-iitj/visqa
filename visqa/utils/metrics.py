"""
Evaluation metrics for video object segmentation and grounding.

Includes:
- J (Jaccard / IoU) for segmentation quality
- F (boundary F-measure) for contour accuracy
- J&F mean (standard VOS benchmark metric)
- Box IoU for grounding evaluation
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute per-frame Intersection over Union (Jaccard index)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / (union + 1e-6))


def compute_j_score(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Compute mean Jaccard (J) score over all frames with valid GT.
    Skips frames where GT mask is all zeros.
    """
    scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        if gt.sum() == 0:
            continue
        scores.append(compute_iou(pred, gt))
    return float(np.mean(scores)) if scores else 0.0


def compute_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray,
                    bound_th: float = 0.008) -> float:
    """
    Compute F-measure (boundary accuracy) for a single frame.

    Based on the DAVIS benchmark implementation.
    """
    def get_boundary(mask: np.ndarray, th: float) -> np.ndarray:
        dil = binary_dilation(mask, iterations=int(th * max(mask.shape)))
        ero = binary_erosion(mask, iterations=int(th * max(mask.shape)))
        return dil.astype(float) - ero.astype(float)

    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 1.0
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return 0.0

    gt_boundary = get_boundary(gt_mask.astype(bool), bound_th)
    pred_boundary = get_boundary(pred_mask.astype(bool), bound_th)

    precision = (gt_boundary * pred_boundary).sum() / (pred_boundary.sum() + 1e-6)
    recall = (gt_boundary * pred_boundary).sum() / (gt_boundary.sum() + 1e-6)

    if precision + recall < 1e-6:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_jf_scores(
    pred_masks: np.ndarray, gt_masks: np.ndarray
) -> dict:
    """
    Compute J, F, and J&F scores for a sequence.

    Args:
        pred_masks: (T, H, W) predicted binary masks
        gt_masks:   (T, H, W) ground truth binary masks

    Returns:
        dict with keys: j_mean, f_mean, jf_mean
    """
    j_scores, f_scores = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        if gt.sum() == 0:
            continue
        j_scores.append(compute_iou(pred, gt))
        f_scores.append(compute_f_score(pred, gt))

    j_mean = float(np.mean(j_scores)) if j_scores else 0.0
    f_mean = float(np.mean(f_scores)) if f_scores else 0.0
    jf_mean = (j_mean + f_mean) / 2.0

    return {"j_mean": j_mean, "f_mean": f_mean, "jf_mean": jf_mean}


def compute_box_iou_sequence(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> float:
    """
    Compute mean box IoU over a sequence.

    Args:
        pred_boxes: (T, 4) xyxy predicted boxes
        gt_boxes:   (T, 4) xyxy ground truth boxes

    Returns:
        mean IoU over frames with valid GT
    """
    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        if gt.sum() == 0:
            continue
        inter_x1 = max(pred[0], gt[0])
        inter_y1 = max(pred[1], gt[1])
        inter_x2 = min(pred[2], gt[2])
        inter_y2 = min(pred[3], gt[3])
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_pred = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
        area_gt = max(0, gt[2] - gt[0]) * max(0, gt[3] - gt[1])
        union = area_pred + area_gt - inter
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def evaluate_predictions(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    pred_boxes: np.ndarray = None,
    gt_boxes: np.ndarray = None,
) -> dict:
    """
    Full evaluation suite.

    Returns dict with all computed metrics.
    """
    metrics = compute_jf_scores(pred_masks, gt_masks)

    if pred_boxes is not None and gt_boxes is not None:
        metrics["box_iou"] = compute_box_iou_sequence(pred_boxes, gt_boxes)

    return metrics
