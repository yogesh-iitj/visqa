"""
Loss functions for training video segmentation + grounding models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation masks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, H, W) or (B, H, W) sigmoid predictions
            target: same shape, binary ground truth
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class GIoULoss(nn.Module):
    """Generalized IoU loss for bounding box regression."""

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes:   (N, 4) xyxy predicted boxes
            target_boxes: (N, 4) xyxy target boxes
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area_pred = ((pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) *
                     (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0))
        area_target = ((target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) *
                       (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0))

        union = area_pred + area_target - inter_area
        iou = inter_area / (union + 1e-6)

        # Enclosing box
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enc_area = ((enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0))

        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        return (1 - giou).mean()


class SegmentationLoss(nn.Module):
    """
    Combined segmentation loss: Dice + Focal.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> dict:
        pred_probs = torch.sigmoid(pred_logits)
        dice_loss = self.dice(pred_probs, target_masks)
        focal_loss = self.focal(pred_logits, target_masks)
        total = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return {
            "total": total,
            "dice": dice_loss.detach(),
            "focal": focal_loss.detach(),
        }


class VisqaLoss(nn.Module):
    """
    Full ViSQA training loss combining segmentation + grounding.
    """

    def __init__(
        self,
        seg_dice_weight: float = 1.0,
        seg_focal_weight: float = 1.0,
        box_giou_weight: float = 2.0,
        box_l1_weight: float = 5.0,
    ):
        super().__init__()
        self.seg_loss = SegmentationLoss(seg_dice_weight, seg_focal_weight)
        self.giou_loss = GIoULoss()
        self.box_giou_weight = box_giou_weight
        self.box_l1_weight = box_l1_weight

    def forward(
        self,
        pred_mask_logits: torch.Tensor,    # (B, T, H, W)
        target_masks: torch.Tensor,        # (B, T, H, W)
        pred_boxes: torch.Tensor = None,   # (B*T, 4) optional
        target_boxes: torch.Tensor = None, # (B*T, 4) optional
    ) -> dict:
        losses = self.seg_loss(pred_mask_logits, target_masks)

        if pred_boxes is not None and target_boxes is not None:
            # Only supervise frames with valid GT boxes
            valid = (target_boxes.sum(dim=-1) > 0)
            if valid.sum() > 0:
                giou = self.giou_loss(pred_boxes[valid], target_boxes[valid])
                l1 = F.l1_loss(pred_boxes[valid], target_boxes[valid])
                losses["giou"] = giou.detach()
                losses["box_l1"] = l1.detach()
                losses["total"] = losses["total"] + self.box_giou_weight * giou + self.box_l1_weight * l1

        return losses
