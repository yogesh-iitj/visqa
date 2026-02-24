"""Tests for ViSQA core utilities."""

import numpy as np
import pytest
import torch

from visqa.utils.box_utils import masks_to_boxes, xywh_to_xyxy, xyxy_to_xywh, box_iou
from visqa.utils.metrics import compute_iou, compute_jf_scores, evaluate_predictions


class TestBoxUtils:
    def test_masks_to_boxes(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:70] = 1
        masks = mask[None]  # (1, H, W)
        boxes = masks_to_boxes(masks)
        assert boxes.shape == (1, 4)
        assert boxes[0, 0] == 30  # x1
        assert boxes[0, 1] == 20  # y1
        assert boxes[0, 2] == 69  # x2
        assert boxes[0, 3] == 49  # y2

    def test_masks_to_boxes_empty(self):
        masks = np.zeros((3, 100, 100), dtype=np.uint8)
        boxes = masks_to_boxes(masks)
        assert (boxes == 0).all()

    def test_xywh_xyxy_roundtrip(self):
        boxes = np.array([[10, 20, 50, 80]], dtype=np.float32)
        xyxy = xywh_to_xyxy(boxes)
        back = xyxy_to_xywh(xyxy)
        np.testing.assert_array_almost_equal(boxes, back)

    def test_box_iou_perfect(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou = box_iou(boxes, boxes)
        assert abs(iou[0, 0] - 1.0) < 1e-5

    def test_box_iou_no_overlap(self):
        a = np.array([[0, 0, 5, 5]], dtype=np.float32)
        b = np.array([[10, 10, 20, 20]], dtype=np.float32)
        iou = box_iou(a, b)
        assert iou[0, 0] == 0.0


class TestMetrics:
    def test_iou_perfect(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert compute_iou(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_iou_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_iou(mask, mask) == pytest.approx(0.0, abs=1e-4)

    def test_iou_partial(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        gt = np.zeros((100, 100), dtype=np.uint8)
        pred[0:50, 0:100] = 1
        gt[0:100, 0:50] = 1
        iou = compute_iou(pred, gt)
        # Intersection = 50x50 = 2500, Union = 50*100 + 100*50 - 2500 = 7500
        assert iou == pytest.approx(2500 / 7500, abs=1e-4)

    def test_jf_scores(self):
        T, H, W = 5, 64, 64
        pred_masks = np.zeros((T, H, W), dtype=np.uint8)
        gt_masks = np.zeros((T, H, W), dtype=np.uint8)
        pred_masks[0:3, 10:30, 10:30] = 1
        gt_masks[0:3, 10:30, 10:30] = 1  # perfect match

        metrics = compute_jf_scores(pred_masks, gt_masks)
        assert metrics["j_mean"] == pytest.approx(1.0, abs=0.01)
        assert metrics["f_mean"] == pytest.approx(1.0, abs=0.05)
        assert metrics["jf_mean"] == pytest.approx(1.0, abs=0.03)

    def test_evaluate_predictions(self):
        T, H, W = 4, 32, 32
        pred = np.zeros((T, H, W), dtype=np.uint8)
        gt = np.zeros((T, H, W), dtype=np.uint8)
        pred[0, 5:15, 5:15] = 1
        gt[0, 5:15, 5:15] = 1

        pred_boxes = np.array([[5, 5, 15, 15]] + [[0, 0, 0, 0]] * 3, dtype=np.float32)
        gt_boxes = np.array([[5, 5, 15, 15]] + [[0, 0, 0, 0]] * 3, dtype=np.float32)

        metrics = evaluate_predictions(pred, gt, pred_boxes, gt_boxes)
        assert "j_mean" in metrics
        assert "f_mean" in metrics
        assert "jf_mean" in metrics
        assert "box_iou" in metrics
        assert metrics["box_iou"] == pytest.approx(1.0, abs=1e-4)
