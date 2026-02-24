"""
ViSQA Pipeline — Main entry point for video segmentation and query anchoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from visqa.models.grounder import GrounderFactory
from visqa.models.segmentor import SAM2Segmentor
from visqa.models.tracker import SAM2Tracker
from visqa.utils.video_io import VideoReader, VideoWriter
from visqa.utils.visualization import VisualizationRenderer
from visqa.utils.box_utils import masks_to_boxes


@dataclass
class QueryResult:
    """Result for a single query across all frames."""
    query: str
    masks: np.ndarray           # (T, H, W) binary masks
    boxes: np.ndarray           # (T, 4) bounding boxes in xyxy format
    scores: np.ndarray          # (T,) confidence scores
    frame_indices: List[int]    # which frames were processed

    def to_dict(self):
        return {
            "query": self.query,
            "masks": self.masks.tolist(),
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "frame_indices": self.frame_indices,
        }


@dataclass
class PipelineResult:
    """Full result for a video + queries run."""
    video_path: str
    queries: List[str]
    results: List[QueryResult]
    output_video_path: Optional[str] = None
    fps: float = 30.0
    width: int = 0
    height: int = 0

    def get_query(self, query: str) -> Optional[QueryResult]:
        for r in self.results:
            if r.query == query:
                return r
        return None


class ViSQAPipeline:
    """
    Main ViSQA pipeline: grounding + segmentation + tracking.

    Example:
        pipeline = ViSQAPipeline()
        result = pipeline.run("video.mp4", ["the person in red", "the dog"])
    """

    def __init__(
        self,
        grounder_type: str = "gdino",
        sam2_model_cfg: str = "sam2_hiera_large.yaml",
        sam2_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        frame_stride: int = 1,
        propagation_mode: str = "full",   # "full" | "keyframe_only"
        max_objects_per_query: int = 3,
        render_output: bool = True,
        output_dir: str = "outputs/",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_stride = frame_stride
        self.propagation_mode = propagation_mode
        self.max_objects_per_query = max_objects_per_query
        self.render_output = render_output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ViSQA] Initializing pipeline on device: {self.device}")

        # Load grounding model
        self.grounder = GrounderFactory.build(
            grounder_type,
            device=self.device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Load SAM2
        self.segmentor = SAM2Segmentor(
            model_cfg=sam2_model_cfg,
            checkpoint=sam2_checkpoint,
            device=self.device,
        )

        # SAM2 video tracker
        self.tracker = SAM2Tracker(
            model_cfg=sam2_model_cfg,
            checkpoint=sam2_checkpoint,
            device=self.device,
        )

        self.renderer = VisualizationRenderer()

    @classmethod
    def from_pretrained(cls, model_name: str = "visqa-base", **kwargs) -> "ViSQAPipeline":
        """Load a pretrained ViSQA pipeline configuration."""
        presets = {
            "visqa-base": {
                "grounder_type": "gdino",
                "sam2_model_cfg": "sam2_hiera_base_plus.yaml",
            },
            "visqa-large": {
                "grounder_type": "gdino",
                "sam2_model_cfg": "sam2_hiera_large.yaml",
            },
            "visqa-fast": {
                "grounder_type": "owlvit",
                "sam2_model_cfg": "sam2_hiera_tiny.yaml",
                "frame_stride": 2,
            },
        }
        config = presets.get(model_name, presets["visqa-base"])
        config.update(kwargs)
        return cls(**config)

    def run(
        self,
        video_path: Union[str, Path],
        queries: Union[str, List[str]],
        output_dir: Optional[str] = None,
        save_masks: bool = True,
        save_video: bool = True,
    ) -> PipelineResult:
        """
        Run the full ViSQA pipeline on a video.

        Args:
            video_path: Path to the input video file.
            queries: One or more text queries describing objects to segment.
            output_dir: Where to save results (defaults to self.output_dir).
            save_masks: Whether to save mask arrays as .npy files.
            save_video: Whether to render and save the output video.

        Returns:
            PipelineResult with masks, boxes, scores for each query.
        """
        if isinstance(queries, str):
            queries = [queries]

        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else self.output_dir / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ViSQA] Processing: {video_path.name}")
        print(f"[ViSQA] Queries: {queries}")

        # Read video
        reader = VideoReader(str(video_path), stride=self.frame_stride)
        frames = reader.read_all()
        fps, W, H = reader.fps, reader.width, reader.height

        print(f"[ViSQA] Video: {W}x{H} @ {fps:.1f}fps, {len(frames)} frames to process")

        # Step 1: Ground objects on representative frames (key frames)
        key_frame_indices = self._select_key_frames(len(frames))
        key_frames = [frames[i] for i in key_frame_indices]

        print(f"[ViSQA] Grounding on {len(key_frames)} key frames...")
        grounding_results = {}
        for q in queries:
            boxes_per_keyframe = []
            for kf in key_frames:
                boxes, scores, labels = self.grounder.predict(kf, q)
                # Keep top-k boxes
                if len(boxes) > self.max_objects_per_query:
                    top_k = np.argsort(scores)[::-1][: self.max_objects_per_query]
                    boxes, scores = boxes[top_k], scores[top_k]
                boxes_per_keyframe.append((boxes, scores))
            grounding_results[q] = boxes_per_keyframe

        # Step 2: SAM2 segmentation on key frames
        print("[ViSQA] Segmenting key frames with SAM2...")
        all_query_results = []

        for q in queries:
            query_masks = np.zeros((len(frames), H, W), dtype=np.uint8)
            query_boxes = np.zeros((len(frames), 4), dtype=np.float32)
            query_scores = np.zeros(len(frames), dtype=np.float32)

            for kf_idx, (kf_frame_idx, (boxes, scores)) in enumerate(
                zip(key_frame_indices, grounding_results[q])
            ):
                if len(boxes) == 0:
                    continue

                # SAM2 segment using best box
                best_box = boxes[np.argmax(scores)]
                mask, seg_score = self.segmentor.predict_from_box(
                    key_frames[kf_idx], best_box
                )
                query_masks[kf_frame_idx] = mask
                query_boxes[kf_frame_idx] = best_box
                query_scores[kf_frame_idx] = seg_score

            # Step 3: Propagate across full video if enabled
            if self.propagation_mode == "full":
                print(f"[ViSQA] Propagating masks for query: '{q}'")
                query_masks, query_scores = self.tracker.propagate(
                    frames=frames,
                    seed_masks=query_masks,
                    seed_frame_indices=key_frame_indices,
                )
                # Recompute boxes from propagated masks
                for i in range(len(frames)):
                    if query_masks[i].sum() > 0:
                        query_boxes[i] = masks_to_boxes(query_masks[i][None])[0]

            result = QueryResult(
                query=q,
                masks=query_masks,
                boxes=query_boxes,
                scores=query_scores,
                frame_indices=list(range(len(frames))),
            )
            all_query_results.append(result)

            if save_masks:
                np.save(out_dir / f"masks_{q[:30].replace(' ', '_')}.npy", query_masks)
                np.save(out_dir / f"boxes_{q[:30].replace(' ', '_')}.npy", query_boxes)

        # Step 4: Render output video
        output_video_path = None
        if save_video:
            output_video_path = str(out_dir / "output.mp4")
            print(f"[ViSQA] Rendering output video to: {output_video_path}")
            self._render_video(
                frames, all_query_results, output_video_path, fps=fps
            )

        pipeline_result = PipelineResult(
            video_path=str(video_path),
            queries=queries,
            results=all_query_results,
            output_video_path=output_video_path,
            fps=fps,
            width=W,
            height=H,
        )

        print("[ViSQA] Done!")
        return pipeline_result

    def _select_key_frames(self, total_frames: int, max_keys: int = 8) -> List[int]:
        """Select evenly-spaced key frames for initial grounding."""
        if total_frames <= max_keys:
            return list(range(total_frames))
        step = total_frames // max_keys
        return list(range(0, total_frames, step))[:max_keys]

    def _render_video(
        self,
        frames: List[np.ndarray],
        results: List[QueryResult],
        output_path: str,
        fps: float = 30.0,
    ):
        H, W = frames[0].shape[:2]
        writer = VideoWriter(output_path, fps=fps, width=W, height=H)
        for i, frame in enumerate(tqdm(frames, desc="Rendering")):
            vis_frame = frame.copy()
            for result in results:
                if result.masks[i].sum() > 0:
                    vis_frame = self.renderer.draw_mask(
                        vis_frame, result.masks[i], label=result.query
                    )
                    vis_frame = self.renderer.draw_box(
                        vis_frame, result.boxes[i], label=result.query,
                        score=result.scores[i]
                    )
            writer.write(vis_frame)
        writer.release()
