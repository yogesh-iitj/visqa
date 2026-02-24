"""
ViSQA Inference Script

Usage:
    python scripts/infer.py --video my_video.mp4 --query "the person in red"
    python scripts/infer.py --video my_video.mp4 --queries "person in red" "dog running"
    python scripts/infer.py --video my_video.mp4 --query "car" --config configs/fast.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visqa import ViSQAPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="ViSQA Video Segmentation Inference")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--query", default=None, help="Single text query")
    parser.add_argument("--queries", nargs="+", default=None, help="Multiple text queries")
    parser.add_argument("--queries_file", default=None, help="File with one query per line")
    parser.add_argument("--output_dir", default="outputs/", help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--grounder", default="gdino", choices=["gdino", "owlvit", "ensemble"])
    parser.add_argument("--sam2_size", default="large", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_video", action="store_true", help="Skip output video rendering")
    parser.add_argument("--save_json", action="store_true", help="Save results as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect queries
    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        queries.extend(args.queries)
    if args.queries_file:
        with open(args.queries_file) as f:
            queries.extend(line.strip() for line in f if line.strip())

    if not queries:
        print("ERROR: Provide at least one query via --query, --queries, or --queries_file")
        sys.exit(1)

    print(f"[ViSQA] Video: {args.video}")
    print(f"[ViSQA] Queries: {queries}")

    sam2_cfg_map = {
        "tiny": "sam2_hiera_tiny.yaml",
        "small": "sam2_hiera_small.yaml",
        "base": "sam2_hiera_base_plus.yaml",
        "large": "sam2_hiera_large.yaml",
    }

    pipeline = ViSQAPipeline(
        grounder_type=args.grounder,
        sam2_model_cfg=sam2_cfg_map[args.sam2_size],
        device=args.device,
        box_threshold=args.box_threshold,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
    )

    result = pipeline.run(
        video_path=args.video,
        queries=queries,
        save_video=not args.no_video,
        save_masks=True,
    )

    print("\n=== Results ===")
    for qr in result.results:
        valid_frames = (qr.masks.sum(axis=(1, 2)) > 0).sum()
        print(f"Query: '{qr.query}'")
        print(f"  Detected in {valid_frames}/{len(qr.masks)} frames")
        print(f"  Avg confidence: {qr.scores[qr.scores > 0].mean():.3f}" if (qr.scores > 0).any() else "  No detections")

    if result.output_video_path:
        print(f"\nOutput video: {result.output_video_path}")

    if args.save_json:
        json_path = Path(args.output_dir) / Path(args.video).stem / "results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            summary = {
                "video": args.video,
                "queries": queries,
                "results": [
                    {
                        "query": qr.query,
                        "boxes": qr.boxes.tolist(),
                        "scores": qr.scores.tolist(),
                    }
                    for qr in result.results
                ]
            }
            json.dump(summary, f, indent=2)
        print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()
