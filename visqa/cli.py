"""
ViSQA command-line interface.

Usage:
  visqa infer --video video.mp4 --query "the person walking"
  visqa demo
  visqa download --models all
  visqa eval --checkpoint best.pth --data_root data/
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="visqa",
        description="ViSQA — Video Segmentation & Query Anchoring",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on a video")
    infer_parser.add_argument("--video", required=True)
    infer_parser.add_argument("--query", default=None)
    infer_parser.add_argument("--queries", nargs="+", default=None)
    infer_parser.add_argument("--output", default="outputs/")
    infer_parser.add_argument("--grounder", default="gdino")
    infer_parser.add_argument("--sam2_size", default="large")
    infer_parser.add_argument("--device", default=None)

    # demo
    demo_parser = subparsers.add_parser("demo", help="Launch Gradio web demo")
    demo_parser.add_argument("--port", type=int, default=7860)
    demo_parser.add_argument("--share", action="store_true")

    # download
    dl_parser = subparsers.add_parser("download", help="Download model weights")
    dl_parser.add_argument("--models", nargs="+", default=["all"])

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate on a dataset")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data_root", required=True)
    eval_parser.add_argument("--split", default="val")

    args = parser.parse_args()

    if args.command == "infer":
        from scripts.infer import main as infer_main
        sys.argv = ["infer"] + sys.argv[2:]
        infer_main()

    elif args.command == "demo":
        from scripts.demo import main as demo_main
        demo_main()

    elif args.command == "download":
        from scripts.download_weights import main as dl_main
        dl_main()

    elif args.command == "eval":
        from scripts.evaluate import main as eval_main
        eval_main()


if __name__ == "__main__":
    main()
