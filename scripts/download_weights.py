"""
Download model weights for ViSQA components.

Usage:
    python scripts/download_weights.py --models all
    python scripts/download_weights.py --models sam2 gdino
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

WEIGHTS_DIR = Path("weights")

MODELS = {
    "sam2": {
        "description": "SAM2 (Segment Anything Model 2) by Meta AI",
        "files": [
            {
                "name": "sam2_hiera_tiny.pt",
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            },
            {
                "name": "sam2_hiera_small.pt",
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
            },
            {
                "name": "sam2_hiera_base_plus.pt",
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
            },
            {
                "name": "sam2_hiera_large.pt",
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            },
        ],
    },
    "gdino": {
        "description": "Grounding DINO by IDEA Research",
        "files": [
            {
                "name": "groundingdino_swint_ogc.pth",
                "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            },
            {
                "name": "groundingdino_swinb_cogcoor.pth",
                "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            },
        ],
    },
}

DEPENDENCIES = {
    "sam2": "pip install git+https://github.com/facebookresearch/sam2.git",
    "gdino": "pip install groundingdino-py",
    "clip": "pip install open_clip_torch",
    "owlvit": "pip install transformers>=4.35.0",
}


def download_file(url: str, dest: Path):
    """Download a file with progress bar using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return

    print(f"  ↓ Downloading: {dest.name}")
    try:
        import urllib.request
        import tqdm

        class DownloadProgressBar(tqdm.tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)

        print(f"  ✓ Saved to: {dest}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"    Manual download: wget {url} -O {dest}")


def install_package(model_key: str):
    cmd = DEPENDENCIES.get(model_key)
    if cmd:
        print(f"  Installing: {cmd}")
        subprocess.run(cmd, shell=True, check=False)


def download_model(model_key: str):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return

    model_info = MODELS[model_key]
    print(f"\n[{model_key.upper()}] {model_info['description']}")

    for file_info in model_info["files"]:
        dest = WEIGHTS_DIR / file_info["name"]
        download_file(file_info["url"], dest)


def main():
    parser = argparse.ArgumentParser(description="Download ViSQA model weights")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        choices=list(MODELS.keys()) + ["all"],
        help="Models to download"
    )
    parser.add_argument(
        "--install_deps", action="store_true",
        help="Also install Python dependencies"
    )
    parser.add_argument(
        "--sam2_size", default="large",
        choices=["tiny", "small", "base", "large", "all"],
        help="Which SAM2 size(s) to download"
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    models_to_download = list(MODELS.keys()) if "all" in args.models else args.models

    print("=" * 50)
    print("ViSQA Weight Downloader")
    print("=" * 50)

    for model_key in models_to_download:
        download_model(model_key)
        if args.install_deps:
            install_package(model_key)

    print("\n✅ All done! Weights saved to ./weights/")
    print("\nNext steps:")
    print("  python scripts/infer.py --video my_video.mp4 --query 'the person walking'")


if __name__ == "__main__":
    main()
