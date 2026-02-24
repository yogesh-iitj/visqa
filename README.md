# 🎬 ViSQA — Video Segmentation & Query Anchoring

<p align="center">
  <img src="docs/assets/banner.png" alt="ViSQA Banner" width="800"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue"/></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green"/></a>
  <a href="#"><img src="https://img.shields.io/badge/models-SAM2%20%7C%20DINO%20%7C%20CLIP-purple"/></a>
</p>

> **Query-driven video segmentation and grounding using open-source models.**  
> Describe what you want to track in natural language — ViSQA segments it, grounds it with bounding boxes, and tracks it across your entire video.

---

## ✨ Features

- 🔍 **Text-Queried Segmentation** — Describe any object in natural language (e.g. *"the red car on the left"*) and get pixel-level masks
- 📦 **Bounding Box Grounding** — Tight bounding boxes derived from segmentation masks for every frame
- 🎯 **Multi-Object Tracking** — Track multiple query objects simultaneously across long videos
- 🕰️ **Temporal Consistency** — SAM2-powered propagation for smooth, consistent masks over time
- 🖼️ **Frame-level & Clip-level** — Works on single frames or full video clips
- 🔌 **Modular Architecture** — Swap in different vision-language backbones (CLIP, DINO, OWL-ViT)
- 🏋️ **Custom Training** — Fine-tune on your own video+annotation datasets
- 🌐 **Gradio Demo** — Interactive browser UI out of the box
- 📊 **Evaluation Suite** — J&F, mIoU, tracking metrics built in

---

## 🏗️ Architecture

```
                     ┌─────────────────────────────────────────┐
                     │              ViSQA Pipeline              │
                     └─────────────────────────────────────────┘

  Text Query ──▶ [ CLIP / DINO Text Encoder ] ──▶ Text Embedding
                                                         │
  Video ────▶ [ Frame Sampler ] ──▶ Key Frames           │
                                         │               ▼
                                    [ Grounding Module (OWL-ViT / Grounding DINO) ]
                                         │
                                         ▼ (bounding box proposals)
                                    [ SAM2 Predictor ]
                                         │
                                         ▼ (masks on key frames)
                                    [ SAM2 Propagator ]
                                         │
                                         ▼
                              Full-Video Mask Sequences
                              + Bounding Box Tracks
                              + Confidence Scores
```

**Core Models Used:**
| Component | Model | Purpose |
|-----------|-------|---------|
| Grounding | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Text → BBox proposals |
| Segmentation | [SAM2](https://github.com/facebookresearch/sam2) | BBox → Pixel masks |
| Tracking | SAM2 Video Predictor | Temporal propagation |
| CLIP Matching | [OpenCLIP](https://github.com/mlfoundations/open_clip) | Query-frame similarity |
| Alt Grounder | [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) | Zero-shot detection |

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/yourname/visqa.git
cd visqa
pip install -e ".[all]"
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate visqa
```

### 2. Download Model Weights

```bash
python scripts/download_weights.py --models all
# or selectively:
python scripts/download_weights.py --models sam2 gdino
```

### 3. Run Inference

```python
from visqa import ViSQAPipeline

pipeline = ViSQAPipeline.from_pretrained("visqa-base")

results = pipeline.run(
    video_path="my_video.mp4",
    queries=["the person in the red jacket", "the dog running"],
    output_dir="outputs/"
)

# results contains:
# - results.masks        — per-frame numpy masks, shape (T, H, W)
# - results.boxes        — per-frame bounding boxes (T, 4) in xyxy format
# - results.scores       — confidence per frame
# - results.video_path   — rendered output video
```

### 4. CLI

```bash
# Single query
visqa infer --video my_video.mp4 --query "the person walking" --output outputs/

# Multiple queries
visqa infer --video my_video.mp4 --queries queries.txt --output outputs/

# With custom config
visqa infer --video my_video.mp4 --query "red car" --config configs/high_quality.yaml

# Batch processing
visqa batch --input_dir videos/ --query "person" --output_dir outputs/
```

### 5. Gradio Demo

```bash
visqa demo
# Opens at http://localhost:7860
```

---

## 📁 Project Structure

```
visqa/
├── visqa/                      # Core library
│   ├── __init__.py
│   ├── pipeline.py             # Main ViSQA pipeline
│   ├── models/
│   │   ├── grounder.py         # Grounding DINO + OWL-ViT wrappers
│   │   ├── segmentor.py        # SAM2 wrapper
│   │   ├── tracker.py          # SAM2 video propagation
│   │   ├── clip_matcher.py     # CLIP-based query matching
│   │   └── ensemble.py         # Multi-model fusion
│   ├── utils/
│   │   ├── video_io.py         # Video reading/writing
│   │   ├── visualization.py    # Mask/box rendering
│   │   ├── box_utils.py        # BBox manipulation
│   │   └── metrics.py          # Evaluation metrics
│   ├── data/
│   │   ├── dataset.py          # Base dataset classes
│   │   ├── ytvis.py            # YouTube-VIS loader
│   │   ├── ref_davis.py        # Ref-DAVIS loader
│   │   └── custom.py           # Custom dataset loader
│   └── training/
│       ├── trainer.py          # Training loop
│       ├── losses.py           # Segmentation + grounding losses
│       └── callbacks.py        # Training callbacks
├── scripts/
│   ├── download_weights.py     # Model weight downloader
│   ├── infer.py                # CLI inference script
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation script
│   └── demo.py                 # Gradio demo launcher
├── configs/
│   ├── default.yaml            # Default pipeline config
│   ├── high_quality.yaml       # High quality (slower) preset
│   ├── fast.yaml               # Fast inference preset
│   └── training/
│       ├── finetune.yaml       # Fine-tuning config
│       └── scratch.yaml        # Train from scratch config
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_custom_training.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_data.py
├── docs/
│   ├── TRAINING.md             # Detailed training guide
│   ├── CUSTOM_DATA.md          # Custom dataset format guide
│   └── API.md                  # API reference
├── environment.yml
├── setup.py
├── pyproject.toml
└── README.md
```

---

## 📊 Supported Datasets & Benchmarks

| Dataset | Task | Status |
|---------|------|--------|
| [Ref-DAVIS 2017](https://davischallenge.org/) | Referring Video Segmentation | ✅ |
| [YouTube-VIS](https://youtube-vos.org/dataset/vis/) | Video Instance Segmentation | ✅ |
| [MeViS](https://henghuiding.github.io/MeViS/) | Motion Expression Segmentation | ✅ |
| [HC-STVG](https://github.com/tzhhhh123/HC-STVG) | Spatio-Temporal Video Grounding | ✅ |
| Custom | Any | ✅ |

---

## 🏋️ Training on Custom Data

See **[docs/TRAINING.md](docs/TRAINING.md)** for the full guide. Quick version:

### Step 1: Prepare your data

```
data/custom/
├── videos/
│   ├── video_001.mp4
│   └── video_002.mp4
└── annotations.json        # COCO-like annotation format
```

### Step 2: Create annotation JSON

```json
{
  "videos": [
    {
      "id": 1,
      "file_name": "video_001.mp4",
      "fps": 30,
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "video_id": 1,
      "query": "the person in the red jacket",
      "frames": [0, 5, 10, 15],
      "segmentations": [ ... ],  // RLE masks per frame
      "bboxes": [[100, 200, 150, 300], ...]  // xyxy per frame
    }
  ]
}
```

### Step 3: Fine-tune

```bash
python scripts/train.py \
  --config configs/training/finetune.yaml \
  --data_root data/custom/ \
  --annotations data/custom/annotations.json \
  --output_dir checkpoints/my_model/ \
  --epochs 10 \
  --batch_size 4
```

### Step 4: Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/my_model/best.pth \
  --data_root data/custom/ \
  --split val
```

---

## 🔧 Configuration

Key config options in `configs/default.yaml`:

```yaml
pipeline:
  grounder: "gdino"          # gdino | owlvit | gdino_owlvit_ensemble
  segmentor: "sam2_large"    # sam2_tiny | sam2_small | sam2_base | sam2_large
  frame_stride: 1            # Process every N frames
  prompt_threshold: 0.3      # Grounding confidence threshold
  propagation_mode: "full"   # full | keyframe_only

grounding_dino:
  model_size: "swinb"        # swint | swinb
  box_threshold: 0.3
  text_threshold: 0.25

sam2:
  model_cfg: "sam2_hiera_large.yaml"
  multimask_output: true
  stability_score_thresh: 0.95
```

---

## 📈 Benchmarks

Results on Ref-DAVIS 2017 val set:

| Method | J Mean | F Mean | J&F Mean |
|--------|--------|--------|----------|
| Baseline (GDINO + SAM) | 67.3 | 71.2 | 69.3 |
| ViSQA-Base | 71.8 | 75.4 | 73.6 |
| ViSQA-Large | **74.2** | **78.1** | **76.2** |

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Run tests
pytest tests/

# Code formatting
black visqa/ scripts/
isort visqa/ scripts/
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This project builds on top of amazing open-source work:
- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research  
- [OpenCLIP](https://github.com/mlfoundations/open_clip) by LAION
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) by Google Research

---

## 📝 Citation

```bibtex
@misc{visqa2024,
  title={ViSQA: Video Segmentation and Query Anchoring},
  author={Your Name},
  year={2024},
  url={https://github.com/yourname/visqa}
}
```
