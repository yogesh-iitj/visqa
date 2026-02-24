# ViSQA Training Guide

This guide covers everything you need to know to fine-tune ViSQA on your own video data.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Preparation](#data-preparation)
3. [Annotation Format](#annotation-format)
4. [Fine-tuning (Recommended)](#fine-tuning-recommended)
5. [Training from Scratch](#training-from-scratch)
6. [Training on Public Datasets](#training-on-public-datasets)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Evaluation](#evaluation)
9. [Tips and Tricks](#tips-and-tricks)

---

## Architecture Overview

ViSQA uses a two-stage approach:

```
Text Query
    ↓
[Grounding DINO] ─────────────── Frame ──▶ Bounding Boxes
                                              ↓
                                           [SAM2] ──▶ Pixel Masks
                                              ↓
                                    [Temporal Adapter] ──▶ Tracked Masks
```

**What gets trained during fine-tuning:**
- A lightweight **Temporal Adapter** (2-layer Transformer) that learns temporal coherence
- Optionally: the top layers of **Grounding DINO** for domain adaptation
- SAM2 is kept **frozen** (it generalizes well and fine-tuning it requires significant VRAM)

---

## Data Preparation

### Directory Structure

```
data/
└── my_dataset/
    ├── videos/
    │   ├── video_001.mp4
    │   ├── video_002.mp4
    │   └── ...
    ├── annotations.json
    └── README.md  (optional, document your dataset)
```

### Video Format Requirements

- **Format:** MP4, AVI, MOV (MP4 recommended)
- **Resolution:** Any, but 480p–1080p works best
- **FPS:** 15–60 FPS (25–30 recommended)
- **Length:** 2–120 seconds per video
- **Codec:** H.264 recommended for compatibility

### Creating Annotations with Label Studio

1. Install Label Studio: `pip install label-studio`
2. Launch: `label-studio start`
3. Create a video project
4. Add a **Video** data type
5. Add a **Brush Segmentation** or **Polygon** label + **Text** label for queries
6. Export as **COCO JSON**
7. Convert to ViSQA format: `python scripts/convert_labelstudio.py --input ls_export.json --output annotations.json`

### Creating Annotations with CVAT

1. Upload videos to CVAT (https://app.cvat.ai)
2. Create a task with "Polygon" annotations
3. Add custom attribute "query" (text type)
4. Annotate objects with polygons per keyframe
5. Export as **COCO 1.0** format
6. Convert: `python scripts/convert_cvat.py --input cvat_export/ --output annotations.json`

### Minimum Annotation Requirements

- **Minimum videos:** 20 (50+ recommended)
- **Keyframe spacing:** Every 5–10 frames
- **Objects per video:** 1–5 queries per video
- **Train/val split:** 90/10 minimum

---

## Annotation Format

ViSQA uses an extended COCO-style JSON format:

```json
{
  "info": {
    "description": "My Custom Dataset",
    "version": "1.0",
    "date_created": "2024-01-01"
  },

  "videos": [
    {
      "id": 1,
      "file_name": "video_001.mp4",
      "fps": 30.0,
      "width": 1920,
      "height": 1080,
      "num_frames": 300,
      "split": "train"
    }
  ],

  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "vehicle"},
    {"id": 3, "name": "animal"}
  ],

  "annotations": [
    {
      "id": 1,
      "video_id": 1,
      "category_id": 1,
      "query": "the person in the red jacket",

      "frames": [0, 10, 20, 30, 40, 50],

      "segmentations": [
        {
          "counts": "...",
          "size": [1080, 1920]
        },
        null,
        {"counts": "...", "size": [1080, 1920]},
        null,
        {"counts": "...", "size": [1080, 1920]},
        {"counts": "...", "size": [1080, 1920]}
      ],

      "bboxes": [
        [120, 85, 280, 410],
        null,
        [130, 90, 275, 420],
        null,
        [145, 95, 270, 415],
        [160, 100, 265, 418]
      ],

      "is_crowd": false,
      "occluded_frames": [1, 3]
    }
  ],

  "split": {
    "train": [1, 2, 3, ...],
    "val": [4, 5, ...]
  }
}
```

**Key points:**
- `frames` lists the annotated frame indices (not required to be contiguous)
- `segmentations` and `bboxes` are arrays aligned with `frames`; use `null` for unannotated frames
- Segmentations can be RLE (dict with `counts`) or polygon (list of [x, y] points)
- `query` is a natural language description of the object — be specific!

### Writing Good Queries

Good queries are specific and discriminative:

| ❌ Bad | ✅ Good |
|-------|---------|
| "person" | "the woman in the blue dress" |
| "car" | "the red sports car on the left lane" |
| "dog" | "the golden retriever running" |
| "ball" | "the soccer ball near the goal" |

---

## Fine-tuning (Recommended)

### Quick Start

```bash
python scripts/train.py \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --output_dir checkpoints/my_model/ \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-5
```

### With Config File

```bash
python scripts/train.py \
  --config configs/training/finetune.yaml \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --output_dir checkpoints/my_model/
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --output_dir checkpoints/my_model/ \
  --batch_size 8
```

### Resume from Checkpoint

```bash
python scripts/train.py \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --output_dir checkpoints/my_model/ \
  --resume checkpoints/my_model/checkpoint_epoch_010.pth
```

---

## Training from Scratch

For heavily customized domains (medical video, satellite imagery, etc.) where pretrained weights don't transfer well:

```bash
python scripts/train.py \
  --config configs/training/scratch.yaml \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --output_dir checkpoints/my_scratch_model/ \
  --epochs 100 \
  --lr 1e-4
```

Requires significantly more data (1000+ videos) and compute (3–5 days on 8xA100).

---

## Training on Public Datasets

### Ref-DAVIS 2017

```bash
# Download
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
unzip DAVIS-2017-trainval-Full-Resolution.zip -d data/

# Convert to ViSQA format
python scripts/convert_davis.py \
  --davis_root data/DAVIS/ \
  --output data/refdavis/

# Train
python scripts/train.py \
  --data_root data/refdavis/ \
  --annotations data/refdavis/annotations.json
```

### YouTube-VIS 2021

```bash
# Download from https://youtube-vos.org/dataset/vis/
python scripts/convert_ytvis.py \
  --ytvis_root data/youtube_vis_2021/ \
  --output data/ytvis/

python scripts/train.py \
  --data_root data/ytvis/ \
  --annotations data/ytvis/annotations.json
```

---

## Hyperparameter Tuning

### Most Important Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lr` | 1e-5 | 1e-6 – 1e-4 | Lower for domain-adapted models |
| `batch_size` | 2 | 1–8 | Limited by GPU VRAM |
| `clip_length` | 8 | 4–16 | Longer = more temporal context |
| `box_threshold` | 0.30 | 0.10–0.50 | Lower = more detections, more noise |
| `dice_weight` | 1.0 | 0.5–2.0 | Increase for cleaner masks |

### VRAM Requirements

| SAM2 Size | Batch Size | VRAM |
|-----------|-----------|------|
| tiny | 4 | ~8 GB |
| small | 4 | ~12 GB |
| base+ | 2 | ~16 GB |
| large | 1 | ~24 GB |
| large | 2 | ~40 GB |

### Mixed Precision Training

Enable AMP for ~2x speed and ~40% memory savings:

```bash
python scripts/train.py ... --fp16
```

---

## Evaluation

### Run Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/my_model/best.pth \
  --data_root data/my_dataset/ \
  --annotations data/my_dataset/annotations.json \
  --split val
```

### Metrics Explained

- **J (Jaccard/IoU):** Overlap between predicted and GT mask. Higher = better segmentation quality.
- **F (F-measure):** Boundary accuracy. Measures how well contours match.
- **J&F Mean:** Average of J and F — the primary benchmark metric for VOS tasks.
- **Box IoU:** Bounding box accuracy for grounding evaluation.

### Expected Performance

After fine-tuning on 100+ videos for 20 epochs:
- J&F improvement over zero-shot: ~5–15 points
- Training time on single A100: ~2–4 hours

---

## Tips and Tricks

### Do's ✅

- **Annotate boundary frames carefully** — the first and last frame of an object's appearance matter most
- **Use specific, descriptive queries** — "the goalkeeper in yellow" not "person"
- **Balance your dataset** — include objects at various scales, occlusions, and lighting conditions
- **Validate with inference** before committing to a full training run
- **Start with fine-tuning** before attempting from-scratch training

### Don'ts ❌

- Don't annotate every single frame — keyframes every 5–10 frames is sufficient
- Don't train SAM2 weights on small datasets (< 500 videos) — it will overfit
- Don't use very low box thresholds (< 0.15) — too much noise
- Don't skip validation — monitor J&F during training to catch overfitting early

### Debugging

If training loss isn't decreasing:
1. Check annotation quality with: `python scripts/visualize_dataset.py --annotations annotations.json`
2. Try a higher learning rate (1e-4)
3. Reduce batch size if training is unstable
4. Check that queries are descriptive and unique per video

If inference quality is poor on custom domain:
1. Lower `box_threshold` to 0.15–0.20 for harder-to-detect objects
2. Increase `frame_stride` for longer videos (reduces memory)
3. Use the `ensemble` grounder for more robust detections
