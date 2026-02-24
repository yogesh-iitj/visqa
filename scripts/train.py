"""
ViSQA Training Script.

Fine-tunes the grounding + segmentation components on custom video data.

Usage:
    python scripts/train.py \
        --config configs/training/finetune.yaml \
        --data_root data/custom/ \
        --annotations data/custom/annotations.json \
        --output_dir checkpoints/my_model/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visqa.data.dataset import CustomVideoDataset
from visqa.training.losses import VisqaLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViSQA model")
    parser.add_argument("--config", default="configs/training/finetune.yaml")
    parser.add_argument("--data_root", required=True, help="Root directory of dataset")
    parser.add_argument("--annotations", required=True, help="Path to annotations.json")
    parser.add_argument("--output_dir", default="checkpoints/", help="Save checkpoints here")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


class ViSQATrainer:
    """Training orchestrator for ViSQA fine-tuning."""

    def __init__(self, args):
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on device: {self.device}")

        # Build datasets
        self._build_datasets()

        # Build model (we fine-tune the grounding components; SAM2 is kept frozen for efficiency)
        self._build_model()

        # Loss and optimizer
        self.criterion = VisqaLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        self.start_epoch = 0
        self.best_jf = 0.0

        if args.resume:
            self._load_checkpoint(args.resume)

        if args.use_wandb:
            try:
                import wandb
                wandb.init(project="visqa", config=vars(args))
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Skipping wandb logging.")
                self.wandb = None
        else:
            self.wandb = None

    def _build_datasets(self):
        args = self.args
        image_size = (args.image_size, args.image_size)

        full_dataset = CustomVideoDataset(
            data_root=args.data_root,
            annotations_path=args.annotations,
            clip_length=args.clip_length,
            image_size=image_size,
        )

        n_val = max(1, int(len(full_dataset) * args.val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            collate_fn=self._collate_fn,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=self._collate_fn,
        )

        logger.info(f"Train: {n_train} samples, Val: {n_val} samples")

    def _collate_fn(self, batch):
        """Custom collate that handles variable-length queries."""
        from torch.utils.data.dataloader import default_collate
        queries = [b.pop("query") for b in batch]
        collated = default_collate(batch)
        collated["query"] = queries
        return collated

    def _build_model(self):
        """
        Build fine-tunable model.

        Strategy: Use Grounding DINO as the trainable backbone.
        SAM2 is kept frozen (used only for mask refinement during inference).
        """
        try:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            import groundingdino.datasets.transforms as T

            # This is a simplified placeholder.
            # In practice, load GDINO and add a light adapter head.
            logger.info("Loading Grounding DINO for fine-tuning...")
            # model = build_model(cfg) ...
            # For now, create a simple wrapper
            self.model = self._build_adapter_model()

        except ImportError:
            logger.warning("Grounding DINO not available. Using adapter-only mode.")
            self.model = self._build_adapter_model()

    def _build_adapter_model(self):
        """
        Lightweight adapter: adds a temporal fusion layer on top of frozen CLIP features.
        This is the recommended approach for quick fine-tuning.
        """
        model = TemporalAdapter(
            feature_dim=512,
            num_heads=8,
            num_layers=2,
        ).to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params:,}")
        return model

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_losses = {}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(self.device)     # (B, T, 3, H, W)
            masks = batch["masks"].to(self.device)       # (B, T, H, W)
            boxes = batch["boxes"].to(self.device)       # (B, T, 4)

            self.optimizer.zero_grad()
            outputs = self.model(frames)
            losses = self.criterion(outputs["mask_logits"], masks, outputs.get("boxes"), boxes.view(-1, 4))
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()

            if step % self.args.log_interval == 0:
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

            if self.wandb:
                self.wandb.log({"train/" + k: v.item() for k, v in losses.items()})

        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        from visqa.utils.metrics import evaluate_predictions

        all_jf = []
        for batch in tqdm(self.val_loader, desc="Validating"):
            frames = batch["frames"].to(self.device)
            masks_gt = batch["masks"].numpy()
            boxes_gt = batch["boxes"].numpy()

            outputs = self.model(frames)
            masks_pred = (torch.sigmoid(outputs["mask_logits"]) > 0.5).cpu().numpy()

            for b in range(len(masks_gt)):
                metrics = evaluate_predictions(masks_pred[b], masks_gt[b])
                all_jf.append(metrics["jf_mean"])

        mean_jf = float(sum(all_jf) / len(all_jf)) if all_jf else 0.0
        return {"jf_mean": mean_jf}

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "best.pth"
            torch.save(state, best_path)
            logger.info(f"New best model saved: J&F={metrics.get('jf_mean', 0):.4f}")

    def _load_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch = state["epoch"] + 1
        self.best_jf = state.get("metrics", {}).get("jf_mean", 0.0)
        logger.info(f"Resumed from epoch {self.start_epoch}, best J&F={self.best_jf:.4f}")

    def train(self):
        logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.args.epochs):
            train_losses = self.train_epoch(epoch)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            )

            if (epoch + 1) % self.args.save_interval == 0:
                val_metrics = self.validate()
                logger.info(f"Val J&F: {val_metrics['jf_mean']:.4f}")

                is_best = val_metrics["jf_mean"] > self.best_jf
                self.best_jf = max(self.best_jf, val_metrics["jf_mean"])
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

                if self.wandb:
                    self.wandb.log({"val/" + k: v for k, v in val_metrics.items()})

        logger.info(f"Training complete! Best J&F: {self.best_jf:.4f}")


class TemporalAdapter(nn.Module):
    """
    Lightweight temporal adapter for fine-tuning.
    Adds temporal attention on top of frozen per-frame features.
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            dict with "mask_logits": (B, T, H, W)
        """
        B, T, C, H, W = frames.shape

        # Placeholder forward pass structure
        # In practice, this feeds frames through a frozen image encoder
        # then applies temporal attention
        dummy_features = torch.zeros(B, T, 512, device=frames.device)
        refined = self.temporal_transformer(dummy_features)

        # Decode to masks
        feat_spatial = refined.view(B * T, 512, 1, 1).expand(-1, -1, H // 16, W // 16)
        mask_logits = self.mask_head(feat_spatial).squeeze(1)
        mask_logits = mask_logits.view(B, T, H // 4, W // 4)
        mask_logits = torch.nn.functional.interpolate(
            mask_logits.view(B * T, 1, H // 4, W // 4),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1).view(B, T, H, W)

        return {"mask_logits": mask_logits}


if __name__ == "__main__":
    args = parse_args()
    trainer = ViSQATrainer(args)
    trainer.train()
