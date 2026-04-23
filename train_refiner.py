from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from depth_utils import (
    list_image_files,
    find_matching_file,
    read_rgb,
    read_depth,
    compute_edge_map,
)
from edge_refiner_model import EdgeGuidedDepthRefiner


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def edge_aware_smoothness_loss(depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """
    depth: B x 1 x H x W
    rgb:   B x 3 x H x W, RGB in [0,1]
    """
    gray = rgb.mean(dim=1, keepdim=True)

    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    gray_dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    gray_dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

    weight_x = torch.exp(-10.0 * gray_dx)
    weight_y = torch.exp(-10.0 * gray_dy)

    return (depth_dx * weight_x).mean() + (depth_dy * weight_y).mean()


class DepthPairDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        raw_dir: str | Path,
        target_dir: str | Path,
        image_size: int = 256,
        augment: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.raw_dir = Path(raw_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment

        self.samples = []
        for img_path in list_image_files(self.image_dir):
            raw_path = find_matching_file(self.raw_dir, img_path.stem)
            tgt_path = find_matching_file(self.target_dir, img_path.stem)
            if raw_path is not None and tgt_path is not None:
                self.samples.append((img_path, raw_path, tgt_path))

        if not self.samples:
            raise RuntimeError(
                "No training pairs found. Check that image names match between "
                "input_images/, output/, and refined_output/refined/."
            )

    def __len__(self):
        return len(self.samples)

    def _augment_np(self, rgb, raw, target, edge):
        if random.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            raw = raw[:, ::-1].copy()
            target = target[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
        return rgb, raw, target, edge

    def __getitem__(self, idx):
        img_path, raw_path, tgt_path = self.samples[idx]

        rgb = read_rgb(img_path, size=self.image_size)      # H W 3
        raw = read_depth(raw_path, size=self.image_size)    # 1 H W
        target = read_depth(tgt_path, size=self.image_size) # 1 H W
        edge = compute_edge_map(rgb, size=self.image_size)  # 1 H W

        if self.augment:
            rgb, raw, target, edge = self._augment_np(rgb, raw, target, edge)

        x = np.concatenate([rgb.transpose(2, 0, 1), raw, edge], axis=0).astype(np.float32)  # 5 H W
        residual_target = (target - raw).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            torch.from_numpy(raw.copy()),
            torch.from_numpy(target.copy()),
            torch.from_numpy(residual_target.copy()),
            img_path.stem,
        )


def split_dataset(dataset, val_ratio=0.2, seed=42):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1 - val_ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1] if len(train_idx) > 1 else train_idx
    return train_idx, val_idx


def save_checkpoint(path, model, epoch, val_loss, image_size, base_channels, residual_scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": float(val_loss),
            "image_size": image_size,
            "base_channels": base_channels,
            "residual_scale": residual_scale,
        },
        str(path),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0

    for x, rgb, raw, target, residual_target, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        rgb = rgb.to(device)
        raw = raw.to(device)
        target = target.to(device)
        residual_target = residual_target.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred_residual = model(x)
        pred_depth = torch.clamp(raw + pred_residual, 0.0, 1.0)

        loss_depth = F.l1_loss(pred_depth, target)
        loss_residual = F.l1_loss(pred_residual, residual_target)
        loss_grad = gradient_loss(pred_depth, target)
        loss_smooth = edge_aware_smoothness_loss(pred_depth, rgb)

        loss = 0.55 * loss_depth + 0.20 * loss_residual + 0.15 * loss_grad + 0.10 * loss_smooth

        loss.backward()
        optimizer.step()

        running += float(loss.item())

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running = 0.0

    for x, rgb, raw, target, residual_target, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        rgb = rgb.to(device)
        raw = raw.to(device)
        target = target.to(device)
        residual_target = residual_target.to(device)

        pred_residual = model(x)
        pred_depth = torch.clamp(raw + pred_residual, 0.0, 1.0)

        loss_depth = F.l1_loss(pred_depth, target)
        loss_residual = F.l1_loss(pred_residual, residual_target)
        loss_grad = gradient_loss(pred_depth, target)
        loss_smooth = edge_aware_smoothness_loss(pred_depth, rgb)

        loss = 0.55 * loss_depth + 0.20 * loss_residual + 0.15 * loss_grad + 0.10 * loss_smooth
        running += float(loss.item())

    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="input_images")
    parser.add_argument("--raw_dir", type=str, default="output")
    parser.add_argument("--target_dir", type=str, default="refined_output/refined")
    parser.add_argument("--save_path", type=str, default="checkpoints/edge_refiner.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DepthPairDataset(
        image_dir=args.image_dir,
        raw_dir=args.raw_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
        augment=True,
    )

    train_idx, val_idx = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EdgeGuidedDepthRefiner(
        in_channels=5,
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                save_path,
                model,
                epoch,
                val_loss,
                args.image_size,
                args.base_channels,
                args.residual_scale,
            )
            print(f"Saved best checkpoint to {save_path}")

    print(f"Training done. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()