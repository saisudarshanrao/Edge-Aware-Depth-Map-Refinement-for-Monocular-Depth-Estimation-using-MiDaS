from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_utils import (
    list_image_files,
    find_matching_file,
    read_rgb,
    read_depth,
    compute_edge_map,
    save_depth16,
    make_comparison_panel,
)
from edge_refiner_model import EdgeGuidedDepthRefiner


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="input_images")
    parser.add_argument("--raw_dir", type=str, default="output")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/edge_refiner.pt")
    parser.add_argument("--out_dir", type=str, default="enhanced_output")
    parser.add_argument("--save_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    image_size = int(ckpt.get("image_size", args.save_size))
    base_channels = int(ckpt.get("base_channels", 32))
    residual_scale = float(ckpt.get("residual_scale", 0.25))

    model = EdgeGuidedDepthRefiner(
        in_channels=5,
        base_channels=base_channels,
        residual_scale=residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_dir = Path(args.image_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    depth_out = out_dir / "depth"
    compare_out = out_dir / "compare"
    depth_out.mkdir(parents=True, exist_ok=True)
    compare_out.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(image_dir)
    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")

    for img_path in image_files:
        raw_path = find_matching_file(raw_dir, img_path.stem)
        if raw_path is None:
            print(f"[skip] No matching raw depth for {img_path.name}")
            continue

        rgb_full = read_rgb(img_path, size=None)
        raw_full = read_depth(raw_path, size=None)[0]  # H x W

        rgb_in = read_rgb(img_path, size=image_size)
        raw_in = read_depth(raw_path, size=image_size)[0]  # H x W
        edge_in = compute_edge_map(rgb_in, size=image_size)[0]  # H x W

        x = np.concatenate(
            [rgb_in.transpose(2, 0, 1), raw_in[None, ...], edge_in[None, ...]],
            axis=0,
        ).astype(np.float32)

        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        pred_residual = model(x_t)[0, 0].cpu().numpy()
        pred_small = np.clip(raw_in + pred_residual, 0.0, 1.0)

        # Upsample back to original resolution for saving and reporting
        final_full = cv2.resize(
            pred_small,
            (raw_full.shape[1], raw_full.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        final_full = np.clip(final_full, 0.0, 1.0)

        save_depth16(depth_out / f"{img_path.stem}.png", final_full)

        panel = make_comparison_panel(rgb_full, raw_full, final_full)
        cv2.imwrite(str(compare_out / f"{img_path.stem}_compare.png"), panel)

        print(f"[ok] {img_path.name}")

    print()
    print(f"Enhanced depth saved to: {depth_out}")
    print(f"Comparison panels saved to: {compare_out}")


if __name__ == "__main__":
    main()