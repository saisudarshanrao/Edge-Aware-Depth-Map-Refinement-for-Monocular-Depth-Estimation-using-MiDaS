#!/usr/bin/env python3
"""
Edge-aware depth refinement for MiDaS outputs.

Best workflow:
1) Run MiDaS and save depth maps.
2) Prefer running MiDaS with --grayscale for depth PNGs.
3) Run this script to refine those depth maps using the RGB image as guide.

Install:
    python3 -m pip install opencv-contrib-python numpy

Example:
    python3 refine_depth.py \
        --input_dir input_images \
        --depth_dir output \
        --output_dir refined_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def denormalize_u16(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 65535.0).round().astype(np.uint16)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth map: {path}")

    # If MiDaS output is colorized, convert to gray for refinement.
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)

    # If it is already 0..1, keep it; otherwise normalize per image.
    if depth.max() > 1.5 or depth.min() < -0.5:
        depth = normalize_01(depth)
    else:
        depth = np.clip(depth, 0.0, 1.0)

    return depth


def find_matching_file(folder: Path, stem: str):
    candidates = list(folder.glob(f"{stem}*"))
    for p in candidates:
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def make_guided_refinement(
    rgb_bgr: np.ndarray,
    depth01: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
    edge_blend_strength: float = 0.45,
) -> np.ndarray:
    """
    Guided filter + slight edge-preserving re-injection near strong RGB edges.
    """
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "cv2.ximgproc not found. Install opencv-contrib-python:\n"
            "python3 -m pip install opencv-contrib-python"
        )

    guide_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if guide_gray.shape != depth01.shape:
        depth01 = cv2.resize(
            depth01,
            (guide_gray.shape[1], guide_gray.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    # Edge-preserving smoothing.
    refined = cv2.ximgproc.guidedFilter(
        guide=guide_gray,
        src=depth01.astype(np.float32),
        radius=radius,
        eps=eps,
    )

    # Small boundary boost: keep original depth a bit more on strong edges.
    guide_u8 = (guide_gray * 255.0).astype(np.uint8)
    edges = cv2.Canny(guide_u8, 60, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    final = refined.copy()
    mask = edges > 0
    final[mask] = (
        (1.0 - edge_blend_strength) * refined[mask]
        + edge_blend_strength * depth01[mask]
    )

    return np.clip(final, 0.0, 1.0)


def colormap_depth(depth01: np.ndarray) -> np.ndarray:
    depth8 = (np.clip(depth01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth8, cv2.COLORMAP_INFERNO)


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def side_by_side(rgb_bgr: np.ndarray, raw_vis: np.ndarray, refined_vis: np.ndarray) -> np.ndarray:
    h = rgb_bgr.shape[0]
    raw_vis = cv2.resize(raw_vis, (rgb_bgr.shape[1], h), interpolation=cv2.INTER_CUBIC)
    refined_vis = cv2.resize(refined_vis, (rgb_bgr.shape[1], h), interpolation=cv2.INTER_CUBIC)
    rgb_disp = cv2.resize(rgb_bgr, (rgb_bgr.shape[1], h), interpolation=cv2.INTER_AREA)

    rgb_disp = put_label(rgb_disp, "RGB")
    raw_vis = put_label(raw_vis, "Raw Depth")
    refined_vis = put_label(refined_vis, "Refined Depth")

    return np.hstack([rgb_disp, raw_vis, refined_vis])


def process_one(
    rgb_path: Path,
    depth_path: Path,
    raw_out_dir: Path,
    refined_out_dir: Path,
    compare_out_dir: Path,
    radius: int,
    eps: float,
) -> None:
    rgb = read_rgb(rgb_path)
    depth = read_depth(depth_path)

    refined = make_guided_refinement(
        rgb_bgr=rgb,
        depth01=depth,
        radius=radius,
        eps=eps,
    )

    raw_u16 = denormalize_u16(depth)
    refined_u16 = denormalize_u16(refined)

    raw_out_dir.mkdir(parents=True, exist_ok=True)
    refined_out_dir.mkdir(parents=True, exist_ok=True)
    compare_out_dir.mkdir(parents=True, exist_ok=True)

    # Save numeric depth maps.
    cv2.imwrite(str(raw_out_dir / f"{rgb_path.stem}.png"), raw_u16)
    cv2.imwrite(str(refined_out_dir / f"{rgb_path.stem}.png"), refined_u16)

    # Save visual comparisons.
    raw_vis = colormap_depth(depth)
    refined_vis = colormap_depth(refined)
    panel = side_by_side(rgb, raw_vis, refined_vis)
    cv2.imwrite(str(compare_out_dir / f"{rgb_path.stem}_compare.png"), panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_images")
    parser.add_argument("--depth_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="refined_output")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)

    raw_out_dir = output_dir / "raw"
    refined_out_dir = output_dir / "refined"
    compare_out_dir = output_dir / "compare"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth folder not found: {depth_dir}")

    image_files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not image_files:
        raise RuntimeError(f"No images found in {input_dir}")

    processed = 0
    skipped = 0

    for rgb_path in image_files:
        depth_path = find_matching_file(depth_dir, rgb_path.stem)
        if depth_path is None:
            print(f"[skip] No matching depth map for {rgb_path.name}")
            skipped += 1
            continue

        try:
            process_one(
                rgb_path=rgb_path,
                depth_path=depth_path,
                raw_out_dir=raw_out_dir,
                refined_out_dir=refined_out_dir,
                compare_out_dir=compare_out_dir,
                radius=args.radius,
                eps=args.eps,
            )
            print(f"[ok] {rgb_path.name}")
            processed += 1
        except Exception as e:
            print(f"[error] {rgb_path.name}: {e}")
            skipped += 1

    print()
    print(f"Done. processed={processed}, skipped={skipped}")
    print(f"Saved raw depth maps to: {raw_out_dir}")
    print(f"Saved refined depth maps to: {refined_out_dir}")
    print(f"Saved comparison panels to: {compare_out_dir}")


if __name__ == "__main__":
    main()