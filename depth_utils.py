from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def find_matching_file(folder: str | Path, stem: str) -> Optional[Path]:
    folder = Path(folder)
    if not folder.exists():
        return None

    # Exact stem match first, then prefix match
    exact_candidates = [
        folder / f"{stem}.png",
        folder / f"{stem}.jpg",
        folder / f"{stem}.jpeg",
        folder / f"{stem}.bmp",
        folder / f"{stem}.tif",
        folder / f"{stem}.tiff",
        folder / f"{stem}.webp",
    ]
    for p in exact_candidates:
        if p.exists():
            return p

    candidates = sorted(folder.glob(f"{stem}*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p

    return None


def read_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def read_depth(path: str | Path, size: int | None = None) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth: {path}")

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)

    mn = float(depth.min())
    mx = float(depth.max())
    if mx - mn < 1e-8:
        depth = np.zeros_like(depth, dtype=np.float32)
    else:
        depth = (depth - mn) / (mx - mn)

    if size is not None:
        depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_CUBIC)

    return np.clip(depth, 0.0, 1.0)[None, ...]


def compute_edge_map(rgb01: np.ndarray, size: int | None = None) -> np.ndarray:
    """
    rgb01: H x W x 3, range [0,1], RGB order
    returns: 1 x H x W, range [0,1]
    """
    gray = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    mx = float(mag.max())
    if mx > 1e-8:
        mag = mag / mx
    else:
        mag = np.zeros_like(mag, dtype=np.float32)

    if size is not None:
        mag = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)

    return np.clip(mag, 0.0, 1.0)[None, ...]


def save_depth16(path: str | Path, depth01: np.ndarray) -> None:
    """
    Save normalized depth [0,1] as 16-bit PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth01 = np.clip(depth01, 0.0, 1.0)
    depth_u16 = (depth01 * 65535.0).round().astype(np.uint16)
    cv2.imwrite(str(path), depth_u16)


def colorize_depth(depth01: np.ndarray) -> np.ndarray:
    """
    depth01: H x W, [0,1]
    returns: BGR colormap image
    """
    depth8 = (np.clip(depth01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth8, cv2.COLORMAP_INFERNO)


def _label_bar(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    bar_h = 34
    out = cv2.copyMakeBorder(out, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
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


def make_comparison_panel(rgb01: np.ndarray, raw01: np.ndarray, final01: np.ndarray) -> np.ndarray:
    """
    rgb01: H x W x 3, RGB [0,1]
    raw01/final01: H x W, [0,1]
    returns: BGR panel
    """
    rgb_bgr = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    raw_col = colorize_depth(raw01)
    final_col = colorize_depth(final01)

    target_h = rgb_bgr.shape[0]

    def resize_keep_h(img):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    rgb_bgr = resize_keep_h(rgb_bgr)
    raw_col = resize_keep_h(raw_col)
    final_col = resize_keep_h(final_col)

    rgb_bgr = _label_bar(rgb_bgr, "RGB Input")
    raw_col = _label_bar(raw_col, "Raw MiDaS Depth")
    final_col = _label_bar(final_col, "Enhanced Depth")

    return np.hstack([rgb_bgr, raw_col, final_col])