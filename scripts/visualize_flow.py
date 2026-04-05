#!/usr/bin/env python3
"""
Visualize precomputed optical flow next to the image pair used to compute it.

For each selected (camera, frame) pair, produces a side-by-side figure:
    [ frame_t  |  frame_t+1  |  flow (HSV colorwheel) ]

Output is saved as a PNG; pass --show to also open an interactive window.

Usage::

    python scripts/visualize_flow.py \\
        --image_dir data/multipleview/background1 \\
        --flow_dir  output/flows \\
        --cam_ids   1 2 3 \\
        --frames    1 5 10 \\
        --out_dir   output/flow_vis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Flow -> RGB colorwheel (Middlebury convention)
# ---------------------------------------------------------------------------

def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert a [2, H, W] flow array to an [H, W, 3] uint8 RGB image.

    Uses an HSV colorwheel: hue encodes direction, brightness encodes magnitude.
    Magnitude is normalised to the 99th percentile so outliers don't wash out
    the visualisation.
    """
    dx, dy = flow[0], flow[1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)  # [-pi, pi]

    # Normalise magnitude to [0, 1] using 99th-percentile clipping
    mag_max = np.percentile(magnitude, 99)
    if mag_max > 0:
        mag_norm = np.clip(magnitude / mag_max, 0.0, 1.0)
    else:
        mag_norm = magnitude

    # HSV: hue = direction, sat = 1, val = magnitude
    hue = (angle + np.pi) / (2 * np.pi)          # [0, 1]
    hsv = np.stack([hue, np.ones_like(hue), mag_norm], axis=-1).astype(np.float32)

    import colorsys
    h, w = hue.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r, g, b = colorsys.hsv_to_rgb(float(hsv[i, j, 0]),
                                           float(hsv[i, j, 1]),
                                           float(hsv[i, j, 2]))
            rgb[i, j] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Vectorised HSV colorwheel — fast version of _flow_to_rgb."""
    import matplotlib.colors as mcolors

    dx, dy = flow[0], flow[1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)                    # [-pi, pi]

    mag_max = np.percentile(magnitude, 99)
    mag_norm = np.clip(magnitude / mag_max, 0.0, 1.0) if mag_max > 0 else magnitude

    hue = (angle + np.pi) / (2 * np.pi)           # [0, 1]
    hsv = np.stack([hue, np.ones_like(hue), mag_norm], axis=-1)
    rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return rgb                                     # [H, W, 3]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
_CAM_PATTERNS = ("cam_{:02d}", "cam{:02d}", "cam_{:d}", "cam{:d}")
_IMG_PATTERNS = ("frame_{:05d}", "{:05d}", "frame_{:04d}", "{:04d}")


def _find_cam_dir(root: Path, cam_id: int) -> Path:
    for pat in _CAM_PATTERNS:
        d = root / pat.format(cam_id)
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No camera directory for cam {cam_id} under {root}")


def _find_image(cam_dir: Path, t: int) -> Path:
    for stem_pat in _IMG_PATTERNS:
        for ext in _EXTENSIONS:
            p = cam_dir / f"{stem_pat.format(t)}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"No image for frame {t} in {cam_dir}")


def _find_flow(flow_cam_dir: Path, t: int) -> Path:
    for stem_pat in ("{:05d}", "{:04d}"):
        p = flow_cam_dir / f"{stem_pat.format(t)}.npy"
        if p.exists():
            return p
    raise FileNotFoundError(f"No flow file for t={t} in {flow_cam_dir}")


# ---------------------------------------------------------------------------
# Per-pair visualisation
# ---------------------------------------------------------------------------

def visualize_pair(
    img1_path: Path,
    img2_path: Path,
    flow_path: Path,
    out_path: Path,
    show: bool = False,
) -> None:
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))
    flow = np.load(flow_path)          # [2, H, W]
    flow_rgb = flow_to_rgb(flow)       # [H, W, 3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1);      axes[0].set_title(f"frame t\n{img1_path.name}");   axes[0].axis("off")
    axes[1].imshow(img2);      axes[1].set_title(f"frame t+1\n{img2_path.name}"); axes[1].axis("off")
    axes[2].imshow(flow_rgb);  axes[2].set_title(f"flow (HSV)\n{flow_path.name}"); axes[2].axis("off")

    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    fig.suptitle(
        f"{flow_path.parent.name} / t={flow_path.stem}  |  "
        f"flow max={mag.max():.2f}px  mean={mag.mean():.2f}px",
        fontsize=10,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize precomputed optical flow next to source images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_dir", type=Path, required=True,
                        help="Root image directory (same as used for preprocess_flow.py).")
    parser.add_argument("--flow_dir",  type=Path, required=True,
                        help="Root flow directory produced by preprocess_flow.py.")
    parser.add_argument("--out_dir",   type=Path, default=Path("output/flow_vis"),
                        help="Directory to write visualisation PNGs into.")
    parser.add_argument("--cam_ids",   type=int, nargs="+", default=[1],
                        help="Camera IDs to visualise.")
    parser.add_argument("--frames",    type=int, nargs="+", default=None,
                        help="Frame indices t to visualise (flow t→t+1). "
                             "Defaults to the first 5 available flows per camera.")
    parser.add_argument("--show",      action="store_true",
                        help="Open an interactive matplotlib window for each pair.")
    args = parser.parse_args()

    for cam_id in args.cam_ids:
        try:
            img_cam_dir  = _find_cam_dir(args.image_dir, cam_id)
            flow_cam_dir = _find_cam_dir(args.flow_dir,  cam_id)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        # Determine which frames to visualise
        if args.frames is not None:
            frames: List[int] = args.frames
        else:
            npy_files = sorted(flow_cam_dir.glob("*.npy"))[:5]
            frames = [int(p.stem) for p in npy_files]

        print(f"\ncam {cam_id}  ({flow_cam_dir.name})  —  frames: {frames}")

        for t in frames:
            try:
                img1_path = _find_image(img_cam_dir, t)
                img2_path = _find_image(img_cam_dir, t + 1)
                flow_path = _find_flow(flow_cam_dir, t)
            except FileNotFoundError as e:
                print(f"  [SKIP] t={t}: {e}")
                continue

            out_path = args.out_dir / flow_cam_dir.name / f"{t:05d}.png"
            visualize_pair(img1_path, img2_path, flow_path, out_path, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
