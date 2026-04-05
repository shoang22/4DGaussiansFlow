#!/usr/bin/env python3
"""
Precompute optical flow for the fixed-camera multi-view dataset.

For each camera and each consecutive frame pair (t, t+1), runs the selected
optical flow model (RAFT or GMFlow) and saves the result as a .npy file.

Output layout::

    <flow_dir>/cam_<cam_id:02d>/<t:04d>.npy    shape: [2, H, W]  (dx, dy pixels)

Usage::

    python preprocess_flow.py \\
        --image_dir /data/scene/images \\
        --flow_dir  /data/scene/flows  \\
        --num_cameras 6                \\
        --num_frames 100               \\
        --model raft                   \\
        --weights /path/to/raft.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from gmflow.config import get_cfg as get_gmflow_cfg
from gmflow.gmflow import build_gmflow
from gmflow.gmflow import GMFlow

# ---------------------------------------------------------------------------
# Named constants — no magic numbers
# ---------------------------------------------------------------------------
CAM_DIR_PATTERN: str = "cam{cam_id:02d}"
IMAGE_NAME_PATTERN: str = "{t:05d}"
FLOW_FILENAME_PATTERN: str = f"{IMAGE_NAME_PATTERN}.npy"
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

_to_tensor = T.ToTensor()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_raft(device: torch.device, weights_path: str | None = None) -> object:
    """Load the RAFT optical flow model.

    Requires RAFT to be importable from PYTHONPATH:
    https://github.com/princeton-vl/RAFT

    Args:
        device: Torch device for model weights.
        weights_path: Optional path to a pretrained ``.pth`` checkpoint.

    Returns:
        RAFT model in ``eval`` mode.

    Raises:
        ImportError: If RAFT is not installed / not on PYTHONPATH.
        FileNotFoundError: If ``weights_path`` is given but does not exist.
    """
    try:
        from raft import RAFT  # type: ignore[import]
        from argparse import Namespace as _RaftArgs
    except ImportError as exc:
        raise ImportError(
            "RAFT is not installed.  Clone https://github.com/princeton-vl/RAFT, "
            "add its `core/` directory to PYTHONPATH, and run: "
            "pip install -r requirements.txt"
        ) from exc

    raft_args = _RaftArgs()
    raft_args.small = False
    raft_args.mixed_precision = False
    raft_args.alternate_corr = False

    model = torch.nn.DataParallel(RAFT(raft_args))
    if weights_path is not None:
        _check_weights(weights_path)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded RAFT weights from {weights_path}")
    model = model.module  # unwrap DataParallel
    model.to(device)
    model.eval()
    return model


def _load_gmflow(device: torch.device, weights_path: str | None = None) -> object:
    """Load the GMFlow optical flow model.

    Requires GMFlow to be importable from PYTHONPATH:
    https://github.com/haofeixu/gmflow

    Args:
        device: Torch device for model weights.
        weights_path: Optional path to a pretrained checkpoint.

    Returns:
        GMFlow model in ``eval`` mode.

    Raises:
        ImportError: If GMFlow is not installed / not on PYTHONPATH.
        FileNotFoundError: If ``weights_path`` is given but does not exist.
    """
    cfg = get_gmflow_cfg()
    flownet = torch.nn.DataParallel(build_gmflow(cfg)) 
    flownet = flownet.module
    checkpoint = torch.load(cfg.model, map_location = 'cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flownet.load_state_dict(weights)
    flownet = flownet.cuda()
    flownet.eval()
    return flownet


def _check_weights(path: str) -> None:
    """Raise :class:`FileNotFoundError` if ``path`` does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Weights file not found: {path}")


def load_flow_model(
    model_name: str,
    device: torch.device,
    weights_path: str | None = None,
) -> object:
    """Load and return the requested optical flow model.

    Args:
        model_name: ``"raft"`` or ``"gmflow"``.
        device: Torch device.
        weights_path: Optional path to pretrained weights.

    Returns:
        Loaded model in ``eval`` mode.

    Raises:
        ValueError: If ``model_name`` is not recognized.
    """
    if model_name == "raft":
        return _load_raft(device, weights_path)
    if model_name == "gmflow":
        return _load_gmflow(device, weights_path)
    raise ValueError(f"Unknown flow model '{model_name}'. Choose 'raft' or 'gmflow'.")


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _find_image(cam_dir: Path, t: int) -> Path:
    """Return the path to the image for frame ``t`` inside ``cam_dir``.

    Tries both ``frame_{t:04d}.<ext>`` and ``{t:04d}.<ext>`` patterns.

    Args:
        cam_dir: Per-camera directory containing frame images.
        t: 0-based frame index.

    Returns:
        Resolved path to the image.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    for ext in SUPPORTED_EXTENSIONS:
        for stem in (f"frame_{IMAGE_NAME_PATTERN}".format(t=t), IMAGE_NAME_PATTERN):
            p = cam_dir / f"{stem}{ext}"
            if p.exists():
                return p
    import pdb; pdb.set_trace()
    raise FileNotFoundError(
        f"No image for frame {t} in {cam_dir}.  "
        f"Tried patterns 'frame_{t:04d}.<ext>' and '{t:04d}.<ext>' "
        f"with extensions {SUPPORTED_EXTENSIONS}."
    )


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load an RGB image as a float32 tensor in the [0, 255] range.

    Most optical-flow networks expect uint8-range inputs scaled to float32.

    Args:
        path: Path to the image file.
        device: Torch device.

    Returns:
        Tensor of shape [1, 3, H, W], dtype float32, values in [0, 255].
    """
    img = Image.open(path).convert("RGB")
    tensor = _to_tensor(img).unsqueeze(0) * 255.0  # [1, 3, H, W], float32
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Per-model flow inference
# ---------------------------------------------------------------------------

def _infer_raft(model: object, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Run RAFT inference on one image pair.

    Args:
        model: Loaded RAFT model.
        img1: [1, 3, H, W] float32 tensor, values in [0, 255].
        img2: [1, 3, H, W] float32 tensor, values in [0, 255].

    Returns:
        Flow tensor of shape [2, H, W] in pixels (dx, dy).
    """
    _flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
    return flow_up[0]  # [2, H, W]


def _infer_gmflow(model: object, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Run GMFlow inference on one image pair.

    GMFlow normalises images internally (expects [0, 255] input, applies
    ImageNet mean/std inside ``normalize_img``).  Pass tensors as-is.

    Args:
        model: Loaded GMFlow model.
        img1: [1, 3, H, W] float32 tensor, values in [0, 255].
        img2: [1, 3, H, W] float32 tensor, values in [0, 255].

    Returns:
        Flow tensor of shape [2, H, W] in pixels (dx, dy).
    """
    if not isinstance(model, GMFlow):
        raise ValueError(
            f"Expected GMFlow model instance, got {type(model)}.  "
            f"Check that GMFlow is installed and weights are loaded correctly."
        )
    results = model(img1, img2)
    return results[-1][0]  # [2, H, W]


def infer_flow(
    model_name: str,
    model: object,
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    """Dispatch flow inference to the correct backend.

    Args:
        model_name: ``"raft"`` or ``"gmflow"``.
        model: Loaded model.
        img1: [1, 3, H, W] float32, values in [0, 255].
        img2: [1, 3, H, W] float32, values in [0, 255].

    Returns:
        Flow tensor [2, H, W] in pixels (dx, dy).

    Raises:
        ValueError: If ``model_name`` is not recognized.
    """
    if model_name == "raft":
        return _infer_raft(model, img1, img2)
    if model_name == "gmflow":
        return _infer_gmflow(model, img1, img2)
    raise ValueError(f"Unknown flow model '{model_name}'.")


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def preprocess_flow(
    image_dir: Path,
    flow_dir: Path,
    num_cameras: int,
    num_frames: int,
    model_name: str,
    weights_path: str | None,
    device: torch.device,
) -> None:
    """Compute and save optical flow for all camera/frame-pair combinations.

    For every camera ``cam_id`` in ``[0, num_cameras)`` and every consecutive
    frame pair ``(t, t+1)`` where ``t`` is in ``[0, num_frames - 2]``:

    * Loads ``image_dir/cam_{cam_id:02d}/frame_{t:04d}.<ext>`` and the
      next-frame image.
    * Runs the flow model (inside ``torch.no_grad()``).
    * Saves the result to ``flow_dir/{CAM_DIR_PATTERN}/{FLOW_FILENAME_PATTERN}`` with
      shape ``[2, H, W]``.

    Already-computed pairs are skipped.  Missing frame files are logged as
    warnings and skipped.

    Args:
        image_dir: Root image directory.
        flow_dir: Output root directory for flow ``.npy`` files.
        num_cameras: Number of cameras.
        num_frames: Total number of frames per camera.
        model_name: ``"raft"`` or ``"gmflow"``.
        weights_path: Optional pretrained weights path.
        device: Torch device for inference.
    """
    model = load_flow_model(model_name, device, weights_path)

    total_pairs = num_cameras * (num_frames - 1)
    pbar = tqdm(total=total_pairs, desc="Computing optical flow", unit="pair")

    for cam_id in range(1, num_cameras + 1):
        cam_img_dir = image_dir / CAM_DIR_PATTERN.format(cam_id=cam_id)
        cam_flow_dir = flow_dir / CAM_DIR_PATTERN.format(cam_id=cam_id)
        cam_flow_dir.mkdir(parents=True, exist_ok=True)

        for t in range(1, num_frames):
            out_path = cam_flow_dir / FLOW_FILENAME_PATTERN.format(t=t)

            if out_path.exists():
                pbar.update(1)
                continue  # resume: skip already-computed pairs

            try:
                img1_path = _find_image(cam_img_dir, t)
                img2_path = _find_image(cam_img_dir, t + 1)
            except FileNotFoundError as err:
                tqdm.write(f"[WARNING] Skipping cam {cam_id:02d}, t={t:04d}: {err}")
                pbar.update(1)
                continue

            with torch.no_grad():
                img1 = load_image_tensor(img1_path, device)
                img2 = load_image_tensor(img2_path, device)
                flow = infer_flow(model_name, model, img1, img2)  # [2, H, W]

            flow_np: np.ndarray = flow.cpu().numpy().astype(np.float32)
            np.save(out_path, flow_np)
            pbar.update(1)

    pbar.close()
    print(f"\nDone.  Flow saved under: {flow_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description=(
            "Precompute optical flow for fixed-camera multi-view dynamic scenes. "
            "Saves per-pair flow as .npy files of shape [2, H, W] (dx, dy in pixels)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help=(
            "Root directory with per-camera image subfolders. "
            "Expected: <image_dir>/cam_<id:02d>/frame_<t:04d>.<ext>"
        ),
    )
    parser.add_argument(
        "--flow_dir",
        type=Path,
        required=True,
        help="Output root for flow files.  Created if absent.",
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        required=True,
        help="Number of cameras.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        help="Total number of frames per camera.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["raft", "gmflow"],
        default="gmflow",
        help="Optical flow model.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained model weights checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    _image_dir = _args.image_dir.resolve()
    _flow_dir = _args.flow_dir.resolve()
    _device = torch.device(_args.device)

    # Basic sanity checks before starting
    if not _image_dir.exists():
        raise FileNotFoundError(f"--image_dir does not exist: {_image_dir}")

    _first_cam = _image_dir / CAM_DIR_PATTERN.format(cam_id=1)
    if not _first_cam.exists():
        raise FileNotFoundError(
            f"Expected first camera directory not found: {_first_cam}.  "
            f"Check --image_dir and --num_cameras."
        )

    print("=" * 60)
    print(f"image_dir   : {_image_dir}")
    print(f"flow_dir    : {_flow_dir}")
    print(f"num_cameras : {_args.num_cameras}")
    print(f"num_frames  : {_args.num_frames}")
    print(f"model       : {_args.model}")
    print(f"weights     : {_args.weights}")
    print(f"device      : {_device}")
    print("=" * 60)

    preprocess_flow(
        image_dir=_image_dir,
        flow_dir=_flow_dir,
        num_cameras=_args.num_cameras,
        num_frames=_args.num_frames,
        model_name=_args.model,
        weights_path=_args.weights,
        device=_device,
    )
