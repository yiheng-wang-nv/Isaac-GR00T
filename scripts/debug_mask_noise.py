#!/usr/bin/env python3
"""
Debug helper: load a few frames + masks and save images after background noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gr00t.utils.video_utils import get_frames_by_indices
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer mask to RGB for visualization."""
    palette = {
        0: (0, 0, 0),        # background
        1: (255, 0, 0),      # red
        2: (0, 255, 0),      # green
        3: (0, 0, 255),      # blue
        4: (255, 255, 0),    # yellow (postprocess fill)
    }
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        rgb[mask == label] = color
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="LeRobot dataset root with meta/info.json",
    )
    parser.add_argument(
        "--camera_key",
        default="head_left_camera_color_optical_frame",
        help="Camera key from meta/modality.json video/mask sections",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 10, 20],
        help="Frame indices to inspect",
    )
    parser.add_argument(
        "--output_dir",
        default="debug_noise_outputs",
        help="Output directory for saved images",
    )
    parser.add_argument(
        "--video_backend",
        default="ffmpeg",
        help="Video backend for decoding",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text())
    chunk_size = info.get("chunks_size", 1000)
    episode_chunk = args.episode // chunk_size
    video_path_pattern = info["video_path"]
    mask_path_pattern = info.get("mask_path")
    if mask_path_pattern is None:
        raise ValueError("mask_path not found in meta/info.json")

    indices = np.array(args.indices, dtype=int)
    video_path = dataset_path / video_path_pattern.format(
        episode_chunk=episode_chunk,
        video_key=f"observation.images.{args.camera_key}",
        episode_index=args.episode,
    )
    mask_path = dataset_path / mask_path_pattern.format(
        episode_chunk=episode_chunk,
        mask_key=f"observation.images.{args.camera_key}",
        episode_index=args.episode,
        video_key=f"observation.images.{args.camera_key}",
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    video = get_frames_by_indices(
        str(video_path),
        indices,
        video_backend=args.video_backend,
    )
    if mask_path.suffix.lower() == ".npz":
        npz_data = np.load(mask_path)
        masks = npz_data["arr_0"] if "arr_0" in npz_data else npz_data[npz_data.files[0]]
    else:
        masks = np.load(mask_path)
    masks = masks[indices]

    images = [Image.fromarray(frame) for frame in video]
    noised = Gr00tN1d6Processor._apply_background_noise(images, list(masks))

    output_dir = Path(args.output_dir)
    for frame_idx, frame, mask, noised_frame in zip(indices, video, masks, noised):
        save_image(output_dir / f"frame_{frame_idx:06d}_orig.png", frame)
        mask_vis = (mask.astype(np.uint8) * 85).clip(0, 255)
        save_image(output_dir / f"frame_{frame_idx:06d}_mask.png", mask_vis)
        save_image(output_dir / f"frame_{frame_idx:06d}_mask_rgb.png", colorize_mask(mask))
        save_image(output_dir / f"frame_{frame_idx:06d}_noised.png", noised_frame)

    print(f"Saved {len(indices)} frames to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

