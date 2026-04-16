#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualize VideoToTensor → VideoCosmosAugmentTransform on real dataset samples.

Loads samples from a LeRobot dataset, applies the two transforms one at a time,
and saves JPEG snapshots at each stage so you can visually confirm Cosmos is
producing photorealistic output.  Also writes a side-by-side comparison grid
per video key across all requested samples.

Prerequisites (Cosmos only):
    Start the Cosmos service in its own terminal (from the Cosmos venv):
        cd third_party/cosmos-transfer2.5
        export HF_HOME=.../cache
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../../scripts/cosmos_service.py --port 5557

Usage (from the repo root, in the gr00t conda env):
    # With Cosmos service running:
    python scripts/test_cosmos_transform_pipeline.py \\
        --dataset-path /healthcareeng_monai/datasets/orca-assemble-trocar-sim/assemble_trocar_sim_box_v3_60 \\
        --output-dir ./output/test_transform_outputs --num-samples 5

    # Without Cosmos service (VideoToTensor only):
    python scripts/test_cosmos_transform_pipeline.py \\
        --dataset-path /path/to/data --output-dir ./transform_outputs --no-cosmos

Output layout:
    output_dir/
        sample_000/
            left_wrist_view_t0_1_raw.jpg       # raw uint8 from dataset
            left_wrist_view_t0_2_tensor.jpg    # after VideoToTensor  (same visually)
            left_wrist_view_t0_3_cosmos.jpg    # after VideoCosmosAugmentTransform
            right_wrist_view_...
            room_view_...
        sample_001/
            ...
        grid_left_wrist_view.jpg               # all samples, 3 columns: raw|tensor|cosmos
        grid_right_wrist_view.jpg
        grid_room_view.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.utils

# Allow "from policy.gr00t_config_cosmos import ..." when running from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.transform.video import VideoCosmosAugmentTransform, VideoToTensor  # noqa: E402
from policy.gr00t_config_cosmos import UnitreeG1SimDataConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_frame(tensor: torch.Tensor, path: Path) -> None:
    """Save a [C, H, W] float32 [0, 1] tensor as JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (tensor.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu()
    tvio.write_jpeg(uint8, str(path))


def _raw_to_chw(frame_hwc: "np.ndarray") -> torch.Tensor:  # noqa: F821
    """Convert a single HWC uint8 numpy frame to CHW float32 [0, 1]."""
    import numpy as np

    return torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save VideoToTensor + VideoCosmosAugmentTransform inputs/outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default=(
            "/healthcareeng_monai/datasets/orca-assemble-trocar-sim/"
            "assemble_trocar_sim_box_v3_60"
        ),
        help="Path to a LeRobot-format dataset whose video keys match UnitreeG1SimDataConfig.",
    )
    parser.add_argument(
        "--output-dir",
        default="./transform_outputs",
        help="Directory to write JPEG outputs and comparison grids.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of dataset samples to process.",
    )
    parser.add_argument(
        "--cosmos-host",
        default="localhost",
        help="Hostname of the Cosmos ZMQ service.",
    )
    parser.add_argument(
        "--cosmos-port",
        type=int,
        default=5557,
        help="Port of the Cosmos ZMQ service.",
    )
    parser.add_argument(
        "--cosmos-cache-dir",
        default="/tmp/cosmos_cache_pipeline_test",
        help="Directory for Cosmos result cache files.",
    )
    parser.add_argument(
        "--no-cosmos",
        action="store_true",
        help="Skip VideoCosmosAugmentTransform (useful when service is not running).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for VideoCosmosAugmentTransform. Default None = random per call.",
    )
    parser.add_argument(
        "--grid-mode",
        action="store_true",
        help="Enable 2×2 grid mode: all video keys are tiled into one Cosmos request.",
    )
    parser.add_argument(
        "--video-backend",
        default="decord",
        choices=["decord", "torchvision_av", "torchcodec"],
        help="Video decoding backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "When > 1, stack that many consecutive samples into a [B, T, C, H, W] batch "
            "and run VideoCosmosAugmentTransform once on the whole batch to test batch support."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ config
    cfg = UnitreeG1SimDataConfig()
    cfg.cosmos_cache_dir = args.cosmos_cache_dir
    cfg.cosmos_host = args.cosmos_host
    cfg.cosmos_ports = [args.cosmos_port]
    cfg.cosmos_probability = 1.0  # always augment so every sample is exercised
    video_keys = cfg.video_keys
    print(f"Video keys : {video_keys}")

    # ----------------------------------------------------------------- dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=cfg.modality_config(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend=args.video_backend,
        transforms=None,  # we apply transforms manually step-by-step
    )
    meta = dataset.metadata
    num_samples = min(args.num_samples, len(dataset))
    print(f"Dataset size: {len(dataset)}  (processing {num_samples} samples)")

    # ------------------------------------------------------------ build transforms
    video_to_tensor = VideoToTensor(apply_to=video_keys)
    video_to_tensor.set_metadata(meta)
    video_to_tensor.train()

    cosmos_transform: VideoCosmosAugmentTransform | None = None
    if not args.no_cosmos:
        cosmos_transform = VideoCosmosAugmentTransform(
            apply_to=video_keys,
            cache_dir=args.cosmos_cache_dir,
            host=args.cosmos_host,
            ports=[args.cosmos_port],
            probability=1.0,
            seed=args.seed,
            grid_mode=args.grid_mode,
        )
        cosmos_transform.set_metadata(meta)
        cosmos_transform.train()
        print(
            f"Cosmos service: {args.cosmos_host}:{args.cosmos_port}"
            f"  cache: {args.cosmos_cache_dir}"
            f"  seed={args.seed}  grid_mode={args.grid_mode}"
        )
    else:
        print("Cosmos augmentation: disabled (--no-cosmos)")

    # --------------------------------------------- per-sample collection for grids
    # grid_frames[key] = list of [C, H, W] tensors in order: raw, tensor, cosmos
    grid_frames: dict[str, list[torch.Tensor]] = {k: [] for k in video_keys}

    # ------------------------------------------------------------------ main loop
    for sample_idx in range(num_samples):
        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        print(f"\n[{sample_idx + 1}/{num_samples}] {sample_dir.name}")

        # Raw data from dataset (numpy uint8, [T, H, W, C])
        data: dict = dataset[sample_idx]

        # ------------ Stage 1: raw (before any transform) ----------------
        for key in video_keys:
            if key not in data:
                continue
            raw_thwc = data[key]  # [T, H, W, C] uint8
            tag = key.replace("video.", "")
            for t in range(raw_thwc.shape[0]):
                frame_chw = _raw_to_chw(raw_thwc[t])
                _save_frame(frame_chw, sample_dir / f"{tag}_t{t:02d}_1_raw.jpg")
                grid_frames[key].append(frame_chw)
        print("  Saved: raw frames")

        # ------------ Stage 2: after VideoToTensor -----------------------
        data = video_to_tensor(data)
        for key in video_keys:
            if key not in data:
                continue
            frames_tchw = data[key]  # [T, C, H, W] float32
            tag = key.replace("video.", "")
            for t in range(frames_tchw.shape[0]):
                _save_frame(frames_tchw[t], sample_dir / f"{tag}_t{t:02d}_2_tensor.jpg")
                grid_frames[key].append(frames_tchw[t])
        print("  Saved: VideoToTensor output")
        for key in data.keys():
            if key in video_keys:
                print(f"  VideoToTensor {key} output shape: {data[key].shape}")        

        # ------------ Stage 3: after VideoCosmosAugmentTransform ---------
        if cosmos_transform is not None:
            t0 = time.perf_counter()
            try:
                data = cosmos_transform(data)
                elapsed = time.perf_counter() - t0
                for key in video_keys:
                    if key not in data:
                        continue
                    frames_tchw = data[key]  # [T, C, H, W] float32
                    tag = key.replace("video.", "")
                    for t in range(frames_tchw.shape[0]):
                        _save_frame(
                            frames_tchw[t], sample_dir / f"{tag}_t{t:02d}_3_cosmos.jpg"
                        )
                        grid_frames[key].append(frames_tchw[t])
                cached = "(cached)" if elapsed < 0.5 else ""
                print(f"  Saved: VideoCosmosAugmentTransform output  [{elapsed:.2f}s {cached}]")
            except Exception as exc:
                print(f"  WARNING: VideoCosmosAugmentTransform failed — {exc}")
                print("  Is the Cosmos service running?  Use --no-cosmos to skip it.")
                # Fill grid slots with blank frames so column alignment stays intact
                for key in video_keys:
                    if key in data:
                        blank = torch.zeros_like(data[key][0])
                        grid_frames[key].append(blank)
        for key in data.keys():
            if key in video_keys:
                print(f"  VideoCosmosAugmentTransform {key} output shape: {data[key].shape}")

    # ---------------------------------------------------------------- save grids
    print("\nSaving comparison grids ...")
    stages = 3 if (cosmos_transform is not None) else 2
    for key in video_keys:
        frames = grid_frames[key]
        if not frames:
            continue
        tag = key.replace("video.", "")
        # frames order: [raw_s0, tensor_s0, cosmos_s0, raw_s1, tensor_s1, cosmos_s1, ...]
        # Arrange as a grid: each row = one sample, columns = raw | tensor | cosmos
        grid = torchvision.utils.make_grid(
            torch.stack(frames),
            nrow=stages,
            padding=4,
            normalize=False,
        )
        grid_path = output_dir / f"grid_{tag}.jpg"
        _save_frame(grid, grid_path)
        print(f"  {grid_path.name}  ({num_samples} rows × {stages} cols)")

    # ---------------------------------------------------------------- summary
    print(f"\nDone.  All outputs saved to: {output_dir.resolve()}")
    col_labels = "raw | tensor | cosmos" if stages == 3 else "raw | tensor"
    print(f"Grid columns: {col_labels}")


if __name__ == "__main__":
    main()
