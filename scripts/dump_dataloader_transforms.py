#!/usr/bin/env python3
"""
Dump transformed samples from the GR00T dataloader pipeline for visualization.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from gr00t.configs.base_config import Config, get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_distributed() -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")


def colorize_mask(mask: np.ndarray) -> Image.Image:
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
    return Image.fromarray(rgb)


def load_config(config_path: Path) -> Config:
    config = get_default_config()
    return config.load(config_path)


def get_image_keys_and_t(config: Config) -> tuple[list[str], int]:
    if not config.data.datasets:
        raise ValueError("Config has no datasets.")
    embodiment_tag = config.data.datasets[0].embodiment_tag
    if embodiment_tag is None:
        raise ValueError("Embodiment tag is missing in config.")
    modality_cfg = config.data.modality_configs[embodiment_tag]
    image_keys = modality_cfg["video"].modality_keys
    temporal_len = len(modality_cfg["video"].delta_indices)
    return image_keys, temporal_len


def build_processor(config: Config) -> Gr00tN1d6Processor:
    processor = Gr00tN1d6Processor(
        modality_configs=config.data.modality_configs,
        statistics=None,
        image_crop_size=config.model.image_crop_size,
        image_target_size=config.model.image_target_size,
        random_rotation_angle=config.model.random_rotation_angle,
        color_jitter_params=config.model.color_jitter_params,
        model_name=config.model.model_name,
        model_type=config.model.backbone_model_type,
        formalize_language=config.model.formalize_language,
        max_state_dim=config.model.max_state_dim,
        max_action_dim=config.model.max_action_dim,
        apply_sincos_state_encoding=config.model.apply_sincos_state_encoding,
        max_action_horizon=config.model.action_horizon,
        use_albumentations=config.model.use_albumentations_transforms,
        background_noise_on_mask=config.model.background_noise_on_mask,
        shortest_image_edge=config.model.shortest_image_edge,
        crop_fraction=config.model.crop_fraction,
        use_relative_action=config.model.use_relative_action,
    )
    processor.train()
    return processor


def sample_and_save(
    config: Config,
    output_dir: Path,
    num_samples: int,
    seed: int,
) -> None:
    ensure_distributed()
    set_seeds(seed)
    processor = build_processor(config)
    dataset_factory = DatasetFactory(config=config)
    train_dataset, _ = dataset_factory.build(processor=processor)

    image_keys, temporal_len = get_image_keys_and_t(config)
    num_views = len(image_keys)

    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_dataset)
    for sample_idx in range(num_samples):
        sample = next(iterator)
        vlm_content = sample["vlm_content"]
        images = vlm_content["images"]
        text = vlm_content["text"]
        (output_dir / f"sample_{sample_idx:03d}_text.txt").write_text(text)

        for idx, img in enumerate(images):
            t = idx // num_views
            v = image_keys[idx % num_views]
            img.save(output_dir / f"sample_{sample_idx:03d}_t{t:02d}_view_{v}.png")

        masks = sample.get("masks")
        if masks is not None:
            masks_np = masks.detach().cpu().numpy()
            for idx in range(min(masks_np.shape[0], temporal_len * num_views)):
                t = idx // num_views
                v = image_keys[idx % num_views]
                mask = masks_np[idx].astype(np.uint8)
                Image.fromarray(mask).save(
                    output_dir / f"sample_{sample_idx:03d}_t{t:02d}_view_{v}_mask.png"
                )
                colorize_mask(mask).save(
                    output_dir / f"sample_{sample_idx:03d}_t{t:02d}_view_{v}_mask_rgb.png"
                )

    print(f"Saved {num_samples} samples to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_paths",
        nargs="+",
        required=True,
        help="One or two experiment config.yaml paths (from experiment_cfg).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for saved samples.",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    for config_path_str in args.config_paths:
        config_path = Path(config_path_str)
        config = load_config(config_path)
        run_name = config.training.output_dir.split("/")[-1]
        out_dir = output_root / run_name
        sample_and_save(config, out_dir, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

