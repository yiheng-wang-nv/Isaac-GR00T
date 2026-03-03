#!/usr/bin/env python3
"""
Dump transformed samples from the GR00T dataloader pipeline for visualization.
Supports loading from a saved experiment config OR building config from CLI args.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from PIL import ImageOps

from gr00t.configs.base_config import Config, get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
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


def parse_color_jitter_params(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) % 2 != 0:
        raise ValueError("color_jitter_params must be key value pairs.")
    params: dict[str, float] = {}
    for key, value in zip(values[0::2], values[1::2]):
        params[key] = float(value)
    return params


def parse_embodiment_tag(tag: str) -> EmbodimentTag:
    try:
        return EmbodimentTag[tag]
    except KeyError:
        pass
    for item in EmbodimentTag:
        if item.value == tag:
            return item
    raise ValueError(f"Unknown embodiment tag: {tag}")


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


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


def build_config_from_args(args: argparse.Namespace) -> Config:
    if args.modality_config_path is not None:
        load_modality_config(args.modality_config_path)

    embodiment_tag = parse_embodiment_tag(args.embodiment_tag).value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": args.dataset_path,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.random_rotation_angle = args.random_rotation_angle
    config.model.color_jitter_params = parse_color_jitter_params(args.color_jitter_params)
    config.model.background_noise_on_mask = args.background_noise_on_mask
    if args.max_state_dim is not None:
        config.model.max_state_dim = args.max_state_dim
    if args.max_action_dim is not None:
        config.model.max_action_dim = args.max_action_dim
    if args.use_albumentations_transforms is not None:
        config.model.use_albumentations_transforms = args.use_albumentations_transforms
    if config.model.background_noise_on_mask:
        config.model.use_albumentations_transforms = True

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    return config


def infer_dims_from_stats(dataset_paths: list[str]) -> tuple[int | None, int | None]:
    max_state_dim = None
    max_action_dim = None
    for dataset_path in dataset_paths:
        stats_path = Path(dataset_path) / "meta" / "stats.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        if "observation.state" in stats and "mean" in stats["observation.state"]:
            dim = len(stats["observation.state"]["mean"])
            max_state_dim = dim if max_state_dim is None else max(max_state_dim, dim)
        if "action" in stats and "mean" in stats["action"]:
            dim = len(stats["action"]["mean"])
            max_action_dim = dim if max_action_dim is None else max(max_action_dim, dim)
    return max_state_dim, max_action_dim


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


def _resize_to_match(original: Image.Image, target: Image.Image) -> Image.Image:
    if original.size == target.size:
        return original
    return original.resize(target.size, resample=Image.BILINEAR)


def _stack_rows(rows: list[Image.Image]) -> Image.Image:
    widths = [row.width for row in rows]
    heights = [row.height for row in rows]
    out = Image.new("RGB", (max(widths), sum(heights)), (0, 0, 0))
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def _compare_original_vs_transformed(
    originals: list[Image.Image], transformed: list[Image.Image]
) -> Image.Image:
    rows = []
    for orig, trans in zip(originals, transformed):
        if not isinstance(orig, Image.Image):
            orig = Image.fromarray(np.asarray(orig))
        if not isinstance(trans, Image.Image):
            trans = Image.fromarray(np.asarray(trans))
        orig = ImageOps.exif_transpose(orig)
        trans = ImageOps.exif_transpose(trans)
        orig = _resize_to_match(orig, trans)
        row = Image.new("RGB", (orig.width + trans.width, max(orig.height, trans.height)), (0, 0, 0))
        row.paste(orig, (0, 0))
        row.paste(trans, (orig.width, 0))
        rows.append(row)
    return _stack_rows(rows)


def save_comparison_per_dataset(
    config: Config,
    output_dir: Path,
    episode_index: int,
) -> None:
    processor = build_processor(config)
    processor.train()

    for dataset_spec in config.data.datasets:
        dataset_paths = dataset_spec.dataset_paths
        embodiment_tag = EmbodimentTag(dataset_spec.embodiment_tag)
        modality_cfg = config.data.modality_configs[embodiment_tag.value]
        image_keys = modality_cfg["video"].modality_keys
        if not image_keys:
            continue
        view = image_keys[0]

        for dataset_path in dataset_paths:
            loader = LeRobotEpisodeLoader(
                dataset_path=dataset_path,
                modality_configs=modality_cfg,
                video_backend=config.data.video_backend,
            )
            df = loader[episode_index]
            if df.empty:
                continue
            length = len(df)
            indices = [0, length // 2, max(0, length - 1)]
            original_images = [df[f"video.{view}"].iloc[i] for i in indices]
            view_masks = None
            mask_col = f"mask.{view}"
            if mask_col in df.columns:
                view_masks = [df[mask_col].iloc[i] for i in indices]

            images_dict = {view: original_images}
            masks_dict = {view: view_masks} if view_masks is not None else None

            vlm_inputs, _ = processor._get_vlm_inputs(
                image_keys=[view],
                images=images_dict,
                masks=masks_dict,
                image_transform=processor.train_image_transform,
                language="",
            )
            transformed_images = vlm_inputs["vlm_content"]["images"]

            comparison = _compare_original_vs_transformed(original_images, transformed_images)
            dataset_path_obj = Path(dataset_path)
            dataset_name = f"{dataset_path_obj.parent.name}_{dataset_path_obj.name}"
            out_path = output_dir / f"{dataset_name}_first_mid_last_comparison.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.save(out_path)
            print(f"Saved comparison image to {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_paths",
        nargs="+",
        help="One or two experiment config.yaml paths (from experiment_cfg).",
    )
    parser.add_argument(
        "--dataset_path",
        nargs="+",
        help="Dataset roots (used when config_paths not provided).",
    )
    parser.add_argument(
        "--embodiment_tag",
        help="Embodiment tag (e.g., NEW_EMBODIMENT) when config_paths not provided.",
    )
    parser.add_argument(
        "--modality_config_path",
        default=None,
        help="Optional modality config .py path.",
    )
    parser.add_argument("--random_rotation_angle", type=int, default=None)
    parser.add_argument(
        "--color_jitter_params",
        nargs="*",
        default=None,
        help="Key-value pairs like: brightness 0.3 contrast 0.4",
    )
    parser.add_argument(
        "--use_albumentations_transforms",
        action="store_true",
        default=None,
        help="Force enabling albumentations transforms.",
    )
    parser.add_argument(
        "--background_noise_on_mask",
        action="store_true",
        help="Replace background (mask==0) with random noise.",
    )
    parser.add_argument("--max_state_dim", type=int, default=None)
    parser.add_argument("--max_action_dim", type=int, default=None)
    parser.add_argument(
        "--comparison_per_dataset",
        action="store_true",
        help="Save a single comparison image per dataset (first/mid/last frames).",
    )
    parser.add_argument(
        "--comparison_episode_index",
        type=int,
        default=0,
        help="Episode index used for comparison output.",
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
    if args.config_paths:
        for config_path_str in args.config_paths:
            config_path = Path(config_path_str)
            config = load_config(config_path)
            run_name = config.training.output_dir.split("/")[-1]
            out_dir = output_root / run_name
            if args.comparison_per_dataset:
                save_comparison_per_dataset(
                    config, out_dir, episode_index=args.comparison_episode_index
                )
                continue
            sample_and_save(config, out_dir, args.num_samples, args.seed)
        return

    if not args.dataset_path or not args.embodiment_tag:
        raise ValueError("Provide --config_paths or both --dataset_path and --embodiment_tag.")

    if args.max_state_dim is None or args.max_action_dim is None:
        inferred_state_dim, inferred_action_dim = infer_dims_from_stats(args.dataset_path)
        if args.max_state_dim is None and inferred_state_dim is not None:
            args.max_state_dim = inferred_state_dim
        if args.max_action_dim is None and inferred_action_dim is not None:
            args.max_action_dim = inferred_action_dim

    config = build_config_from_args(args)
    run_name = "preview_from_args"
    out_dir = output_root / run_name
    if args.comparison_per_dataset:
        save_comparison_per_dataset(
            config, out_dir, episode_index=args.comparison_episode_index
        )
        return
    sample_and_save(config, out_dir, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

