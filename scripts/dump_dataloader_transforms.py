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
import imageio.v2 as imageio
import cv2

import albumentations as A

from gr00t.configs.base_config import Config, get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.gr00t_n1d6.image_augmentations import (
    BackgroundNoiseTransform,
    MaskedColorTransform,
)


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


def build_mask_only_transforms(extra_augmentation_config: dict):
    """Build transforms that only apply mask-based augmentations (no resize/crop)."""
    mask_transforms = []
    
    # Background noise on mask (replaces mask==0 with random noise)
    bg_noise = extra_augmentation_config.get("background_noise_on_mask")
    if bg_noise:
        p = bg_noise if isinstance(bg_noise, (int, float)) else 1.0
        mask_transforms.append(BackgroundNoiseTransform(p=float(p)))
    
    # Masked region transforms
    for transform_cfg in extra_augmentation_config.get("masked_region_transforms", []):
        target_mask_values = transform_cfg.get("target_mask_values", [])
        p = transform_cfg.get("p", 0.5)
        grayscale_prob = transform_cfg.get("grayscale_prob", 0.5)
        alpha_range = tuple(transform_cfg.get("alpha_range", [0.3, 1.0]))
        
        mask_transforms.append(
            MaskedColorTransform(
                target_mask_values=target_mask_values,
                grayscale_prob=grayscale_prob,
                alpha_range=alpha_range,
                p=p,
            )
        )
    
    return mask_transforms


def apply_mask_only_transforms(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None,
    mask_transforms: list,
) -> list[np.ndarray]:
    """Apply only mask-based transforms without resize/crop."""
    results = []
    for idx, img in enumerate(images):
        img_array = np.asarray(img)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        mask = masks[idx] if masks is not None else None
        if mask is not None:
            mask_array = np.asarray(mask)
        else:
            mask_array = None
        
        # Apply each mask transform
        for transform in mask_transforms:
            # Check if transform should be applied (based on p)
            if np.random.random() < transform.p:
                img_array = transform.apply(img_array, mask=mask_array)
        
        results.append(img_array)
    
    return results


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
        extra_augmentation_config=config.model.extra_augmentation_config,
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
    
    # Parse extra_augmentation_config from JSON string
    import json
    if args.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(args.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None
    
    if args.max_state_dim is not None:
        config.model.max_state_dim = args.max_state_dim
    if args.max_action_dim is not None:
        config.model.max_action_dim = args.max_action_dim
    if args.use_albumentations_transforms is not None:
        config.model.use_albumentations_transforms = args.use_albumentations_transforms
    if config.model.extra_augmentation_config:
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


def _create_grid_comparison(
    originals: list[Image.Image | np.ndarray],
    transformed: list[Image.Image | np.ndarray],
    grid_size: int = 3,
    cell_width: int | None = None,
) -> Image.Image:
    """Create a grid comparison image (e.g., 3x3) with original|transformed pairs.
    
    Args:
        originals: List of original images
        transformed: List of transformed images
        grid_size: Number of rows and columns (default 3 for 3x3 = 9 cells)
        cell_width: Width for each cell (original + transformed). If None, auto-compute.
    
    Returns:
        Grid image with original on left, transformed on right for each cell
    """
    n_cells = grid_size * grid_size
    
    # Convert all to PIL Images
    def to_pil(img):
        if isinstance(img, Image.Image):
            return ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    
    orig_pils = [to_pil(img) for img in originals[:n_cells]]
    trans_pils = [to_pil(img) for img in transformed[:n_cells]]
    
    # Determine cell size
    if cell_width is None:
        # Use first transformed image size as reference
        ref_w, ref_h = trans_pils[0].size
        cell_width = ref_w * 2  # original + transformed side by side
        cell_height = ref_h
    else:
        cell_height = int(cell_width * trans_pils[0].height / (trans_pils[0].width * 2))
    
    single_w = cell_width // 2
    single_h = cell_height
    
    # Create the grid
    grid_w = cell_width * grid_size
    grid_h = cell_height * grid_size
    grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    
    for idx in range(min(len(orig_pils), n_cells)):
        row = idx // grid_size
        col = idx % grid_size
        
        orig = orig_pils[idx].resize((single_w, single_h), Image.BILINEAR)
        trans = trans_pils[idx].resize((single_w, single_h), Image.BILINEAR)
        
        x = col * cell_width
        y = row * cell_height
        
        grid_img.paste(orig, (x, y))
        grid_img.paste(trans, (x + single_w, y))
    
    return grid_img


def save_grid_comparison(
    config: Config,
    output_dir: Path,
    dataset_index: int,
    episode_index: int,
    view: str | None,
    num_frames: int,
    output_name: str | None,
    no_resize: bool = False,
) -> None:
    """Sample frames uniformly and save a grid comparison image."""
    dataset_spec = config.data.datasets[0]
    dataset_paths = dataset_spec.dataset_paths
    if dataset_index < 0 or dataset_index >= len(dataset_paths):
        raise ValueError(f"dataset_index {dataset_index} out of range (0..{len(dataset_paths)-1})")
    dataset_path = dataset_paths[dataset_index]

    embodiment_tag = EmbodimentTag(dataset_spec.embodiment_tag)
    modality_cfg = config.data.modality_configs[embodiment_tag.value]
    image_keys = modality_cfg["video"].modality_keys
    if not image_keys:
        raise ValueError("No video modality keys found.")
    view = view or image_keys[0]
    if view not in image_keys:
        raise ValueError(f"View '{view}' not in modality config: {image_keys}")

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_cfg,
        video_backend=config.data.video_backend,
    )
    df = loader[episode_index]
    if df.empty:
        raise ValueError(f"Episode {episode_index} is empty.")
    
    total_frames = len(df)
    # Sample uniformly
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    original_images = [df[f"video.{view}"].iloc[i] for i in indices]
    view_masks = None
    mask_col = f"mask.{view}"
    if mask_col in df.columns:
        view_masks = [df[mask_col].iloc[i] for i in indices]

    if no_resize:
        # Only apply mask-based transforms without resize/crop
        extra_aug_config = config.model.extra_augmentation_config or {}
        if not extra_aug_config:
            print("Warning: --no_resize specified but no extra_augmentation_config provided.")
            transformed_images = original_images
        else:
            mask_transforms = build_mask_only_transforms(extra_aug_config)
            transformed_images = apply_mask_only_transforms(
                original_images, view_masks, mask_transforms
            )
    else:
        processor = build_processor(config)
        processor.train()
        
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
    
    # Determine grid size (ceil of sqrt)
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    
    grid_img = _create_grid_comparison(original_images, transformed_images, grid_size=grid_size)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path_obj = Path(dataset_path)
    dataset_name = f"{dataset_path_obj.parent.name}_{dataset_path_obj.name}"
    view_tag = view.replace(".", "_")
    default_name = f"{dataset_name}_episode_{episode_index:06d}_{view_tag}_grid_{num_frames}frames.png"
    output_path = output_dir / (output_name or default_name)
    grid_img.save(output_path)
    print(f"Saved grid comparison ({grid_size}x{grid_size}, {num_frames} frames) to {output_path.resolve()}")


def _write_video(
    frames: list[Image.Image],
    output_path: Path,
    fps: int,
    side_by_side: bool = False,
    originals: list[Image.Image] | None = None,
    codec: str = "libx264",
) -> None:
    def to_rgb_array(item: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(item, Image.Image):
            item = ImageOps.exif_transpose(item).convert("RGB")
            arr = np.asarray(item)
        else:
            if isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
            else:
                arr = np.asarray(item)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                try:
                    arr = np.asarray(Image.fromarray(arr.squeeze()).convert("RGB"))
                except Exception:
                    if arr.ndim == 1 and arr.size % 3 == 0:
                        arr = arr.reshape(-1, 3)
                        arr = np.stack([arr] * 3, axis=-1)
                    else:
                        raise
        arr = np.ascontiguousarray(arr)
        return arr

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get first frame to determine video size
    first_arr = to_rgb_array(frames[0])
    if side_by_side and originals:
        first_orig = to_rgb_array(originals[0])
        first_orig_img = Image.fromarray(first_orig)
        first_img = Image.fromarray(first_arr)
        first_orig_img = _resize_to_match(first_orig_img, first_img)
        width = first_orig_img.width + first_img.width
        height = max(first_orig_img.height, first_img.height)
    else:
        height, width = first_arr.shape[:2]

    # Use cv2 for more reliable video writing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx, frame in enumerate(frames):
        frame_arr = to_rgb_array(frame)
        if side_by_side:
            if originals is None or idx >= len(originals):
                continue
            orig_arr = to_rgb_array(originals[idx])
            orig_img = Image.fromarray(orig_arr)
            frame_img = Image.fromarray(frame_arr)
            orig_img = _resize_to_match(orig_img, frame_img)
            combined = Image.new(
                "RGB",
                (orig_img.width + frame_img.width, max(orig_img.height, frame_img.height)),
                (0, 0, 0),
            )
            combined.paste(orig_img, (0, 0))
            combined.paste(frame_img, (orig_img.width, 0))
            frame_arr = np.asarray(combined)
        # cv2 expects BGR, convert from RGB
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def save_video_from_episode(
    config: Config,
    output_dir: Path,
    dataset_index: int,
    episode_index: int,
    view: str | None,
    fps: int,
    side_by_side: bool,
    output_name: str | None,
    no_resize: bool = False,
) -> None:
    dataset_spec = config.data.datasets[0]
    dataset_paths = dataset_spec.dataset_paths
    if dataset_index < 0 or dataset_index >= len(dataset_paths):
        raise ValueError(f"dataset_index {dataset_index} out of range (0..{len(dataset_paths)-1})")
    dataset_path = dataset_paths[dataset_index]

    embodiment_tag = EmbodimentTag(dataset_spec.embodiment_tag)
    modality_cfg = config.data.modality_configs[embodiment_tag.value]
    image_keys = modality_cfg["video"].modality_keys
    if not image_keys:
        raise ValueError("No video modality keys found.")
    view = view or image_keys[0]
    if view not in image_keys:
        raise ValueError(f"View '{view}' not in modality config: {image_keys}")

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_cfg,
        video_backend=config.data.video_backend,
    )
    df = loader[episode_index]
    if df.empty:
        raise ValueError(f"Episode {episode_index} is empty.")
    original_images = list(df[f"video.{view}"].values)
    view_masks = None
    mask_col = f"mask.{view}"
    if mask_col in df.columns:
        view_masks = list(df[mask_col].values)

    if no_resize:
        # Only apply mask-based transforms without resize/crop
        extra_aug_config = config.model.extra_augmentation_config or {}
        if not extra_aug_config:
            print("Warning: --no_resize specified but no extra_augmentation_config provided. Using original images.")
            transformed_images = original_images
        else:
            mask_transforms = build_mask_only_transforms(extra_aug_config)
            transformed_images = apply_mask_only_transforms(
                original_images, view_masks, mask_transforms
            )
        print(f"Processing {len(transformed_images)} frames (mask-only transforms, no resize)...")
    else:
        processor = build_processor(config)
        processor.train()
        
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
        print(f"Processing {len(transformed_images)} frames...")

    dataset_path_obj = Path(dataset_path)
    dataset_name = f"{dataset_path_obj.parent.name}_{dataset_path_obj.name}"
    view_tag = view.replace(".", "_")
    default_name = f"{dataset_name}_episode_{episode_index:06d}_{view_tag}_noised.mp4"
    output_path = output_dir / (output_name or default_name)
    _write_video(
        frames=transformed_images,
        output_path=output_path,
        fps=fps,
        side_by_side=side_by_side,
        originals=original_images if side_by_side else None,
    )
    print(f"Saved video to {output_path.resolve()}")


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
        "--extra_augmentation_config",
        type=str,
        default=None,
        help=(
            "JSON config for extra augmentations. Example: "
            '\'{"background_noise_on_mask": true, "masked_region_transforms": '
            '[{"type": "hue_shift", "target_mask_values": [5], "p": 0.5}]}\''
        ),
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
    parser.add_argument("--save_video", action="store_true", help="Save a full transformed video.")
    parser.add_argument("--video_dataset_index", type=int, default=0, help="Dataset index for video.")
    parser.add_argument("--video_episode_index", type=int, default=0, help="Episode index for video.")
    parser.add_argument("--video_view", type=str, default=None, help="Camera view for video.")
    parser.add_argument("--video_fps", type=int, default=20, help="FPS for output video.")
    parser.add_argument(
        "--video_side_by_side",
        action="store_true",
        help="Output side-by-side original vs transformed video.",
    )
    parser.add_argument("--video_output_name", type=str, default=None, help="Output video file name.")
    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Only apply mask-based transforms (noise, color change) without resize/crop. Keeps original resolution.",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save a grid comparison image with uniformly sampled frames.",
    )
    parser.add_argument(
        "--grid_num_frames",
        type=int,
        default=9,
        help="Number of frames to sample for grid comparison (default: 9 for 3x3 grid).",
    )
    parser.add_argument(
        "--grid_output_name",
        type=str,
        default=None,
        help="Output filename for grid comparison image.",
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
            if args.save_video:
                save_video_from_episode(
                    config=config,
                    output_dir=out_dir,
                    dataset_index=args.video_dataset_index,
                    episode_index=args.video_episode_index,
                    view=args.video_view,
                    fps=args.video_fps,
                    side_by_side=args.video_side_by_side,
                    output_name=args.video_output_name,
                    no_resize=args.no_resize,
                )
            if args.save_grid:
                save_grid_comparison(
                    config=config,
                    output_dir=out_dir,
                    dataset_index=args.video_dataset_index,
                    episode_index=args.video_episode_index,
                    view=args.video_view,
                    num_frames=args.grid_num_frames,
                    output_name=args.grid_output_name,
                    no_resize=args.no_resize,
                )
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
    if args.save_video:
        save_video_from_episode(
            config=config,
            output_dir=out_dir,
            dataset_index=args.video_dataset_index,
            episode_index=args.video_episode_index,
            view=args.video_view,
            fps=args.video_fps,
            side_by_side=args.video_side_by_side,
            output_name=args.video_output_name,
            no_resize=args.no_resize,
        )
    if args.save_grid:
        save_grid_comparison(
            config=config,
            output_dir=out_dir,
            dataset_index=args.video_dataset_index,
            episode_index=args.video_episode_index,
            view=args.video_view,
            num_frames=args.grid_num_frames,
            output_name=args.grid_output_name,
            no_resize=args.no_resize,
        )
    if args.comparison_per_dataset:
        save_comparison_per_dataset(
            config, out_dir, episode_index=args.comparison_episode_index
        )
        return
    sample_and_save(config, out_dir, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

