# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualise ChangeBackgroundTransform on random frames from a dataset.

Produces one 3x3 collage per camera view (9 samples each) with a fresh random
background template per (sample, view). Bundles the three collages into a zip.

Run:
    python scripts/viz_bg_transform.py
"""

import argparse
import copy
import importlib
import os
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.transform.video import ChangeBackgroundTransform

# Make the user's config modules importable (they live at the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


VIDEO_KEYS = ["video.room_view", "video.left_wrist_view", "video.right_wrist_view"]


def _resize(img: np.ndarray, target_w: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = int(h * target_w / w)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _make_grid(tiles: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """Arrange len(rows*cols) equal-shape tiles into a `rows`x`cols` grid."""
    assert len(tiles) == rows * cols
    row_imgs = [
        np.concatenate(tiles[r * cols : (r + 1) * cols], axis=1) for r in range(rows)
    ]
    return np.concatenate(row_imgs, axis=0)


def _resolve_bg_transform_from_config(data_config_spec: str) -> ChangeBackgroundTransform:
    """Instantiate the data config and return the ChangeBackgroundTransform used in training."""
    module_name, class_name = data_config_spec.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    cfg = cls()
    composed = cfg.transform()
    bg_tfs = [
        t for t in composed.transforms if isinstance(t, ChangeBackgroundTransform)
    ]
    if not bg_tfs:
        raise RuntimeError(
            f"{data_config_spec}.transform() has no ChangeBackgroundTransform"
        )
    return bg_tfs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="/localhome/local-vennw/data/trocar_parallel_combined_success",
    )
    parser.add_argument(
        "--data-config",
        default="gr00t_mask_config:UnitreeG1SimMaskDataConfig",
        help="Training data config in module:Class form. The ChangeBackgroundTransform "
        "actually used during finetuning is pulled from this config so the viz matches "
        "the real training pipeline.",
    )
    parser.add_argument("--out-dir", default="/localhome/local-vennw/code/bg_viz")
    parser.add_argument("--zip-path", default="/localhome/local-vennw/code/bg_viz.zip")
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-apply",
        action="store_true",
        default=True,
        help="Override transform p=1.0 for viz (every sample swapped). Default True.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Pull the actual transform instance used in training, so we honor its
    # template_folder, target_mask_values, feather_radius, apply_to, etc.
    bg_transform = _resolve_bg_transform_from_config(args.data_config)
    orig_p = bg_transform.p
    if args.force_apply:
        bg_transform.p = 1.0
    # Make sure it's in training mode — ChangeBackgroundTransform is a no-op in eval.
    bg_transform.train()
    print(
        f"Pulled transform from {args.data_config}: "
        f"apply_to={bg_transform.apply_to} targets={bg_transform.target_mask_values} "
        f"p={orig_p}->{bg_transform.p} feather={bg_transform.feather_radius} "
        f"templates={len(bg_transform._template_images)}"
    )

    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
    }
    ds = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord",
    )
    print(f"Dataset loaded: {len(ds)} steps across {len(ds.trajectory_ids)} episodes")
    assert ds._has_masks, "Dataset has no masks/ directory"

    step_indices = rng.sample(range(len(ds)), 9)
    picks = [ds.all_steps[i] for i in step_indices]
    print("picked samples:", picks)

    swapped_samples = []
    for traj_id, base_idx in picks:
        raw = ds.get_step_data(traj_id, base_idx)
        swapped_samples.append(bg_transform.apply(copy.deepcopy(raw)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view_pngs: list[Path] = []
    for video_key in VIDEO_KEYS:
        view_name = video_key.replace("video.", "")
        tiles = [
            _resize(sample[video_key][0], args.panel_width) for sample in swapped_samples
        ]
        grid = _make_grid(tiles, rows=3, cols=3)
        out_path = out_dir / f"{view_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        view_pngs.append(out_path)
        print(f"saved: {out_path} shape={grid.shape}")

    # Per-sample view comparison: each row is one sample with all three views
    # side by side, so you can eyeball whether the three views independently
    # sampled different templates.
    per_sample_tiles: list[np.ndarray] = []
    for sample in swapped_samples:
        for video_key in VIDEO_KEYS:
            per_sample_tiles.append(_resize(sample[video_key][0], args.panel_width))
    per_sample_grid = _make_grid(per_sample_tiles, rows=len(swapped_samples), cols=len(VIDEO_KEYS))
    per_sample_path = out_dir / "per_sample_3views.png"
    cv2.imwrite(str(per_sample_path), cv2.cvtColor(per_sample_grid, cv2.COLOR_RGB2BGR))
    view_pngs.append(per_sample_path)
    print(f"saved: {per_sample_path} shape={per_sample_grid.shape}")

    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in view_pngs:
            zf.write(p, arcname=p.name)
    print(f"zip: {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
