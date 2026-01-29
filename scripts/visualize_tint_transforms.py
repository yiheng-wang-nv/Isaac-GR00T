#!/usr/bin/env python3
"""Visualize random_tint and grayscale_tint transforms side by side."""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.model.gr00t_n1d6.image_augmentations import MaskedColorTransform
try:
    from gr00t.utils.video_utils import get_frames_by_indices
except ImportError:
    import cv2
    def get_frames_by_indices(video_path, indices):
        """Fallback video frame extraction."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.array(frames)


def load_random_frames(dataset_path, video_key, num_frames=4, seed=42):
    """Load random frames directly from video and mask files."""
    np.random.seed(seed)
    
    dataset_path = Path(dataset_path)
    
    # Load metadata
    meta_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(meta_path, "r") as f:
        for line in f:
            episodes.append(json.loads(line))
    
    # Find an episode with enough frames
    episode = None
    for _ in range(50):
        idx = np.random.randint(0, len(episodes))
        ep = episodes[idx]
        if ep.get("length", 0) >= num_frames:
            episode = ep
            break
    
    if episode is None:
        episode = episodes[0]
    
    episode_idx = episode["episode_index"]
    episode_length = episode["length"]
    print(f"Episode {episode_idx} has {episode_length} frames")
    
    # Evenly distributed frame indices (start, middle, end)
    if episode_length <= num_frames:
        frame_indices = list(range(episode_length))
    else:
        # Evenly sample across the video
        frame_indices = np.linspace(0, episode_length - 1, num_frames, dtype=int).tolist()
    
    # Load video frames - try multiple path patterns
    video_paths_to_try = [
        dataset_path / "videos" / "chunk-000" / f"observation.images.{video_key}" / f"episode_{episode_idx:06d}.mp4",
        dataset_path / "videos" / f"{video_key}_episode_{episode_idx:06d}.mp4",
        dataset_path / "videos" / f"episode_{episode_idx:06d}.mp4",
    ]
    
    video_path = None
    for vp in video_paths_to_try:
        if vp.exists():
            video_path = vp
            break
    
    if video_path is None:
        raise FileNotFoundError(f"Video not found. Tried: {[str(p) for p in video_paths_to_try]}")
    
    print(f"Loading video from: {video_path}")
    frames = get_frames_by_indices(str(video_path), frame_indices)
    
    # Load mask - try multiple path patterns
    mask_paths_to_try = [
        dataset_path / "masks" / "chunk-000" / f"observation.images.{video_key}" / f"episode_{episode_idx:06d}_masks.npz",
        dataset_path / "masks" / f"{video_key}_episode_{episode_idx:06d}.npz",
        dataset_path / "masks" / f"episode_{episode_idx:06d}.npz",
    ]
    
    masks = []
    mask_loaded = False
    for mask_path in mask_paths_to_try:
        if mask_path.exists():
            print(f"Loading mask from: {mask_path}")
            npz_data = np.load(mask_path)
            all_masks = npz_data["arr_0"] if "arr_0" in npz_data.files else npz_data[npz_data.files[0]]
            for fidx in frame_indices:
                masks.append(all_masks[fidx])
            mask_loaded = True
            break
    
    if not mask_loaded:
        print(f"No mask file found. Tried: {[str(p) for p in mask_paths_to_try]}")
        masks = [None] * len(frames)
    
    return list(frames), masks, frame_indices, episode_idx


def create_comparison_grid(frames, masks, transform, title, target_mask_values):
    """Create a comparison grid: original | transformed for each frame."""
    rows = []
    
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        # Original
        orig = frame.copy()
        
        # Apply transform
        if mask is not None:
            result = transform(image=frame, mask=mask)
            transformed = result["image"]
        else:
            transformed = frame.copy()
        
        # Concatenate horizontally: original | transformed
        row = np.concatenate([orig, transformed], axis=1)
        rows.append(row)
    
    # Stack all rows vertically
    grid = np.concatenate(rows, axis=0)
    
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./debug_tint_comparison")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--target_mask_values", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_key", type=str, default="ego_view")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data directly from files
    print(f"Loading dataset from {args.dataset_path}")
    frames, masks, frame_indices, episode_idx = load_random_frames(
        args.dataset_path, args.video_key, num_frames=args.num_frames, seed=args.seed
    )
    print(f"Loaded {len(frames)} frames from episode {episode_idx}, indices: {frame_indices}")
    
    # Create transforms for comparison
    # 1. Pure random_tint mode (grayscale_prob=0)
    random_tint_only = MaskedColorTransform(
        target_mask_values=args.target_mask_values,
        grayscale_prob=0.0,  # Always random_tint
        alpha_range=(0.0, 1.0),
        p=1.0,
    )
    
    # 2. Pure grayscale_tint mode (grayscale_prob=1)
    grayscale_tint_only = MaskedColorTransform(
        target_mask_values=args.target_mask_values,
        grayscale_prob=1.0,  # Always grayscale_tint
        alpha_range=(0.0, 1.0),
        p=1.0,
    )
    
    # 3. Mixed mode (50/50) - what you'll use in training
    mixed_transform = MaskedColorTransform(
        target_mask_values=args.target_mask_values,
        grayscale_prob=0.5,  # 50% grayscale, 50% random_tint
        alpha_range=(0.0, 1.0),
        p=1.0,
    )
    
    # Generate random_tint comparison
    print("Generating random_tint comparison (color overlay, partial texture)...")
    np.random.seed(args.seed)
    grid = create_comparison_grid(
        frames, masks, random_tint_only, 
        f"Random Tint (alpha 0-1)", 
        args.target_mask_values
    )
    out_path = output_dir / f"random_tint.png"
    Image.fromarray(grid).save(out_path)
    print(f"  Saved: {out_path}")
    
    # Generate grayscale_tint comparison
    print("Generating grayscale_tint comparison (preserves texture, changes color)...")
    np.random.seed(args.seed)
    grid = create_comparison_grid(
        frames, masks, grayscale_tint_only,
        f"Grayscale Tint",
        args.target_mask_values
    )
    out_path = output_dir / f"grayscale_tint.png"
    Image.fromarray(grid).save(out_path)
    print(f"  Saved: {out_path}")
    
    # Generate mixed comparison
    print("Generating mixed comparison (50% grayscale / 50% random_tint)...")
    np.random.seed(args.seed + 1000)
    grid = create_comparison_grid(
        frames, masks, mixed_transform,
        f"Mixed (50/50)",
        args.target_mask_values
    )
    out_path = output_dir / f"mixed_transform.png"
    Image.fromarray(grid).save(out_path)
    print(f"  Saved: {out_path}")
    
    print(f"\nDone! Check {output_dir} for results.")
    print("Each image shows: [Original | Transformed] for 4 frames stacked vertically")


if __name__ == "__main__":
    main()
