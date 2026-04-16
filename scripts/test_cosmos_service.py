#!/usr/bin/env python3
"""
Test client for scripts/cosmos_service.py.

Sends one or more real images to the running Cosmos ZMQ service and measures
end-to-end latency (including serialization but excluding model load time).

Prerequisites:
    Start the Cosmos service first (in its own terminal, from the Cosmos venv):
        cd third_party/cosmos-transfer2.5
        export HF_HOME=/workspace/code/Isaac-GR00T/third_party/cosmos-transfer2.5/cache
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../../scripts/cosmos_service.py --port 5557

Usage (from the Isaac-GR00T repo root, in the GR00T conda env):
    # Use bundled Cosmos sample image (no --image needed)
    python scripts/test_cosmos_service.py

    # Send a custom image or video (first frame used for video files)
    python scripts/test_cosmos_service.py --image path/to/image.jpg
    python scripts/test_cosmos_service.py --image path/to/clip.mp4

    # Run 5 timed inference calls (default: 3) and save the augmented output
    python scripts/test_cosmos_service.py --image path/to/image.jpg --runs 5 --save-output
    python scripts/test_cosmos_service.py --image path/to/image.jpg --save-output --save-dir /path/to/output

    # Connect to a remote or non-default-port service
    python scripts/test_cosmos_service.py --host 192.168.1.10 --port 5558

    # Override input resolution (default: 480x854)
    python scripts/test_cosmos_service.py --image path/to/image.jpg --height 720 --width 1280
"""

import argparse
import hashlib
import random
import struct
import time

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
import zmq


def load_image_as_tensor(path: str, height: int = 480, width: int = 854) -> torch.Tensor:
    """Load an image file and return float32 [1, C, H, W] tensor in [0, 1]."""
    img = tvio.read_image(path)  # uint8 [C, H, W]
    img = TF.resize(img, [height, width])
    return img.unsqueeze(0).float() / 255.0  # [1, C, H, W]


def load_video_first_frame(path: str, height: int = 480, width: int = 854) -> torch.Tensor:
    """Load the first frame of a video and return float32 [1, C, H, W] in [0, 1]."""
    frames, _, _ = tvio.read_video(path, start_pts=0, end_pts=0, pts_unit="sec")
    frame = frames[0].permute(2, 0, 1)  # [C, H, W]
    frame = TF.resize(frame, [height, width])
    return frame.unsqueeze(0).float() / 255.0  # [1, C, H, W]


def send_recv(sock: zmq.Socket, frames: torch.Tensor, seed: int) -> torch.Tensor:
    """Send frames to the service and return the augmented tensor."""
    T, C, H, W = frames.shape
    # Wire format: 20-byte header (5 × uint32: T, C, H, W, seed) + float32 payload
    msg = struct.pack("5I", T, C, H, W, seed) + frames.numpy().tobytes()
    sock.send(msg)
    reply = sock.recv()
    rT, rC, rH, rW = struct.unpack("4I", reply[:16])
    out = torch.from_numpy(
        np.frombuffer(reply[16:], dtype=np.float32).copy()
    ).reshape(rT, rC, rH, rW)
    return out


def main():
    parser = argparse.ArgumentParser(description="Test client for cosmos_service.py")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an image (.jpg/.png) or video (.mp4) to send. "
             "Defaults to the bundled Cosmos sample image.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of inference calls to time")
    parser.add_argument(
        "--height", type=int, default=480, help="Resize input to this height before sending"
    )
    parser.add_argument(
        "--width", type=int, default=854, help="Resize input to this width before sending"
    )
    parser.add_argument("--save-output", action="store_true", help="Save augmented image to disk")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/tmp",
        help="Directory to save augmented image when --save-output is set (default: /tmp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to pass to the Cosmos service. Defaults to a random uint32 (printed to stdout).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------------
    if args.image is None:
        # Fall back to the bundled Cosmos sample
        import os
        sample = os.path.join(
            os.path.dirname(__file__),
            "../third_party/cosmos-transfer2.5/assets/image_example/coastal_highway.mp4",
        )
        print(f"No --image specified, using bundled sample: {sample}")
        frames = load_video_first_frame(sample, args.height, args.width)
    elif args.image.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frames = load_image_as_tensor(args.image, args.height, args.width)
    else:
        frames = load_video_first_frame(args.image, args.height, args.width)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    T, C, H, W = frames.shape
    payload_mb = frames.numel() * 4 / 1024 / 1024
    print(f"Input:  shape={tuple(frames.shape)}  dtype={frames.dtype}  payload={payload_mb:.1f} MB")
    print(f"Seed:   {seed}{' (random)' if args.seed is None else ' (fixed)'}")
    print(f"Connecting to tcp://{args.host}:{args.port} ...")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")

    # ------------------------------------------------------------------
    # Warmup (not timed)
    # ------------------------------------------------------------------
    print("Warmup call ... ", end="", flush=True)
    t0 = time.perf_counter()
    out = send_recv(sock, frames, seed)
    warmup_s = time.perf_counter() - t0
    print(f"{warmup_s:.2f}s")

    # ------------------------------------------------------------------
    # Timed runs
    # ------------------------------------------------------------------
    latencies = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = send_recv(sock, frames, seed)
        latencies.append(time.perf_counter() - t0)
        print(f"  run {i+1}/{args.runs}: {latencies[-1]:.3f}s")

    print(f"\nLatency over {args.runs} run(s):")
    print(f"  min  {min(latencies):.3f}s")
    print(f"  mean {sum(latencies)/len(latencies):.3f}s")
    print(f"  max  {max(latencies):.3f}s")
    print(f"\nOutput: shape={tuple(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")

    # ------------------------------------------------------------------
    # Verify output integrity
    # ------------------------------------------------------------------
    in_hash  = hashlib.md5(frames.numpy().tobytes()).hexdigest()[:8]
    out_hash = hashlib.md5(out.numpy().tobytes()).hexdigest()[:8]
    same = in_hash == out_hash
    print(f"Input hash: {in_hash}  Output hash: {out_hash}  {'(UNCHANGED — service echoed input?)' if same else '(different ✓)'}")

    # ------------------------------------------------------------------
    # Optionally save output
    # ------------------------------------------------------------------
    if args.save_output:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "cosmos_test_out.jpg")
        out_uint8 = (out[0] * 255).to(torch.uint8)  # [C, H, W]
        tvio.write_jpeg(out_uint8, out_path)
        print(f"Saved augmented image to {out_path}")


if __name__ == "__main__":
    main()
