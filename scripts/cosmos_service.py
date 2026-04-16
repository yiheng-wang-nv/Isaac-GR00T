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

"""
Cosmos-Transfer2.5 image2image ZMQ service for GR00T data augmentation.

This script runs inside the Cosmos .venv (incompatible with GR00T's conda env).
Launch one process per GPU; GR00T DataLoader workers connect to the services
round-robin via VideoCosmosAugmentTransform.

Usage:
    cd third_party/cosmos-transfer2.5
    export HF_HOME=/workspace/code/cosmos-transfer2.5-gr00t/cache
    export HF_TOKEN="$HF_TOKEN"
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ../../scripts/cosmos_service.py --port 5557 --control-type edge --num-steps 10 --guidance 3

    # For multiple GPUs:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../../scripts/cosmos_service.py --port 5557 &
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python ../../scripts/cosmos_service.py --port 5558 &
"""

import argparse
import struct
import tempfile
from pathlib import Path

# Log appears immediately when stdout is a file (long import phase otherwise looks "empty").
print("cosmos_service: importing torch / Cosmos (can take minutes)...", flush=True)

import numpy as np
import torch
import torchvision.io as tvio
import zmq
from cosmos_transfer2.config import SetupArguments
from cosmos_transfer2.inference import Control2WorldInference

DEFAULT_ROBOTICS_PROMPT = "A four-panel multi-view robotics scene. \
Top-left is left_wrist: close wrist-camera view of a gripper holding a small white cylindrical part. \
Top-right is right_wrist: wrist-camera view of the other gripper guiding the part toward the target area. \
Bottom-left is head: wider head-camera view of both robot arms coordinating over the workspace. \
Bottom-right is blank."

DEFAULT_PROMPT = "A scenic drive unfolds along a coastal highway. The video captures a smooth, continuous journey along a multi-lane road, with the camera positioned as if from the perspective of a vehicle traveling in the right lane. The road is bordered by a tall, green mountain on the right, which casts a shadow over part of the highway, while the left side opens up to a view of the ocean, visible in the distance beyond a row of low-lying vegetation and a sidewalk. Several vehicles, including two red vehicles, travel ahead, maintaining a steady pace. The road is well-maintained, with clear white lane markings and a concrete barrier separating the lanes from the mountain covered by trees on the right. Utility poles and power lines run parallel to the road on the left, adding to the infrastructure of the scene. The camera remains static, providing a consistent view of the road and surroundings, emphasizing the serene and uninterrupted nature of the drive."
DEFAULT_NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos-Transfer2.5 ZMQ inference service")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ REP port to bind")
    parser.add_argument(
        "--control-type",
        type=str,
        default="edge",
        choices=["edge", "depth", "seg", "vis"],
        help="Cosmos control modality",
    )
    parser.add_argument("--num-steps", type=int, default=5, help="Diffusion steps (5=~1.15s, 10=~2s)")
    parser.add_argument("--guidance", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for generation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/cosmos_service_out",
        help="Temp output dir for Cosmos internals",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading Cosmos-Transfer2.5 ({args.control_type}, {args.num_steps} steps)...")
    setup_args = SetupArguments(
        output_dir=args.output_dir,
        model=args.control_type,
    )
    engine = Control2WorldInference(
        args=setup_args,
        batch_hint_keys=[args.control_type],
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"Cosmos service ready on port {args.port}")

    while True:
        # Wire format: 20-byte header (5 x uint32: T, C, H, W, seed) + float32 payload
        msg = sock.recv()
        T, C, H, W, seed = struct.unpack("5I", msg[:20])
        frames_t = torch.from_numpy(
            np.frombuffer(msg[20:], dtype=np.float32).copy()
        ).reshape(T, C, H, W)  # float32 [T, C, H, W] in [0, 1]
        print(f"Received frames: {frames_t.shape}")

        # Write to a temp mp4 for Cosmos file-based API
        frames_uint8 = (frames_t.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        tvio.write_video(tmp_path, frames_uint8, fps=30)

        try:
            output, _, _, _, _ = engine.inference_pipeline.generate_img2world(
                prompt=DEFAULT_ROBOTICS_PROMPT, #args.prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                video_path=tmp_path,
                max_frames=1, # for image inference, we only need to pass in the first frame
                num_video_frames_per_chunk=1, # for image inference, we only need to pass in the first frame
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=seed,
                keep_input_resolution=True,
                resolution="480",
                hint_key = [f"{args.control_type}"],
                input_control_video_paths={f"{args.control_type}": None},
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # output: (1, C, T, H, W) float32 in [-1, 1]
        out = output[0].permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        out = (out * 0.5 + 0.5).clamp(0.0, 1.0).float()   # [0, 1]
        # Guard: resize if Cosmos changed resolution despite keep_input_resolution=True
        if out.shape[-2:] != (H, W):
            import torchvision.transforms.functional as TF
            out = TF.resize(out, [H, W])

        rT, rC, rH, rW = out.shape
        sock.send(struct.pack("4I", rT, rC, rH, rW) + out.numpy().tobytes())


if __name__ == "__main__":
    main()
