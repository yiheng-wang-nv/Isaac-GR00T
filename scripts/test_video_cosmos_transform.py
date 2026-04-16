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
"""Tests for VideoCosmosAugmentTransform.

Runs fully offline — no GPU or Cosmos service required.  ZMQ sockets are mocked
throughout so these tests exercise cache logic, wire format, probability gating,
and port-selection without any external dependencies.

Run with:
    pytest -v --color=yes scripts/test_video_cosmos_transform.py
"""

import hashlib
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatistics,
    DatasetStatisticalValues,
    StateActionMetadata,
    VideoMetadata,
)
from gr00t.data.transform.video import VideoCosmosAugmentTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frames(T: int = 4, C: int = 3, H: int = 8, W: int = 8) -> torch.Tensor:
    """Return a [T, C, H, W] float32 tensor with values in [0, 1]."""
    return torch.rand(T, C, H, W, dtype=torch.float32)


def _make_zmq_reply(tensor: torch.Tensor) -> bytes:
    """Encode a tensor into the ZMQ wire format used by cosmos_service.py."""
    T, C, H, W = tensor.shape
    return struct.pack("4I", T, C, H, W) + tensor.numpy().tobytes()


def _content_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DUMMY_STAT = DatasetStatisticalValues(
    max=np.array([1.0]),
    min=np.array([0.0]),
    mean=np.array([0.5]),
    std=np.array([0.1]),
    q01=np.array([0.01]),
    q99=np.array([0.99]),
)

_STATE_ACTION_META = StateActionMetadata(
    absolute=True,
    rotation_type=None,
    shape=(7,),
    continuous=True,
)


@pytest.fixture
def dataset_metadata():
    return DatasetMetadata(
        statistics=DatasetStatistics(
            state={"arm": _DUMMY_STAT},
            action={"arm": _DUMMY_STAT},
        ),
        modalities=DatasetModalities(
            video={
                "ego_view": VideoMetadata(resolution=(8, 8), channels=3, fps=30.0),
                "left_wrist_view": VideoMetadata(resolution=(8, 8), channels=3, fps=30.0),
                "right_wrist_view": VideoMetadata(resolution=(8, 8), channels=3, fps=30.0),
            },
            state={"arm": _STATE_ACTION_META},
            action={"arm": _STATE_ACTION_META},
        ),
        embodiment_tag=EmbodimentTag.GR1,
    )


@pytest.fixture
def transform_factory(dataset_metadata):
    """Return a factory that builds a configured VideoCosmosAugmentTransform."""

    def factory(
        cache_dir: str,
        probability: float = 0.5,
        apply_to: list[str] | None = None,
        ports: list[int] | None = None,
        seed: int | None = None,
        grid_mode: bool = False,
    ) -> VideoCosmosAugmentTransform:
        t = VideoCosmosAugmentTransform(
            apply_to=apply_to or ["video.ego_view"],
            cache_dir=cache_dir,
            host="localhost",
            ports=ports or [5557],
            probability=probability,
            seed=seed,
            grid_mode=grid_mode,
        )
        t.set_metadata(dataset_metadata)
        return t

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_mode_skip(transform_factory):
    """In eval mode, apply() returns the data dict unchanged and never connects."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=1.0)
        t.eval()

        frames = _make_frames()
        data = {"video.ego_view": frames}
        result = t.apply(data)

        assert result["video.ego_view"] is frames, "eval mode must return same object"
        assert t._socket is None, "no socket should be created in eval mode"


def test_cache_hit_reads_from_disk(transform_factory):
    """When a cache file already exists, return it without connecting to ZMQ."""
    with tempfile.TemporaryDirectory() as cache_dir:
        # Use a fixed seed so we know exactly which cache filename the transform will look for.
        t = transform_factory(cache_dir, probability=1.0, seed=42)

        frames = _make_frames()
        cached = torch.ones_like(frames) * 0.42  # recognisable sentinel value
        cache_path = Path(cache_dir) / f"{_content_hash(frames)}_42.pt"
        torch.save(cached, cache_path)

        result = t.apply({"video.ego_view": frames})

        assert torch.equal(result["video.ego_view"], cached), "should return cached tensor"
        assert t._socket is None, "no socket should be created on cache hit"


def test_cache_miss_calls_zmq_and_saves(transform_factory):
    """On a cache miss: correct wire format is sent, reply is decoded, .pt is saved."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=1.0, seed=99)

        frames = _make_frames()
        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)
        t._socket = mock_socket

        result = t.apply({"video.ego_view": frames})

        # Check send was called once with correct wire format:
        # 20-byte header = 5 x uint32 (T, C, H, W, seed) + float32 payload
        assert mock_socket.send.call_count == 1
        sent = mock_socket.send.call_args[0][0]
        T, C, H, W, sent_seed = struct.unpack("5I", sent[:20])
        assert (T, C, H, W) == frames.shape
        assert sent_seed == 99, "seed should be passed through the wire format"
        assert len(sent[20:]) == T * C * H * W * 4  # float32 bytes

        # Check recv was called
        assert mock_socket.recv.call_count == 1

        # Check output shape and dtype
        assert result["video.ego_view"].shape == frames.shape
        assert result["video.ego_view"].dtype == torch.float32

        # Check cache file was written with {hash}_{seed}.pt naming
        pt_files = list(Path(cache_dir).glob("*.pt"))
        assert len(pt_files) == 1, "exactly one .pt file should be saved"
        assert pt_files[0].stem.endswith("_99"), "cache filename must include the seed"

        # No leftover .tmp files
        tmp_files = list(Path(cache_dir).glob("*.tmp"))
        assert len(tmp_files) == 0, "no .tmp files should remain after atomic write"


def test_probability_zero_never_augments(transform_factory):
    """With probability=0, apply() never touches data and never creates a socket."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=0.0)

        for _ in range(50):
            frames = _make_frames()
            data = {"video.ego_view": frames}
            result = t.apply(data)
            assert result["video.ego_view"] is frames

        assert t._socket is None
        assert list(Path(cache_dir).glob("*.pt")) == []


def test_probability_one_always_augments(transform_factory):
    """With probability=1 and a fixed seed, first call is a cache miss; second hits cache."""
    with tempfile.TemporaryDirectory() as cache_dir:
        # Fixed seed ensures both calls look for the same {hash}_{seed}.pt file.
        t = transform_factory(cache_dir, probability=1.0, seed=7)

        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)

        def _inject(self_inner):
            self_inner._socket = mock_socket

        frames = _make_frames()

        with patch.object(VideoCosmosAugmentTransform, "_connect", _inject):
            # First call — cache miss → ZMQ
            t.apply({"video.ego_view": frames})
            assert mock_socket.send.call_count == 1

            # Second call with same frames and same fixed seed — cache hit → no additional ZMQ
            t.apply({"video.ego_view": frames})
            assert mock_socket.send.call_count == 1, "second call must hit cache"


def test_atomic_write_no_tmp_leftover(transform_factory):
    """After a successful cache-miss write, no .tmp files remain on disk."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=1.0)

        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)
        t._socket = mock_socket

        t.apply({"video.ego_view": _make_frames()})

        all_files = list(Path(cache_dir).iterdir())
        assert all(f.suffix != ".tmp" for f in all_files), "no .tmp files should remain"
        pt_files = [f for f in all_files if f.suffix == ".pt"]
        assert len(pt_files) == 1


def test_output_shape_dtype_preserved(transform_factory):
    """Output tensor has same shape and float32 dtype for various input shapes."""
    shapes = [(1, 3, 8, 8), (4, 3, 8, 8), (8, 3, 16, 16)]
    for shape in shapes:
        with tempfile.TemporaryDirectory() as cache_dir:
            t = transform_factory(cache_dir, probability=1.0)
            frames = torch.rand(*shape, dtype=torch.float32)
            augmented = torch.rand(*shape, dtype=torch.float32)
            mock_socket = MagicMock()
            mock_socket.recv.return_value = _make_zmq_reply(augmented)
            t._socket = mock_socket

            result = t.apply({"video.ego_view": frames})

            assert result["video.ego_view"].shape == torch.Size(shape), f"shape mismatch for {shape}"
            assert result["video.ego_view"].dtype == torch.float32, f"dtype mismatch for {shape}"


def test_set_metadata_stores_resolutions(dataset_metadata):
    """set_metadata() correctly populates original_resolutions per apply_to key."""
    # Use different resolutions to confirm each key maps correctly
    meta = DatasetMetadata(
        statistics=DatasetStatistics(
            state={"arm": _DUMMY_STAT},
            action={"arm": _DUMMY_STAT},
        ),
        modalities=DatasetModalities(
            video={
                "ego_view": VideoMetadata(resolution=(480, 640), channels=3, fps=30.0),
                "left_wrist_view": VideoMetadata(resolution=(240, 320), channels=3, fps=30.0),
                "right_wrist_view": VideoMetadata(resolution=(360, 480), channels=3, fps=30.0),
            },
            state={"arm": _STATE_ACTION_META},
            action={"arm": _STATE_ACTION_META},
        ),
        embodiment_tag=EmbodimentTag.GR1,
    )
    t = VideoCosmosAugmentTransform(
        apply_to=["video.ego_view", "video.left_wrist_view", "video.right_wrist_view"],
        cache_dir="/tmp/test_cache",
        probability=0.5,
    )
    t.set_metadata(meta)

    assert t.original_resolutions == {
        "video.ego_view": (480, 640),
        "video.left_wrist_view": (240, 320),
        "video.right_wrist_view": (360, 480),
    }
    assert t.training is True
    assert t._dataset_metadata is meta


def test_multi_port_round_robin(transform_factory):
    """_connect() assigns workers to ports via worker_id % len(ports)."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, ports=[5557, 5558, 5559])

        cases = [(0, 5557), (1, 5558), (2, 5559), (3, 5557), (4, 5558)]
        for worker_id, expected_port in cases:
            t._socket = None  # reset so _connect runs fresh

            mock_sock = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.socket.return_value = mock_sock

            with patch("zmq.Context", return_value=mock_ctx):
                with patch(
                    "torch.utils.data.get_worker_info",
                    return_value=MagicMock(id=worker_id),
                ):
                    t._connect()

            connect_url = mock_sock.connect.call_args[0][0]
            assert connect_url == f"tcp://localhost:{expected_port}", (
                f"worker {worker_id} should connect to port {expected_port}, got {connect_url}"
            )


def test_multi_port_no_worker_info(transform_factory):
    """When get_worker_info() returns None (main process), use ports[0]."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, ports=[5557, 5558])

        mock_sock = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_sock

        with patch("zmq.Context", return_value=mock_ctx):
            with patch("torch.utils.data.get_worker_info", return_value=None):
                t._connect()

        connect_url = mock_sock.connect.call_args[0][0]
        assert connect_url == "tcp://localhost:5557"


def test_cache_miss_connect_called_lazily(transform_factory):
    """_connect() is called exactly once even with multiple cache misses."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=1.0, seed=1)

        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)

        connect_calls = 0

        def _inject_and_count(self_inner):
            nonlocal connect_calls
            connect_calls += 1
            self_inner._socket = mock_socket

        with patch.object(VideoCosmosAugmentTransform, "_connect", _inject_and_count):
            # Two different tensors → two cache misses (different content hashes)
            t.apply({"video.ego_view": _make_frames()})
            t.apply({"video.ego_view": _make_frames()})

        assert connect_calls == 1, "_connect should be called lazily only once"
        assert mock_socket.send.call_count == 2, "both misses should send to the service"


# ---------------------------------------------------------------------------
# Seed tests
# ---------------------------------------------------------------------------


def test_fixed_seed_deterministic_cache(transform_factory):
    """Same input + fixed seed always resolves to the same cache filename."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(cache_dir, probability=1.0, seed=42)

        frames = _make_frames()
        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)
        t._socket = mock_socket

        # First call — miss, writes cache
        t.apply({"video.ego_view": frames})
        pt_files_after_first = list(Path(cache_dir).glob("*.pt"))
        assert len(pt_files_after_first) == 1
        first_path = pt_files_after_first[0]

        # Second call — must hit exactly the same file, no new file created
        t.apply({"video.ego_view": frames})
        pt_files_after_second = list(Path(cache_dir).glob("*.pt"))
        assert len(pt_files_after_second) == 1, "no new cache file should appear"
        assert pt_files_after_second[0] == first_path, "same file should be reused"
        assert mock_socket.send.call_count == 1, "second call must be a cache hit"


def test_random_seed_different_cache_files(transform_factory):
    """Same input with seed=None produces different cache files across calls."""
    with tempfile.TemporaryDirectory() as cache_dir:
        # seed=None → random seed per call
        t = transform_factory(cache_dir, probability=1.0, seed=None)

        frames = _make_frames()
        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)
        t._socket = mock_socket

        # Patch _next_seed to return two distinct values deterministically
        seeds_to_emit = iter([111, 222])
        with patch.object(VideoCosmosAugmentTransform, "_next_seed", lambda _: next(seeds_to_emit)):
            t.apply({"video.ego_view": frames})
            t.apply({"video.ego_view": frames})

        pt_files = sorted(Path(cache_dir).glob("*.pt"))
        assert len(pt_files) == 2, "each distinct seed should produce its own cache file"
        assert pt_files[0].stem.endswith("_111")
        assert pt_files[1].stem.endswith("_222")
        assert mock_socket.send.call_count == 2, "both calls are cache misses (different seeds)"


# ---------------------------------------------------------------------------
# Grid mode tests
# ---------------------------------------------------------------------------


def test_grid_mode_build_and_split(transform_factory):
    """_build_grid assembles a 2×2 grid; extracting quadrants recovers originals."""
    T, C, H, W = 2, 3, 8, 10
    frames = [torch.rand(T, C, H, W) for _ in range(3)]

    grid = VideoCosmosAugmentTransform._build_grid(frames)

    assert grid.shape == (T, C, 2 * H, 2 * W), "grid should be [T, C, 2H, 2W]"

    # Extract quadrants and verify they match the originals
    assert torch.equal(grid[:, :, :H, :W], frames[0]), "top-left must match key0"
    assert torch.equal(grid[:, :, :H, W:], frames[1]), "top-right must match key1"
    assert torch.equal(grid[:, :, H:, :W], frames[2]), "bottom-left must match key2"
    # Bottom-right slot (index 3, absent) must be zero-filled
    assert torch.equal(grid[:, :, H:, W:], torch.zeros(T, C, H, W)), "empty slot must be zeros"


def test_grid_mode_end_to_end(transform_factory):
    """grid_mode=True: both video keys are updated; shapes match originals."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(
            cache_dir,
            probability=1.0,
            apply_to=["video.left_wrist_view", "video.right_wrist_view"],
            seed=5,
            grid_mode=True,
        )

        T, C, H, W = 2, 3, 8, 8
        left = torch.rand(T, C, H, W)
        right = torch.rand(T, C, H, W)

        # Build the grid so the mock can return a plausible augmented version
        grid = VideoCosmosAugmentTransform._build_grid([left, right])
        augmented_grid = torch.rand_like(grid)
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented_grid)
        t._socket = mock_socket

        data = {"video.left_wrist_view": left, "video.right_wrist_view": right}
        result = t.apply(data)

        # One ZMQ call for the whole grid (not two separate calls)
        assert mock_socket.send.call_count == 1, "grid mode should make exactly one ZMQ call"

        # Verify the sent payload is grid-shaped
        sent = mock_socket.send.call_args[0][0]
        sT, sC, sH, sW, sent_seed = struct.unpack("5I", sent[:20])
        assert (sT, sC, sH, sW) == (T, C, 2 * H, 2 * W), "grid dimensions should be sent"
        assert sent_seed == 5

        # Each key is restored to its original per-view shape
        assert result["video.left_wrist_view"].shape == (T, C, H, W)
        assert result["video.right_wrist_view"].shape == (T, C, H, W)

        # Values should come from the augmented grid quadrants
        assert torch.equal(result["video.left_wrist_view"], augmented_grid[:, :, :H, :W])
        assert torch.equal(result["video.right_wrist_view"], augmented_grid[:, :, :H, W:])


def test_grid_mode_single_key_fallback(transform_factory):
    """grid_mode=True with only 1 key present falls back to per-key augmentation."""
    with tempfile.TemporaryDirectory() as cache_dir:
        t = transform_factory(
            cache_dir,
            probability=1.0,
            apply_to=["video.left_wrist_view", "video.right_wrist_view"],
            seed=3,
            grid_mode=True,
        )

        frames = _make_frames()
        augmented = _make_frames()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = _make_zmq_reply(augmented)
        t._socket = mock_socket

        # Only one key present — should fall back gracefully to per-key path
        data = {"video.left_wrist_view": frames}  # right_wrist_view absent
        result = t.apply(data)

        assert mock_socket.send.call_count == 1, "single key should still call the service once"
        assert result["video.left_wrist_view"].shape == frames.shape
        assert "video.right_wrist_view" not in result
