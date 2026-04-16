# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionSinCosTransform, StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCosmosAugmentTransform,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class UnitreeG1SimDataConfig(BaseDataConfig):
    video_keys = ["video.left_wrist_view", "video.right_wrist_view", "video.room_view"] 
    # video_keys order defines the grid order of VideoCosmosAugmentTransform in grid mode: top-left, top-right, bottom-left
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Cosmos-Transfer2.5 augmentation settings.
    # Set cosmos_cache_dir to a persistent path to benefit from caching across runs.
    # List one port per GPU service (e.g. [5557, 5558] for two GPUs).
    cosmos_cache_dir: str = "/tmp/cosmos_cache"
    cosmos_host: str = "localhost"
    cosmos_ports: list[int] = [5557, 5558]
    cosmos_probability: float = 1.0


    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            # Cosmos augmentation runs at full resolution before the random crop,
            # so the model sees unclipped context. VideoCrop + VideoResize follow,
            # preserving per-step stochasticity. Results are cached by content hash;
            # first epoch is slow (~2s/frame), all later epochs load from cache.
            VideoCosmosAugmentTransform(
                apply_to=self.video_keys,
                cache_dir=self.cosmos_cache_dir,
                host=self.cosmos_host,
                ports=self.cosmos_ports,
                probability=self.cosmos_probability,
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP["unitree_g1_sim"] = UnitreeG1SimDataConfig()