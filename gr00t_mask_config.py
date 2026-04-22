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

"""Data config that mirrors `gr00t_config.py` but adds ChangeBackgroundTransform.

Mask pixel values (per category_mapping.json):
    0 = background, 1 = ground, 2 = robot, 3 = trocar_1, 4 = trocar_2,
    5 = tray, 6 = cart, 7 = instrument_trolley
We replace background + ground (0, 1); robot / trocars / tray / cart / trolley stay.
"""

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.stage import StageTransform
from gr00t.data.transform.state_action import StateActionSinCosTransform, StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    ChangeBackgroundTransform,
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig
from gr00t.model.transforms import GR00TTransform


# Templates for background replacement. Point this at a directory of .jpg/.png images
# (e.g. room photos, plain colors, or frames extracted from a template video).
DEFAULT_TEMPLATE_FOLDER = "/localhome/local-vennw/code/cosmos_gr00t/HealthSurgiBench_1e4/media"

# Category IDs in the masks.npz files that should be treated as background and replaced.
# Keep foreground categories (robot, trocars, tray, cart, instrument_trolley) untouched.
BACKGROUND_MASK_VALUES = [0, 1]


class UnitreeG1SimMaskDataConfig(BaseDataConfig):
    video_keys = ["video.left_wrist_view", "video.right_wrist_view", "video.room_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    stage_keys = ["stage.current_stage"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Background-swap hyperparameters (override per-instance after construction if needed).
    template_folder: str = DEFAULT_TEMPLATE_FOLDER
    bg_mask_values: list[int] = BACKGROUND_MASK_VALUES
    bg_change_prob: float = 0.9
    bg_feather_radius: int = 3

    # When False, the `stage` column is skipped entirely (no ModalityConfig entry,
    # no StageTransform, no `stage` tensor in the batch). Required to be True for
    # the auxiliary stage classifier head.
    use_stage: bool = True

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

        configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        if self.use_stage:
            configs["stage"] = ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.stage_keys,
            )
        return configs

    def transform(self):
        transforms = [
            # Background replacement runs FIRST, on raw uint8 [T,H,W,C] with masks still in data.
            ChangeBackgroundTransform(
                apply_to=self.video_keys,
                template_folder=self.template_folder,
                target_mask_values=self.bg_mask_values,
                p=self.bg_change_prob,
                feather_radius=self.bg_feather_radius,
            ),
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
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
        ]
        if self.use_stage:
            # stage label: merge rare stage 5 (trailing frame of `place trocar`) into 4
            transforms.append(StageTransform(apply_to=self.stage_keys, merge_map={5: 4}))
        transforms += [
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


DATA_CONFIG_MAP["unitree_g1_sim_mask"] = UnitreeG1SimMaskDataConfig()
