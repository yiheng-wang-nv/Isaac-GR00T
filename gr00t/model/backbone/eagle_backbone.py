# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        return_pure_vision_features: bool = False,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()
        self.return_pure_vision_features = return_pure_vision_features

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _pack_pure_vision_features(
        self, eagle_input: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract pre-language vision features and pack them by sample."""
        pixel_values = eagle_input.get("pixel_values")
        input_ids = eagle_input.get("input_ids")
        if pixel_values is None or input_ids is None:
            return None, None

        vision_features = self.eagle_model.extract_feature(pixel_values)
        image_flags = eagle_input.get("image_flags")
        if image_flags is not None:
            vision_features = vision_features[image_flags.view(-1) == 1]
        vision_features = self.eagle_linear(vision_features)

        image_token_index = getattr(
            self.eagle_model, "image_token_index", self.eagle_model.config.image_token_index
        )
        token_counts = (input_ids == image_token_index).sum(dim=1)
        total_tokens = int(token_counts.sum().item())
        flat_features = vision_features.reshape(-1, vision_features.shape[-1])[:total_tokens]
        if flat_features.shape[0] < total_tokens:
            raise RuntimeError(
                "Not enough pure vision tokens to match the image-token placeholders "
                f"({flat_features.shape[0]} < {total_tokens})."
            )

        batch_size = input_ids.shape[0]
        max_tokens = int(token_counts.max().item()) if batch_size > 0 else 0
        packed_features = flat_features.new_zeros(
            (batch_size, max_tokens, flat_features.shape[-1])
        )
        packed_mask = torch.arange(max_tokens, device=input_ids.device).unsqueeze(0) < token_counts.unsqueeze(1)

        offset = 0
        for batch_idx, token_count in enumerate(token_counts.tolist()):
            token_count = int(token_count)
            if token_count == 0:
                continue
            next_offset = offset + token_count
            packed_features[batch_idx, :token_count] = flat_features[offset:next_offset]
            offset = next_offset

        return packed_features, packed_mask

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        vision_features = None
        vision_mask = None
        if self.return_pure_vision_features:
            vision_features, vision_mask = self._pack_pure_vision_features(eagle_input)

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"], vision_features, vision_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask, vision_embeds, vision_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        output = {"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        if vision_embeds is not None and vision_mask is not None:
            output["backbone_vision_features"] = vision_embeds
            output["backbone_vision_attention_mask"] = vision_mask
        return BatchFeature(data=output)  # [B, T2, hidden_size]
