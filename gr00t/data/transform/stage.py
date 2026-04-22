# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from pydantic import Field

from gr00t.data.transform.base import ModalityTransform


class StageTransform(ModalityTransform):
    """Per-frame stage label transform.

    - Optionally merges a trailing stage id into another (e.g. 5 -> 4 to collapse a
      rare trailing-frame label into the main "place" stage).
    - Squeezes the time axis for `delta_indices=[0]` so the downstream collate
      produces a (B,) int64 tensor.
    """

    apply_to: list[str] = Field(default_factory=list)
    merge_map: dict[int, int] = Field(
        default_factory=dict,
        description=(
            "Map of source stage id -> target stage id. Applied elementwise before "
            "emitting the label."
        ),
    )
    squeeze_time: bool = Field(
        default=True,
        description="Squeeze the time axis so each sample emits a scalar int64 label.",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in list(data.keys()):
            if key not in self.apply_to:
                continue
            arr = np.asarray(data[key])
            if self.merge_map:
                arr = arr.copy()
                for src, dst in self.merge_map.items():
                    arr[arr == src] = dst
            # Shape coming in is (T, 1) from the dataset loader. Drop the trailing
            # singleton dim and optionally the time dim when it is 1.
            if arr.ndim >= 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            if self.squeeze_time and arr.ndim >= 1 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            data[key] = arr.astype(np.int64)
        return data
