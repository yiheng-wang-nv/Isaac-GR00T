"""
Modality configuration for Unitree G1 + Dex3 dual-arm manipulation dataset.
Dataset: install_trocar_from_tray_realsense_lerobot
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


unitree_g1_dex3_config = {
    # Video modality: 3 cameras
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=[
            "left_wrist_view",   # Must match key in meta/modality.json "video"
            "right_wrist_view",
            "room_view",
        ],
    ),
    
    # State modality: proprioceptive observations
    "state": ModalityConfig(
        delta_indices=[0],  # Current state
        modality_keys=[
            "left_arm",    # 7 dims: shoulder/elbow/wrist joints
            "right_arm",   # 7 dims
            "left_hand",   # 7 dims: thumb/index/middle fingers
            "right_hand",  # 7 dims
        ],
        # Use sin/cos encoding for ALL state keys (all are joint angles in radians)
        # This matches the old config: StateActionSinCosTransform(apply_to=self.state_keys)
        sin_cos_embedding_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    
    # Action modality: 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # Predict 16 steps into the future
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # left_arm: use absolute actions (matching old config with min_max normalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,  # Joint space control
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: use absolute actions
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: use absolute actions
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: use absolute actions
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    
    # Language modality: task description
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Register the configuration
register_modality_config(unitree_g1_dex3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

