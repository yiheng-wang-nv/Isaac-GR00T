#!/usr/bin/env python3
"""
SO101 Real Robot Evaluation Script for GR00T Policy

Based on eval_gr00t_so100.py but adapted for SO101 robot.
This script connects to a GR00T inference server and runs policy evaluation on SO101.

Usage:
    # Start GR00T server first:
    python scripts/inference_service.py --server \
        --model_path /localhome/local-vennw/code/Isaac-GR00T/so101_wrist_finetune/checkpoint-10000 \
        --embodiment_tag new_embodiment \
        --data_config so101_wrist \
        --host 0.0.0.0 \
        --port 5555 \
        --denoising_steps 4

    # Then run this client:
    ssh -L 5555:localhost:5555 local-vennw@10.176.195.216
    python eval_gr00t_so101.py \
        --host 127.0.0.1 \
        --port 5555 \
        --camera_index 0 \
        --port_follower /dev/ttyACM0 \
        --task_description "Grip a straight scissor and put it in the box." \
        --actions_to_execute 300
"""

import argparse
import os
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
from tqdm import tqdm

# Import GR00T client
from gr00t.eval.service import ExternalRobotInferenceClient


class SO101Robot:
    """SO101 Robot controller for GR00T policy evaluation."""
    
    def __init__(self, port_follower: str = "/dev/ttyACM1", calibrate: bool = False, 
                 enable_camera: bool = True, cam_idx: int = 0):
        self.config = So101RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.cam_idx = cam_idx
        self.port_follower = port_follower
        
        # Configure robot
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {"wrist": OpenCVCameraConfig(cam_idx, 30, 640, 480, "rgb")}
        
        # Inference mode: no leader arms needed
        self.config.leader_arms = {}
        
        # Set follower arm port
        self.config.follower_arms["main"].port = port_follower
        
        # Remove calibration folder if requested
        if self.calibrate:
            import shutil
            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so101")
            print(f"========> Deleting calibration_folder: {calibration_folder}")
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)
        
        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        """Context manager for robot activation."""
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        """Connect to SO101 robot."""
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "SO101Robot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the motor bus
        self.motor_bus.connect()

        # Disable torque for calibration
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Run calibration
        self.robot.activate_calibration()

        # Set robot preset
        self.set_so101_robot_preset()

        # Enable torque
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("SO101 present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        # Connect camera
        self.camera = self.robot.cameras["wrist"] if self.enable_camera else None
        if self.camera is not None:
            self.camera.connect()
        
        print("================> SO101 Robot is fully connected =================")

    def set_so101_robot_preset(self):
        """Set SO101-specific motor configurations."""
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration for faster response
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        """Move robot to initial pose."""
        print("-------------------------------- Moving to initial pose")
        # SO101 initial pose (adjust these values as needed)
        initial_state = torch.tensor([90, 90, 90, 90, -70, 30], dtype=torch.float32)
        self.robot.send_action(initial_state)
        time.sleep(2)

    def go_home(self):
        """Move robot to home pose."""
        print("-------------------------------- Moving to home pose")
        # SO101 home pose (adjust these values as needed)
        home_state = torch.tensor([88.0, 156.0, 135.0, 83.0, -89.0, 16.0], dtype=torch.float32)
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        """Get robot observation."""
        return self.robot.capture_observation()

    def get_current_state(self):
        """Get current robot state as numpy array."""
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        """Get current camera image as RGB numpy array."""
        if not self.enable_camera:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        img = self.get_observation()["observation.images.wrist"].data.numpy()
        # Convert to HWC format if needed
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
            img = img.transpose(1, 2, 0)  # Convert to HWC
        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        return img

    def set_target_state(self, target_state: torch.Tensor):
        """Send target state to robot."""
        self.robot.send_action(target_state)

    def enable(self):
        """Enable motor torque."""
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        """Disable motor torque."""
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        """Disconnect robot."""
        self.disable()
        if self.robot.is_connected:
            self.robot.disconnect()
            self.robot.is_connected = False
        print("================> SO101 Robot disconnected")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'robot') and self.robot.is_connected:
            self.disconnect()


class Gr00tSO101InferenceClient:
    """GR00T inference client for SO101."""
    
    def __init__(self, host: str = "localhost", port: int = 5555, 
                 language_instruction: str = "Pick up the object and place it in the box."):
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        print(f"Connected to GR00T server at {host}:{port}")
        print(f"Task: {language_instruction}")

    def get_action(self, img: np.ndarray, state: np.ndarray) -> dict:
        """Get action from GR00T policy."""
        # Format observation for SO101 wrist config
        obs_dict = {
            "video.wrist": img[np.newaxis, :, :, :],  # Add batch dimension
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),  # 5 arm joints
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),   # 1 gripper joint
            "annotation.human.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_language_instruction(self, instruction: str):
        """Update task instruction."""
        self.language_instruction = instruction
        print(f"Task updated: {instruction}")


def view_img(img, title: str = "Camera View"):
    """Display image using matplotlib (non-blocking)."""
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear for next frame


def main():
    parser = argparse.ArgumentParser(description="SO101 GR00T Policy Evaluation")
    
    # GR00T server connection
    parser.add_argument("--host", type=str, default="localhost", help="GR00T server host")
    parser.add_argument("--port", type=int, default=5555, help="GR00T server port")
    
    # Robot configuration
    parser.add_argument("--port_follower", type=str, default="/dev/ttyACM1", help="SO101 serial port")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index")
    
    # Task configuration
    parser.add_argument("--task_description", type=str, 
                       default="Grip a straight scissor and put it in the box.",
                       help="Task description for the policy")
    
    # Execution parameters
    parser.add_argument("--actions_to_execute", type=int, default=300, help="Number of action steps")
    parser.add_argument("--action_horizon", type=int, default=12, help="Actions to execute from chunk")
    parser.add_argument("--calibrate", action="store_true", help="Run robot calibration")
    parser.add_argument("--record_images", action="store_true", help="Save images during execution")
    parser.add_argument("--output_dir", type=str, default="eval_so101_images", help="Output directory for images")
    
    args = parser.parse_args()

    print(f"Task: {args.task_description}")
    print(f"Actions to execute: {args.actions_to_execute}")
    print(f"Action horizon: {args.action_horizon}")

    # Initialize GR00T client
    client = Gr00tSO101InferenceClient(
        host=args.host,
        port=args.port,
        language_instruction=args.task_description
    )

    # Setup image recording if requested
    if args.record_images:
        os.makedirs(args.output_dir, exist_ok=True)
        # Clear existing images
        for file in os.listdir(args.output_dir):
            if file.endswith(('.jpg', '.png')):
                os.remove(os.path.join(args.output_dir, file))
        print(f"Recording images to: {args.output_dir}")

    # Initialize robot
    robot = SO101Robot(
        port_follower=args.port_follower,
        calibrate=args.calibrate,
        enable_camera=True,
        cam_idx=args.camera_index
    )

    image_count = 0
    MODALITY_KEYS = ["single_arm", "gripper"]

    try:
        with robot.activate():
            print("Starting policy execution...")
            
            for i in tqdm(range(args.actions_to_execute), desc="Executing actions"):
                # Get current observation
                img = robot.get_current_img()
                state = robot.get_current_state()
                
                # Display image
                view_img(img, f"Step {i}/{args.actions_to_execute}")
                
                # Get action from policy
                action_start_time = time.time()
                action = client.get_action(img, state)
                
                # Execute action chunk
                execution_start_time = time.time()
                for j in range(args.action_horizon):
                    # Concatenate action components
                    concat_action = np.concatenate([
                        np.atleast_1d(action[f"action.{key}"][j]) 
                        for key in MODALITY_KEYS
                    ], axis=0)
                    
                    assert concat_action.shape == (6,), f"Expected (6,) but got {concat_action.shape}"
                    
                    # Send to robot
                    robot.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.02)  # Small delay between actions
                    
                    # Update display
                    img = robot.get_current_img()
                    view_img(img, f"Step {i}/{args.actions_to_execute} - Action {j}/{args.action_horizon}")
                    
                    # Save image if recording
                    if args.record_images:
                        # Resize and save
                        img_save = cv2.resize(img, (320, 240))
                        img_save_bgr = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{args.output_dir}/img_{image_count:06d}.jpg", img_save_bgr)
                        image_count += 1
                
                action_time = time.time() - action_start_time
                execution_time = time.time() - execution_start_time
                
                if i % 10 == 0:  # Print every 10 steps
                    print(f"Step {i}: Action time: {action_time:.3f}s, "
                          f"Execution time: {execution_time:.3f}s")

            print("Policy execution completed!")
            
            # Return to home position
            print("Returning to home position...")
            robot.go_home()
            
            if args.record_images:
                print(f"Saved {image_count} images to {args.output_dir}")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
    finally:
        print("Cleaning up...")


if __name__ == "__main__":
    main() 