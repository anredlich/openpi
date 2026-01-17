# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Protocol

from robots.configs import ( #lerobot.common.robot_devices.
    AlohaRobotConfig,
    KochBimanualRobotConfig,
    KochRobotConfig,
    LeKiwiRobotConfig,
    ManipulatorRobotConfig,
    MossRobotConfig,
    RobotConfig,
    So100RobotConfig,
    StretchRobotConfig,
    TrossenAIMobileRobotConfig,
    TrossenAISoloRobotConfig,
    TrossenAIStationaryRobotConfig,
)


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "aloha":
        return AlohaRobotConfig(**kwargs)
    elif robot_type == "koch":
        return KochRobotConfig(**kwargs)
    elif robot_type == "koch_bimanual":
        return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss":
        return MossRobotConfig(**kwargs)
    elif robot_type == "so100":
        return So100RobotConfig(**kwargs)
    elif robot_type == "stretch":
        return StretchRobotConfig(**kwargs)
    elif robot_type == "lekiwi":
        return LeKiwiRobotConfig(**kwargs)
    elif robot_type == "trossen_ai_stationary":
        return TrossenAIStationaryRobotConfig(**kwargs)
    elif robot_type == "trossen_ai_solo":
        return TrossenAISoloRobotConfig(**kwargs)
    elif robot_type == "trossen_ai_mobile":
        return TrossenAIMobileRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from robots.manipulator import ManipulatorRobot #lerobot.common.robot_devices.

        return ManipulatorRobot(config)
    # elif isinstance(config, LeKiwiRobotConfig):
    #     from lerobot.common.robot_devices.robots.mobile_manipulator import MobileManipulator

    #     return MobileManipulator(config)
    # elif isinstance(config, TrossenAIMobileRobotConfig):
    #     from lerobot.common.robot_devices.robots.trossen_ai_mobile import TrossenAIMobile

    #     return TrossenAIMobile(config)
    else:
        return None
        # from lerobot.common.robot_devices.robots.stretch import StretchRobot

        # return StretchRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)

#############################################################################################

import torch

TROSSEN_AI_STATIONARY_JOINT_MIN = torch.tensor([
    -3.05433, 0.0, 0.0, -1.5708, -1.5708, -3.14159, 0.0,
    -3.05433, 0.0, 0.0, -1.5708, -1.5708, -3.14159, 0.0,
], dtype=torch.float32)

TROSSEN_AI_STATIONARY_JOINT_MAX = torch.tensor([
    3.05433, 3.14159, 2.35619, 1.5708, 1.5708, 3.14159, 0.044,
    3.05433, 3.14159, 2.35619, 1.5708, 1.5708, 3.14159, 0.044,
], dtype=torch.float32)

TROSSEN_AI_STATIONARY_GRIPPER_INDICES = [6,13]

def normalize_batch(batch: dict):
    if 'Trossen AI Stationary' not in batch['task'][0]:
        return batch

    batch = batch.copy()  # Shallow copy of dict
    batch['action'] = batch['action'].clone()  # Deep copy of tensor
    batch['observation.state'] = batch['observation.state'].clone()  # Deep copy of tensor

    #normalize the gripper actions to 0-1
    for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
        batch['action'][:, :, idx] = (
            (batch['action'][:, :, idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx]) / 
            (TROSSEN_AI_STATIONARY_JOINT_MAX[idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx])
        )

    #normalize the gripper observation state to 0-1
    for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
        batch['observation.state'][:, idx] = (
            (batch['observation.state'][:, idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx]) / 
            (TROSSEN_AI_STATIONARY_JOINT_MAX[idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx])
        )

    return batch

def normalize_state(state: dict):
    #if 'Trossen AI Stationary' not in batch['task'][0]:
    #    return batch

    # batch = batch.copy()  # Shallow copy of dict
    # batch['action'] = batch['action'].clone()  # Deep copy of tensor
    # batch['observation.state'] = batch['observation.state'].clone()  # Deep copy of tensor

    #normalize the gripper observation state to 0-1
    if state.ndim==1:
        for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
            state[idx] = (
                (state[idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx]) / 
                (TROSSEN_AI_STATIONARY_JOINT_MAX[idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx])
            )
    elif state.ndim==2:
        for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
            state[:,idx] = (
                (state[:,idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx]) / 
                (TROSSEN_AI_STATIONARY_JOINT_MAX[idx] - TROSSEN_AI_STATIONARY_JOINT_MIN[idx])
            )

    return state

def unnormalize_numpy_action(action: dict):
    #if 'Trossen AI Stationary' not in batch['task'][0]:
    #    return batch

    # batch = batch.copy()  # Shallow copy of dict
    # batch['action'] = batch['action'].clone()  # Deep copy of tensor
    # batch['observation.state'] = batch['observation.state'].clone()  # Deep copy of tensor

    #normalize the gripper observation state to 0-1
    if action.ndim==1:
        for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
            action[idx] = TROSSEN_AI_STATIONARY_JOINT_MIN[idx].numpy() + action[idx] * (TROSSEN_AI_STATIONARY_JOINT_MAX[idx].numpy() - TROSSEN_AI_STATIONARY_JOINT_MIN[idx].numpy())
    elif action.ndim==2:
        for idx in TROSSEN_AI_STATIONARY_GRIPPER_INDICES:
            action[:,idx] = TROSSEN_AI_STATIONARY_JOINT_MIN[idx].numpy() + action[:,idx] * (TROSSEN_AI_STATIONARY_JOINT_MAX[idx].numpy() - TROSSEN_AI_STATIONARY_JOINT_MIN[idx].numpy())
        
    return action
