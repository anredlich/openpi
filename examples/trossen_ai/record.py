#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Usage:
    python main.py --mode autonomous --task_prompt "grab and handover red cube"

    Test mode (no movement):
    python main.py --mode test --task_prompt "grab and handover red cube"
"""

import argparse
from collections import defaultdict
import logging
import time
import torch

import cv2
cv2.namedWindow('_init', cv2.WINDOW_NORMAL)
cv2.destroyAllWindows()
# from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
# from lerobot.robots import make_robot_from_config
# from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
from robots.configs import TrossenAIStationaryRobotConfig
from robots.utils import make_robot_from_config
import numpy as np
from openpi_client import websocket_client_policy
from scipy.interpolate import PchipInterpolator
from utils import init_keyboard_listener
import openpi_client.image_tools as image_tools
#from voice_command import VoiceCommandListener
from utils import say_gtts, say_tts
import os
os.environ["SVT_LOG"] = "0"

import trossen_arm as trossen

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",  # "autonomous" or "test"
        max_steps: int = 1000,
        action_chunk_size: int = 50,
        max_relative_target: float = 0.05,
        adjust_for_sim_to_real: bool = False,
        record_mode: str = "rollout",
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        self.adjust_for_sim_to_real = adjust_for_sim_to_real #False #hack to adjust joints to better match sim to real
        self.display = True

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        #Get metadata from server:
        metadata = self.policy_client.get_server_metadata()
        self.crop_cameras = metadata.get("crop_cameras", {}) if metadata else {}
        logger.info(f"Crop config from server: {self.crop_cameras}")
 
        robot_config=TrossenAIStationaryRobotConfig(max_relative_target,home_pose=[0, 0.261799, 0.261799, 0, 0, 0, 0.044]) #max_relative_target=0.05
        self.robot = make_robot_from_config(robot_config)
        self.dataset_features = self.robot.features.copy()  # Save before clearing leader_arms
        # Add dtype for image features
        for key, ft in self.dataset_features.items():
            if 'images' in key:
                self.dataset_features[key] = {
                    'dtype': 'video',
                    **ft,
                }
        #logger.info(f"Saved dataset_features: {self.dataset_features}")
        self.record_mode=record_mode
        if record_mode == 'rollout':
            self.robot.leader_arms = {} #[]
        self.robot.connect(hold=True)
        #self.hold_leaders()

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.action_chunk_size = (
            action_chunk_size #50  # Number of actions per chunk from the policy (Defined by the policy server in this case 50)
        )
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = self.action_chunk_size #50  # Number of control steps per policy inference (matches README and Pi-0 paper)

        self.temporal_ensemble_coefficient = None  # Temporal ensembling weight (can be set to None for no ensembling)

        # FIFO Buffer for actions
        self.action_buffer = defaultdict(list)
        self.action_buffer_size = (
            self.max_steps + self.action_chunk_size
        )  # Buffer size to hold actions for the entire episode

        #self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms
        self.action_dim = len(self.robot.features['action']['names'])

        #speech listener
        self.speech_listener = None
        self.gemini_planner = None
        self.gemini_annotated_frame = None
        self.gemini_running = False
        self.gemini_next_prompt = None
        self.control_mode = "speech"  # default, overridden from __main__

    # def execute_action(self, action: np.ndarray):
    #     """Execute action on the arm."""
    #     full_action = action.copy()
    #     full_action = torch.from_numpy(full_action).float()

    #     if self.test_mode == "test":
    #         logger.info(f"TEST MODE: Would execute action: {full_action}")
    #         return
    #     if self.test_mode == "autonomous":
    #         #joint_features = list(self.robot._joint_ft.keys())
    #         #action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
    #         #self.robot.send_action(action_dict)
    #         self.robot.send_action(full_action)
    #     else:
    #         logger.error(f"Unknown mode: {self.test_mode}. No action executed.")
    def execute_action(self, action: np.ndarray):
        full_action = action.copy()
        full_action = torch.from_numpy(full_action).float()
        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return full_action
        if self.test_mode == "autonomous":
            return self.robot.send_action(full_action)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")
            return full_action
    
    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits)."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        # Example stage_pose for bimanual WidowX arms.
        # Each value corresponds to a joint position (in radians) for the 14 joints:
        # [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_left_carriage_joint,
        #  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_left_carriage_joint]
        # The values below represent a "stage" pose, e.g. arms up and open, ready for task start.
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def run_episode(self, task_prompt: str = "look down", dataset=None, events=None):
        """Run a single episode of policy execution."""
        #logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        #listener, events = init_keyboard_listener()
        if events is None:
            events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
                    "switch_to_teleop": False, "switch_to_rollout": False}

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            #cv2.namedWindow(cam, cv2.WINDOW_AUTOSIZE)                
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            #cv2.resizeWindow(cam, 224, 224)
            cv2.resizeWindow(cam, 640, 480)

        if self.gemini_planner:
            cv2.namedWindow("gemini_view", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("gemini_view", 640, 480)
            gemini_high_cam = [c for c in camera_features if 'high' in c][0]
            gemini_low_cam = [c for c in camera_features if 'low' in c][0]

            # Synchronous first Gemini call in auto mode to get initial task
            if self.control_mode == "gemini_auto":
                logger.info("Starting autonomous mode")
                obs = self.robot.capture_observation()
                #img_rgb = obs[gemini_high_cam].numpy()
                gemini_images = {
                    'cam_high': obs[gemini_high_cam].numpy(),
                    'cam_low': obs[gemini_low_cam].numpy(),
                }
                try:
                    #new_task, narration, objects = self.gemini_planner.plan_next_task(img_rgb, self.gemini_planner.allowed_prompts)
                    new_task, narration, objects = self.gemini_planner.plan_next_task(gemini_images, self.gemini_planner.allowed_prompts)
                    if narration:
                        logger.info(f"Gemini: {narration}")
                    if new_task:
                        say_tts('starting episode')
                        time.sleep(2)
                        task_prompt = new_task
                        logger.info(f"'{task_prompt}'")
                        say_tts(task_prompt)
                        print()
                    #self.gemini_annotated_frame = self.gemini_planner.draw_annotations(img_rgb, objects)
                    self.gemini_annotated_frame = self.gemini_planner.draw_annotations(gemini_images['cam_high'], objects)
                except Exception as e:
                    logger.warning(f"Gemini initial query failed: {e}")

        #speech to text
        colors = ['red','blue','pink','yellow','brown']
        template = "pick up {} cube and place in green bucket"
        if self.control_mode in ("speech", "speech_narrate", "speech_objects"):
            if not self.speech_listener:
                from voice_command import VoiceCommandListener
                self.speech_listener = VoiceCommandListener()
            self.speech_listener.start()
            while True:
                command = self.speech_listener.get_command_blocking(timeout=1.0)
                if command:
                    print(f"\n  >> COMMAND READY: \"{command}\"\n")
                    color = next((c for c in colors if c in command), None)
                    if color:
                        task_prompt=template.format(color)
                        logger.info(f"'{task_prompt}'")
                    break

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            if self.speech_listener:
                command = self.speech_listener.get_command()
                if command:
                    print(f"\n  >> COMMAND READY: \"{command}\"\n")
                    color = next((c for c in colors if c in command.lower()), None)
                    if color:
                        task_prompt=template.format(color)
                        logger.info(f"'{task_prompt}'")
                    elif 'exit' in command.lower() or 'stop' in command.lower():
                        break

            if self.gemini_planner and self.gemini_planner.should_query() and not self.gemini_running:                
                import threading
                obs = self.robot.capture_observation()
                #img_rgb = obs[gemini_high_cam].numpy()
                gemini_images = {
                    'cam_high': obs[gemini_high_cam].numpy(),
                    'cam_low': obs[gemini_low_cam].numpy(),
                }

                if self.control_mode == "speech_narrate":
                    def _gemini_worker(img, task):
                        self.gemini_running = True
                        try:
                            narration, objects, success = self.gemini_planner.narrate_and_annotate(img, task)
                            if success:
                                logger.info(f"Gemini: Task succeeded! '{task}'")
                            if narration:
                                logger.info(f"Gemini: {narration}")
                            self.gemini_annotated_frame = self.gemini_planner.draw_annotations(img['cam_high'], objects)
                        except Exception as e:
                            logger.warning(f"Gemini query failed: {e}")
                        finally:
                            self.gemini_running = False
                    threading.Thread(target=_gemini_worker, args=(gemini_images, task_prompt), daemon=True).start()

                elif self.control_mode == "speech_objects":
                    def _gemini_worker(img, task):
                        self.gemini_running = True
                        try:
                            narration, objects = self.gemini_planner.detect_objects(img, task)
                            if narration:
                                logger.info(f"Gemini: {narration}")
                            self.gemini_annotated_frame = self.gemini_planner.draw_annotations(img['cam_high'], objects)
                        except Exception as e:
                            logger.warning(f"Gemini query failed: {e}")
                        finally:
                            self.gemini_running = False
                    threading.Thread(target=_gemini_worker, args=(gemini_images, task_prompt), daemon=True).start()

                elif self.control_mode == "gemini_auto":
                    def _gemini_worker(img, current_prompt):
                        self.gemini_running = True
                        try:
                            new_task, narration, objects = self.gemini_planner.plan_next_task(img, self.gemini_planner.allowed_prompts)
                            if narration:
                                logger.info(f"Gemini: {narration}")
                            if new_task: # and new_task != current_prompt:
                                self.gemini_next_prompt = new_task
                            elif new_task is None:
                                logger.info("All cubes done!")
                                events["exit_early"] = True
                            self.gemini_annotated_frame = self.gemini_planner.draw_annotations(img['cam_high'], objects)
                        except Exception as e:
                            logger.warning(f"Gemini query failed: {e}")
                        finally:
                            self.gemini_running = False
                    #threading.Thread(target=_gemini_worker, args=(img_rgb, task_prompt), daemon=True).start()
                    threading.Thread(target=_gemini_worker, args=(gemini_images, task_prompt), daemon=True).start()

            if self.gemini_annotated_frame is not None:
                cv2.imshow("gemini_view", self.gemini_annotated_frame)
                cv2.waitKey(1)

            if self.gemini_next_prompt:
                if self.gemini_next_prompt != task_prompt:
                    say_tts(self.gemini_next_prompt)
                task_prompt = self.gemini_next_prompt
                logger.info(f"'{task_prompt}'")
                print()
                self.gemini_next_prompt = None

            if self.display:
                display_224=False
                observation_dict = self.robot.capture_observation()
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    if not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if display_224:
                        image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                        cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            # Request new action chunk after consuming the previous one
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                #observation_dict = self.robot.get_observation()
                observation_dict = self.robot.capture_observation()

                # Extract joint positions from observation
                #joint_pos_keys = [k for k in observation_dict.keys() if k.endswith(".pos")]
                #joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])
                joint_positions = observation_dict['observation.state'].numpy()

                # Transform and resize images from all cameras
                # cameras = list(self.robot._cameras_ft.keys())
                #cameras = list(self.robot.cameras.keys())
                #display_224=False
                debug_dir = "./debug_images"
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    # Apply crop if configured by training, this crop only executes if crop_size is in the server metadata
                    cam_short = cam.replace('observation.images.', '')
                    for pattern, crop_size in self.crop_cameras.items():
                        if pattern in cam_short:
                            h, w = image_hwc.shape[:2]
                            y_start = (h - crop_size) // 2
                            x_start = (w - crop_size) // 2
                            image_hwc = image_hwc[y_start:y_start + crop_size, x_start:x_start + crop_size]
                            break
                    if 0: #not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    # convert BGR to RGB
                    #image_resized = cv2.resize(image_hwc, (224, 224))
                    image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                    ##image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    if 1: #display_224:
                        #cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{debug_dir}/{cam}_sample{self.episode_step}.png", cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        #cv2.waitKey(1)
                    image_chw = np.transpose(image_resized, (2, 0, 1))
                    observation_dict[cam] = image_chw

                # Create observation for policy to follow the ALOHA format
                observation = {
                    "state": joint_positions,
                    "images": {cam.replace('observation.images.', ''): observation_dict[cam] for cam in camera_features},
                    "prompt": task_prompt,
                }
                #logger.info(f"Actual: '{task_prompt}'")

                #logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]

                for k in range(self.action_chunk_size):
                    future_t = self.episode_step + k
                    if future_t < self.action_buffer_size:
                        self.action_buffer[future_t].append(self.current_action_chunk[k])

                self.action_chunk_idx = 0
                #logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            # Select action using temporal ensembling if enabled
            if self.temporal_ensemble_coefficient is not None:
                if len(self.action_buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.action_buffer[self.episode_step])  # shape: (N, 14)
                    weights = self._get_weights(len(candidates))  # shape: (N,)
                    a_t = np.average(candidates, axis=0, weights=weights)  # shape: (14,)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]
            # Execute the current action
            # if is_first_step:
            #     logger.info("Moving to start position to avoid large jumps...")
            #     self.move_to_start_position(a_t, duration=5.0)
            #     is_first_step = False
            # else:
            if self.adjust_for_sim_to_real:
                a_t=a_t.copy()
                a_t[7]=1.05*(a_t[7]+0.01)
                a_t[8]=a_t[8]-0.025
                a_t[9]=a_t[9]+0.025
            actual_action = self.execute_action(a_t)

            if dataset is not None:
                obs = self.robot.capture_observation()
                frame = {
                    **obs,
                    "action": actual_action,
                    "task": task_prompt,
                }
                dataset.add_frame(frame)

            self.action_chunk_idx += 1
            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            #logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            if events["exit_early"]:
                events["exit_early"] = False
                break


        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def run_episode_teleoperate(self, task_prompt: str = "look down", dataset=None, events=None):
        """Run a single episode of teleoperation."""
        self.episode_step = 0
        self.is_running = True
        is_first_step = True

        #listener, events = init_keyboard_listener()
        if events is None:
                events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 640, 480)

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            if self.display:
                display_224=False
                observation_dict = self.robot.capture_observation()
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    if not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if display_224:
                        image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                        cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            observation, action = self.robot.teleop_step(record_data=True)

            if dataset is not None:
                #obs = self.robot.capture_observation()
                frame = {
                    **observation,
                    "action": action["action"],
                    "task": task_prompt,
                }
                dataset.add_frame(frame)

            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
 
            if events["exit_early"]:
                events["exit_early"] = False
                break

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def run_episode_intervention(self, task_prompt: str = "look down", dataset=None, events=None):
        """Run a single episode of policy execution."""
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        #listener, events = init_keyboard_listener()
        if events is None:
            events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
                    "switch_to_teleop": False, "switch_to_rollout": False}

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 640, 480)

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            if self.display:
                display_224=False
                observation_dict = self.robot.capture_observation()
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    if not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if display_224:
                        image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                        cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            # Request new action chunk after consuming the previous one
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                #observation_dict = self.robot.get_observation()
                observation_dict = self.robot.capture_observation()

                # Extract joint positions from observation
                #joint_pos_keys = [k for k in observation_dict.keys() if k.endswith(".pos")]
                #joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])
                joint_positions = observation_dict['observation.state'].numpy()

                # Transform and resize images from all cameras
                # cameras = list(self.robot._cameras_ft.keys())
                #cameras = list(self.robot.cameras.keys())
                #display_224=False
                debug_dir = "./debug_images"
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    # Apply crop if configured by training, this crop only executes if crop_size is in the server metadata
                    cam_short = cam.replace('observation.images.', '')
                    for pattern, crop_size in self.crop_cameras.items():
                        if pattern in cam_short:
                            h, w = image_hwc.shape[:2]
                            y_start = (h - crop_size) // 2
                            x_start = (w - crop_size) // 2
                            image_hwc = image_hwc[y_start:y_start + crop_size, x_start:x_start + crop_size]
                            break
                    if 0: #not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    # convert BGR to RGB
                    #image_resized = cv2.resize(image_hwc, (224, 224))
                    image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                    ##image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    if 1: #display_224:
                        #cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{debug_dir}/{cam}_sample{self.episode_step}.png", cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        #cv2.waitKey(1)
                    image_chw = np.transpose(image_resized, (2, 0, 1))
                    observation_dict[cam] = image_chw

                # Create observation for policy to follow the ALOHA format
                observation = {
                    "state": joint_positions,
                    "images": {cam.replace('observation.images.', ''): observation_dict[cam] for cam in camera_features},
                    "prompt": task_prompt,
                }
                #logger.info(f"Actual: '{task_prompt}'")

                #logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]

                for k in range(self.action_chunk_size):
                    future_t = self.episode_step + k
                    if future_t < self.action_buffer_size:
                        self.action_buffer[future_t].append(self.current_action_chunk[k])

                self.action_chunk_idx = 0
                #logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            # Select action using temporal ensembling if enabled
            if self.temporal_ensemble_coefficient is not None:
                if len(self.action_buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.action_buffer[self.episode_step])  # shape: (N, 14)
                    weights = self._get_weights(len(candidates))  # shape: (N,)
                    a_t = np.average(candidates, axis=0, weights=weights)  # shape: (14,)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]
            if self.adjust_for_sim_to_real:
                a_t=a_t.copy()
                a_t[7]=1.05*(a_t[7]+0.01)
                a_t[8]=a_t[8]-0.025
                a_t[9]=a_t[9]+0.025
            actual_action = self.execute_action(a_t)

            if dataset is not None:
                obs = self.robot.capture_observation()
                frame = {
                    **obs,
                    "action": actual_action,
                    "task": task_prompt,
                }
                dataset.add_frame(frame)

            self.action_chunk_idx += 1
            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            #logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if events["switch_to_teleop"]:
                break

        if not events["switch_to_teleop"]:
            self.is_running = False
            logger.info(f"Episode completed after {self.episode_step} steps")
            return

        #transition to teleoperate mode
        events["switch_to_teleop"]=False
        #move leaders to follower positions
        for name in self.robot.follower_arms:
            follower_pos = self.robot.follower_arms[name].read("Present_Position")
            self.robot.leader_arms[name].driver.set_all_modes(trossen.Mode.position)
            self.robot.leader_arms[name].driver.set_all_positions(follower_pos, 5.0, False)
        time.sleep(2)
        logger.info("ready for teleop")
        say_tts("ready for teleop")
        #pause until ready, hit down arrow again to start
        while not events["switch_to_teleop"]:
            time.sleep(0.1)
        #release leaders
        for name in self.robot.leader_arms:
           self.robot.leader_arms[name].write("Torque_Enable", 0)
        
        #restarted, but now teleoperating
        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            if self.display:
                display_224=False
                observation_dict = self.robot.capture_observation()
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    if not display_224:
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if display_224:
                        image_resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_hwc, 224, 224))
                        cv2.imshow(cam, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            observation, action = self.robot.teleop_step(record_data=True)

            if dataset is not None:
                #obs = self.robot.capture_observation()
                frame = {
                    **observation,
                    "action": action["action"],
                    "task": task_prompt,
                }
                dataset.add_frame(frame)

            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
 
            if events["exit_early"]:
                events["exit_early"] = False
                break

        events["switch_to_teleop"]=False

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()

    def autonomous_mode(self, task_prompt: str = "look down", dataset=None, num_episodes=1):
        """Run in autonomous mode, optionally recording episodes."""
        logger.info("Starting autonomous mode")
        listener, events = init_keyboard_listener()

        reset_time_s=10

        #say_tts("Reset environment")
        say_tts("Reset environment")
        time.sleep(reset_time_s)

        recorded_episodes = 0
        while recorded_episodes < num_episodes:

            if events["stop_recording"]:
                say_tts("stopped recording")
                break

            ep_index = dataset.num_episodes if dataset else recorded_episodes
            logger.info(f"Recording episode {ep_index} ({recorded_episodes}/{num_episodes-1})")
            say_tts(f"starting episode {ep_index}")            
            time.sleep(2)
            
            if self.record_mode == "teleoperate":
                self.release_leaders()
                self.run_episode_teleoperate(task_prompt=task_prompt, dataset=dataset, events=events)
            elif self.record_mode == "intervention":
                self.run_episode_intervention(task_prompt=task_prompt, dataset=dataset, events=events)
            else:
                self.run_episode(task_prompt=task_prompt, dataset=dataset, events=events)
 
            self.robot.teleop_safety_stop()
            self.hold_leaders()                

            if not events["stop_recording"] and (
                recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
            ):
                #self.robot.teleop_safety_stop()
                #self.hold_leaders()                
                say_tts("Reset environment")
                time.sleep(reset_time_s)
                if events["exit_early"]:
                    events["exit_early"] = False

            if events["rerecord_episode"]:
                logger.info("Re-recording episode")
                say_tts("Re-record episode")
                time.sleep(3)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if dataset is not None:
                    dataset.clear_episode_buffer()
                continue

            if dataset is not None:
                logger.info(f"Saving episode {ep_index}")
                say_tts(f"Saving episode {ep_index}")
                time.sleep(2)
                dataset.save_episode()
                logger.info(f"Finished saving episode {ep_index}")
                say_tts(f"Finished saving episode {ep_index}")
                time.sleep(3)
        
            recorded_episodes += 1

            if events["stop_recording"]:
                say_tts("stopped recording")
                time.sleep(2)
                break

        if listener is not None:
            listener.stop()
        #logger.info(f"Completed {num_episodes} episodes")

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()
        if self.speech_listener:
            self.speech_listener.stop()

    def hold_leaders(self):
        for name in self.robot.leader_arms:
            self.robot.leader_arms[name].driver.set_all_modes(trossen.Mode.position)

    def release_leaders(self):
        for name in self.robot.leader_arms:
           self.robot.leader_arms[name].write("Torque_Enable", 0)

def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Stationary Kit <-> OpenPI Policy Server Bridge")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="Operation mode: autonomous (execute) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="Transfer cube", help="Task description for the policy") # default="place lid on pot"#default="Transfer cube" #default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--action_chunk_size", type=int, default=50, help="Action chunk size to call and use")
    parser.add_argument("--max_relative_target", type=float, default=0.05, help="Max delta action for robot safety and stability")
    parser.add_argument("--adjust_for_sim_to_real", type=bool, default=False, help="True for sim to real")
    #GEMINI_API_KEY = "key"  # Your Google AI Studio API key
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    parser.add_argument("--gemini_api_key", default=GEMINI_API_KEY, help="Google API key for Gemini Robotics-ER")
    parser.add_argument("--high_level_task", default=None, help="High-level task for Gemini planner")
    parser.add_argument("--control_mode", default="none", choices=["none", "speech", "speech_narrate", "speech_objects", "gemini_auto"],
                        help="Mode 1: speech only, Mode 2: speech + gemini narration, Mode 3: gemini autonomous")    
    #lerobot dataset args:
    parser.add_argument("--repo_id", default=None, help="Dataset repo ID (e.g. ANRedlich/my_dataset)")
    parser.add_argument("--dataset_root", default=None, help="Local dataset root path")
    parser.add_argument("--resume", type=parse_bool, default=False, help="Resume recording")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--use_videos", type=parse_bool, default=True, help="Save videos")
    parser.add_argument("--record_mode", default="rollout", 
                        choices=["rollout", "intervention", "teleoperate"],
                        help="Recording mode: rollout, intervention (policy+teleop switch), teleoperate")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        max_relative_target=args.max_relative_target,
        adjust_for_sim_to_real=args.adjust_for_sim_to_real,
        record_mode=args.record_mode,
    )

    bridge.control_mode = args.control_mode

    if args.gemini_api_key and args.high_level_task and not bridge.control_mode=='speech' and not bridge.control_mode=='none':
        from gemini_planner import GeminiPlanner
        bridge.gemini_planner = GeminiPlanner(
            api_key=args.gemini_api_key,
            high_level_task=args.high_level_task,
            allowed_prompts=[
                "pick up red cube and place in green bucket",
                "pick up blue cube and place in green bucket",
                "pick up pink cube and place in green bucket",
                "pick up yellow cube and place in green bucket",
                "pick up brown cube and place in green bucket",
            ]
        )
        # Warmup call to avoid cold start latency
        logger.info("Warming up Gemini API...")
        try:
            obs = bridge.robot.capture_observation()
            # cam = [c for c in bridge.robot.camera_features.keys() if 'high' in c][0]
            # img = obs[cam].numpy()
            # bridge.gemini_planner.narrate_and_annotate(img, "warming up")
            cam_high = [c for c in bridge.robot.camera_features.keys() if 'high' in c][0]
            cam_low = [c for c in bridge.robot.camera_features.keys() if 'low' in c][0]
            gemini_images = {'cam_high': obs[cam_high].numpy(), 'cam_low': obs[cam_low].numpy()}
            bridge.gemini_planner.narrate_and_annotate(gemini_images, "warming up")            
            logger.info("Gemini warmup complete")
        except Exception as e:
            logger.warning(f"Gemini warmup failed: {e}")

    dataset = None
    if args.repo_id and args.dataset_root:
        if args.resume:
            dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
            dataset.start_image_writer(
                num_processes=1,
                num_threads=4 * len(bridge.robot.cameras),
            )
        else:
            dataset = LeRobotDataset.create(
                args.repo_id,
                args.control_freq,
                root=args.dataset_root,
                #robot=bridge.robot,
                robot_type=bridge.robot.robot_type,
                features=bridge.dataset_features,
                use_videos=args.use_videos,
                image_writer_processes=1,
                image_writer_threads=4 * len(bridge.robot.cameras),
            )
    else:
        logger.info("No repo_id/dataset_root specified — running without recording")

    #debug
    #logger.info(f"Dataset features: {dataset.features}")

    bridge.autonomous_mode(task_prompt=args.task_prompt, dataset=dataset, num_episodes=args.num_episodes)
    #bridge.autonomous_mode(task_prompt=args.task_prompt)

    if dataset is not None:
        dataset.stop_image_writer()

    bridge.cleanup()
