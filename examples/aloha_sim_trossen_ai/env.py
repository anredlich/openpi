import gym_aloha  # noqa: F401
print(f"gym_aloha location: {gym_aloha.__file__}")
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
from utils_ import plot_observation_images
import matplotlib.pyplot as plt

class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0, display: bool = False) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x0/10000
        #self._gym = gymnasium.make(task, obs_type=obs_type, max_episode_steps=600, box_size=[0.02,0.02,0.02]) #, tabletop='my_desktop')

        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x1/19999"
        self._gym = gymnasium.make(task, obs_type=obs_type, max_episode_steps=600,
                                   box_size=[0.02,0.02,0.02], box_pos=[0,0,-0.02], tabletop='my_desktop', backdrop='my_backdrop', lighting=[[0.3,0.3,0.3],[0.3,0.3,0.3]])

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

        self.display = display #anr
        self.plt_imgs = [] #anr
        self.steps = 0 #anr

        self.cam_list = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] #anr

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0
        #self.cam_list = list(gym_obs["pixels"].keys()) #anr

        # temp_images={
        #         cam: np.transpose(self._last_obs["images"][cam].copy(),(1, 2, 0)) 
        #         for cam in self.cam_list}

        #setup display #anr
        self.steps = 0
        if self.display:
            #if "top" in gym_obs["pixels"]:
            #    cam_list = ["top"]
            #else:
            #    cam_list = ["cam_high"]
            self.plt_imgs = plot_observation_images(gym_obs['pixels'], self.cam_list)
            #self.plt_imgs = plot_observation_images(temp_images, self.cam_list)
            plt.pause(0.02)
 
    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated or reward==5 #reward=5 is hack so either finger in transfer task is ok, anr 1/12/26
        self._episode_reward = max(self._episode_reward, reward)

        self.steps += 1
        #display #anr
        if self.display:
            #if "top" in gym_obs["pixels"]:
            #    cam_list = ["top"]
            #else:
            #    cam_list = ["cam_high"]
            for i in range(len(self.cam_list)):
                self.plt_imgs[i].set_data(gym_obs['pixels'][self.cam_list[i]])
            plt.pause(0.02)
            print(f"step= {self.steps} reward={reward}")

    def _convert_observation(self, gym_obs: dict) -> dict:
        def process_image(img):
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            # Convert axis order from [H, W, C] --> [C, H, W]
            return np.transpose(img, (2, 0, 1))
        
        return {
            "state": gym_obs["agent_pos"],
            "images": {
                cam: process_image(gym_obs["pixels"][cam]) 
                for cam in self.cam_list
            },
        }
    
    # def _convert_observation(self, gym_obs: dict) -> dict:
    #     if "top" in gym_obs["pixels"]:
    #         img = gym_obs["pixels"]["top"]
    #     else:
    #         img = gym_obs["pixels"]["cam_high"]
    #     img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    #     # Convert axis order from [H, W, C] --> [C, H, W]
    #     img = np.transpose(img, (2, 0, 1))

    #     return {
    #         "state": gym_obs["agent_pos"],
    #         "images": {"cam_high": img},
    #     }
