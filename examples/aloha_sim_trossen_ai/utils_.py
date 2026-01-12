#tools to display images #anr

import numpy as np
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt

def plot_observation_images(observation: dict, cam_list: list[str]) -> list[AxesImage]:
    """
    Plot observation images from multiple camera viewpoints.

    :param observation: The observation data containing images.
    :param cam_list: List of camera names used for capturing images.
    :return: A list of AxesImage objects for dynamic updates.
    """
    #images = observation.get("images", {}) #anr, removed
    #if len(images)==0:      #anr in this case the observation is the top camera
    #    images=observation  #anr
    images = {key: observation[key] for key in cam_list if key in observation} #anr added

    # Define the layout based on the provided camera list
    num_cameras = len(cam_list)

    if num_cameras == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, num_cameras)  # Maximum of 3 columns
        rows = (num_cameras + cols - 1) // cols  # Compute rows dynamically
    _, axs = plt.subplots(rows, cols, figsize=(10, 10))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else np.array(axs).flatten() #[axs] anr

    plt_imgs: list[AxesImage] = []
    titles = {
        "cam_high": "Camera High",
        "cam_low": "Camera Low",
        "cam_teleop": "Teleoperator POV",
        "cam_left_wrist": "Left Wrist Camera",
        "cam_right_wrist": "Right Wrist Camera",
        "top": "Top Camera",
    }

    for i, cam in enumerate(cam_list):
        if cam in images:
            plt_imgs.append(axs[i].imshow(images[cam]))
            axs[i].set_title(titles.get(cam, cam))

    for ax in axs.flat:
        ax.axis("off")

    plt.ion()
    return plt_imgs

