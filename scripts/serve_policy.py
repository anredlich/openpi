import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    ALOHA_SIM_X = "aloha_sim_x"
    ALOHA_SIM_TROSSEN_AI = "aloha_sim_trossen_ai"
    ALOHA_SIM_TROSSEN_AI_PI05 = "aloha_sim_trossen_ai_pi05"
    ALOHA_SIM_TROSSEN_AI_FINETUNE = "aloha_sim_trossen_ai_finetune"
    ALOHA_SIM_TROSSEN_AI_FULL_FINETUNE = "aloha_sim_trossen_ai_full_finetune"
    ALOHA_SIM_LORA_FINETUNE = "aloha_sim_low_mem_finetune"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.ALOHA_SIM_X: Checkpoint(
        config="pi0_aloha_sim_x",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        #dir="gs://openpi-assets/checkpoints/pi0_base",
        #dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM_TROSSEN_AI: Checkpoint(
        config="pi0_aloha_sim_trossen_ai",
        #dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_base",
        #dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM_TROSSEN_AI_FINETUNE: Checkpoint(
        config="pi0_aloha_sim_trossen_ai_mem_finetune_v2",
        #dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        #dir="gs://openpi-assets/checkpoints/pi0_base",
        #dir="gs://openpi-assets/checkpoints/pi05_base",
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x0/10000"
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x1/19999" #trossen_ai_stationary_sim_pi013
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x3/19999" #trossen_ai_stationary_sim_pi013
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x2/19999" #trossen_ai_stationary_place_lids_04
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x2/39999" #trossen_ai_stationary_place_lids_04
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x2/59999" #trossen_ai_stationary_place_lids_04
        dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v2/trossen_ai_stationary_x4/9999" #act_trossen_ai_stationary_real_03
        #dir="./checkpoints/hf_checkpoint"
    ),
    EnvMode.ALOHA_SIM_TROSSEN_AI_FULL_FINETUNE: Checkpoint(
        #config="pi0_aloha_sim_trossen_ai_full_finetune_v0", #for 'place lid on pot'
        #config="pi0_aloha_sim_trossen_ai_mem_finetune_v3", #actually full fine tune for 'pick up yellow cube and place in silver pan' etc
        config="pi0_aloha_sim_trossen_ai_full_finetune_v4", #for 'pick up yellow cube and place in silver pan' etc
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_full_finetune_v0/trossen_ai_stationary_x5/19999" #trossen_ai_stationary_place_lids_04
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v3/trossen_ai_stationary_x6/19999" #trossen_ai_stationary_pick_and_place_07
        #dir="./checkpoints/pi0_aloha_sim_trossen_ai_mem_finetune_v3/trossen_ai_stationary_x6/39999" #trossen_ai_stationary_pick_and_place_07
        dir="./checkpoints/pi0_aloha_sim_trossen_ai_full_finetune_v4/trossen_ai_stationary_x7/19999" #trossen_ai_stationary_pick_and_place_07
    ),
    EnvMode.ALOHA_SIM_TROSSEN_AI_PI05: Checkpoint(
        config="pi05_aloha_sim_trossen_ai",
        #dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        #dir="gs://openpi-assets/checkpoints/pi0_base",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM_LORA_FINETUNE: Checkpoint(
        config="pi0_aloha_sim_low_mem_finetune",
        dir="./checkpoints/pi0_aloha_sim_low_mem_finetune/my_lora_experiment/19999",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
