from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf


@dataclass
class EnvParams:
    environment_location: Optional[str] = None
    environment_name: Optional[str] = None
    algorithm: str = "PPO"
    warmup_time: int = 3600
    sims_per_step: int = 1
    horizon: int = 3600
    reward_class: str = "FCIC"
    clip_actions: bool = True
    num_rollouts: int = 50
    cpu_num: int = 1
    checkpoint_path: Optional[str] = None
    video_dir: Optional[str] = None


@dataclass
class SimParams:
    settings_dict: dict
    sim_state_dir: Path
    start_time: str
    nema_file_map: Dict[Any, Path]
    gui: bool = False
    port: int = 0
    net_file: Optional[str] = None
    route_file: Optional[str] = None
    additional_files: Optional[List[str]] = None
    additional_args: Optional[List[str]] = None
    sim_step: Optional[float] = None
    warmup_time: float = 0
    sim_length: int = 0
    no_actor: bool = False
    emissions: Optional[str] = None
    pre_processing_tasks: Optional[List[dict]] = None




@dataclass
class RLSUMOConfig:
    Name: str
    Environment: EnvParams
    Simulation: SimParams


def get_parameters(input_object: str or dict) -> Tuple[EnvParams, SimParams]:
    """Loads the parameters from the rllib configuration dump."""

    if isinstance(input_object, dict):
        settings_dict = input_object
    else:
        with open(input_object, "r") as f:
            settings_dict = OmegaConf.load(f)

    rl_sumo_config = OmegaConf.structured(RLSUMOConfig)
    rl_sumo_config = OmegaConf.merge(rl_sumo_config, settings_dict)

    return rl_sumo_config.Environment, rl_sumo_config.Simulation

