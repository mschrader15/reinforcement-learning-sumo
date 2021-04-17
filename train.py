from my_gym.parameters import EnvParams
from my_gym.parameters import SimParams
from helpers import execute_preprocessing_tasks


def get_parameters(path_to_settings):

    """Get the environment parameters"""
    env_params = EnvParams(parameter_file=path_to_settings)

    """Get the simulation parameters"""
    sim_params = SimParams(env_params=env_params, parameter_file=path_to_settings)

    return env_params, sim_params


def preprocessing(sim_params, *args, **kwargs):

    execute_preprocessing_tasks([['tools.preprocessing.generate_input_files', (sim_params, )]])


def register_env()
