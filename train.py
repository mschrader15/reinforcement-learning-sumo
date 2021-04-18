from my_gym.parameters import EnvParams
from my_gym.parameters import SimParams
from helpers import make_create_env
from helpers import execute_preprocessing_tasks
from ray import tune
from ray.tune.registry import register_env
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


def get_parameters(path_to_settings):

    """Get the environment parameters"""
    env_params = EnvParams(parameter_file=path_to_settings)

    """Get the simulation parameters"""
    sim_params = SimParams(env_params=env_params, parameter_file=path_to_settings)

    return env_params, sim_params


def preprocessing(sim_params, *args, **kwargs):

    execute_preprocessing_tasks([['tools.preprocessing.generate_input_files', (sim_params, )]])


def preregister(sim_params, env_params):

    create_env, gym_name = make_create_env(env_params, sim_params)

    # Register as rllib env
    register_env(gym_name, create_env)


def run_no_rl(sim_params, env_params):

    # from my_gym.environment import TLEnv

    tl_env = __import__(sim_params['environment_location'])

    for _ in range(3000):
    reward, _, _, _ =  tl_env.step(action=[])







