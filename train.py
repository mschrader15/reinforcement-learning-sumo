import importlib
import argparse
import sys

from my_gym.parameters import EnvParams
from my_gym.parameters import SimParams
from helpers import make_create_env
from helpers import execute_preprocessing_tasks
# from ray import tune
# from ray.tune.registry import register_env
# try:
#     from ray.rllib.agents.agent import get_agent_class
# except ImportError:
#     from ray.rllib.agents.registry import get_agent_class


def get_parameters(path_to_settings):

    """Get the environment parameters"""
    env_params = EnvParams(parameter_file=path_to_settings)

    """Get the simulation parameters"""
    sim_params = SimParams(env_params=env_params, parameter_file=path_to_settings)

    return env_params, sim_params


def preprocessing(sim_params, *args, **kwargs):

    execute_preprocessing_tasks([['tools.preprocessing.generate_input_files', (sim_params, )]])


# def preregister(sim_params, env_params):
#
#     create_env, gym_name = make_create_env(env_params, sim_params)
#
#     # Register as rllib env
#     register_env(gym_name, create_env)


def run_no_rl(sim_params, env_params):

    gym_name, create_env = make_create_env(env_params, sim_params)

    env = create_env()

    # with env(env_params=env_params, sim_params=sim_params) as e:

    # for _ in range(10):
    for _ in range(1000):

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('done')

    env.close()


def commandline_parser(args):

    parser = argparse.ArgumentParser(
        # formatter_class=argparse.RawDescriptionHelpFormatter,
        # description="Parse argument used when running a Flow simulation.",
        # epilog="python train.py EXP_CONFIG"
    )

    parser.add_argument(
        '--config_path', type=str, default='settings/4_16_2020.json',
        help='path to the configuration file'
    )

    return parser.parse_known_args(args)[0]


def main(cmd_line_args):

    # parse the command line arguments
    args = commandline_parser(cmd_line_args)

    # get the sim and environment parameters
    env_params, sim_params = get_parameters(args.config_path)

    # TODO: dump json of the parameters

    # preprocessing
    # preprocessing(sim_params)

    # run no RL
    run_no_rl(sim_params, env_params)


if __name__ == "__main__":

    main(sys.argv[2:])








