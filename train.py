import importlib
import argparse
import sys
from copy import deepcopy
from my_gym.helpers import make_create_env
from my_gym.helpers import execute_preprocessing_tasks
from my_gym.helpers import get_parameters
from trainers import TRAINING_FUNCTIONS


def preprocessing(sim_params, *args, **kwargs):
    """
    Execute preprocessing tasks
    
    """
    # add the root location to the path
    sys.path.insert(0, sim_params.root)

    execute_preprocessing_tasks([['tools.preprocessing.generate_input_files', (sim_params, )]])


def commandline_parser(args):

    parser = argparse.ArgumentParser(
        # formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG"
    )

    parser.add_argument(
        '--config_path', type=str, default='./settings/4_16_2020.json', help='path to the configuration file'
    )

    return parser.parse_known_args(args)[0]


def main(cmd_line_args):

    # parse the command line arguments
    args = commandline_parser(cmd_line_args)

    # get the sim and environment parameters
    env_params, sim_params = get_parameters(args.config_path)

    # preprocessing
    # preprocessing(sim_params)

    TRAINING_FUNCTIONS[env_params.algorithm.lower()](sim_params, env_params)

if __name__ == "__main__":

    main(sys.argv[1:])








