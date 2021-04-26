import importlib
import argparse
import sys
import click
from copy import deepcopy
from rl_sumo.helpers import make_create_env
from rl_sumo.helpers import execute_preprocessing_tasks
from rl_sumo.helpers import get_parameters
from trainers import TRAINING_FUNCTIONS


def preprocessing(sim_params, *args, **kwargs):
    """
    Execute preprocessing tasks
    
    """
    # add the root location to the path
    sys.path.insert(0, sim_params.root)

    execute_preprocessing_tasks([['tools.preprocessing.generate_input_files', (sim_params, )]])


@click.option('--config_path', help="Path to the configuration file", )
def _main(config_path):
    

    # get the sim and environment parameters
    env_params, sim_params = get_parameters(config_path)

    # preprocessing
    preprocessing(sim_params)

    TRAINING_FUNCTIONS[env_params.algorithm.lower()](sim_params, env_params)


# this is to bypass the pylint errors
main = click.command()(_main)

if __name__ == "__main__":

    main()








