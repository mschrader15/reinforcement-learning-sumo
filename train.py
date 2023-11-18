import sys
import click
from rl_sumo.helpers.preprocessing import execute_preprocessing_tasks
from rl_sumo.parameters.params import get_parameters
from trainers import TRAINING_FUNCTIONS


def preprocessing(sim_params, *args, **kwargs):
    """Execute preprocessing tasks. They should be passed to the configuration
    like { ... "Simulation": { "pre_processing_tasks": [ {"python_path":
    "tools.preprocessing.my_custom_function", "module_path": "<absolute path to
    module root>" }, ... ], "file_root": ... }

    The entire simulation object is passed to the function.
    """
    # add the root location to the path
    if sim_params["pre_processing_tasks"]:
        for task in sim_params.pre_processing_tasks:
            sys.path.insert(0, task["module_path"])
            execute_preprocessing_tasks([[task["module_path"], (sim_params,)]])


@click.option(
    "--config_path",
    help="Path to the JSON configuration file",
)
def _main(config_path):
    """This script runs the desired RL training.

    Before it does the training, it will create environment and
    simulation parameter classes based on the configuration file input
    and run desired preprocessing tasks

    The actual RL training functions should live in
    ./trainers/training_functions.py
    """

    # get the sim and environment parameters
    env_params, sim_params = get_parameters(config_path)

    # preprocessing
    preprocessing(sim_params)

    TRAINING_FUNCTIONS[env_params.algorithm.lower()](sim_params, env_params)


# this is to bypass the pylint errors
main = click.command()(_main)

if __name__ == "__main__":
    main()
