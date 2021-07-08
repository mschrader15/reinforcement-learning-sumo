import os
import json
from typing import Tuple, List
from rl_sumo.parameters import SimParams
from rl_sumo.parameters import EnvParams


def get_parameters(input_object: str or dict) -> Tuple[EnvParams, SimParams]:
    """
    loads the parameters from the rllib configuration dump

    """

    if isinstance(input_object, dict):
        settings_dict = input_object
    else:
        with open(input_object, 'rb') as f:
            settings_dict = json.load(f)

    """Get the environment parameters"""
    env_params = EnvParams(settings_dict=settings_dict)

    """Get the simulation parameters"""
    sim_params = SimParams(env_params=env_params, settings_dict=settings_dict)

    return env_params, sim_params


def execute_preprocessing_tasks(fn_list: List[List[str, tuple]]) -> None:
    """
    This is a helper function for executing an arbitrary number of preprocessing functions.
    They should be past as a list of strings like: ['lib.module.function', ]
    @param fn_list: a list of lists, inner list being ['path.to.function', (argument1, argument2, ...)]
    @return: None
    """
    for fn_string, args in fn_list:
        fns = fn_string.split('.')
        fn = __import__(fns[0])
        for fn_next in fns[1:]:
            fn = getattr(fn, fn_next)
        fn(*args)
