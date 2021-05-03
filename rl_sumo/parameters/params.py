import os
from distutils import util
from copy import deepcopy
from datetime import datetime


def safe_getter(_dict: dict, param: str):
    """
    Get and remove an item from a dictionary. 
    If the item doesn't exist, return None

    Args:
        _dict (dict): The dictionary from which to get item
        param (str): the dictionary key

    Returns:
        [type]: either None or _dict[param]
    """
    try:
        return _dict.pop(param)
    except KeyError:
        return None


def make_directory(path):
    """
    make a directory if it doesn't exist

    Args:
        path ([type]): absolute path of the directory to create
    """

    os.makedirs(path, exist_ok=True)


class EnvParams(object):
    def __init__(self, settings_dict: dict):
        """
        The EnvParams used throughout rl_sumo. 
        Contains various information. 

        Args:
            settings_dict (dict): a python dictionary from json.load
        """

        # import the parameters
        self.json_input = settings_dict

        self.name = self.json_input['Name']

        # seperate the dictionary that is being deconstructed from the real input
        params = deepcopy(self.json_input['Environment'])

        self.environment_location: str = safe_getter(params, 'environment_location')

        self.environment_name: str = safe_getter(params, 'environment_name')

        self.algorithm: str = safe_getter(params, 'algorithm') or 'PPO'

        self.warm_up_time: int = safe_getter(params, 'warmup_time') or 3600

        self.sims_per_step: int = safe_getter(params, 'sims_per_step') or 1

        # the horizon is entered as a time. It is divided by the simulation step to get
        # the number of steps required by the RL learner
        self.horizon: int = int((safe_getter(params, 'horizon') or 3600) / \
            self.json_input['Simulation']['sim_step'])

        self.reward_class: str = safe_getter(params, 'reward_class') or 'FCIC'

        self.clip_actions: str = safe_getter(params, 'clip_actions') or True

        self.num_rollouts: int = safe_getter(params, 'num_rollouts') or 50

        self.cpu_num: int = safe_getter(params, 'cpu_num') or 1

        # pass the remaining items in the json input as parameters too
        for key, value in params.items():
            self.__dict__[key] = value

    def __getitem__(self, item):
        return getattr(self, item, None)


class SimParams(object):
    def __init__(self, env_params: EnvParams, settings_dict: dict):
        """
        The EnvParams used throughout rl_sumo. 
        Listed are the must-have parameters but it can be extended 

        Args:
            env_params (EnvParams): 
            settings_dict (dict): a python dictionary from json.load
        """

        # import the parameters
        params = deepcopy(settings_dict['Simulation'])

        try:
            # one could pass the file root as an executable bit of python code
            root = exec(params['file_root'])
        except Exception:
            root = params['file_root']

        self.sim_state_dir: str = os.path.join(root, 'reinforcement-learning-sumo', 'tmp', 'sim_state')

        # make the directory
        make_directory(self.sim_state_dir)

        self.root = root

        self.gui = bool(util.strtobool(str(safe_getter(params, 'gui'))))

        self.port: int = 0

        self.net_file: str = os.path.join(root, safe_getter(params, 'net_file'))

        self.route_file: str = os.path.join(root, safe_getter(params, 'route_file'))

        self.additional_files: [str] = [os.path.join(root, file) for file in safe_getter(params, 'additional_files')]

        self.tl_ids: [str] = safe_getter(params, 'tl_ids')

        self.tl_settings_file: str = os.path.join(root, safe_getter(params, 'tl_settings'))

        self.tl_file: str = os.path.join(root, safe_getter(params, 'tl_file'))

        self.sim_step: float = safe_getter(params, 'sim_step')

        self.warmup_time: float = env_params.warm_up_time

        # sum the warmup time, sims per step * horizon and an extra 1000
        self.sim_length: int = env_params.warm_up_time + (env_params.sims_per_step * env_params.horizon) + 1000

        # determine if the actor is desired or not
        # using this for offline analysis of the reward
        self.no_actor = safe_getter(params, "no_actor") or False

        emissions = safe_getter(params, 'emissions')

        if emissions:
            emissions_path = os.path.join(*os.path.split(emissions)[:-1], env_params.name,
                                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            make_directory(emissions_path)
            setattr(self, 'emissions', os.path.join(emissions_path, os.path.split(emissions)[-1]))

        # add in the rest of the stuff in the configuration file
        for key, value in params.items():
            self.__dict__[key] = value

    def __getitem__(self, item):
        return getattr(self, item, None)
