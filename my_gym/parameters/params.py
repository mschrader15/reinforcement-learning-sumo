import os
import json
import pendulum


def load_json(file):
    with open(file, 'rb') as f:
        return json.load(f)


def safe_getter(_dict, param):
    try:
        return _dict[param]
    except KeyError:
        return None


class EnvParams(object):

    def __init__(self, parameter_file):

        # import the parameters
        params = load_json(parameter_file)['Environment']

        self.algorithm: str = safe_getter(params, 'algorithm') or 'PPO'

        self.warm_up_time: int = safe_getter(params, 'warmup_time') or 3600

        self.sims_per_step: int = safe_getter(params, 'sims_per_step') or 1

        self.horizon: int = safe_getter(params, 'horizon') or 3600

        self.reward_function: str = safe_getter(params, 'reward_function') or 'minimize_fuel'


class SimParams(object):

    def __init__(self, env_params: EnvParams, parameter_file: str):

        # import the parameters
        params = load_json(parameter_file)['Simulation']

        try:
            root = exec(params['file_root'])
        except Exception as e:
            root = params['file_root']

        self.gui = params['gui']

        self.port: int = 0

        self.net_file: str = os.path.join(root, params['net_file'])

        self.route_file: str = os.path.join(root, params['route_file'])

        self.additional_files: [str] = [os.path.join(root, file) for file in params['additional_files']]

        self.tl_ids: [str] = os.path.join(root, params['tl_ids'])

        self.tl_settings_file: str = os.path.join(root, params['tl_settings'])

        self.sim_step: float = params['sim_step']

        self.sim_length: int = env_params.warm_up_time + (env_params.sims_per_step * env_params.horizon) + 5  # adding 5 seconds to be safe on reset

        self.start_time: pendulum.DateTime = pendulum.parse(params['start_datetime'])

