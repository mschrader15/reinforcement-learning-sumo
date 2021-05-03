import gym
from gym.envs.registration import register


def make_create_env(env_params, sim_params, version=0) -> (str, object):
    """
    This function makes the create_env() function that is used by ray.tune.registry.register_env

    Args:
        env_params: EnvParams class
        sim_params: SimParams class
        version: 0

    Returns:
        fn: a create_env() function
    """

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(env_params.environment_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(env_params.environment_name, version)

    def create_env(*_):

        try:
            entry_point = f"{env_params.environment_location}:{env_params.environment_name}"

            register(
                id=env_name,
                entry_point=entry_point,
                kwargs={
                    "env_params": env_params,
                    "sim_params": sim_params,
                }
            )

            _env = gym.envs.make(env_name)

        except ModuleNotFoundError:

            entry_point = f"rl_sumo.environment:{env_params.environment_name}"

            register(
                id=env_name,
                entry_point=entry_point,
                kwargs={
                    "env_params": env_params,
                    "sim_params": sim_params,
                }
            )

            _env = gym.envs.make(env_name)
        
        return _env

    return env_name, create_env
