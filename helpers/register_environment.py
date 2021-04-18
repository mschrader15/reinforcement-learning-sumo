import gym
from gym.envs.registration import register


def make_create_env(env_params, sim_params, version=0):

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(env_params['environment_name'], version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(env_params['environment_name'], version)

    def create_env(*_):

        # env_name = "{}-v{}".format(base_env_name, version)

        entry_point = f"{env_params['environment_location']}:{env_params['environment_name']}"

        register(
            id=env_params['environment_name'],
            entry_point=entry_point,
            kwargs={
                "env_params": env_params,
                "sim_params": sim_params,
            }
        )

        return gym.envs.make(env_params['environment_name'])

    return create_env, env_name
