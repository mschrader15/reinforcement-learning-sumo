import gym
from gym.envs.registration import register

def register_creator():

    entry_point =

    register(
        id=env_name,
        entry_point=entry_point,
        kwargs={
            "env_params": env_params,
            "sim_params": sim_params,
            "network": network,
            "simulator": params['simulator']
        })

    return gym.envs.make(env_name)
