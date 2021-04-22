import os
from copy import deepcopy
from helpers import make_create_env


def run_no_rl(sim_params, env_params):
    # pylint: disable=unused-variable
    gym_name, create_env = make_create_env(env_params, sim_params)

    env = create_env()

    # with env(env_params=env_params, sim_params=sim_params) as e:

    # for _ in range(10):
    for _ in range(3):

        done = False

        while not done:

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

        env.reset()

    env.close()


def run_rllib_es(sim_params, env_params):

    import ray
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class
    from ray.tune import run_experiments

    alg_run = 'ES'

    # start ray
    ray.init()

    # initialize the gym
    gym_name, create_env = make_create_env(env_params, sim_params)

    # get the agent that is desired
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = 1 #min(env_params.cpu_num, env_params.num_rollouts)
    config["episodes_per_batch"] = env_params.num_rollouts
    config["eval_prob"] = 0.05
    # optimal parameters
    config["noise_stdev"] = 0.02
    config["stepsize"] = 0.02

    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["observation_filter"] = "NoFilter"

    # register the environment
    register_env(gym_name, create_env)

    # experiment
    exp_tag = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 25,
        "max_failures": 999,
        "stop": {"training_iteration": 500},
        "num_samples": 1,
    }

    #
    trials = run_experiments({
        env_params.name: exp_tag
    })



TRAINING_FUNCTIONS = {'no-rl': run_no_rl,
                      'es': run_rllib_es}
