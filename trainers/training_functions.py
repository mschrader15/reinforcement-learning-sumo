import os
from copy import deepcopy
from rl_sumo.helpers import make_create_env


def run_no_rl(sim_params, env_params):
    import csv
    from rl_sumo.helpers import xml2csv
    import time

    # pylint: disable=unused-variable
    gym_name, create_env = make_create_env(env_params, sim_params)

    env = create_env()

    # with env(env_params=env_params, sim_params=sim_params) as e:

    # for _ in range(10):
    for i in range(1):

        env.reset()

        rewards = []

        done = False
        while not done:

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if sim_params['gather_reward']:
                rewards.append([env.k.sim_time, reward])

        if sim_params['emissions']:
            reward_file = os.path.join(*os.path.split(sim_params['emissions'])[:-1], f'rewards_run_{i}.csv')
            with open(reward_file, 'w') as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerows(rewards)

    env.close()

    if sim_params['emissions']:

        # let sumo finish building the output file
        time.sleep(0.5)

        emission_path_csv = sim_params['emissions'][:-4] + ".csv"
        xml2csv(sim_params['emissions'], 'emissions', emission_path_csv)
        os.remove(sim_params['emissions'])

        # print the location of the emission csv file
        print("\nGenerated emission file at " + emission_path_csv)


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

    # force no gui, crashes computer if so many instances spawn
    sim_params.gui = False

    # initialize the gym
    gym_name, create_env = make_create_env(env_params, sim_params)

    # get the agent that is desired
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config['horizon'] = env_params.horizon
    config["num_workers"] = min(env_params.cpu_num, env_params.num_rollouts)
    config["episodes_per_batch"] = env_params.num_rollouts
    config["eval_prob"] = 0.05
    # optimal parameters
    config["noise_stdev"] = 0.02
    config["stepsize"] = 0.02

    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config['clip_actions'] = False
    config["observation_filter"] = "NoFilter"

    # add the environment parameters to the config settings so that they will be saved
    # Helps simplify replay
    config['env_config']['settings_input'] = env_params.json_input

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
        "stop": {
            "training_iteration": 500
        },
        "num_samples": 1,
    }

    #
    trials = run_experiments({env_params.name: exp_tag})


def run_rllib_ppo(sim_params, env_params):

    import ray
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class
    from ray.tune import run_experiments

    alg_run = "PPO"

    ray.init()

    # force no gui, crashes computer if so many instances spawn
    sim_params.gui = False

    # initialize the gym
    gym_name, create_env = make_create_env(env_params, sim_params)

    # get the agent that is desired
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = min(env_params.cpu_num, env_params.num_rollouts)
    config["train_batch_size"] = env_params.horizon * env_params.num_rollouts
    config["use_gae"] = True
    config["horizon"] = env_params.horizon
    # gae_lambda = 0.97
    # step_size = 5e-4
    # if benchmark_name == "grid0":
    gae_lambda = 0.5
    step_size = 5e-5
    # elif benchmark_name == "grid1":
    #     gae_lambda = 0.3
    config["lambda"] = gae_lambda
    config["lr"] = step_size
    config["vf_clip_param"] = 1e6
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"

    # save the flow params for replay
    config['env_config']['settings_input'] = env_params.json_input

    # Register as rllib env
    register_env(gym_name, create_env)

    exp_tag = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 25,
        "max_failures": 999,
        "stop": {
            "training_iteration": 500
        },
        "num_samples": 3,
    }

    trials = run_experiments({env_params.name: exp_tag})


TRAINING_FUNCTIONS = {'no-rl': run_no_rl, 'es': run_rllib_es, 'ppo': run_rllib_ppo}
