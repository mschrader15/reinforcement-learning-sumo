"""This function contains the main training functions ran by train.py."""

import os
from copy import deepcopy
from rl_sumo.helpers.register_environment import make_create_env

# The frequency with which rllib checkpoints are saved
CHECKPOINT_FREQEUNCY = 10


def run_no_rl(sim_params, env_params):
    """Run the environment specified in env_params without reinforcement
    learning.

    The actions will be a random sample of the action space
    unless an override is desired

    Args:
        sim_params
        env_params
    """
    import time

    # pylint: disable=unused-variable
    gym_name, create_env = make_create_env(env_params, sim_params)

    env = create_env()

    for i in range(1):
        env.reset()

        rewards = []

        done = False
        while not done:
            # step and act on the evironment
            action = env.action_space.sample()
            observation, reward, done, truncate, info = env.step(action)
            # record the reward
            rewards.append([env.k.sim_time, reward])

            # this is custom for recording video (taking SUMO gui pictures)
            # if (
            #     env_params["video_dir"]
            #     and not env.k.sim_time % 1
            #     and env.k.sim_time < 300
            # ):
            #     env.k.traci_c.gui.screenshot(
            #         "View #0",
            #         os.path.join(
            #             env_params["video_dir"], "frame_%06d.png" % env.k.sim_time
            #         ),
            #     )

        # env.reset()

        # save the rewards if emissions are also required
        # if sim_params["emissions"]:
        #     reward_file = os.path.join(
        #         *os.path.split(sim_params["emissions"])[:-1], f"rewards_run_{i}.csv"
        #     )
        #     with open(reward_file, "w") as f:
        #         writer = csv.writer(f, dialect="excel")
        #         writer.writerows(rewards)

    # close the environment
    env.close()

    if sim_params["emissions"]:
        # let sumo finish building the output file
        time.sleep(0.5)

        emission_path_csv = sim_params["emissions"][:-4] + ".csv"
        # xml2csv(sim_params['emissions'], 'emissions', emission_path_csv)
        os.remove(sim_params["emissions"])

        # print the location of the emission csv file
        print("\nGenerated emission file at " + emission_path_csv)


def run_rllib_es(sim_params, env_params):
    """Run Rllib train with Evolutionary Strategies Algorithm.

    Args:
        sim_params
        env_params
    """

    import ray
    from ray.tune.registry import register_env

    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class
    from ray.tune import run_experiments

    alg_run = "ES"

    # start ray
    ray.init()

    # force no gui, crashes computer if so many instances spawn
    sim_params.gui = False

    # initialize the gym
    gym_name, create_env = make_create_env(env_params, sim_params)

    # get the agent that is desired
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["horizon"] = env_params.horizon
    config["num_workers"] = min(env_params.cpu_num, env_params.num_rollouts)
    config["episodes_per_batch"] = env_params.num_rollouts
    config["eval_prob"] = 0.05
    # optimal parameters
    config["noise_stdev"] = 0.02
    config["stepsize"] = 0.02

    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["clip_actions"] = False
    config["observation_filter"] = "NoFilter"

    # add the environment parameters to the config settings so that they will be saved
    # Helps simplify replay
    config["env_config"]["settings_input"] = env_params.json_input

    # register the environment
    register_env(gym_name, create_env)

    # experiment
    exp_tag = {
        "run": alg_run,
        "env": gym_name,
        "config": {**config},
        "checkpoint_freq": CHECKPOINT_FREQEUNCY,
        "max_failures": 999,
        "stop": {"training_iteration": 500},
        "num_samples": 1,
    }

    if env_params["restore_checkpoint"]:
        exp_tag["restore"] = env_params["restore_checkpoint"]

    # pylint: disable=unused-variable
    run_experiments({env_params.name: exp_tag})


def run_rllib_ppo(sim_params, env_params):
    """Run Rllib train with PPO Algorithm.

    Args:
        sim_params
        env_params
    """

    import ray
    from ray.tune.registry import register_env
    from ray.rllib.algorithms import ppo
    from models.base_model import IPPO
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model("ippo", IPPO)

    ray.init()
    # # force no gui, crashes computer if so many instances spawn
    # sim_params.gui = False

    # initialize the gym
    gym_name, create_env = make_create_env(env_params, sim_params)

    # register the environment
    register_env(gym_name, create_env)

    config = (
        ppo.PPOConfig()
        .environment(
            gym_name,
            disable_env_checking=True,
        )
        .resources(
            # num_workers=env_params.cpu_num - 1,
            # num_gpus=z1,
            # num_gpus_per_learner_worker=1,
        )
        .rollouts(
            num_rollout_workers=env_params.cpu_num - 1,
            enable_connectors=True,
            batch_mode="truncate_episodes",
        )
        .framework("torch")
        .training(_enable_learner_api=False)
        .rl_module(_enable_rl_module_api=False)
        .training(
            gamma=0.99,
            lr=5e-5,
            model={
                "custom_model": "ippo",
                # Extra kwargs to be passed to your model's c'tor.
                "custom_model_config": {},
            },
        )
    )

    algo = config.build()
    algo.train()


# a helper dictionary to make selecting the desired algorithm easier
TRAINING_FUNCTIONS = {"no-rl": run_no_rl, "es": run_rllib_es, "ppo": run_rllib_ppo}
