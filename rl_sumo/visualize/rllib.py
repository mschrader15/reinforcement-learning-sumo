"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import click
import gym
import numpy as np
import os
import sys
import time
import csv
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from rl_sumo.helpers import make_create_env
from rl_sumo.helpers import get_rllib_config
from rl_sumo.helpers import get_rllib_pkl
from rl_sumo.helpers import xml2csv
from rl_sumo.helpers import get_parameters

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


@click.argument('result_dir', nargs=1, help="Path to the rllib results folder", type=click.Path(exists=True))
@click.argument('checkpoint_num', nargs=1, help="The number of the checkpoint to simulate")
def get_config(result_dir, checkpoint_num):

    result_dir = result_dir if result_dir[-1] != '/' else result_dir[:-1]

    config = get_rllib_pkl(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    try:
        env_params, sim_params = get_parameters(config['env_config']['settings_input'])
    except KeyError:
        env_params, sim_params = get_parameters(input("Config file cannot be found. Enter the path: "))

    # HACK: for old environment names
    if 'my_gym' in env_params.environment_location:
        env_params.environment_location = 'rl_sumo.environment'

    agent_cls = get_agent_class(env_params.algorithm)

    gym_name, create_env = make_create_env(env_params=env_params, sim_params=sim_params)
    register_env(gym_name, create_env)

    agent = agent_cls(env=gym_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
    agent.restore(checkpoint)

    return agent, gym_name, config, multiagent, env_params, sim_params, result_dir



@click.option('--emissions_output', default=None)
@click.option('--horizon', type=int, help='Specifies the horizon.', default=None)
@click.option('--num_rollouts', type=int, help='The number of rollouts to visualize.', default=1)
def _visualizer_rllib(emissions_output, horizon, num_rollouts):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.

    example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

    Here the arguments are:
    1 - the path to the simulation results
    2 - the number of the checkpoint

    """
    # start up ray
    ray.init(num_cpus=1)
    

    # pylint: disable=no-value-for-parameter
    agent, gym_name, config, multiagent, env_params, sim_params, result_dir = get_config()

    setattr(sim_params, 'emissions_path', emissions_output)

    # set the gui to true
    sim_params.gui = True

    # lower the horizon if testing
    if horizon:
        config['horizon'] = horizon
        env_params.horizon = horizon

    # just use 1 cpu for replay
    config['num_workers'] = 1 if env_params.algorithm.lower() in 'es' else 0

    if hasattr(agent, "local_evaluator") and os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(gym_name)

    if multiagent:
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        rets = {key: [] for key in config['multiagent']['policies'].keys()}
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
            size = config['model']['lstm_cell_size']
            state_init = {
                key: [np.zeros(size, np.float32), np.zeros(size, np.float32)]
                for key in config['multiagent']['policies'].keys()
            }

        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    for i in range(num_rollouts):

        rewards = [["sim_time", "reward"]]

        state = env.reset()
        ret = {key: [0] for key in rets.keys()} if multiagent else 0
        for _ in range(env_params.horizon):
            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(state[agent_id], policy_id=policy_map_fn(agent_id))
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                action = agent.compute_action(state)
                state, reward, done, _ = env.step(action)
                rewards.append([env.k.sim_time, reward])
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)

    if emissions_output:
        path = result_dir if '/' not in emissions_output else emissions_output

        reward_file = os.path.join(*os.path.split(path)[:-1], f'rewards_run_{i}.csv')
        with open(reward_file, 'w') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerows(rewards)

    print('==== Summary of results ====')
    print("Return:")
    # print(mean_speed)
    if multiagent:
        for agent_id, rew in rets.items():
            print('For agent', agent_id)
            print(rew)
            print('Average, std return: {}, {} for agent {}'.format(np.mean(rew), np.std(rew), agent_id))
    else:
        print(rets)
        print('Average, std: {}, {}'.format(np.mean(rets), np.std(rets)))

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if emissions_output:

        time.sleep(0.5)

        emission_path_csv = emissions_output[:-4] + ".csv"
        xml2csv(emissions_output, 'emissions', emission_path_csv)
        os.remove(emissions_output)

        # print the location of the emission csv file
        print("\nGenerated emission file at " + emission_path_csv)


main = click.command()(_visualizer_rllib)

if __name__ == '__main__':
    
    main()
