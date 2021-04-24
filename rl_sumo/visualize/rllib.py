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

import argparse
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

def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_pkl(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # flow_params = get_agent_class(config)

    # this is the input file that was passed to train
    # master_input = config['env_config']['settings_input']

    try:

        env_params, sim_params = get_parameters(config['env_config']['settings_input'])
    except KeyError:
        env_params, sim_params = get_parameters(input("Config file cannot be found. Enter the path: "))

    # HACK: for old environment names
    if 'my_gym' in env_params.environment_location:
        env_params.environment_location = 'rl_sumo.environment'

    agent_cls = get_agent_class(env_params.algorithm)

    # sim_params.restart_instance = True
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    setattr(sim_params, 'emissions_path', args.emissions_output)

    # set the gui to true
    sim_params.gui = True

    gym_name, create_env = make_create_env(env_params=env_params, sim_params=sim_params)
    register_env(gym_name, create_env)

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # just use 1 cpu for replay
    config['num_workers'] = 1 if env_params.algorithm.lower() in 'es' else 0

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=gym_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
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

    for i in range(args.num_rollouts):

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

    if args.emissions_output:
        path = result_dir if '/' not in args.emissions_output else args.emissions_output

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
    if args.emissions_output:

        time.sleep(0.5)

        emission_path_csv = args.emissions_output[:-4] + ".csv"
        xml2csv(args.emissions_output, 'emissions', emission_path_csv)
        os.remove(args.emissions_output)

        # print the location of the emission csv file
        print("\nGenerated emission file at " + emission_path_csv)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='[Flow] Evaluates a reinforcement learning agent '
                                     'given a checkpoint.',
                                     epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument('result_dir', type=str, help='Directory containing results')

    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument('--run',
                        type=str,
                        help='The algorithm or model to train. This may refer to '
                        'the name of a built-on algorithm (e.g. RLLib\'s DQN '
                        'or PPO), or a user-defined trainable function or '
                        'class registered in the tune registry. '
                        'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument('--num_rollouts', type=int, default=1, help='The number of rollouts to visualize.')
    parser.add_argument(
        '--emissions_output',
        type=str,
        default=None,  # this is obvi inconsistant with the type but :shrug:
        help='Specifies whether to generate an emission file from the '
        'simulation')
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='Specifies whether to use the \'evaluate\' reward '
                        'for the environment.')
    parser.add_argument('--render_mode',
                        type=str,
                        default='sumo_gui',
                        help='Pick the render mode. Options include sumo_web3d, '
                        'rgbd and sumo_gui')
    parser.add_argument('--save_render',
                        action='store_true',
                        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
                        'with pyglet rendering.')
    parser.add_argument('--horizon', type=int, help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    visualizer_rllib(args)
