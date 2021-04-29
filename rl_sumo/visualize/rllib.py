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


def get_config(result_dir, emissions_output, gui_config_file, tls_record_file):
    """Generates the configuration

    Args:
        result_dir: Path to the rllib results folder
        checkpoint_num: The number of the checkpoint to simulate
    """

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

    if gui_config_file:
        setattr(sim_params, 'gui_config_file', gui_config_file)

    if tls_record_file:
        setattr(sim_params, 'tls_record_file', tls_record_file)


    # HACK: for old environment names
    if 'my_gym' in env_params.environment_location:
        env_params.environment_location = 'rl_sumo.environment'

    agent_cls = get_agent_class(env_params.algorithm)

    # set the gui to true
    sim_params.gui = True

    if emissions_output:
        emissions_output = os.path.join(result_dir, emissions_output)
        # add the emissions path to the environment parameters
        setattr(sim_params, 'emissions', emissions_output)

    gym_name, create_env = make_create_env(env_params=env_params, sim_params=sim_params)
    register_env(gym_name, create_env)

    agent = agent_cls(env=gym_name, config=config)

    return agent, gym_name, config, multiagent, env_params, result_dir, sim_params


def restore_checkpoint(agent, result_dir, checkpoint_num):
    checkpoint = result_dir + '/checkpoint_' + checkpoint_num + '/checkpoint-' + checkpoint_num
    agent.restore(checkpoint)
    return agent


def make_video_directory(video_dir, checkpoint_num):
    current_vid_dir = os.path.join(video_dir, checkpoint_num)
    # try:
    os.mkdir(current_vid_dir)
    # except
    return current_vid_dir


def run_simulation(agent, env, multiagent, config, env_params, use_lstm, state_init, video_dir):

    if multiagent:
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        rets = {key: [] for key in config['multiagent']['policies'].keys()}
    else:
        rets = []

    # for i in range(num_rollouts):

    rewards = [["sim_time", "reward"]]

    state = env.reset()
    # ret = {key: [0] for key in rets.keys()} if multiagent else 0
    for _ in range(env_params.horizon):

        # if multiagent:
        #     action = {}
        #     for agent_id in state.keys():
        #         if use_lstm:
        #             action[agent_id], state_init[agent_id], _ = \
        #                 agent.compute_action(
        #                 state[agent_id], state=state_init[agent_id],
        #                 policy_id=policy_map_fn(agent_id))
        #         else:
        #             action[agent_id] = agent.compute_action(state[agent_id], policy_id=policy_map_fn(agent_id))
        #     for actor, rew in reward.items():
        #         ret[policy_map_fn(actor)][0] += rew

        # else:
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        rewards.append([env.k.sim_time, reward])
        # ret += reward

        # save the image
        if video_dir and not env.k.sim_time % 1:
            env.k.traci_c.gui.screenshot("View #0", os.path.join(video_dir, "frame_%06d.png" % env.k.sim_time))

        if multiagent and done['__all__']:
            break
        if not multiagent and done:
            break

    # if multiagent:
    #     for key in rets.keys():
    #         rets[key].append(ret[key])
    # else:
    #     rets.append(ret)

    return rewards


@click.argument('result_dir', type=click.Path(exists=True))
@click.argument('checkpoints', nargs=-1)
@click.option('--emissions_output', default=None)
@click.option('--horizon', type=int, help='Specifies the horizon.', default=None)
@click.option('--video_dir', type=click.Path(exists=True), help='The number of rollouts to visualize.', default=None)
@click.option('--gui_config_file',
              type=str,
              help='A SUMO configuration file pointing a simulation and a network',
              default=None)
@click.option('--tls_record_file',
              type=str,
              help='An additional file to record traffic light states',
              default=None)
def _visualizer_rllib(result_dir, checkpoints, emissions_output, horizon, video_dir, gui_config_file, tls_record_file):
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
    agent, gym_name, config, multiagent, env_params, result_dir, sim_params = get_config(
        result_dir, emissions_output, gui_config_file, tls_record_file)

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

    if config['model']['use_lstm']:
        use_lstm = True
        # if multiagent:
        #     # map the agent id to its policy
        #     policy_map_fn = config['multiagent']['policy_mapping_fn']
        #     size = config['model']['lstm_cell_size']
        #     state_init = {
        #         key: [np.zeros(size, np.float32), np.zeros(size, np.float32)]
        #         for key in config['multiagent']['policies'].keys()
        #     }

        # else:
        state_init = [
            np.zeros(config['model']['lstm_cell_size'], np.float32),
            np.zeros(config['model']['lstm_cell_size'], np.float32)
        ]
    else:
        use_lstm = False
        state_init = []

    for checkpoint in checkpoints:

        agent = restore_checkpoint(agent, result_dir, checkpoint)

        video_dir_chekpoint = make_video_directory(video_dir, checkpoint) if video_dir else None

        rewards = run_simulation(agent, env, multiagent, config, env_params, use_lstm, state_init, video_dir_chekpoint)

        if emissions_output:
            reward_file = os.path.join(
                result_dir,
                f'rewards_run_checkpoint_{checkpoint}.csv') if '/' not in emissions_output else os.path.join(
                    *os.path.split(emissions_output)[:-1], f'rewards_run_checkpoint_{checkpoint}.csv')
            with open(reward_file, 'w') as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerows(rewards)

        # print('==== Summary of results ====')
        # print("Return:")
        # # print(mean_speed)
        # if multiagent:
        #     for agent_id, rew in rets.items():
        #         print('For agent', agent_id)
        #         print(rew)
        #         print('Average, std return: {}, {} for agent {}'.format(np.mean(rew), np.std(rew), agent_id))
        # else:
        #     print(rets)
        #     print('Average, std: {}, {}'.format(np.mean(rets), np.std(rets)))

        # terminate the environment
        env.unwrapped.terminate()

        # if prompted, convert the emission file into a csv file
        if emissions_output:

            time.sleep(0.5)

            emission_path_csv = os.path.splitext(sim_params['emissions'])[0] + f"_checkpoint_{checkpoint}.csv"
            xml2csv(sim_params['emissions'], 'emissions', emission_path_csv)
            os.remove(sim_params['emissions'])

            # print the location of the emission csv file
            print("\nGenerated emission file at " + emission_path_csv)

        time.sleep(0.5)


main = click.command()(_visualizer_rllib)

if __name__ == '__main__':

    main()
