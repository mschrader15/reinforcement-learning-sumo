import gym
import sumolib
import atexit

import traci.exceptions
from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
import numpy as np
import traceback
from copy import deepcopy
from math import floor
from my_gym.core import Kernel
from my_gym.core import GlobalObservations
from my_gym.core import GlobalActor
from my_gym.core import rewarder
from abc import ABCMeta, abstractmethod


class TLEnv(gym.Env, metaclass=ABCMeta):

    def __init__(self,
                 env_params,
                 sim_params, ):

        # counters
        self.step_counter = 0
        self.time_counter = 0

        # read in the parameters
        self.env_params = deepcopy(env_params)
        self.sim_params = deepcopy(sim_params)

        # calculate the "true" simulation horizon
        self.horizon = self.env_params.sims_per_step * self.env_params.horizon

        # find an open port
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()

        # instantiate the kernel
        self.k = Kernel(self.sim_params)

        # create the observer
        self.observer = GlobalObservations(net_file=sim_params.net_file, tl_ids=sim_params.tl_ids, name="Global")

        # create the action space
        self.actor = GlobalActor(tl_settings_file=sim_params.tl_settings_file)

        # run the simulation to create the traci connection. I think this makes everything pickle-able!
        traci_c = self.k.start_simulation()

        # pass the traci connection back to the kernel
        self.k.pass_traci_kernel(traci_c)

        # pass the observer traci function call back to the kernel
        self.k.add_traci_call(self.observer.register_traci(traci_c))

        # register the actor
        self.actor.register_traci(traci_c)

        # create the reward function
        self.rewarder = getattr(rewarder, self.env_params.reward_class)(sim_params, env_params)

        # pass the rewarder traci function call back to the kernel
        self.k.add_traci_call(self.rewarder.register_traci(traci_c))

        # terminate sumo on exit
        atexit.register(self.terminate)

    @property
    def action_space(self):
        return MultiDiscrete([*self.actor.discrete_space_shape])
        # return Box(
        #     low=0,
        #     high=self.actor.max_value,
        #     shape=(self.actor.size,),
        #     dtype=np.float32,
        # )

    @property
    def observation_space(self):

        # traffic_light_shapes = self.actor.size['']

        traffic_light_states = MultiDiscrete([*self.actor.discrete_space_shape])

        traffic_light_colors = MultiDiscrete(self.actor.size['color'])

        # # MultiDiscrete()
        # traffic_light_states = Box(
        #     low=0,
        #     high=6383,  # this is per the enumeration format in the observer class
        #     shape=self.action_space.shape,
        #     dtype=np.float32
        # )

        traffic_light_times = Box(
            low=0,
            high=self.sim_params.sim_length,
            # the value is actually the time delta since the start of the last green state but theoretical max is sim length
            shape=self.action_space.shape,
            dtype=np.float32
        )

        # traffic_light_transition_active = Box(
        #     low=0,
        #     high=1,
        #     shape=self.action_space.shape,
        #     dtype=np.float32
        # )

        vehicle_num = Box(
            low=0,
            # unrealistic, but setting the maximum number of cars in any lane = to the distance that the camera can see in meters
            high=self.observer.distance_threshold,
            shape=(self.observer.get_lane_count(),),
            dtype=np.float32,
        )

        return Tuple((traffic_light_states,  traffic_light_times, traffic_light_colors, vehicle_num))

    def apply_rl_actions(self, rl_actions):
        """Specify the actions to be performed by the rl agent(s).

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm
        """
        if rl_actions is None:
            return

        actions = self.clip_actions(rl_actions) if self.env_params.clip_actions else rl_actions

        # convert the actions to integers
        # actions = list(map(floor, rl_actions))
        

        # update the lights
        if not self.sim_params.no_actor:
            self.actor.update_lights(action_list=actions, sim_time=self.k.sim_time)

    def get_state(self, subscription_data):
        """
        Return the state of the simulation as perceived by the RL agent.

        Returns
        -------
        state : array_like, in the shape of self.action_space
        """

        # prompt the observer class to find all counts
        count_list = self.observer.get_counts(subscription_data)

        # get the current traffic light states, a tuple of lists is returned
        tl_states = self.actor.get_current_state()

        return (*tl_states, count_list)

    def clip_actions(self, rl_actions=None):
        """Clip the actions passed from the RL agent.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm

        Returns
        -------
        array_like
            The rl_actions clipped according to the box or boxes
        """
        return rl_actions
        # ignore if no actions are issued
        # if rl_actions is None:
        #     return

        # return np.clip(rl_actions,
        #                a_min=self.action_space.low,
        #                a_max=self.action_space.high
        #                )

    def reset(self, ):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        # reset the time counter
        # self.time_counter = 0

        # restart completely if we should restart
        if self.step_counter > 1e6:
            self._hard_reset()

        # # else reset the simulation
        else:
            try:
                self.k.reset_simulation()
                self._reset_action_obs_rewarder()
            except traci.exceptions.FatalTraCIError:
                self.reset()
        subscription_data = self.k.simulation_step()
        return self.get_state(subscription_data)

    def _hard_reset(self):
        """
        This function is called when SUMO needs to be tore down and rebuilt

        @return: None
        """

        self.step_counter = 0
        self.k.close_simulation()
        traci_c = self.k.start_simulation()
        self.k.pass_traci_kernel(traci_c)
        self._reset_action_obs_rewarder()

        # pass traci to the observer and the corresponding functions back to the
        self.k.add_traci_call(self.observer.register_traci(traci_c))

        # pass traci to the actor
        self.actor.register_traci(traci_c)

        # pass traci to the rewarder and then register the calls
        self.k.add_traci_call(self.rewarder.register_traci(traci_c))

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        for _ in range(self.env_params.sims_per_step):

            # increment the step counter
            self.step_counter += 1

            # self.time_counter += self.sim_params.sim_step

            # apply the rl agent actions
            self.apply_rl_actions(rl_actions=action)

            # step the simulation
            subscription_data = self.k.simulation_step()

            # check for collisions and kill the simulation if so
            crash = self.k.check_collision()
            if crash:
                print("There was a crash")
                break

        observation = self.get_state(subscription_data)
        reward = self.calculate_reward(subscription_data)

        done = (self.k.sim_time > self.horizon) or crash

        info = {
            'sim_time': self.k.sim_time,
        }

        return observation, reward, done, info

    def calculate_reward(self, subscription_data) -> float:
        return self.rewarder.get_reward(subscription_data)

    def _reset_action_obs_rewarder(self, ):
        self.observer.re_initialize()
        self.actor.re_initialize()
        self.rewarder.re_initialize()

    def terminate(self, ):
        try:
            self.k.close_simulation()

        except FileNotFoundError:
            # Skip automatic termination. Connection is probably already closed
            print(traceback.format_exc())
