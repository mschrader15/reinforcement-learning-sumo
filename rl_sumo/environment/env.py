import gymnasium
from rl_sumo.core.actors.DualRingActor import GlobalDualRingActor
from rl_sumo.core.observers.per_phase_observer import GlobalPhaseObservations
from rl_sumo.parameters.params import EnvParams, SimParams
import sumolib
import atexit
from gymnasium.spaces import Box, MultiDiscrete, Dict
from gymnasium.utils import seeding
import numpy as np
import traceback
from copy import deepcopy
from rl_sumo.core import Kernel
from rl_sumo.core import rewarder
from abc import ABCMeta

from traci.constants import VAR_LANES, VAR_VEHICLE


MAX_SPEED = 30
AVG_VEHICLE_LENGTH = 4.5  # meters
MIN_GAP = 2


class TLEnv(gymnasium.Env, metaclass=ABCMeta):
    def __init__(
        self,
        env_params: EnvParams,
        sim_params: SimParams,
    ):
        """
        This is a gym environment that creates the OpenAI gym used in
        https://maxschrader.io/reinforcement_learning_and_sumo

        Args:
            env_params: an instance of EnvParams class
            sim_params: an instance of SimParams class
        """

        # counters
        self.step_counter = 0
        self.time_counter = 0
        self.master_reset_count = 0

        # read in the parameters
        self.env_params = deepcopy(env_params)
        self.sim_params = deepcopy(sim_params)

        # calculate the "true" simulation horizon
        self.horizon = self.env_params.horizon

        # find an open port
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()

        # instantiate the kernel
        self.k = Kernel(self.sim_params)

        # create the observer
        self.observer = GlobalPhaseObservations(
            net_file=self.sim_params.net_file,
            nema_file_map=self.sim_params.nema_file_map,
            name="GlobalPhaseObservations",
        )

        # create the action space
        self.actor = GlobalDualRingActor(
            nema_file_map=self.sim_params.nema_file_map,
            network_file=self.sim_params.net_file,
            subscription_method=True,
        )

        # create the reward function
        self.rewarder = getattr(rewarder, self.env_params.reward_class)(
            sim_params, env_params
        )

        # MAX VECTOR LENGTH
        self.max_vector_length = int(
            (self.observer.distance_threshold / (AVG_VEHICLE_LENGTH + MIN_GAP))
        )

        # terminate sumo on exit
        atexit.register(self.terminate)

    @property
    def action_space(self):
        return MultiDiscrete(self.actor.discrete_space_shape)

    @property
    def observation_space(self) -> Dict:
        # max_shape = self.max_vector_length
        return Dict(
            {
                "traffic_light_colors": Box(
                    low=0,
                    high=2,
                    shape=(
                        len(self.actor.tls),
                        max(self.actor.discrete_space_shape),
                        2,
                    ),
                ),
                "radar_states": Box(
                    low=0,
                    high=max(MAX_SPEED, self.observer.distance_threshold, 2),
                    shape=(self.observer.get_phase_count(), self.max_vector_length, 3),
                    dtype=np.float32,
                ),
            }
        )

    def apply_rl_actions(self, rl_actions):
        """Specify the actions to be performed by the rl agent(s).

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm
        """
        if rl_actions is None:
            return

        actions = (
            self.clip_actions(rl_actions)
            if self.env_params.clip_actions
            else rl_actions
        )

        # update the lights
        if not self.sim_params.no_actor:
            self.actor.update_lights(
                action_list=actions,
            )

    def get_state(self, raw_obs) -> Dict:
        """Return the state of the simulation as perceived by the RL agent.

        Returns
        -------
        state : array_like, in the shape of self.action_space
        """
        # prompt the observer class to find all counts
        radar_measures = raw_obs
        # the radar measures need to be padded to the max length
        radar_data = np.stack(
            [
                np.stack(
                    [
                        np.pad(r[k], (self.max_vector_length - len(r[k]) - 1, 1))
                        for r in radar_measures
                    ],
                )
                for k in radar_measures[0].keys()
            ],
            axis=2,
        ).astype(np.float32)

        # get the current traffic light states, a tuple of lists is returned
        tl_states = self.actor.get_current_state()
        tl_array = np.zeros(
            (
                len(self.actor.tls),
                max(self.actor.discrete_space_shape),
                2,
            ),
            dtype=np.float32,
        )
        for i, tl in enumerate(tl_states[0]):
            for j, p in enumerate(tl):
                tl_array[i, p - 1, j] = tl_states[1][i][j]

        # build the observation
        return {
            "traffic_light_colors": np.array(tl_array, dtype=np.float32),
            "radar_states": radar_data,
            # "radar_distances": ,
        }

        # return (*tl_states, count_list)

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

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
            info (dict): the initial info object.
        """
        if seed is not None:
            self.seed(seed)
        # reset the time counter
        # self.time_counter = 0

        # restart completely if we should restart
        if (self.step_counter > 1e6) or (self.master_reset_count < 1):
            self._hard_reset()
        # # else reset the simulation
        else:
            try:
                self.k.reset_simulation()
                self._reset_action_obs_rewarder()
                self._subscribe_n_pass_traci()
            except Exception:
                print("I am hard reseting b.c. of kernel exception")
                self._hard_reset()

        subscription_data = self.k.simulation_step()
        self.actor.get_current_state(self.k.sim_time, subscription_data)
        ok_2_switch = self.actor.get_okay_2_switch(self.k.sim_time)

        if not subscription_data:
            self.reset()

        # reset the counters
        self.master_reset_count += 1
        self.step_counter = 0

        # reset the reward class
        self.rewarder.re_initialize()

        return self.get_state(subscription_data, ok_2_switch), {}

    def _subscribe_n_pass_traci(
        self,
    ):
        self.k.add_traci_call(self.observer.register_traci(self.k.traci_c))
        self.observer.freeze()

        # pass traci to the actor
        self.k.add_traci_call(self.actor.register_traci(self.k.traci_c))
        # take control of the traffic lights
        self.actor.initialize_control()

        if reward_calls := self.rewarder.register_traci(self.k.traci_c):
            self.k.add_traci_call(reward_calls)

    def _hard_reset(self):
        """This function is called when SUMO needs to be tore down and rebuilt.

        @return: None
        """
        self.step_counter = 0
        self.k.close_simulation()
        traci_c = self.k.start_simulation()
        self._reset_action_obs_rewarder()
        self.k.pass_traci_kernel(traci_c)

        # re-pass traci to the actor and the observer
        self._subscribe_n_pass_traci()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # TODO: The seeing function returns a # too large to convert to C long
        # for now just taking first 8 integers
        self.k.set_seed(int(str(seed)[:8]))
        return [
            seed,
        ]

    def _get_raw_obs(self, subscription_data):
        """Return the raw observation space.

        Returns
        -------
        array_like
            The raw observation space
        """
        return self.observer.get_counts(subscription_data)

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further
                step() calls will return undefined results
            truncated (bool): a boolean, true if the trajectory went out of bounds
            info (dict): contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)
        """
        sim_broke = False
        crash = False
        truncate = False

        # this happens before step so that we can get the okay 2 switch
        # when the action is applied
        # okay_2_switch = self.actor.get_okay_2_switch(self.k.sim_time,)
        # apply the rl agent actions
        self.apply_rl_actions(rl_actions=action, sim_time=self.k.sim_time)

        for _ in range(self.env_params.sims_per_step):
            # increment the step counter
            self.step_counter += 1

            # step the simulation
            subscription_data = self.k.simulation_step()

            # check to see if there was a failure
            if not subscription_data:
                sim_broke = True
                break

            # check for collisions and kill the simulation if so.
            crash = self.k.check_collision()

            if crash:
                print("There was a crash")
                break

        if not sim_broke:
            raw_obs = self._get_raw_obs(subscription_data)
            observation = self.get_state(
                subscription_data=subscription_data,
                raw_obs=raw_obs,
                okay_2_switch=okay_2_switch,
            )
            reward = self.calculate_reward(
                subscription_data=subscription_data,
                obs_data=raw_obs,
                actions=action,
                okay_2_switch=okay_2_switch,
            )
        else:
            observation = []
            reward = 1
            truncate = True

        # check long delays
        if self.k.check_long_delay(subscription_data):
            truncate = True

        done = (self.step_counter * self.k.sim_step_size) >= self.horizon

        info = {
            "sim_time": self.k.sim_time,
            "broken": sim_broke,
            "reward": reward,
            # "all_de"
        }

        return observation, reward, done, truncate or crash, info

    def calculate_reward(
        self, subscription_data, obs_data, actions, *args, **kwargs
    ) -> float:
        return self.rewarder.get_reward(
            subscription_data,
            obs_data,
            [tl.action_space[act] for act, tl in zip(actions, self.actor.tls)],
            *args,
            **kwargs,
        )

    def _reset_action_obs_rewarder(
        self,
    ):
        self.observer.re_initialize()
        self.actor.re_initialize()
        self.rewarder.re_initialize()

    def terminate(
        self,
    ):
        try:
            self.k.close_simulation()

        except FileNotFoundError:
            # Skip automatic termination. Connection is probably already closed
            print(traceback.format_exc())

    def close(self):
        """Terminate the simulation."""
        self.terminate()


class TLEnvFlat(TLEnv):
    def __init__(self, *args, **kwargs):
        super(TLEnvFlat, self).__init__(*args, **kwargs)

    @property
    def observation_space(self) -> Dict:
        # max_shape = self.max_vector_length
        # flatten everything
        shape = (len(self.actor.tls) * max(self.actor.discrete_space_shape) * 2) + (
            self.observer.get_phase_count() * self.max_vector_length * 3
        )

        return Box(
            low=0,
            high=max(MAX_SPEED, self.observer.distance_threshold, 2),
            shape=(shape,),
            dtype=np.float32,
        )

    def get_state(self, subscription_data, raw_obs) -> Dict:
        state_dict = super().get_state(subscription_data, raw_obs)

        # flatten everything
        tl_array = state_dict["traffic_light_colors"].flatten()
        radar_array = state_dict["radar_states"].flatten()

        return np.concatenate((tl_array, radar_array))


class RescoEnv(TLEnv):
    def __init__(self, *args, **kwargs):
        super(RescoEnv, self).__init__(*args, **kwargs)

    @property
    def observation_space(self) -> Dict:
        # max_shape = self.max_vector_length
        # flatten everything
        # shape = (len(self.actor.tls), max(self.actor.discrete_space_shape), 5)
        shape = (len(self.actor.tls) * max(self.actor.discrete_space_shape) * 6,)

        return Box(
            low=0,
            high=1,
            shape=(*shape,),
            dtype=np.float32,
        )

    def get_state(self, subscription_data, okay_2_switch, *args, **kwargs) -> Dict:
        states, colors = self.actor.get_current_state(
            subscription_results=subscription_data, sim_time=self.k.sim_time
        )

        empty_state = np.zeros(
            (
                len(self.actor.tls),
                max(self.actor.discrete_space_shape),
                6,
            ),
            dtype=np.float32,
        )

        tl_to_pos_dict = [
            {p.name: i for i, p in enumerate(tls._children)}
            for tls in self.observer.tls
        ]

        for i, tls in enumerate(self.observer.tls):
            res = tls.update_counts(
                lane_info=subscription_data[VAR_LANES],
                vehicle_info=subscription_data[VAR_VEHICLE],
            )
            current_state_pos = [tl_to_pos_dict[i][s] for s in states[i]]
            empty_state[i, current_state_pos, 1] = okay_2_switch[i] * 1
            empty_state[i, current_state_pos, 0] = 1
            # set the okay to switch info
            for j, phase in enumerate(res):
                waiting_time_array = np.array(phase["waiting_time"])
                num_lanes = len(tls._children[j]._children)

                if len(waiting_time_array):
                    empty_state[i, j, 2] = (
                        waiting_time_array < self.k.sim_step_size
                    ).sum() / (
                        self.max_vector_length * num_lanes
                    )  # the number of cars moving
                    empty_state[i, j, 3] = waiting_time_array.sum() / (
                        self.max_vector_length * num_lanes * self.env_params.horizon
                    )  # the total waiting time
                    empty_state[i, j, 4] = (
                        waiting_time_array >= self.k.sim_step_size
                    ).sum() / (
                        self.max_vector_length * num_lanes
                    )  # the number of cars stopped
                    empty_state[i, j, 5] = (
                        sum(phase["speeds"])
                        / max(len(phase["speeds"]), 1)
                        / (self.max_vector_length * num_lanes)
                    )  # the average speed)

        # normalize

        empty_state[:, :, 5] /= MAX_SPEED

        # return np.expand_dims(empty_state, axis=0)
        return empty_state.flatten()

        # dims are tl, lanes,
