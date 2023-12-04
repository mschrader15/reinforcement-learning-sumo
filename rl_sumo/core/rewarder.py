from pathlib import Path
from typing import Dict
import numpy as np
from rl_sumo.parameters.params import SimParams
import traci.constants as tc
from copy import deepcopy
from scipy.ndimage.filters import uniform_filter1d
import sumolib


def minimize_fuel(subscription_values):
    """This is a very simple function that minimizes the fuel consumption of
    the network.

    @param subscription_values: @return:
    """
    vehicle_list = list(subscription_values[tc.VAR_VEHICLE].values())
    fc = sum(vehicle_data[tc.VAR_FUELCONSUMPTION] for vehicle_data in vehicle_list)
    return -1 * fc


class Rewarder:
    def __init__(
        self,
    ):
        pass

    def register_traci(self, traci_c):
        return None

    def get_reward(self, *args, **kwargs):
        pass

    def re_initialize(
        self,
    ):
        pass


class PureFuelMin(Rewarder):
    def __init__(self, sim_params, *args, **kwargs):
        super(PureFuelMin, self).__init__()
        self.sim_step = deepcopy(sim_params.sim_step)
        self.normailizer = 10  # ml_s

    def get_reward(self, subscription_dict):
        vehicle_list = list(subscription_dict[tc.VAR_VEHICLE].values())
        fc = sum(
            vehicle_data[tc.VAR_FUELCONSUMPTION] * self.sim_step
            for vehicle_data in vehicle_list
        ) / len(vehicle_list)

        return (-1 * fc) / self.normailizer


class DelayMin(Rewarder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._action_penalty = ActionPenalty()

        self._running_mean = np.zeros(1)  # 50 seconds

    def get_reward(
        self,
        subscription_dict,
        observation_space,
        # action,  # okay_2_switch
        *args,
        **kwargs,
    ) -> None:
        # use the observation space to get the lanes that we are interested in
        # vehicle_list = list(subscription_dict[tc.VAR_VEHICLE].values())
        total_wait = sum(
            sum(approach["waiting_time"])
            for tl_obs in observation_space
            for approach in tl_obs
        )
        total_veh = sum(
            len(approach["waiting_time"])
            for tl_obs in observation_space
            for approach in tl_obs
        )

        # rel_speeds = [
        #         sc_results[_id][tc.VAR_SPEED] / sc_results[_id][tc.VAR_ALLOWED_SPEED] for tl_obs in observation_space for approach in tl_obs for _id in approach["ids"]
        #     ]
        #     # compute values corresponding to summary-output
        #     running = len(rel_speeds)
        #     # stopped = len([1 for d in sc_results.values() if d[tc.VAR_SPEED] < 0.1])
        #     mean_speed_relative = sum(rel_speeds) / running
        #     return (1 - mean_speed_relative) * running * self.sim_step

        wait_penalty = -1 * (
            total_wait / max(total_veh, 1) / 600
        )  # divide by 300 to normalize by the worst case scenario

        self._running_mean = np.roll(self._running_mean, -1)
        self._running_mean[-1] = wait_penalty

        return self._running_mean.mean()


class ActionPenalty(Rewarder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._action_last = None

    def get_reward(self, subscription_dict, action, okay_2_switch) -> None:
        # use the observation space to get the lanes that we are interested in
        # vehicle_list = list(subscription_dict[tc.VAR_VEHICLE].values())
        if self._action_last is None:
            self._action_last = action
            return 0
        else:
            penalty = 0
            for i, tl_act in enumerate(zip(action, self._action_last)):
                for j, act in enumerate(tl_act):
                    if act != self._action_last[i][j] and not okay_2_switch[i][j]:
                        penalty -= 0.5
            self._action_last = action
            return penalty


class SpeedMax(Rewarder):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_reward(self, subscription_dict, observation_space, *args, **kwargs) -> None:
        # use the observation space to get the lanes that we are interested in
        total_speed = sum(
            sum(approach["speeds"])
            for tl_obs in observation_space
            for approach in tl_obs
        )
        total_veh = sum(
            len(approach["speeds"]) for tl_obs in observation_space for approach in tl_obs
        )

        return total_speed / max(total_veh, 1) / 30


class FCIC(Rewarder):
    """
    From: https://journals.sagepub.com/doi/full/10.1177/03611981211004181

    @return:
    """

    def __init__(self, sim_params: SimParams, *args, **kwargs):
        super(FCIC, self).__init__()
        # self.junction_id = deepcopy(sim_params.central_junction)
        self._junctions = list(sim_params.nema_file_map.keys())
        self.sim_step = deepcopy(sim_params.sim_step)
        self._k_mapping = self._build_k_mapping(sim_params.net_file)
        # normailize by the worst case scenario
        self.window_size = int(100 / self.sim_step)
        self._reward_array = []

    @staticmethod
    def k_func(speed_mps: float) -> float:
        # convert mps to mph
        speed = speed_mps * 2.23694
        # this is the k function from the paper
        return 17 * np.exp(0.0531 * speed)

    def _build_k_mapping(self, net_file: Path) -> Dict[str, float]:
        """This function builds a mapping between the road id and the k value
        for that road.

        @param net_file:
        @return:
        """
        net = sumolib.net.readNet(str(net_file), withInternal=True)
        return {edge.getID(): self.k_func(edge.getSpeed()) for edge in net.getEdges()}

    def re_initialize(self):
        self._reward_array.clear()

    def _running_mean(
        self,
    ):
        """
        From https://stackoverflow.com/a/43200476/13186064
        """
        return uniform_filter1d(
            self._reward_array,
            self.window_size,
            mode="constant",
            origin=-(self.window_size // 2),
        )[: -(self.window_size - 1)]

    def register_traci(self, traci_c):
        self._reward_array.clear()
        traci_calls = []
        for junction_id in self._junctions:
            traci_c.junction.subscribeContext(
                junction_id,
                tc.CMD_GET_VEHICLE_VARIABLE,
                200,
                [
                    tc.VAR_SPEED,
                    tc.VAR_ALLOWED_SPEED,
                    tc.VAR_WAITING_TIME,
                    tc.VAR_ROAD_ID,
                ],
            )
            traci_calls.append(
                [
                    traci_c.junction.getContextSubscriptionResults,
                    (junction_id,),
                    junction_id,
                ]
            )
        return traci_calls

    def _get_junction_reward(self, subscription_dict, junction_id):
        # D_i + K_i * S_i
        # we are summing over all the edges within 300m of the junction
        total_delay = self._get_delay(subscription_dict[junction_id])
        # get the sum of stop time
        k_s = sum(
            (subscription_dict[junction_id][d][tc.VAR_WAITING_TIME] > 0)
            * self._k_mapping[subscription_dict[junction_id][d][tc.VAR_ROAD_ID]]
            for d in subscription_dict[junction_id]
        )
        return total_delay + k_s

    def get_reward(self, subscription_dict):
        reward = [
            self._get_junction_reward(subscription_dict, junction_id)
            for junction_id in self._junctions
        ]
        self._reward_array.append(sum(reward))
        r_array = self._running_mean()
        r_r = r_array[-1] if len(r_array) else self._reward_array[-1]
        # print(r_r)
        return -1 * r_r / 1e6

    def get_stops(self):
        pass

    def _get_delay(self, sc_results):
        """
        From https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html#retrieve_the_timeloss_for_all_vehicles_currently_in_the_network
        @param sc_results:
        @return:
        """
        if sc_results:
            rel_speeds = [
                d[tc.VAR_SPEED] / d[tc.VAR_ALLOWED_SPEED] for d in sc_results.values()
            ]
            # compute values corresponding to summary-output
            running = len(rel_speeds)
            # stopped = len([1 for d in sc_results.values() if d[tc.VAR_SPEED] < 0.1])
            mean_speed_relative = sum(rel_speeds) / running
            return (1 - mean_speed_relative) * running * self.sim_step
        return 0
