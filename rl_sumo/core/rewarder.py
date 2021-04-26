import traci.constants as tc
from copy import deepcopy
from scipy.ndimage.filters import uniform_filter1d


def minimize_fuel(subscription_values):
    """
    this is a very simple function that minimizes the fuel consumption of the network

    @param subscription_values:
    @return:
    """
    vehicle_list = list(subscription_values[tc.VAR_VEHICLE].values())
    fc = sum(vehicle_data[tc.VAR_FUELCONSUMPTION] for vehicle_data in vehicle_list)
    return -1 * fc


class Rewarder:
    def __init__(self, ):
        pass

    @staticmethod
    def register_traci(self, traci_c):
        return [[]]

    def get_reward(self, *args, **kwargs):
        pass

    def re_initialize(self, ):
        pass


class FCIC(Rewarder):
    """
    From: https://journals.sagepub.com/doi/full/10.1177/03611981211004181

    @return:
    """
    def __init__(self, sim_params, *args, **kwargs):
        super(FCIC, self).__init__()
        self.junction_id = deepcopy(sim_params.central_junction)
        self.sim_step = deepcopy(sim_params.sim_step)
        self.k_array = [[[
            'gneE0.12', 'gneE17', '-638636924#1.9', '660891910#1.19', '660891910#1', 'gneE18', 'gneE13', 'gneE20',
            '-8867312#6', 'gneE22', 'gneE22.27'
        ], 100], [[], 190]]
        # normailize by the worst case scenario
        self.min_reward = -200
        self.window_size = int(10 / self.sim_step)
        self._reward_array = []

    def re_initialize(self):
        self._reward_array.clear()


    def _running_mean(self, ):
        """
        From https://stackoverflow.com/a/43200476/13186064
        """
        return uniform_filter1d(self._reward_array, self.window_size, mode='constant',
                                origin=-(self.window_size // 2))[:-(self.window_size - 1)]

    def register_traci(self, traci_c):

        self._reward_array.clear()

        traci_c.junction.subscribeContext(self.junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000000,
                                          [tc.VAR_SPEED, tc.VAR_ALLOWED_SPEED, tc.VAR_ROAD_ID])

        return [[traci_c.junction.getContextSubscriptionResults, (self.junction_id, ), self.junction_id]]

    def get_reward(self, subscription_dict):
        relevant_data = subscription_dict[self.junction_id]
        delay = self._get_delay(relevant_data)
        k_s = self._get_sorted_stopped(relevant_data)
        r = -1 * (delay + k_s / 3600)
        self._reward_array.append(r)

        r_array = self._running_mean()

        r_r = r_array[-1] if len(r_array) else r
        
        # print("reward", r_r)
        # print("min_reward", self.min_reward)
        
        # self.min_reward = min(r_r, self.min_reward)
        
        return -1 * (r_r / self.min_reward)

    def get_stops(self):
        pass

    def _get_delay(self, sc_results):
        """
        From https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html#retrieve_the_timeloss_for_all_vehicles_currently_in_the_network
        @param sc_results:
        @return:
        """
        if sc_results:
            rel_speeds = [d[tc.VAR_SPEED] / d[tc.VAR_ALLOWED_SPEED] for d in sc_results.values()]
            # compute values corresponding to summary-output
            running = len(rel_speeds)
            # stopped = len([1 for d in sc_results.values() if d[tc.VAR_SPEED] < 0.1])
            mean_speed_relative = sum(rel_speeds) / running
            return (1 - mean_speed_relative) * running * self.sim_step
        return 0

    def _get_sorted_stopped(self, sc_results):
        k_s = [0] * len(self.k_array)
        if sc_results:
            for d in sc_results.values():
                if d[tc.VAR_SPEED] < 0.1:
                    if d[tc.VAR_ROAD_ID] in self.k_array[0][0]:
                        k_s[0] += self.k_array[0][1]
                    else:
                        k_s[1] += self.k_array[1][1]
            return sum(k_s) * self.sim_step
        return sum(k_s)
