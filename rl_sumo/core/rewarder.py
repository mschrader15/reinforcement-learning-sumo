import traci.constants as tc
from copy import deepcopy


def minimize_fuel(subscription_values):
    """
    this is a very simple function that minimizes the fuel consumption of the network

    @param subscription_values:
    @return:
    """
    vehicle_list = list(subscription_values[tc.VAR_VEHICLE].values())
    fc = 0
    for vehicle_data in vehicle_list:
        fc += vehicle_data[tc.VAR_FUELCONSUMPTION]
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

    def register_traci(self, traci_c):
        
        traci_c.junction.subscribeContext(self.junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000000,
                                          [tc.VAR_SPEED, tc.VAR_ALLOWED_SPEED, tc.VAR_ROAD_ID])

        return [[traci_c.junction.getContextSubscriptionResults, (self.junction_id, ), self.junction_id]]

    def get_reward(self, subscription_dict):
        relevant_data = subscription_dict[self.junction_id]
        delay = self._get_delay(relevant_data)
        k_s = self._get_sorted_stopped(relevant_data)
        return -1 * (delay + k_s / 3600)

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
            delay = (1 - mean_speed_relative) * running * self.sim_step
            return delay
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
            return sum([k * self.sim_step for k in k_s])
        return sum(k_s)


# def fcic_pi(subscription_values, ):
#         nonlocal x
#         print(x)
#         x += 1
#
#     return fcic_pi
