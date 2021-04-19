import traci.constants as tc

#
# class Rewarder(object):
#
#     def __init__(self, method):
#
#         self.execute_er = getattr(self, method)
#
#     def calculate(self, subscription_values):
#
#         self.execute_er(subscription_values)


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

