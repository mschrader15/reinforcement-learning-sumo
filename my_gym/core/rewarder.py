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

    return -1 * sum(subscription_values[tc.VAR_VEHICLE][tc.VAR_FUELCONSUMPTION].values())

