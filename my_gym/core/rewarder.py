import traci.constants as tc


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


def fcic_pi_parent():

    """
    From: https://journals.sagepub.com/doi/full/10.1177/03611981211004181

    @return:
    """
    x = 0

    def fcic_pi(subscription_values, ):
        nonlocal x
        print(x)
        x += 1

    return fcic_pi
