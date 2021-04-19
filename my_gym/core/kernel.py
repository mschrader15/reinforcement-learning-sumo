import os
import signal
import traci
import traci.constants as tc
import logging
from random import randint
import time
from copy import deepcopy
from sumolib import checkBinary
import subprocess


def sumo_cmd_line(params):
    cmd = ['-n', params.net_file, '-e', str(params.sim_length), '--step-length', str(params.sim_step),
           '-r', params.route_file, '-a', ", ".join(params.additional_files), '--remote-port', str(params.port),
           '--seed', str(randint(0, 10000))
           ]
    if params.gui:
        cmd.extend(['--start'])
    return cmd


VEHICLE_SUBSCRIPTIONS = [tc.VAR_POSITION, tc.VAR_FUELCONSUMPTION]


class Kernel(object):
    """
    This class is the core for interfacing with the simulation
    """

    def __init__(self, sim_params):

        self.traci_c = None
        self.sumo_proc = None
        self.parent_fns = []
        self.sim_params = deepcopy(sim_params)
        self.sim_step_size = self.sim_params.sim_step
        # self.subscription_data = {}
        self.sim_time = 0
        self.traci_calls = []

    def pass_traci_kernel(self, traci_c):
        """
        This is the method that FLOW uses. Causes traci to "live" at the parent level
        @param traci_c:
        @return:
        """
        self.traci_c = traci_c

    def start_simulation(self, ):

        # find SUMO
        sumo_binary = checkBinary('sumo-gui') if self.sim_params.gui else checkBinary('sumo')

        # create the command line call
        sumo_call = [sumo_binary] + sumo_cmd_line(self.sim_params)

        # start the process
        self.sumo_proc = subprocess.Popen(
            sumo_call,
            stdout=subprocess.DEVNULL
        )

        # sleep before trying to connect with TRACI
        time.sleep(1)

        # connect to traci
        traci_c = traci.connect(port=self.sim_params.port, numRetries=100)

        # TODO: is this needed?? It throws an error
        # traci_c.setOrder(0)

        traci_c.simulationStep()

        # set the traffic lights to the default behaviour and run for warm up period
        for tl_id in self.sim_params.tl_ids:
            traci_c.trafficlight.setProgram(tl_id, f'{tl_id}-1')

        # run for an hour to warm up the simulation
        for _ in range(int(self.sim_params.warmup_time * 1 / self.sim_step_size)):
            traci_c.simulationStep()

        # subscribe to all the vehicles in the network at this state
        for veh_id in traci_c.vehicle.getIDList():
            traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

        # set the traffic lights to the all green program
        for tl_id in self.sim_params.tl_ids:
            traci_c.trafficlight.setProgram(tl_id, f'{tl_id}-2')

        # saving the beginning state of the simulation
        traci_c.simulation.saveState('start_state.xml')

        return traci_c

    @staticmethod
    def _subscribe_to_vehicles(traci_c):
        # subscribe to all new vehicle positions and fuel consumption
        for veh_id in traci_c.simulation.getDepartedIDList():
            traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

    def reset_simulation(self, ):
        self.sim_time = 0
        # self.traci_c.close()
        self.traci_c.simulation.clearPending()
        # self.traci_c.close()
        # self.start_simulation()
        logging.info('resetting the simulation')
        self.traci_c.simulation.loadState('start_state.xml')

        # # subscribe to all vehicles in the simulation at this point
        # # subscribe to all new vehicle positions and fuel consumption
        # for veh_id in self.traci_c.simulation.getAllI():
        #     self.traci_c.subscribe(veh_id, tc.VAR_VEHICLE, [tc.VAR_POSITION, tc.VAR_FUELCONSUMPTION])
        self.simulation_step()

    def kill_simulation(self, ):
        try:
            os.killpg(self.sumo_proc.pid, signal.SIGTERM)
            os.remove('')
        except Exception as e:
            print("Error during teardown: {}".format(e))

    def register_traci_function(self, fn: object):
        self.parent_fns.append(fn)

    def _execute_traci_fns(self):
        for fn in self.parent_fns:
            fn(self.traci_c)

    def close_simulation(self, ):
        logging.info('closing the simulation')
        self.traci_c.close()
        self.traci_calls.clear()

    def simulation_step(self, ):
        # step the simulation
        self.traci_c.simulationStep()

        self._subscribe_to_vehicles(self.traci_c)

        self.sim_time += self.sim_step_size

        return self.get_traci_data()

        # return get the simulation subscriptions
        # self.subscription_data = self.traci_c.simulation.getAllSubscriptionResults()

    def get_traci_data(self, ):
        return {key: fn(*args) for fn, args, key in self.traci_calls}

    def add_traci_call(self, traci_module):
        self.traci_calls.extend(traci_module)

