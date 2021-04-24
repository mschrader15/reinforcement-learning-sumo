import os
import signal
import traci
import traci.constants as tc
import logging
import time
from copy import deepcopy
from sumolib import checkBinary
import subprocess


def sumo_cmd_line(params, kernel):

    cmd = [
        '-n',
        params.net_file,
        '-e',
        str(params.sim_length),
        '--step-length',
        str(params.sim_step),
        '-a',
        ", ".join(params.additional_files + [params.route_file]),
        '--remote-port',
        str(params.port),
        '--seed',
        str(kernel.seed),
        "--time-to-teleport",
        "-1",
        "--collision.action",
        "remove",
    ]
    if params.gui:
        cmd.extend(['--start'])

    if params['emissions']:
        cmd.extend(['--emission-output', params['emissions']])
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
        self.state_file = os.path.join(sim_params.sim_state_dir,
                                       f"start_state_{sim_params.port}.xml")
        self.sim_time = 0
        self.seed = 5
        self.traci_calls = []
        self.sim_data = {}

        self._initial_tl_colors = {}

    def set_seed(self, seed):
        self.seed = seed

    def pass_traci_kernel(self, traci_c):
        """
        This is the method that FLOW uses. Causes traci to "live" at the parent level
        @param traci_c:
        @return:
        """
        self.traci_c = traci_c

    def start_simulation(self, ):
        # pylint: disable=maybe-no-member

        # find SUMO
        sumo_binary = checkBinary(
            'sumo-gui') if self.sim_params.gui else checkBinary('sumo')

        # create the command line call
        sumo_call = [sumo_binary] + sumo_cmd_line(self.sim_params, self)

        # start the process
        self.sumo_proc = subprocess.Popen(sumo_call, stdout=subprocess.DEVNULL)

        # sleep before trying to connect with TRACI
        time.sleep(1)

        # connect to traci
        traci_c = traci.connect(port=self.sim_params.port, numRetries=100)
        traci_c.setOrder(0)
        traci_c.simulationStep()

        # set the traffic lights to the default behaviour and run for warm up period
        for tl_id in self.sim_params.tl_ids:
            traci_c.trafficlight.setProgram(tl_id, f'{tl_id}-1')

        # run for an hour to warm up the simulation
        for _ in range(
                int(self.sim_params.warmup_time * 1 / self.sim_step_size)):
            traci_c.simulationStep()

        # subscribe to all the vehicles in the network at this state
        for veh_id in traci_c.vehicle.getIDList():
            traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

        # get the light states
        # for tl_id in self.sim_params.tl_ids:
        #     self._initial_tl_colors[tl_id] = traci_c.trafficlight.getRedYellowGreenState(tl_id)

        # set the traffic lights to the all green program
        if not self.sim_params.no_actor:
            for tl_id in self.sim_params.tl_ids:
                traci_c.trafficlight.setProgram(tl_id, f'{tl_id}-2')
        
        # overwrite the default traffic light states to what they where
        for tl_id in self.sim_params.tl_ids:
            traci_c.trafficlight.setPhase(tl_id, 0)

        # saving the beginning state of the simulation
        traci_c.simulation.saveState(self.state_file)

        traci_c.simulation.subscribe([tc.VAR_COLLIDING_VEHICLES_NUMBER])

        self.add_traci_call([
            [
                traci_c.lane.getAllSubscriptionResults, (),
                tc.VAR_COLLIDING_VEHICLES_NUMBER
            ],
        ])

        return traci_c

    @staticmethod
    def _subscribe_to_vehicles(traci_c):
        # subscribe to all new vehicle positions and fuel consumption
        for veh_id in traci_c.simulation.getDepartedIDList():
            traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

    def reset_simulation(self, ):
        self.sim_time = 0
        # self.traci_c.close()

        # self.traci_c.load(sumo_cmd_line(self.sim_params))
        self.traci_c.simulation.clearPending()

        # unsubscribe from all the vehicles at the end state
        for veh_id in self.traci_c.vehicle.getIDList():
            self.traci_c.vehicle.unsubscribe(veh_id)

        logging.info('resetting the simulation')
        self.traci_c.simulation.loadState(self.state_file)

        # set the traffic lights to the correct program
                # set the traffic lights to the all green program
        if not self.sim_params.no_actor:
            for tl_id in self.sim_params.tl_ids:
                self.traci_c.trafficlight.setProgram(tl_id, f'{tl_id}-2')

        # overwrite the default traffic light states to what they where
        for tl_id in self.sim_params.tl_ids:
            self.traci_c.trafficlight.setPhase(tl_id, 0)

        # subscribe to all vehicles in the simulation at this point
        # subscribe to all new vehicle positions and fuel consumption
        self.simulation_step()

        # subscribe to all of the vehicles again
        # unsubscribe from all the vehicles at the end state
        for veh_id in self.traci_c.vehicle.getIDList():
            self.traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

    def kill_simulation(self, ):
        try:
            self.sumo_proc.kill()
            os.killpg(self.sumo_proc.pid, signal.SIGTERM)
        except Exception as e:
            print("Error during teardown: {}".format(e))

    # def register_traci_function(self, fn: object):
    #     self.parent_fns.append(fn)

    def _execute_traci_fns(self):
        for fn in self.parent_fns:
            fn(self.traci_c)

    def close_simulation(self, ):
        logging.info('closing the simulation')
        # kill the simulation if using the gui.
        if self.sim_params.gui:
            # self.traci_c.close()
            self.kill_simulation()
        else:
            if self.traci_c:
                self.traci_c.close()
        self.traci_calls.clear()

    def simulation_step(self, ):
        # step the simulation
        self.traci_c.simulationStep()

        self._subscribe_to_vehicles(self.traci_c)

        self.sim_time += self.sim_step_size

        self.sim_data = self.get_traci_data()

        return self.sim_data

    def get_traci_data(self, ):
        return {key: fn(*args) for fn, args, key in self.traci_calls}

    def add_traci_call(self, traci_module):
        self.traci_calls.extend(traci_module)

    def check_collision(self):
        """See parent class."""
        return False  # self.sim_data[tc.VAR_COLLIDING_VEHICLES_NUMBER] != 0
