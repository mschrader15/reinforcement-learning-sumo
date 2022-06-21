import contextlib
import os
import signal
import traci
import traci.constants as tc
import logging
from copy import deepcopy
from sumolib import checkBinary

# should I try and use libsumo?
# Most of the libsumo code is take directly from https://github.com/LucasAlegre/sumo-rl/blob/master/sumo_rl/environment/env.py
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def sumo_cmd_line(params, kernel):

    cmd = ["-c", params["gui_config_file"]] if params["gui_config_file"] else []

    cmd.extend(
        [
            "-n",
            params.net_file,
            "-e",
            str(params.sim_length),
            "--step-length",
            str(params.sim_step),
            "--seed",
            str(kernel.seed),
            "--time-to-teleport",
            "-1",
            "--collision.action",
            "remove",
        ]
    )

    additional_files = ", ".join(params.additional_files + [params.route_file])

    if params["tls_record_file"]:
        additional_files = ", ".join([additional_files] + [params["tls_record_file"]])

    cmd.extend(["-a", additional_files])

    if params.gui:
        cmd.extend(["--start"])

    if params["emissions"]:
        cmd.extend(["--emission-output", params["emissions"]])

    return cmd


VEHICLE_SUBSCRIPTIONS = [tc.VAR_POSITION, tc.VAR_FUELCONSUMPTION, tc.VAR_SPEED]


class Kernel(object):
    """
    This class is the core for interfacing with the simulation
    """

    CONNECTION_NUMBER = 0

    def __init__(self, sim_params):

        self.traci_c = None
        self.sumo_proc = None
        self.parent_fns = []
        self.sim_params = deepcopy(sim_params)
        self.sim_step_size = self.sim_params.sim_step
        self.state_file = os.path.join(
            sim_params.sim_state_dir, f"start_state_{sim_params.port}.xml"
        )
        self.sim_time = 0
        self.seed = 5
        self.traci_calls = []
        self.sim_data = {}

        self._initial_tl_colors = {}

        self._sumo_conn_label = str(Kernel.CONNECTION_NUMBER)
        # increment the connection
        Kernel.CONNECTION_NUMBER += 1

    def set_seed(self, seed):
        self.seed = seed

    def pass_traci_kernel(self, traci_c):
        """
        This is the method that FLOW uses. Causes traci to "live" at the parent level
        @param traci_c:
        @return:
        """
        self.traci_c = traci_c

    def start_simulation(
        self,
    ):
        # find SUMO
        sumo_binary = (
            checkBinary("sumo-gui") if self.sim_params.gui else checkBinary("sumo")
        )

        # create the command line call
        sumo_call = [sumo_binary] + sumo_cmd_line(self.sim_params, self)

        if LIBSUMO:
            sumo_call.extend(
                [
                    "--remote-port",
                    str(self.sim_params.port),
                ]
            )
            traci.start(
                sumo_call,
            )
            traci_c = traci
        else:
            traci.start(sumo_call, label=self._sumo_conn_label)
            traci_c = traci.getConnection(self._sumo_conn_label)

        # connect to traci
        traci_c.simulationStep()

        # set the traffic lights to the default behaviour and run for warm up period
        for tl_id in self.sim_params.tl_ids:
            traci_c.trafficlight.setProgram(tl_id, f"{tl_id}-1")

        # run for an hour to warm up the simulation
        for _ in range(int(self.sim_params.warmup_time * 1 / self.sim_step_size)):
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
                traci_c.trafficlight.setProgram(tl_id, f"{tl_id}-2")

            # overwrite the default traffic light states to what they where
            for tl_id in self.sim_params.tl_ids:
                traci_c.trafficlight.setPhase(tl_id, 0)

        # saving the beginning state of the simulation
        traci_c.simulation.saveState(self.state_file)

        traci_c.simulation.subscribe([tc.VAR_COLLIDING_VEHICLES_NUMBER])

        self.add_traci_call(
            [
                [
                    traci_c.lane.getAllSubscriptionResults,
                    (),
                    tc.VAR_COLLIDING_VEHICLES_NUMBER,
                ],
            ]
        )

        self.sim_time = 0

        return traci_c

    # @staticmethod
    def _subscribe_to_vehicles(
        self,
    ):
        # subscribe to all new vehicle positions and fuel consumption
        for veh_id in self.traci_c.simulation.getDepartedIDList():
            self.traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

    def reset_simulation(
        self,
    ):

        try:
            self.sim_time = 0
            # self.traci_c.close()

            # self.traci_c.load(sumo_cmd_line(self.sim_params))
            with contextlib.suppress(AttributeError):
                self.traci_c.simulation.clearPending()
            # unsubscribe from all the vehicles at the end state
            for veh_id in self.traci_c.vehicle.getIDList():
                self.traci_c.vehicle.unsubscribe(veh_id)

            logging.info("resetting the simulation")
            self.traci_c.simulation.loadState(self.state_file)

            # set the traffic lights to the correct program
            # set the traffic lights to the all green program
            if not self.sim_params.no_actor:

                for tl_id in self.sim_params.tl_ids:
                    self.traci_c.trafficlight.setProgram(tl_id, f"{tl_id}-2")
                    self.traci_c.trafficlight.setPhase(tl_id, 0)

            # subscribe to all vehicles in the simulation at this point
            # subscribe to all new vehicle positions and fuel consumption
            self.simulation_step()

            # subscribe to all of the vehicles again
            # unsubscribe from all the vehicles at the end state
            for veh_id in self.traci_c.vehicle.getIDList():
                self.traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

        except Exception as e:
            print("Something in TRACI failed")
            raise e from e

    def kill_simulation(
        self,
    ):
        for fn, args in [
            [self._kill_sumo_proc, ()],
            [self._os_pg_killer, ()],
            [self._close_traci, ()],
        ]:
            with contextlib.suppress(Exception):
                fn(*args)

    def _kill_sumo_proc(
        self,
    ):
        if self.sumo_proc:
            self.sumo_proc.kill()

    def _close_traci(
        self,
    ):
        if self.traci_c:
            self.traci_c.close()

    def _os_pg_killer(
        self,
    ):
        if self.sumo_proc:
            os.killpg(self.sumo_proc.pid, signal.SIGTERM)

    def _execute_traci_fns(self):
        for fn in self.parent_fns:
            fn(self.traci_c)

    def close_simulation(
        self,
    ):
        logging.info("closing the simulation")
        # kill the simulation if using the gui.
        self.kill_simulation()
        self.traci_calls.clear()

    def simulation_step(
        self,
    ):
        # step the simulation
        try:
            self.traci_c.simulationStep()
        except traci.exceptions.FatalTraCIError:
            logging.error("sumo crashed on a step")
            return False

        self._subscribe_to_vehicles()

        self.sim_time += self.sim_step_size

        self.sim_data = self.get_traci_data()

        return self.sim_data

    def get_traci_data(
        self,
    ):
        return {key: fn(*args) for fn, args, key in self.traci_calls}

    def add_traci_call(self, traci_module):
        self.traci_calls.extend(traci_module)

    def check_collision(self):
        """See parent class."""
        return False  # self.sim_data[tc.VAR_COLLIDING_VEHICLES_NUMBER] != 0
