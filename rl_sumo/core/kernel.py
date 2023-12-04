import contextlib
import os
from pathlib import Path
import signal
from typing import Any, Dict, List
from rl_sumo.parameters.params import SimParams
import traci.constants as tc
import logging
from copy import deepcopy
from sumolib import checkBinary

import libsumo as traci  # noqa: E402


def sumo_cmd_line(params: SimParams, kernel):
    cmd = [
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
        # "--collision.action",
        # "remove",
        *["-a", ", ".join(params.additional_files + [params.route_file])],
    ]

    cmd.extend(params.additional_args)

    if params.gui:
        cmd.extend(["--start"])

    return cmd


VEHICLE_SUBSCRIPTIONS = [
    tc.VAR_POSITION,
    tc.VAR_FUELCONSUMPTION,
    tc.VAR_SPEED,
    tc.VAR_VEHICLECLASS,
    tc.VAR_WAITING_TIME,
]


class Kernel(object):
    """This class is the core for interfacing with the simulation."""

    # fuck it we only support libsumo
    def __init__(self, sim_params):
        self.traci_c = None
        self.sumo_proc = None
        self.parent_fns = []
        self.sim_params = deepcopy(sim_params)
        self.sim_step_size = self.sim_params.sim_step
        self.state_file = (
            Path(sim_params.sim_state_dir) / f"start_state_{sim_params.port}.xml"
        )

        if not self.state_file.parent.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.sim_time = 0

        self.seed: int = None
        self.set_seed(5)

        self.traci_calls = []
        self.sim_data = {}

        self._initial_tl_colors = {}

        self._loaded_tl_programs: Dict[str, List[Any]] = {}

        self._controlled_signals: List[str] = list(self.sim_params.nema_file_map.keys())

    def set_seed(self, seed):
        self.seed = seed
        self.state_file = (
            Path(self.sim_params.sim_state_dir) / f"start_state_{self.seed}.xml"
        )

    def pass_traci_kernel(self, traci_c):
        """This is the method that FLOW uses. Causes traci to "live" at the
        parent level.

        @param traci_c: @return:
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
        traci.start(
            sumo_call,
        )

        traci_c = traci

        # connect to traci
        traci_c.simulationStep()

        # set the traffic lights to the default behaviour and run for warm up period
        for tl_id in self._controlled_signals:
            # get all the programs
            self._loaded_tl_programs[tl_id] = traci_c.trafficlight.getAllProgramLogics(
                tl_id
            )
            if len(self._loaded_tl_programs[tl_id]) != 3:
                raise ValueError("I don't know how to handle > 3 programs")
            # set the default program to the second one, as this is
            # the default NEMA program
            traci_c.trafficlight.setProgram(
                tl_id, self._loaded_tl_programs[tl_id][1].programID
            )

        # run for an hour to warm up the simulation
        # for _ in range(int(self.sim_params.warmup_time * 1 / self.sim_step_size)):
        traci_c.simulationStep(self.sim_params.warmup_time)

        # subscribe to all the vehicles in the network at this state
        for veh_id in traci_c.vehicle.getIDList():
            traci_c.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIPTIONS)

        # set the traffic lights to the all green program
        if not self.sim_params.no_actor:
            self._prep_signals_4_control(traci_c)

        # saving the beginning state of the simulation
        if not self.state_file.exists():
            traci_c.simulation.saveState(str(self.state_file))

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

    def _prep_signals_4_control(self, traci_c: traci.connection.Connection = None):
        pass

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
            self.traci_c.simulation.loadState(str(self.state_file))

            # set the traffic lights to the correct program
            # set the traffic lights to the all green program
            if not self.sim_params.no_actor:
                self._prep_signals_4_control()

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
        if self.traci_c and not self.sim_params.gui:
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
        return self.traci_c.simulation.getCollidingVehiclesNumber() > 0

    def check_long_delay(self, sim_data, threshold=300):
        """See parent class."""
        return any(
            veh[tc.VAR_WAITING_TIME] > threshold
            for veh in sim_data[tc.VAR_VEHICLE].values()
        )
