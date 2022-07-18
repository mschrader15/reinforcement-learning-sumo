from ast import Call
import copy
import enum
import itertools
from typing import Any, Dict, Iterator, List, OrderedDict, Set, Tuple, Union, Callable

from rl_sumo.helpers.utils import read_nema_config

import sumolib
import traci
from traci.constants import TL_RED_YELLOW_GREEN_STATE, VAR_NAME, TL_PROGRAM
import xmltodict

COLOR_ENUMERATE = {"s": 0, "r": 0, "y": 1, "G": 2, "g": 2}


class _Base:
    """
    This base class freezes and unfreezes the data
    """

    def __init__(
        self,
    ):
        # self.parent = parent
        self.init_state = copy.deepcopy(self)

    def _re_initialize(
        self,
    ):
        """
        This function "unfreeezes" the initial values that were saved in the call to freeze

        Always update with a copy, otherwise the core will be ovewritten
        """
        self.__dict__.update(copy.deepcopy(self.init_state.__dict__))
            # self.__dict__[name] = value

    def freeze(
        self,
    ):
        """
        This is gets called later than the init state. and freezes all the values in self
        """
        self.init_state = copy.deepcopy(self)


class DualRingActor(_Base):
    def __init__(
        self, tl_id, nema_config_xml: str, net_file_xml: str, subscription_method: bool
    ) -> None:
        self.tl_id = tl_id
        # A list of all detectors controlling the traffic light
        self._all_detectors: List[str] = []
        # A list of unique detectors to phase maps
        # (different than above as one phase can have multiple detectors, but we only need one for controll)
        self._phase_2_detect: Dict[int, str] = {}

        # the current active phases
        self._requested_state: Set[int, int] = set([])

        # the action space of traffic light. A list of valid phase combinations
        self._action_space: List[Tuple[int, int]] = []

        # the sim time that the actual state was recorded and the actual active phases
        self._sumo_active_state: Tuple[float, Tuple[int, int]] = (0, ())
        # track the last iterations phase
        self._last_sumo_phase: Tuple[int, int] = ()

        # a map of phase name to light string index
        self._p_string_map: Dict[int, int] = {}

        # sorted phases in the order that SUMO renders them
        self._phases_sumo_order: List[int] = ()

        # the programID
        self._programID: str = ""

        # Am I currently in control
        self.controlled: bool = False

        # Whether or not to use subscriptions for getting SUMO data
        self._subscriptions: bool = subscription_method

        # The default state that the traffic light should rest in. Assumed to be the coordinated phases
        self._default_state: Tuple[int, int] = ()

        # store the (minimum phase durations, last switch time), so that we can know if its okay to switch or not
        self._time_tracker: Dict[int, List[float, float]] = {}

        # build the control based on the traffic light settings file
        self._build(nema_config_xml, net_file_xml)
        
        # traci         
        self._traci_c: traci = None

        # freeze the initial settings
        super().__init__()

    @property
    def default_state(
        self,
    ) -> Tuple[int, int]:
        return self._default_state

    @property
    def sumo_active_state(
        self,
    ) -> Tuple[int, int]:
        return self._sumo_active_state[1]

    @property
    def action_space(
        self,
    ) -> List[Tuple[int]]:
        return self._action_space

    @property
    def action_space_length(
        self,
    ) -> int:
        return len(self._action_space)

    @property
    def requested_state(
        self,
    ) -> Set[int]:
        return self._requested_state

    @requested_state.setter
    def requested_state(self, l: Union[list, Set[int]]) -> None:
        self._requested_state = set(l)

    def _build(self, nema_config_xml: str, net_file_xml: str) -> None:
        """
        builds a mapping of NEMA phase # to the actuating detectors

        Args:
            nema_config_xml (str): the SUMO additional file describing the NEMA behavior
            net_file_xml (str): the
        """

        tl_dict = read_nema_config(nema_config_xml)

        # set the program id
        self._programID = tl_dict["@programID"]

        # open the network file to get the order of the lanes
        lane_order = {
            i: l_list[0][0]
            for i, l_list in sumolib.net.readNet(net_file_xml)
            .getTLS(tl_dict["@id"])
            .getLinks()
            .items()
        }

        # loop through the traffic light phases, find the order that they control and then the "controllng" detectors
        for phase_name, phase in tl_dict["phase"].items():
            phase_int = int(phase_name)
            # set the minimum green time
            self._time_tracker[phase_int] = [float(phase["@minDur"]), 0]

            # save the index of the light head string
            self._p_string_map[phase_int] = phase["controlling_index"]
            # loop through the controlled lanes
            for lane_index in phase["controlling_index"]:
                if detect_id := tl_dict["param"].get(
                    lane_order[lane_index].getID(), ""
                ):
                    # the phase is controlled by a custom detector
                    detect_name = detect_id
                else:
                    # the phase is controlled by a generated detector. They are generated according to https://github.com/eclipse/sumo/issues/10045#issuecomment-1022207944
                    detect_name = (
                        tl_dict["@id"]
                        + "_"
                        + tl_dict["@programID"]
                        + "_D"
                        + str(phase_int)
                        + "."
                        + str(lane_order[lane_index].getIndex())
                    )

                # add the detector as a controlled detector
                self._all_detectors.append(detect_name)
                # add the NEMA phase to detector mapping
                if phase_int not in self._phase_2_detect.keys():
                    self._phase_2_detect[phase_int] = detect_name

        # create a list of the sumo order of phases
        temp_list = sorted(
            ((p, indexes) for p, indexes in self._p_string_map.items()),
            key=lambda x: x[1][0],
        )
        self._phases_sumo_order = [int(t[0]) for t in temp_list]

        # find the barriers
        bs = [
            tl_dict["param"]["barrierPhases"],
            tl_dict["param"].get("barrier2Phases", ""),
        ]

        bs[-1] = bs[-1] if bs[-1] != "" else tl_dict["param"]["coordinatePhases"]

        # convert the barriers to a list of integers
        bs = [tuple(map(int, b.split(","))) for b in bs]

        # set the default state = to the barriers
        self._default_state = bs[-1]

        # Create the action space
        # using the rings and barrier phases parameter\
        rings = [[[], []], [[], []]]
        for i, ring in enumerate(rings):
            r = tl_dict["param"][f"ring{i+1}"]
            # ring[0].extend(map(int, p.split(",")))
            b_num = 0
            for _p in map(int, r.split(",")):
                if _p > 0:
                    ring[b_num].append(_p)
                    if _p in bs[0] or _p in bs[1]:
                        b_num = 1

        # compose the combinations
        # add the barrier pairs as the first items in the action space, to give preference
        self._action_space.extend(bs[::-1])
        for i, _ in enumerate(bs):
            for pair in itertools.product(rings[0][i], rings[1][i]):
                if pair not in self._action_space:
                    self._action_space.append(pair)
            

    def re_initialize(self) -> None:
        """
        Re-loads the default settings from the frozen instance.
        Can be used for quick re-initialization in maching learning applications

        Returns:
            None
        """
        return self._re_initialize()

    def set_traci(self, traci_c: traci) -> Tuple[Callable, Tuple, int]:
        """
        pass the current traci instance to the class
        """
        self._traci_c = traci_c

        if self._subscriptions:
            # subscribe to the red yellow green state and subscribe to the current phase name
            traci_c.trafficlight.subscribe(
                self.tl_id, [TL_RED_YELLOW_GREEN_STATE, VAR_NAME]
            )
            # # subscribe to the current phase name
            # traci_c.trafficlight.subscribe(self.tl_id, TL_CURRENT_PHASE)
            return (traci_c.trafficlight.getAllSubscriptionResults, (), TL_PROGRAM)
        return ()

    def initialize_control(self, gracefully: bool) -> None:
        """
        This function initializes control of the traffic light. After

        Args:
            gracefully (bool):
        """
        if gracefully:
            # wait for a barrier cross event (all light heads will be 'r' or 's')
            current_str = self._traci_c.trafficlight.getRedYellowGreenState(self.tl_id)
            if any(l in current_str for l in ["G", "g", "y"]):
                return False
        # set the traffic light program id to the desired one
        self._traci_c.trafficlight.setProgram(self.tl_id, self._programID)
        self.controlled = True
        # overwrite all of the detectors to 0
        self._take_control()
        # force the traffic light to move initially by writing detector calls on the default state
        return self.try_switch(self.default_state)

    def _take_control(
        self,
    ) -> None:
        """
        Take control of all of the NEMA lights by first switching control to the controlled logic id and then
        overriding the detection to 0, meaning that the light will not be actuated by simulation traffic

        Returns:
            bool: True
        """
        for detect in self._all_detectors:
            self._traci_c.lanearea.overrideVehicleNumber(detect, 0)
        # return True
        self.controlled *= True

    def release_control(
        self,
    ) -> None:
        """
        Release control of the traffic light's actuating detectors.
        The SUMO traffic light will then behave according to the setting file

        Returns:
            None:
        """
        for detect in self._all_detectors:
            self._traci_c.lanearea.overrideVehicleNumber(detect, -1)

    def try_switch(
        self, requested_state: Tuple[int]
    ) -> None:
        """
        This function is called to change the light state

        Args:
            requested_state (List[int, int]): a list of requested states.
                Should likely have length 2 but there could be longer or shorter scenarios
        """
        if not self.controlled:
            # Don't take any action if I am not controlled
            return

        # turn off the diff detectors in the current state
        new_state = set(requested_state)
        for s in self.requested_state - new_state:
            self._traci_c.lanearea.overrideVehicleNumber(self._phase_2_detect[s], 0)

        # turn on the detectors for the new phase in requested state
        for s in new_state - self.requested_state:
            self._traci_c.lanearea.overrideVehicleNumber(self._phase_2_detect[s], 1)

        # pass the new state as the current state
        self.requested_state = new_state

    def get_requested_state(
        self,
    ) -> List[int]:
        """
        Returns the current state that the actor thinks the traffic light is in.
        This is not the same thing as the actual light state,
        as the light might be transitioning from the last state

        Returns:
            List[int]: _description_
        """
        return list(self._requested_state)

    def get_sumo_state(
        self, sim_time: float, sub_res: Dict[int, Dict] = {}
    ) -> Tuple[int, int]:
        """
        Get the actual NEMA state in integer list format

        Returns:
            Tuple[int]: the active phases as integers
        """
        self._last_sumo_phase = self.sumo_active_state

        self._sumo_active_state = (
            sim_time,
            tuple(
                int(p)
                for p in (
                    sub_res[TL_PROGRAM][self.tl_id][VAR_NAME]
                    if self._subscriptions
                    else self._traci_c.trafficlight.getPhaseName(self.tl_id)
                ).split("+")
            ),
        )

        if len(self._last_sumo_phase) == len(self.sumo_active_state):
            for p, p_old in zip(self.sumo_active_state, self._last_sumo_phase):
                if p != p_old:
                    # this means that the light changed and we should record this as it's start time
                    self._time_tracker[p][1] = sim_time

        return self.sumo_active_state

    def get_actual_color(self, sim_time: float, sub_res: Dict[int, Dict] = None) -> Tuple[int]:
        """
        Get a list of COLOR_ENUMERATE light head states ('r' -> 0, etc..)

        Returns:
            List[int]
        """
        if sub_res is None:
            sub_res = {}
        if not self._sumo_active_state or self._sumo_active_state[0] != sim_time:
            self.get_sumo_state(sim_time, sub_res)

        light_str = (
            sub_res[TL_PROGRAM][self.tl_id][TL_RED_YELLOW_GREEN_STATE]
            if self._subscriptions
            else self.self._traci_c.trafficlight.getRedYellowGreenState(self.tl_id)
        )
        return tuple(COLOR_ENUMERATE[light_str[self._p_string_map[s][0]]] for s in self.sumo_active_state)

    def okay_2_switch(self, sim_time: float) -> bool:
        # sourcery skip: raise-specific-error
        if not self._sumo_active_state:
            raise Exception(
                "You first need to pass the simulation dict before checking if it is okay to try and switch the traffic lights"
            )

        # both phases are passed their minimum timer aka (current_time - start_time) > min_time
        return all(
            (sim_time - self._time_tracker[p][1]) > self._time_tracker[p][0]
            for p in self.sumo_active_state
        )

    def get_phase_active_time(self, p: int, current_time: float) -> float:
        return current_time - self._time_tracker[p][1]

    def list_2_phase(
        self, count_list: List[int], per_phase: bool = False
    ) -> Dict[int, int]:
        """
        helper function to turn a list of lane counts (# of vehicles in each lane) to a per-phase count

        Returns:
            List[Tuple(phase (int), count (int))]:
        """
        if per_phase:
            return dict(zip(self._phases_sumo_order, count_list))
        else:
            return {
                p: sum(count_list[p_inds[0] : p_inds[-1] + 1])
                for p, p_inds in self._p_string_map.items()
            }


class GlobalDualRingActor:
    """
    A global actor for controlling all network traffic lights through a standard API
    """

    def __init__(
        self,
        nema_file_map: Dict[str, str],
        network_file: str,
        subscription_method=False,
    ):
        """
        Initializes the GlobalDualRingActor class.

        Args:
            nema_file_map (Dict[str, str]): a dictionary like: {<traffic-light-id>: <path to sumo additional file describing traffic light>, ...}. Can contain an arbitrary # of traffic lights
            network_file (str): the path to SUMO .net.xml file
        """
        self.tls: List[DualRingActor] = self.create_tl_managers(
            nema_file_map, network_file, subscription_method
        )

    def __iter__(self) -> Iterator[DualRingActor]:
        yield from self.tls

    def __getitem__(self, item: str) -> DualRingActor:
        """
        Emulates a dictionary

        @param item:
        @return: an instance of the TrafficLightManager class
        """
        return [tl for tl in self.tls if tl.tl_id == item][0]

    def register_traci(
        self, traci_c: object
    ) -> Union[List, Tuple[Tuple[Callable, Tuple, int]]]:
        """
        pass traci to all the children

        @param traci_c: a traci connection object
        @return:
        """
        for tl_manager in self.tls:
            tl_manager.set_traci(traci_c)
        return ((traci_c.trafficlight.getAllSubscriptionResults, (), TL_PROGRAM),)

    def re_initialize(
        self,
    ) -> None:
        """
        This functions reinitializes everything to its default values
        @return:
        """
        for tl_manager in self:
            tl_manager.re_initialize()

    @property
    def size(
        self,
    ) -> Dict[str, Any]:
        return {
            "state": [tl_manager.action_space_length for tl_manager in self],
            "color": [3 for _ in range(len(self.tls))],
        }

    @property
    def discrete_space_shape(self) -> int:
        return [tl_manager.action_space_length for tl_manager in self]

    @staticmethod
    def create_tl_managers(
        nema_file_map: Dict[str, str], net_file: str, subscriptions: bool
    ) -> List[DualRingActor,]:
        return [
            DualRingActor(tl_id, nema_file, net_file, subscriptions)
            for tl_id, nema_file in nema_file_map.items()
        ]

    def update_lights(
        self,
        action_list: list,
    ) -> None:
        """
        Update the state of all the lights

        Example:
        actions = [[2, 6], [1, 5]]
        GlobalDualRingActor(*args).update_lights(actions)

        Args:
            action_list (list): a list of actions to take
        """
        for action, tl_manager in zip(action_list, self.tls):
            tl_manager.try_switch(
                action,
            )

    def get_sumo_state(
        self,
        sim_time: float,
        subscription_results: Dict[int, Dict] = None,
    ) -> List[int,]:
        """
        get the states of all the traffic lights in the network

        @return: list of int
        """
        if subscription_results is None:
            subscription_results = {}
        states = []
        light_head_colors = []

        for tl in self.tls:
            states.append(tl.get_sumo_state(sim_time, subscription_results))
            light_head_colors.append(
                tl.get_actual_color(sim_time, subscription_results)
            )
        return states, light_head_colors

    def initialize_control(self, gracefully=False) -> bool:
        for tl in self.tls:
            if not tl.controlled:
                tl.initialize_control(gracefully)
        return all(tl.controlled for tl in self.tls)
