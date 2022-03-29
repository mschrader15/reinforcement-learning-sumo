import enum
import json
import copy
from distutils.util import strtobool
from typing import Any, Dict, List, OrderedDict, Set, Tuple, Union
from xml.dom import minidom
from attr import attr
import traci
import sumolib
import xmltodict
import itertools


COLOR_ENUMERATE = {"r": 0, "y": 1, "G": 2}


def first(s):
    """Return the first element from an ordered collection
    or an arbitrary element from an unordered collection.
    Raise StopIteration if the collection is empty.
    """
    return next(iter(s))


def find_matching_param(
    params: List[OrderedDict], attr: str, value: str
) -> OrderedDict:
    """
    Helper function to find a xmltodict parameter that contains the given key and value

    Args:
        params (List[OrderedDict]): a list of xmltodict ordered dictionaries that all have the same tag
        attr (str): the attribute to look for
        value (str): the attribute value to look for

    Returns:
        OrderedDict: xmltodict object
    """
    for p in params:
        if p.get(attr, "") == value:
            return p
    return OrderedDict()


class _Base:
    """
    This base class freezes and unfreezes the data
    """

    def __init__(
        self,
    ):
        # self.parent = parent
        self.init_state = copy.deepcopy(self.__dict__)

    def _re_initialize(
        self,
    ):
        """
        This function "unfreeezes" the initial values that were saved in the call to freeze
        """
        for name, value in self.init_state.items():
            self.__dict__[name] = value

    def freeze(
        self,
    ):
        """
        This is gets called later than the init state. and freezes all the values in self
        """
        self.init_state = copy.deepcopy(self.__dict__)


class DualRingActor(_Base):
    def __init__(self, tl_id, nema_config_xml: str, net_file_xml: str) -> None:
        self.tl_id = tl_id
        # A list of all detectors controlling the traffic light
        self._all_detectors: List[str] = []
        # A list of unique detectors to phase maps
        # (different than above as one phase can have multiple detectors, but we only need one for controll)
        self._phase_2_detect: Dict[int, str] = {}

        # the current active phases
        self._current_state: Set[int, int] = []

        # the action space of traffic light. A list of valid phase combinations
        self._action_space: List[List[int]] = []

        # the actual active phases
        self._active_state: List[int] = []

        # a map of phase name to light string index
        self._p_string_map: Dict[int, int] = {}

        # build the control based on the traffic light settings file
        self._build(nema_config_xml, net_file_xml)

        # freeze the initial settings
        super().__init__()

        self._traci_c: traci = None

    @property
    def action_space(
        self,
    ) -> List[List[int]]:
        return self._action_space

    @property
    def action_space_length(
        self,
    ) -> int:
        return len(self._action_space)

    @property.setter
    def _current_state(self, l: Union[list, set]) -> None:
        self._current_state = set(l)

    def _build(self, nema_config_xml: str, net_file_xml: str) -> None:
        """
        builds a mapping of NEMA phase # to the actuating detectors

        Args:
            nema_config_xml (str): the SUMO additional file describing the NEMA behavior
            net_file_xml (str): the
        """

        # get the lanes that the traffic light controls
        with open(nema_config_xml, "r") as f:
            raw = xmltodict.parse(f.read())
            tl_dict = raw[next(raw)]["tlLogic"]

        # open the network file to get the order of the lanes
        lane_order = {
            i: l_list[0]
            for i, l_list in sumolib.net.readNet(net_file_xml)
            .getTLS(tl_dict["@id"])
            .getLinks()
            .items()
        }

        # loop through the traffic light phases, find the order that they control and then the "controllng" detectors
        for phase in tl_dict["phase"]:
            light_str = phase["@state"]
            controlled_lane_index = [i for i, s in enumerate(light_str) if s == "G"]
            # save the index of the light head string
            self._p_string_map[phase] = controlled_lane_index[0]
            # loop through the controlled lanes
            for lane_index in controlled_lane_index:
                # try to find a matching parameter in the traffic light configuration file
                param = find_matching_param(
                    tl_dict["param"], attr="@key", value=lane_order[lane_index].getID()
                )
                if param.get("@key", "") == lane_order[lane_index].getID():
                    # the phase is controlled by a custom detector
                    detect_name = param.get("@value")
                else:
                    # the phase is controlled by a generated detector. They are generated according to https://github.com/eclipse/sumo/issues/10045#issuecomment-1022207944
                    detect_name = (
                        tl_dict["@id"]
                        + "_"
                        + tl_dict["@programID"]
                        + "_D"
                        + phase
                        + "."
                        + lane_order[lane_index].getIndex()
                    )

                # add the detector as a controlled detector
                self._all_detectors.append(detect_name)
                # add the NEMA phase to detector mapping
                if not int(phase) in self._phase_2_detect.keys():
                    self._phase_2_detect[int(phase)] = detect_name

        # find the barriers
        bs = []
        bs.append(
            find_matching_param(
                tl_dict["param"], attr="@key", value=f"barrierPhases"
            ).get("@value", "")
        )
        # b2 can either be called barrier2Phases or coordinatePhases
        bs.append(
            find_matching_param(
                tl_dict["param"], attr="@key", value=f"barrier2Phases"
            ).get("@value", "")
        )
        bs[-1] = (
            bs[-1]
            if bs[-1] == ""
            else find_matching_param(
                tl_dict["param"], attr="@key", value=f"coordinatePhases"
            ).get("@value", "")
        )

        # Create the action space
        # using the rings and barrier phases parameter\
        rings = [[[] * 2] * 2]
        for i, ring in enumerate(rings):
            p = find_matching_param(tl_dict["param"], attr="@key", value=f"ring{i+1}")
            # ring[0].extend(map(int, p.split(",")))
            b_num = 0
            for _p in map(int, p.split(",")):
                ring[b_num].append(_p)
                if _p in bs[i]:
                    b_num = 1

        # compose the combinations
        for i, _ in enumerate(bs):
            self._action_space.extend(itertools.product(rings[0][i], rings[1][i]))

        # set the current state to the barrier phase
        # self._current_state = bs[-1]

    def re_initialize(self) -> None:
        """
        Re-loads the default settings from the frozen instance.
        Can be used for quick re-initialization in maching learning applications

        Returns:
            None
        """
        return self._re_initialize()

    def set_traci(
        self,
    ) -> None:
        """
        pass the current traci instance to the class
        """
        self._traci_c = traci

    def intialize_control(
        self,
    ) -> None:
        """
        Take control of all of the NEMA lights by overriding the detection to 0,
        meaning that the light will not be actuated by simulation traffic
        """
        for detect in self._all_detectors:
            self._traci_c.lanearea.overrideVehicleNumber(detect, 0)

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

    def try_switch(self, requested_state: List[int]) -> None:
        """
        This function is called to change the light state

        Args:
            requested_state (List[int, int]): a list of requested states.
                Should likely have length 2 but there could be longer or shorter scenarios
        """
        # turn off the diff detectors in the current state
        new_state = set(requested_state)
        for s in self._current_state - new_state:
            self._traci_c.lanearea.overrideVehicleNumber(self._phase_2_detect[s], 0)

        # turn on the detectors for the new phase in requested state
        for s in new_state - self._current_state:
            self._traci_c.lanearea.overrideVehicleNumber(self._phase_2_detect[s], 0)

        # pass the new state as the current state
        self._current_state = new_state

    def get_current_state(
        self,
    ) -> List[int]:
        """
        Returns the current state that the actor thinks the traffic light is in.
        This is not the same thing as the actual light state,
        as the light might be transitioning from the last state

        Returns:
            List[int]: _description_
        """
        return list(self._current_state)

    def get_actual_state(
        self,
    ) -> List[int]:
        """
        Get the actual NEMA state in integer list format

        Returns:
            List[int]: the active phases as integers
        """
        self._active_state = [
            int(p)
            for p in self._traci_c.trafficlight.getPhaseName(self.tl_id).split("+")
        ]
        return self._active_state

    def get_actual_color(
        self,
    ) -> List[int]:
        """
        Get a list of COLOR_ENUMERATE light head states ('r' -> 0, etc..)

        Returns:
            List[int] 
        """
        if not len(self._active_state):
            self.get_actual_state()

        light_str = self._traci_c.trafficlight.getRedYellowGreenState(self.tl_id)
        light_states = [
            COLOR_ENUMERATE[light_str[self._p_string_map[s]]]
            for s in self._active_state
        ]
        # clear the active state so that it is freshly read
        self._active_state.clear()
        return light_states


class GlobalDualRingActor:
    def __init__(self, nema_file_map: Dict[str, str], network_file: str):
        self.tls = self.create_tl_managers(nema_file_map, network_file)

    def __iter__(self) -> DualRingActor:
        yield from self.tls

    def __getitem__(self, item: str) -> DualRingActor:
        """
        Emulates a dictionary

        @param item:
        @return: an instance of the TrafficLightManager class
        """
        return [tl for tl in self.tls if tl.tl_id == item][0]

    def register_traci(self, traci_c: object) -> None:
        """
        pass traci to all the children

        @param traci_c: a traci connection object
        @return:
        """
        for tl_manager in self:
            tl_manager.set_traci(traci_c)

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
        nema_file_map: Dict[str, str], net_file: str
    ) -> List[DualRingActor,]:
        return [
            DualRingActor(tl_id, nema_file, net_file)
            for tl_id, nema_file in nema_file_map.items()
        ]

    def update_lights(self, action_list: list) -> None:
        """
        Update the state of all the lights

        Example:
        actions = [[2, 6], [1, 5]]
        GlobalDualRingActor(*args).update_lights(actions)

        Args:
            action_list (list): a list of actions to take  
        """
        for action, tl_manager in zip(action_list, self.tls):
            tl_manager.try_switch(action)

    def get_current_state(
        self,
    ) -> List[int,]:
        """
        get the states of all the traffic lights in the network

        @return: list of int
        """
        states = []
        light_head_colors = []

        for tl in self.tls:
            states.append(tl.get_actual_state())
            light_head_colors.append(tl.get_actual_color())
        return states, light_head_colors
