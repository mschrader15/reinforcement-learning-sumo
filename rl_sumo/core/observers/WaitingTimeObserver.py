from bdb import Breakpoint
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple, Union
import sumolib
from traci.constants import (
    LAST_STEP_VEHICLE_ID_LIST,
    VAR_VEHICLE,
    VAR_LANES,
    VAR_POSITION,
    VAR_WAITING_TIME,
)
from copy import deepcopy

from rl_sumo.helpers.utils import read_nema_config
from .observer import DISTANCE_THRESHOLD, Lane, LaneType, xy_to_m
from .per_phase_observer import GlobalPhaseObservations, Phase, PhaseTLObservations


class WaitingTimeLane(Lane):
    """
    This is the base class for the observable environment, containing:
    1. the count of vehicles in the lane
    2. the SUMO name of the lane
    3. the waiting time on the lane

    It can be extended in the future
    """

    def __init__(self, lane_list: List[str], direction: LaneType = ..., *args, **kwargs):
        super().__init__(lane_list, direction, *args, **kwargs)
        
        # a store of the waiting time on each lane
        self.waiting_time = 0

    def _subscribe_2_lanes(self, traci_c):
        """
        This function is called once to subscribe to the lanes

        @return:
        """
        for lane in self._lane_list:
            traci_c.lane.subscribe(lane, [LAST_STEP_VEHICLE_ID_LIST, VAR_WAITING_TIME])

    def update_counts(
        self, center: tuple, lane_info: dict, vehicle_info: dict
    ) -> Tuple[float, float]:
        """
        this function redefines the _Base update_counts and implements the logic for each lane

        @param vehicle_positions: {ids: positions}
        @param lane_ids: {lane_ids: {18: [id_list]}}
        @param center: the center of the intersection (simulating where a camera would be placed)
        @return: None
        """
        # call traci to get the ids of vehicles in each of the lanes
        ids = []
        for lane in self._lane_list:
            ids.extend(lane_info[lane][18])
        # loop through the ids, only checking the distance for those that are "new" to the network
        new_ids = []
        if len(ids):
            for _id in ids:
                # if it was there last time, it will be there this timestep. Assuming that cars do not travel backwards
                if _id in self._last_ids:
                    new_ids.append(_id)

                elif (
                    xy_to_m(*center, *vehicle_info[_id][VAR_POSITION])
                    <= DISTANCE_THRESHOLD
                ):
                    new_ids.append(_id)

        # assign the waiting time
        # TODO make this have a distance horizon
        self.waiting_time = sum(
            lane_info[_lane][VAR_WAITING_TIME] for _lane in self._lane_list
        )

        # assign these new ids to the history
        self._last_ids = new_ids
        self.count = len(new_ids)
        return (self.count, self.waiting_time)

    def get_counts(
        self,
    ) -> Tuple[int, float]:
        """
        a public function for getting the counts

        @return: the instance's last count
        """
        return (self.count, self.waiting_time)

    def register_traci(self, traci_c):
        # register traci
        # self.traci_c = traci_c
        # subscribe to the lane that I am in charge of
        self._subscribe_2_lanes(traci_c)


class WaitingTimePhase(Phase):
    def __init__(
        self,
        phase_name: Union[str, int],
        tls_object: sumolib.net.TLS,
        camera_position: tuple,
        phase_2_lane: Tuple[int],
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            phase_name,
            tls_object,
            camera_position,
            phase_2_lane,
            outgoing_too=False,
            *args,
            **kwargs
        )


    def _lane_factory(self, camera_position, lane: sumolib.net.lane.Lane, *args, **kwargs) -> Lane:
        return WaitingTimeLane(
            self._recursive_lane_getter([lane], camera_position, *args, **kwargs),
            *args,
            **kwargs
        )

    def update_counts(self, **kwargs):
        return super().update_counts(**kwargs)

    def get_waiting_time(
        self,
    ):
        return [sum(l.waiting_time) for l in self._children]

    def get_value(self, param: str, mapped: bool = False):
        return sum(getattr(c, param) for c in self._children)


class WaitingTimeTLObservations(PhaseTLObservations):
    """
    This class handles individual traffic lights
    """

    def __init__(self, net_obj: sumolib.net.Net, tl_id: str, nema_config_dict: OrderedDict, *args, **kwargs):
        super().__init__(net_obj, tl_id, nema_config_dict, *args, **kwargs)

    def _phase_factory(self, *args, **kwargs) -> Phase:
        return WaitingTimePhase(camera_position=self._center, *args, **kwargs)

    def compose_approaches(self, net_obj: sumolib.net.TLS) -> list:
        """
        This function is called only once, it creates a list of Approaches

        @param net_obj: net object
        @return: a list of Approaches
        """
        return_list = []
        edge_list = []
        for lane0, _, _ in sorted(
            (v[0] for v in net_obj.getLinks().values()), key=lambda x: x[-1]
        ):
            edge = lane0.getEdge()
            if edge not in edge_list:
                edge_list.append(edge)
                return_list.append(
                    WaitingTimePhase(approach_obj=edge, camera_position=self._center)
                )
        return return_list

    def get_waiting_time(
        self, mapped_method: bool = False
    ) -> Union[List[int], Dict[str, int]]:
        if mapped_method:
            return {t.name: t.get_waiting_time() for t in self._children}
        else:
            return (l[1] for a in self.count_list for l in a.count_list)


class GlobalWaitingTimeObserver(GlobalPhaseObservations):
    """
    The overall observation space class
    """

    def __init__(self, net_file: str, tl_ids: list, name: str):
        super().__init__(net_file, tl_ids, name)
    
    def _compose_tls(
        self, net_obj: sumolib.net.Net, nema_file_map: Dict[str, str]
    ) -> dict:
        """
        This function is called only once and it creates a list of TLObservations

        @param net_obj: the sumolib.net object
        @return: a list
        """
        # return {tls: TLObservations(net_obj=net_obj, tl_id=tls, ) for tls in self._tl_ids}
        return [
            WaitingTimeTLObservations(
                net_obj=net_obj,
                tl_id=tls,
                nema_config_dict=read_nema_config(nema_file_map[tls]),
            )
            for tls in self._tl_ids
        ]

    def register_traci(self, traci_c: object) -> Tuple[Tuple[object, tuple, int]]:
        """
        pass traci to the children and return the functions that the core traci module should execute.

        @param traci_c:
        @return:
        """
        # self.traci_c = traci_c
        for child in self:
            child.register_traci(traci_c)

        return (
            (traci_c.lane.getAllSubscriptionResults, (), VAR_LANES),
            (traci_c.vehicle.getAllSubscriptionResults, (), VAR_VEHICLE),
        )

    def get_waiting_time(
        self, mapped_method: bool = False
    ) -> Union[List[List[int]], Dict[str, Dict[str, int]]]:
        if mapped_method:
            return {t.name: t.get_waiting_time(mapped_method) for t in self._children}
        else:
            return (l[1] for a in self.count_list for l in a.count_list)
