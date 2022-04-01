"""
Based on:
- https://doi.org/10.1016/j.trc.2013.08.014 
- https://dl.acm.org/doi/pdf/10.1145/3292500.3330949
- https://github.com/docwza/sumolights 
"""


from cmath import phase
import math
from typing import Dict, Iterable, List, OrderedDict, Tuple, Union
from prometheus_client import Enum
import sumolib
from traci.constants import (
    LAST_STEP_VEHICLE_ID_LIST,
    VAR_VEHICLE,
    VAR_LANES,
    VAR_POSITION,
)
from copy import deepcopy

from ..helpers.utils import read_nema_config
from .observer import Approach, GlobalObservations, Lane, TLObservations

DISTANCE_THRESHOLD = 100  # in meters
VEHICLE_LENGTH = 5  # meters


def read_net(path: str) -> sumolib.net:
    """
    Read in the net file with the sumolib utility

    Args:
        path (str): path

    Returns:
        sumolib.net: sumolib net object
    """
    return sumolib.net.readNet(path)


def xy_to_m(x0, y0, x1, y1):
    """

    @param x0:
    @param y0:
    @param x1:
    @param y1:
    @return: a distance from (x0, y0) to (x1, y1) in meters
    """
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)


class LaneType(Enum):
    OUTGOING = -1
    INCOMING = 1


class MaxPressureLane(Lane):
    """
    This is the base class for the observable environment, containing:
    1. the count of vehicles in the lane
    2. the SUMO name of the lane

    It can be extended in the future
    """

    def __init__(self, lane_list: List[sumolib.net.lane.Lane], direction: LaneType):
        """
        Initialising the base class

        @param lane_list: a list of lanes
        """
        super(Lane, self).__init__(name=lane_list[0].getID(), children=[])
        # a "lane" can actually be composed of multiple lanes in the SUMO network, aka depending on the distance backwards that we should "observe"
        self._lane_list: List[str] = [l.getID() for l in lane_list]

        # the density of cars
        self.density: int = 0

        # the ids of the cars in the lane during the last time step
        self._last_ids: List[str] = []

        # a storage of the direction (either incoming or outgoing)
        self._direction: LaneType = direction

        # a constant factor to determin the "density" of the approach
        # TODO: Consider juction dimensions as well
        self._max_permissible_vehicles: int = max(
            math.floor(
                (
                    min(sum(l.getLength() for l in lane_list), DISTANCE_THRESHOLD)
                    / VEHICLE_LENGTH
                )
            ),
            1,
        )

    def get_lane_count(
        self,
    ):
        """
        @return: 1
        """
        return 1

    def _subscribe_2_lanes(self, traci_c):
        """
        This function is called once to subscribe to the lanes

        @return:
        """
        for lane in self._lane_list:
            traci_c.lane.subscribe(lane, [LAST_STEP_VEHICLE_ID_LIST])

    def update_density(
        self, center: tuple, lane_ids: dict, vehicle_positions: dict
    ) -> None:
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
            ids.extend(lane_ids[lane][18])
        # loop through the ids, only checking the distance for those that are "new" to the network
        new_ids = []
        if len(ids):
            for _id in ids:
                # if it was there last time, it will be there this timestep. Assuming that cars do not travel backwards
                if _id in self._last_ids:
                    new_ids.append(_id)

                elif (
                    xy_to_m(*center, *vehicle_positions[_id][VAR_POSITION])
                    <= DISTANCE_THRESHOLD
                ):
                    new_ids.append(_id)
        # assign these new ids to the history
        self._last_ids = new_ids
        self.density = (len(new_ids) / self._max_permissible_vehicles) * self._direction
        return self.density

    def get_density(
        self,
    ) -> int:
        """
        a public function for getting the counts

        @return: the instance's last count
        """
        return self.density

    def get_direction(
        self,
    ) -> str:

        raise self._direction

    def register_traci(self, traci_c):
        # register traci
        # self.traci_c = traci_c
        # subscribe to the lane that I am in charge of
        self._subscribe_2_lanes(traci_c)

    def update_counts(self, *args, **kwargs) -> float:
        """
        Override super update count method to return a float

        Returns:
            float: the density of the lane
        """
        return self.update_density(*args, **kwargs)


class MaxPressurePhase(Approach):
    """
    A class for each NEMA dual ring phase.

    #TODO Figure out how to handle right turns

    """

    def __init__(
        self,
        phase_name: Union[str, int],
        tls_object: sumolib.net.TLS,
        camera_position: tuple,
        phase_2_lane: Tuple[int],
    ) -> None:
        """
        Initializing the Approach class

        @param approach_obj: a sumolib.net.edge object
        @param camera_position: the x, y position of the camera
        """
        super(Approach, self).__init__(
            name=int(phase_name),
            children=self.compose_lanes(tls_object, camera_position, phase_2_lane),
        )

    def compose_lanes(
        self,
        tls_object: sumolib.net.TLS,
        camera_position: tuple,
        phase_2_lane: Tuple[int],
    ) -> tuple:
        """
        This function is called once to compose a list of Lanes, including both the incoming and outgoing

        @param camera_position: the position of the camera. needed in the _recursive_lane_getter_function
        @param edge_obj: a sumolib.net.edge object
        @return: a list of Lane objects
        """
        # loop the lanes in the
        lanes = []
        names = []
        for p in phase_2_lane:
            in_lane, out_lane, _ = tls_object.getLinks()[p][0]
            # add the incoming lanes
            for l, d in zip(
                (in_lane, out_lane), (LaneType.INCOMING, LaneType.OUTGOING)
            ):
                # Only count a lane once.
                if l.getID() not in names:
                    # TODO: How does this handle multiple outgoing connections?
                    lanes.append(
                        MaxPressureLane(
                            self._recursive_lane_getter([l], camera_position, d), d
                        )
                    )
                    names.append(l.getID())
        return lanes

    @staticmethod
    def _get_straight_connection(lanes, direction: LaneType):
        """

        @param lanes: a list of lanes (of which we only care about the last one)
        @return: the previous lane that connects to our lane of interest with a "straight" connection, end flag
        """
        for connect in (
            lanes[-1].getIncoming()
            if direction == LaneType.INCOMING
            else lanes[-1].getOutgoing()
        ):
            connect_connect = (
                connect.getConnection(lanes[-1])
                if direction == LaneType.INCOMING
                else connect
            )
            if "s" in connect_connect.getDirection():
                lanes.append(
                    connect if direction == LaneType.INCOMING else connect.getToLane()
                )
                return lanes
        raise TypeError

    def update_counts(self, **kwargs) -> float:
        """
        Overrides the parent method. Returns the sum of inward and outward pressure

        Returns:
            float: the phase pressure, which is = to the density in - density out
        """
        return sum(super().update_counts(**kwargs))


class MaxPressureTLObservations(TLObservations):
    """
    This class handles individual traffic lights
    """

    def __init__(
        self, net_obj: sumolib.net.Net, tl_id: str, nema_config_dict: OrderedDict
    ):
        self._tl_id = tl_id
        self._center = self.calc_center(net_obj.getNode(tl_id))

        super(TLObservations, self).__init__(
            name=tl_id,
            children=self.compose_phases(net_obj.getTLS(tl_id), nema_config_dict),
        )

        # print("I am ", self._tl_id)
        # for phase in self._children:
        #     in_lanes = [l.lane_list for l in phase if l._direction > 0]
        #     out_lanes = [l.lane_list for l in phase if l._direction < 0]
        #     print(f"Phase #{phase.name} comprised of in lanes {in_lanes} and out lanes {out_lanes}")

        self._clear_duplicate_lanes()

        # print(self._children)

    def compose_phases(
        self, tls_obj: sumolib.net.TLS, nema_config_dict: OrderedDict
    ) -> list:
        """
        This function is called only once, it creates a list of Approaches

        @param net_obj: net object
        @return: a list of Approaches
        """
        return [MaxPressurePhase(
                    phase_name=phase_name,
                    tls_object=tls_obj,
                    phase_2_lane=phase_info["controlling_index"],
                    camera_position=self._center,
                ) for phase_name, phase_info in nema_config_dict["phase"].items()]

    def update_pressure(self, *args, **kwargs):
        return self.update_counts(*args, **kwargs)

    @property
    def pressure_list(
        self,
    ) -> List[float]:
        return self.count_list

    def update_counts(
        self, **kwargs
    ) -> List[List,]:
        """
        This function calls update_counts on the children and passes the center coordinates

        @param kwargs: a forgiving list of inputs
        @return: a list of lists
        """
        self.count_list.clear()
        for child in self:
            self.count_list.append(child.update_counts(center=self._center, **kwargs))
        return self.count_list.copy()

    def get_pressure(self, mapped_method) -> Union[List[int], Dict[int, float]]:
        if mapped_method:
            return {t.name: sum(t.count_list) for t in self._children}
        else:
            return self.count_list


class MaxPressureGlobalObservations(GlobalObservations):
    """
    The overall observation space class
    """

    distance_threshold = DISTANCE_THRESHOLD

    def __init__(
        self,
        net_file: str,
        nema_file_map: Dict[str, str],
        name: str,
    ):
        """
        Instantiating the GlobalObservations class

        @param net_file: the net file path
        @param tl_ids: a list of traffic light ids
        @param name: the name of the object
        @param traci_instance: the traci instance
        """
        self._tl_ids = list(nema_file_map.keys())
        super(GlobalObservations, self).__init__(
            name, children=self._compose_tls(read_net(net_file), nema_file_map)
        )
        # freeze all the initial values
        self.freeze()

    @property
    def tls(
        self,
    ) -> Iterable[MaxPressureTLObservations]:
        yield from self._children

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
            MaxPressureTLObservations(
                net_obj=net_obj,
                tl_id=tls,
                nema_config_dict=read_nema_config(nema_file_map[tls]),
            )
            for tls in self._tl_ids
        ]

    def get_pressure(self, sim_dict) -> List[float]:
        """
        update the density for all phases for all traffic lights and get the pressure (density in - density out)

        @return: self.count_list
        """
        counts = []
        # print("sim_counts", sim_dict[VAR_LANES])
        for child in self.tls:
            counts.extend(
                # update counts really updates the density
                child.update_pressure(
                    lane_ids=sim_dict[VAR_LANES],
                    vehicle_positions=sim_dict[VAR_VEHICLE],
                )
            )

        # return the pre-constructed count dictionary
        return counts

    def get_counts(self, sim_dict) -> list:
        """
        update the counts for all lanes by passing the subscription updates

        @return: self.count_list
        """
        return self.get_density(sim_dict)

    def update_counts(self, **kwargs):
        return self.get_pressure(**kwargs)
