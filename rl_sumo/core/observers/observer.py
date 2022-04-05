from bdb import Breakpoint
from collections import OrderedDict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union
import sumolib
from traci.constants import (
    LAST_STEP_VEHICLE_ID_LIST,
    VAR_VEHICLE,
    VAR_LANES,
    VAR_POSITION,
)
from copy import deepcopy

DISTANCE_THRESHOLD = 100  # in meters



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


class _Base:
    def __init__(self, name: str, children: list):
        """
        This class serves as a base for the following classes.
        It simplifies the iteration as there are 3 different layers

        @param name: a unique name for each class. This is what will be written in the output
        """
        self.name = name
        self._children: List[_Base] = children
        self.count_list = [child.count_list for child in self]
        self.init_state = None

        self._val_map = {}

    def freeze(
        self,
    ):
        self.init_state = deepcopy(self.__dict__)
        for child in self:
            child.freeze()

    def re_initialize(self):

        for name, value in deepcopy(self.init_state).items():
            self.__dict__[name] = value

        self.freeze()

    def get_lane_count(
        self,
    ):
        return sum(child.get_lane_count() for child in self)

    def register_traci(self, traci_c):
        # self.traci_c = traci_c
        for child in self:
            child.register_traci(traci_c)

    def __getitem__(self, item):
        for child in self._children:
            if child.name == item:
                return child

    def __iter__(self):
        yield from self._children

    def update_counts(self, **kwargs):
        """
        This function generates a dictionary output for all lanes in the
        network when called from the GlobalObservations level

        @param kwargs: a forgiving list of inputs
        @return:
        """
        self.count_list.clear()
        for child in self:
            self.count_list.append(child.update_counts(**kwargs))
        return self.count_list


    def get_value(self, param: str, mapped: bool = False):
        # if mapped:
        return {c.name: c.get_value(param, mapped) for c in self._children}
        # else:
        #     return [c.get_value(param) for c in self._children]


class Lane(_Base):
    """
    This is the base class for the observable environment, containing:
    1. the count of vehicles in the lane
    2. the SUMO name of the lane

    It can be extended in the future
    """

    def __init__(self, lane_list: List[str], direction: LaneType = LaneType.INCOMING, *args, **kwargs):
        """
        Initialising the base class

        @param lane_list: a list of lanes
        """
        super(Lane, self).__init__(name=lane_list[0].getID(), children=[])
        # a "lane" can actually be composed of multiple lanes in the SUMO network
        self._lane_list = [l.getID() for l in lane_list]

        # the count of cars in each lane (list to emulate a pointer)
        self.count = 0
        # the ids of the cars in the lane during the last time step
        self._last_ids = []
        # subscribe to all of the lanes
        # self._subscribe_2_lanes()
        
        # a storage of the direction (either incoming or outgoing)
        self._direction: LaneType = direction

    @property
    def lanes(
        self,
    ) -> List[str]:
        return self._lane_list

    @lanes.setter
    def lanes(self, val: List[str]) -> None:
        self._lane_list = val

    def get_lane_count(
        self,
    ):
        """
        The lane count is the

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

    def update_counts(self, center: tuple, lane_info: dict, vehicle_info: dict) -> None:
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

        # assign these new ids to the history
        self._last_ids = new_ids
        self.count = len(new_ids)
        return self.count

    def get_counts(
        self,
    ) -> int:
        """
        a public function for getting the counts

        @return: the instance's last count
        """
        return self.count

    def get_direction(
        self,
    ) -> str:

        raise NotImplementedError("This function hasn't been implemented")

    def register_traci(self, traci_c):
        # register traci
        # self.traci_c = traci_c
        # subscribe to the lane that I am in charge of
        self._subscribe_2_lanes(traci_c)

    def contain_lane(self, lane_id: str):
        return lane_id in self._lane_list

    def get_value(self, param: str):
        return getattr(self, param)


class Approach(_Base):
    """
    A class for each approach
    """

    def __init__(
        self, approach_obj: sumolib.net.edge, camera_position: tuple, *args, **kwargs
    ) -> None:
        """
        Initializing the Approach class

        @param approach_obj: a sumolib.net.edge object
        @param camera_position: the x, y position of the camera
        """
        super().__init__(
            name=approach_obj.getID(),
            children=self.compose_lanes(approach_obj, camera_position),
        )
        self.direction = self.name
        # composing a dict of lanes. keys are the SUMO name
        self._last_vehicle_num = 0
        # self.count_dict = {child.name: child.count for child in self}

    @property
    def lane_objs(
        self,
    ) -> List[Lane]:
        return self._children

    def compose_lanes(
        self, edge_obj: sumolib.net.edge, camera_position: tuple
    ) -> tuple:
        """
        This function is called once to compose a list of Lanes

        @param camera_position: the position of the camera. needed in the _recursive_lane_getter_function
        @param edge_obj: a sumolib.net.edge object
        @return: a list of Lane objects
        """
        return [
            Lane(
                self._recursive_lane_getter([lane], camera_position),
            )
            for lane in edge_obj.getLanes()
        ]

    def _recursive_lane_getter(
        self, lanes: List[object], camera_position: tuple, *args, **kwargs
    ):
        """

        @param lanes: a list of lanes that can be extended
        @param camera_position: a tuple that represents the positition of the camera
        @return: an extended list of lanes
        """
        # calculate the distance to the "from" node
        try:
            # if the distance to that node is less than the threshold distance, then we need to look at the next
            distance = xy_to_m(
                *camera_position, *lanes[-1].getEdge().getFromNode().getCoord()
            )

            # lane for vehicles too
            while distance < DISTANCE_THRESHOLD:
                # continue calling the function until the distance is greater than the threshold distance
                try:
                    return self._recursive_lane_getter(
                        lanes=self._get_straight_connection(lanes, *args, **kwargs),
                        camera_position=camera_position,
                    )
                except TypeError:
                    # break the while loop and return where we are at.
                    # This happens from the error raised in _get_straight_connection
                    break

        except TypeError:
            # this exception occurs when we have reached the end of the network. No need to look back anymore
            pass

        # check to make sure that none of the other phases contain this lane yet. AKA the lane can only be counted once
        return lanes

    @staticmethod
    def _get_straight_connection(lanes, *args, **kwargs):
        """

        @param lanes: a list of lanes (of which we only care about the last one)
        @return: the previous lane that connects to our lane of interest with a "straight" connection, end flag
        """
        for connect in lanes[-1].getIncoming():
            if "s" in connect.getConnection(lanes[-1]).getDirection():
                lanes.append(connect)
                return lanes
        raise TypeError


class TLObservations(_Base):
    """
    This class handles individual traffic lights
    """

    def __init__(self, net_obj: sumolib.net, tl_id: list, *args, **kwargs):
        """
        Instantiating this class

        @param net_obj: a sumolib.net object
        @param tl_id: the traffic light id
        """
        self._tl_id = tl_id
        self._center = self.calc_center(net_obj.getNode(tl_id))
        super().__init__(
            name=tl_id,
            children=self.compose_approaches(net_obj.getTLS(tl_id), *args, **kwargs),
        )

        # loop through the children and try to remove duplicate lanes from children
        self._clear_duplicate_lanes()

    def _clear_duplicate_lanes(
        self,
    ) -> None:
        running_list = []
        for approach in self:
            for lane in approach:
                # check if any of the sequentially prior lanes have the same lane id
                new_lanes = []
                for l in lane.lanes:
                    if l not in running_list:
                        new_lanes.append(l)
                lane.lanes = new_lanes
                running_list.extend(lane.lanes)

    def _child_factory(self, edge, *args, **kwargs) -> Approach:

        return Approach(
            approach_obj=edge, camera_position=self._center, *args, **kwargs
        )

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
                return_list.append(self._child_factory(edge=edge))
        return return_list

    @staticmethod
    def calc_center(net_obj):
        """
        Calculating the center (in x,y) of the traffic light

        @param net_obj: net object
        @return: a tuple of (x, y)
        """
        return net_obj.getCoord()

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
            self.count_list.extend(child.update_counts(center=self._center, **kwargs))
        return self.count_list.copy()

    def get_counts(
        self, mapped_method: bool = False
    ) -> Union[List[int], Dict[str, int]]:
        if mapped_method:
            return {t.name: t.count_list for t in self._children}
        else:
            return self.count_list


class GlobalObservations(_Base):
    """
    The overall observation space class
    """

    distance_threshold = DISTANCE_THRESHOLD

    def __init__(
        self,
        net_file: str,
        tl_ids: list,
        name: str,
    ):
        """
        Instantiating the GlobalObservations class

        @param net_file: the net file path
        @param tl_ids: a list of traffic light ids
        @param name: the name of the object
        @param traci_instance: the traci instance
        """
        self._tl_ids = tl_ids
        super().__init__(name, children=self._compose_tls(read_net(net_file)))
        # freeze all the initial values
        self.freeze()

    def __iter__(self) -> Iterable[TLObservations]:
        yield from super().__iter__()

    @property
    def tls(
        self,
    ) -> Iterable[TLObservations]:
        yield from self._children

    @property
    def size(self):
        return

    def _compose_tls(self, net_obj, *args, **kwargs) -> dict:
        """
        This function is called only once and it creates a list of TLObservations

        @param net_obj: the sumolib.net object
        @return: a list
        """
        # return {tls: TLObservations(net_obj=net_obj, tl_id=tls, ) for tls in self._tl_ids}
        return [
            TLObservations(
                net_obj=net_obj,
                tl_id=tls,
            )
            for tls in self._tl_ids
        ]

    def get_counts(self, sim_dict) -> list:
        """
        update the counts for all lanes by passing the subscription updates

        @return: self.count_list
        """

        # get a list of ids in each lane
        # lane_ids = self.traci_c.lane.getAllSubscriptionResults()

        # get the position of all vehicles in the network
        # vehicle_positions = self.traci_c.vehicle.getAllSubscriptionResults()

        # pass the lane_ids and vehicle_positions to the distance calculation
        counts = []
        # print("sim_counts", sim_dict[VAR_LANES])
        for child in self:
            counts.extend(
                child.update_counts(
                    lane_info=sim_dict[VAR_LANES], vehicle_info=sim_dict[VAR_VEHICLE]
                )
            )

        # return the pre-constructed count dictionary
        return counts

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

    @property
    def vehicle_subscriptions(
        self,
    ) -> List[int]:
        return [VAR_POSITION]
