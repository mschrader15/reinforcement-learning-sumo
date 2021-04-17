import sumolib
from traci.constants import LAST_STEP_VEHICLE_ID_LIST
from copy import deepcopy
from tools.sumo_tools import read_net


DISTANCE_THRESHOLD = 100  # in meters

# LAST_STEP_VEHICLE_IDS = tc.LAST_STEP_VEHICLE_ID_LIST

def xy_to_m(x0, y0, x1, y1):
    """

    @param x0:
    @param y0:
    @param x1:
    @param y1:
    @return: a distance from (x0, y0) to (x1, y1) in meters
    """
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)


class _Base:

    def __init__(self, name: str, children: list):
        """
        This class serves as a base for the following classes.
        It simplifies the iteration as there are 3 different layers

        @param name: a unique name for each class. This is what will be written in the output
        """
        self.name = name
        self._children: _Base = children
        self.count_list = [child.count_list for child in self]
        self.traci_c = None
        self.init_state = None

    def freeze(self, ):
        self.init_state = deepcopy(self.__dict__)
        for child in self:
            child.freeze()

    def re_initialize(self):
        for name, value in self.init_state.items():
            self.__dict__[name] = value

    def get_lane_count(self, ):
        return sum([child.get_lane_count() for child in self])

    def register_traci(self, traci_c):
        self.traci_c = traci_c
        for child in self:
            child.register_traci(traci_c)

    def __getitem__(self, item):
        for child in self._children:
            if child.name == item:
                return child

    def __iter__(self):
        for child in self._children:
            yield child

    def update_counts(self, **kwargs):
        """
        This function generates a dictionary output for all lanes in the
        network when called from the GlobalObservations level

        @param kwargs: a forgiving list of inputs
        @return: a dictionary with children and their get_counts result
        """
        for child in self:
            child.update_counts(**kwargs)


class Lane(_Base):
    """
    This is the base class for the observable environment, containing:
    1. the count of vehicles in the lane
    2. the SUMO name of the lane

    It can be extended in the future
    """

    def __init__(self, lane_list: [str], ):
        """
        Initialising the base class

        @param lane_list: a list of lanes
        """
        super(Lane, self).__init__(name=lane_list[0], children=[])
        # a "lane" can actually be composed of multiple lanes in the SUMO network
        self._lane_list = lane_list
        # the count of cars in each lane (list to emulate a pointer)
        self.count = [0]
        # the ids of the cars in the lane during the last time step
        self._last_ids = []
        # subscribe to all of the lanes
        self._subscribe_2_lanes()

    def get_lane_count(self, ):
        """
        The lane count is the

        @return: 1
        """
        return 1

    def _subscribe_2_lanes(self, ):
        """
        This function is called once to subscribe to the lanes

        @return:
        """
        for lane in self._lane_list:
            self.traci_c.lane.subscribe(lane, [LAST_STEP_VEHICLE_ID_LIST])

    def update_counts(self, center: tuple, lane_ids: dict, vehicle_positions: dict) -> None:
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
                if _id not in self._last_ids:
                    # check to see if inside the distance threshold (i.e. the "camera" can see them)
                    if xy_to_m(*center, *vehicle_positions[_id][66]) <= DISTANCE_THRESHOLD:
                        new_ids.append(_id)
                else:
                    new_ids.append(_id)

        # assign these new ids to the history
        self._last_ids = new_ids
        self.count[0] = len(new_ids)

    def get_counts(self, ) -> int:
        """
        a public function for getting the counts

        @return: the instance's last count
        """
        return self.count

    def get_direction(self, ) -> str:

        raise NotImplementedError("This function hasn't been implemented")


class Approach(_Base):
    """
    A class for each approach
    """

    def __init__(self, approach_obj: sumolib.net.edge, camera_position: tuple) -> None:
        """
        Initializing the Approach class

        @param approach_obj: a sumolib.net.edge object
        @param camera_position: the x, y position of the camera
        """
        super().__init__(name=approach_obj.getID(), children=self.compose_lanes(approach_obj, camera_position))
        self.direction = self.name
        # composing a dict of lanes. keys are the SUMO name
        self._last_vehicle_num = 0
        self.count_dict = {child.name: child.count for child in self}

    def compose_lanes(self, edge_obj: sumolib.net.edge, camera_position: tuple) -> tuple:
        """
        This function is called once to compose a list of Lanes

        @param camera_position: the position of the camera. needed in the _recursive_lane_getter_function
        @param edge_obj: a sumolib.net.edge object
        @return: a list of Lane objects
        """
        return [Lane(self._recursive_lane_getter([lane], camera_position), ) for lane in edge_obj.getLanes()]

    def _recursive_lane_getter(self, lanes: [sumolib.net.lane], camera_position: tuple):
        """

        @param lanes: a list of lanes that can be extended
        @param camera_position: a tuple that represents the positition of the camera
        @return: an extended list of lanes
        """
        # calculate the distance to the "from" node
        try:
            distance = xy_to_m(*camera_position, *lanes[-1].getEdge().getFromNode().getCoord())
        except TypeError:
            # this exception occurs when we have reached the end of the network. No need to look back anymore
            return [lane.getID() for lane in lanes]
        # if the distance to that node is less than the threshold distance, then we need to look at the next
        # lane for vehicles too
        while distance < DISTANCE_THRESHOLD:
            # continue calling the function until the distance is greater than the threshold distance
            try:
                return self._recursive_lane_getter(lanes=self._get_straight_connection(lanes),
                                                   camera_position=camera_position)
            except TypeError:
                # break the while loop and return where we are at.
                # This happens from the error raised in _get_straight_connection
                break
        # the normal return. the function found enough lanes to satisfy the distance requirement
        return [lane.getID() for lane in lanes]

    @staticmethod
    def _get_straight_connection(lanes, ):
        """

        @param lanes: a list of lanes (of which we only care about the last one)
        @return: the previous lane that connects to our lane of interest with a "straight" connection, end flag
        """
        for connect in lanes[-1].getIncoming():
            if 's' in connect.getConnection(lanes[-1]).getDirection():
                lanes.append(connect)
                return lanes
        raise TypeError


class TLObservations(_Base):
    """
    This class handles individual traffic lights
    """

    def __init__(self, net_obj: sumolib.net, tl_id: list, ):
        """
        Instantiating this class

        @param net_obj: a sumolib.net object
        @param tl_id: the traffic light id
        """
        self._tl_id = tl_id
        self._center = self.calc_center(net_obj.getNode(tl_id))
        super().__init__(name=tl_id, children=self.compose_approaches(net_obj.getTLS(tl_id)))

    def compose_approaches(self, net_obj: sumolib.net.TLS) -> list:
        """
        This function is called only once, it creates a list of Approaches

        @param net_obj: net object
        @return: a list of Approaches
        """
        return_list = []  # {}
        for lane0, _, _ in net_obj.getConnections():
            edge = lane0.getEdge()
            if edge.getToNode().getID() in self._tl_id:
                return_list.append(Approach(approach_obj=edge, camera_position=self._center))
        return return_list

    @staticmethod
    def calc_center(net_obj):
        """
        Calculating the center (in x,y) of the traffic light

        @param net_obj: net object
        @return: a tuple of (x, y)
        """
        return net_obj.getCoord()

    def update_counts(self, **kwargs):
        """
        This function calls update_counts on the children and passes the center coordinates

        @param kwargs: a forgiving list of inputs
        @return: None
        """
        for child in self:
            child.update_counts(center=self._center, **kwargs)


class GlobalObservations(_Base):
    """
    The overall observation space class
    """

    distance_threshold = DISTANCE_THRESHOLD

    def __init__(self, net_file: str, tl_ids: list, name: str, ):
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

    @property
    def size(self):
        return

    def _compose_tls(self, net_obj) -> dict:
        """
        This function is called only once and it creates a list of TLObservations

        @param net_obj: the sumolib.net object
        @return: a list
        """
        # return {tls: TLObservations(net_obj=net_obj, tl_id=tls, ) for tls in self._tl_ids}
        return [TLObservations(net_obj=net_obj, tl_id=tls, ) for tls in self._tl_ids]

    def get_counts(self, subscription_results: dict) -> list:
        """
        update the counts for all lanes by passing the subscription updates

        @return: self.count_dict
        """

        # get a list of ids in each lane
        lane_ids = subscription_results[LAST_STEP_VEHICLE_ID_LIST]

        # get the position of all vehicles in the network
        vehicle_positions = self.traci_c.vehicle.getAllSubscriptionResults()

        # pass the lane_ids and vehicle_positions to the distance calculation
        for child in self:
            child.update_counts(lane_ids=lane_ids, vehicle_positions=vehicle_positions)

        # return the pre-constructed count dictionary
        return self.count_list

    # @staticmethod
    # def subscribe_2_all_vehicles():
    #     """
    #     Subscribe to all new vehicles that have entered the network
    #
    #     @return: None
    #     """
    #     for veh_id in _Base.traci_c.simulation.getDepartedIDList():
    #         _Base.traci_c.vehicle.subscribe(veh_id, [_Base.traci_c.constants.VAR_POSITION])
