from collections import OrderedDict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union

import sumolib
from rl_sumo.core.observers.observer import Approach, GlobalObservations, Lane, LaneType, TLObservations, read_net
from rl_sumo.helpers.utils import read_nema_config



class Phase(Approach):
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
        outgoing_too: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Initializing the Approach class

        @param approach_obj: a sumolib.net.edge object
        @param camera_position: the x, y position of the camera
        """
        self._outgoing_too = outgoing_too

        super(Approach, self).__init__(
            name=int(phase_name),
            children=self.compose_lanes(tls_object, camera_position, phase_2_lane),
        )

    def _lane_factory(
        self, camera_position, lane: sumolib.net.lane.Lane, *args, **kwargs
    ) -> Lane:
        return Lane(
            self._recursive_lane_getter([lane], camera_position, *args, **kwargs),
            *args,
            **kwargs
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
            iterator = (
                zip((in_lane, out_lane), (LaneType.INCOMING, LaneType.OUTGOING))
                if self._outgoing_too
                else zip((in_lane,), (LaneType.INCOMING,))
            )
            for l, d in iterator:
                # Only count a lane once.
                if l.getID() not in names:
                    # TODO: How does this handle multiple outgoing connections?
                    lanes.append(self._lane_factory(camera_position, l, d))
                    names.append(l.getID())
        return lanes

    @staticmethod
    def _get_straight_connection(lanes, direction: LaneType):
        """
        @param lanes: a list of lanes (of which we only care about the last one)
        @return: the previous lane that connects to our lane of interest with a "straight" connection, end flag
        """
        for connect in (
            lanes[-1].getIncoming(onlyDirect=True)
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
        return super().update_counts(**kwargs)

    def get_values(self, param: str) -> List[float]:
        return [getattr(c, param) for c in self._children]


class PhaseTLObservations(TLObservations):
    """
    This class handles individual traffic lights
    """

    def __init__(
        self, net_obj: sumolib.net.Net, tl_id: str, nema_config_dict: OrderedDict, *args, **kwargs
    ):
        self._tl_id = tl_id
        self._center = self.calc_center(net_obj.getNode(tl_id))

        super(TLObservations, self).__init__(
            name=tl_id,
            children=self.compose_phases(net_obj.getTLS(tl_id), nema_config_dict),
        )

        self._clear_duplicate_lanes()

    def _phase_factory(self, *args, **kwargs) -> Phase:
        return Phase(
                camera_position=self._center,
                *args,
                **kwargs
            )

    def compose_phases(
        self, tls_obj: sumolib.net.TLS, nema_config_dict: OrderedDict
    ) -> list:
        """
        This function is called only once, it creates a list of Approaches

        @param net_obj: net object
        @return: a list of Approaches
        """
        return [
            self._phase_factory(
                phase_name=phase_name,
                tls_object=tls_obj,
                phase_2_lane=phase_info["controlling_index"],
            )
            for phase_name, phase_info in nema_config_dict["phase"].items()
        ]

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

    def get_values(self, param: str = 'counts', mapped_method: bool = False) -> Union[List[int], Dict[int, float]]:
        if mapped_method:
            return {t.name: sum(t.get_values(param)) for t in self._children}
        else:
            return [sum(t.get_values(param)) for t in self._children]

class GlobalPhaseObservations(GlobalObservations):
    """
    The overall observation space class
    """

    distance_threshold = GlobalObservations.distance_threshold

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
        # super(GlobalObservations, self).__init__()
        super(GlobalObservations, self).__init__(
            name, children=self._compose_tls(read_net(net_file), nema_file_map)
        )
        # freeze all the initial values
        self.freeze()

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
            PhaseTLObservations(
                net_obj=net_obj,
                tl_id=tls,
                nema_config_dict=read_nema_config(nema_file_map[tls]),
            )
            for tls in self._tl_ids
        ]
