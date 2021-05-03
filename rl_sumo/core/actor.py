import enum
import json
import copy
from distutils.util import strtobool
from xml.dom import minidom
import traci


def read_settings(settings_path):
    """Read in the traffic light settings file (this is probably custom 
    to the UAs effort)

    Args:
        settings_path: path to json file

    Returns:
        [type]: [description]
    """
    with open(settings_path, 'rb') as f:
        return json.load(f, )


def _value_error_handler(tuple_obj, default):
    """
    This function handles trying to unpack an empty tuple
    :param tuple_obj: tuple to unpack
    :param default: the empty tuple value
    :return:
    """
    try:
        x, y = tuple_obj
        return x, y
    except ValueError:
        return default


def tls_file(path):
    """
    This function reads in a SUMO tls phasing file

    It yields each phase and it's index

    Args:
        path (str): path to file
    Yields:
        phase: a mindom element
        i: the phase object index
    """
    tls_obj = minidom.parse(path)
    for i, phase in enumerate(tls_obj.getElementsByTagName("phase")):
        yield phase, i


def safe_int(prospective_int):
    """
    safely convert a number to an integer

    Args:
        prospective_int (any)
    Returns:
        [any]: eithter int(number) or non-number
    """
    try:
        return int(prospective_int)
    except (ValueError, TypeError):
        return prospective_int


class _Timer:
    """
    This class creates a global timer

    Returns:
        float: a copy of the current time
    """
    time = 0.

    @staticmethod
    def get_time():
        return copy.copy(_Timer.time)


class _Base:
    """
    This base class freezes and unfreezes the data
    """
    def __init__(self, ):
        # self.parent = parent
        self.init_state = copy.deepcopy(self.__dict__)

    def _re_initialize(self, ):
        """
        This function "unfreeezes" the initial values that were saved in the call to freeze
        """
        for name, value in self.init_state.items():
            self.__dict__[name] = value

    def freeze(self, ):
        """
        This is gets called later than the init state. and freezes all the values in self 
        """
        self.init_state = copy.deepcopy(self.__dict__)


class TrafficLightManager(_Base):
    def __init__(self, tl_id, tl_details, tl_file):
        """
        TrafficLightManager represents a controller sitting at each traffic light

        Args:
            tl_id (str): a unique id for the traffic light 
            tl_details (dict): a dictionary containing setup information about the traffic light
            tl_file [str]: a string pointing to the xml file describing traffic light states 
        """

        self.tl_id = tl_id
        self.current_state: list = [2, 6]
        self.tl_details = tl_details
        self.potential_movements = list(map(int, tl_details['phase_order']))
        self.action_space, self.action_space_index_dict = self._create_states()
        self.action_space_length = len(self.action_space)
        self.phase_num_name_eq = self.read_in_tls_xml(tl_file)
        self._task_list = []
        self._last_green_time = 0
        # self._transition_active = False
        self._sim_time = 0
        self._last_changed_time = 0
        self._color = 'g'
        self._minimum_times = {
            'r': 3,  # this is really the time from yellow -> red
            'y': 5,  # this is green -> yellow
            'g': 1  # this is red -> green
        }
        self._color_int = {'r': 0, 'y': 1, 'g': 2}

        super().__init__()

        self.traci_c = None

    def compose_minimum_times(self, ):
        pass

    def re_initialize(self, ):
        self._re_initialize()

    def _set_initial_states(self, light_string: str):
        """
        This function sets the initial states to what they are in the simulation when the reinforcement learning algorithm takes over. 
        It is called when traci is passed to this class

        Args:
            light_string (str): the string returned from traci.trafficlight.getRedYellowGreenState()
        """
        start_index = 0
        actual_state = []
        for phase in self.potential_movements:
            end_index = self.tl_details['phases'][str(phase)]['lane_num'] + 1 if bool(
                strtobool(self.tl_details['phases'][str(phase)]['right_on_red'])) else self.tl_details['phases'][str(
                    phase)]['lane_num']
            end_index = start_index + end_index
            substring = light_string[start_index:end_index]
            if 'G' in substring:
                actual_state.append(phase)
            start_index = end_index
        if actual_state in self.action_space:
            self.current_state = actual_state
        else:
            print('uh oh. Alert Max')

    def set_traci(self, traci_c):
        """
        This function is called by the parent class to pass Traci. Called on Environment resets

        It passes traci to the class, and also sets the light heads to the state that they are in the simulation

        Args:
            traci_c ([type]): A traci connection object
        """
        self.traci_c = traci_c
        self._set_initial_states(self.traci_c.trafficlight.getRedYellowGreenState(self.tl_id))

    def _int_to_action(self, action: int) -> list:
        """
        convert an integer to an action

        Args:
            action (int): an integer from the RL-algorithm

        Returns:
            list: a list of [phase, phase]
        """
        return self.action_space[action]

    def _create_states(self, ):
        """
        This is a helper function to create a list of possible states.
        It is not pretty and based completely on the self.potential movements format

        Returns:
            possible states and an dict pointing to each states
        """
        mainline = [move for move in self.potential_movements if move in [1, 2, 5, 6]]
        secondary = [move for move in self.potential_movements if move in [3, 4, 7, 8]]
        possible_states = []
        for j in mainline:
            for i in mainline:
                if j in [1, 2] and i in [5, 6]:
                    possible_states.append([j, i])
                elif len(mainline) < 2:
                    possible_states.append([j])
        for j in secondary:
            for i in secondary:
                if j in [3, 4] and i in [7, 8]:
                    possible_states.append([j, i])
                elif len(secondary) < 2:
                    possible_states.append([j])
        return possible_states, {tuple(state): i for i, state in enumerate(possible_states)}

    def read_in_tls_xml(self, file_path):
        """
        This function reads in the traffic light settings

        Args:
            file_path (str): file path to traffic light simulation 

        Returns:
            dict: a dictionary pointing to the phase and its index
        """
        phase_dict = {}
        for phase, i in tls_file(file_path):
            name = phase.getAttribute('name').split("-")
            split_name = [inner_data.split('+') for inner_data in name]
            flattened_name = [safe_int(inner_2) for inner_data in split_name for inner_2 in inner_data]
            if len(flattened_name) < 3:
                flattened_name.extend(['g'])
            self.recursive_dict_constructor(phase_dict, flattened_name, i)
        return phase_dict

    def recursive_dict_constructor(self, _dict, keys, value):
        """
        construct the output of self.read_in_tls_xml recursively

        Args:
            _dict ([dictionary]): 
            keys ([]): 
            value ([type]): 
        """
        if len(keys) > 1:
            # try:
            try:
                result = _dict[keys[0]]
                keys.pop(0)
            except KeyError:
                _dict[keys[0]] = {}
                result = _dict
            self.recursive_dict_constructor(result, keys, value)
        else:
            _dict[keys[0]] = value

    def tasks_are_empty(self, ):
        # check to see if the task list is empty
        return not len(self._task_list)

    def update_state(self, action, sim_time):
        success = False
        desired_state = self._int_to_action(action)
        self._sim_time = sim_time
        if (
            (desired_state != self.current_state)
            and (desired_state in self.action_space)
            and self.tasks_are_empty()
        ):
            # set the transition to being active
            # self._transition_active = True

            states = [*self.current_state, *desired_state]

            state_progression = [[[self.set_light_state, states, 'y']], [[self.set_light_state, states, 'r']],
                                 [[self.set_light_state, desired_state, 'g'], [self._update_state, desired_state],
                                  [self._update_timer, ()]]]

            self._task_list.extend(state_progression)
            success = True

        light_heads_success = self._step()
        return success * light_heads_success

    def _update_state(self, state):
        self.current_state = state
        # self._transition_active = False
        return True

    def _update_timer(self, *args, **kwargs):
        self._last_green_time = _Timer.get_time()
        return True

    def _step(self, ):
        result = True
        if not self.tasks_are_empty():
            task_list = self._task_list[0]
            result = 1
            for fn, *args in task_list:
                result *= fn(*args)
            if result:
                del self._task_list[0]
            # self.update_sumo()
        return True * result

    def _check_timer(self, color):
        return self._sim_time - self._last_changed_time >= self._minimum_times[color]

    def set_light_state(self, phase_list, color):
        if self._check_timer(color):
            self._last_changed_time = self._sim_time
            success = False
            while not success:
                try:
                    self.traci_c.trafficlight.setPhase(self.tl_id, self._get_index(phase_list, color))
                    success = True
                except traci.exceptions.TraCIException:
                    self.traci_c.trafficlight.setProgram(self.tl_id, f'{self.tl_id}-2')
            # print(self.traci_c._socket.getpeername(), "set successfully")
            self._color = color
            return True
        return False

    def _get_index(self, phase_list, color):
        phase_dict = self.phase_num_name_eq[phase_list[0]]
        if len(phase_list) > 1:
            # iter_list = phase_list[1:-1] if color in 'g' else phase_list[1:]
            for phase in phase_list[1:]:
                phase_dict = phase_dict[phase]
        return phase_dict[color]

    def get_current_state(self, ):
        return self.action_space_index_dict[tuple(self.current_state)]

    def get_last_green_time(self, ):
        return _Timer.time - self._last_green_time

    def get_light_head_color(self, ):
        return self._color_int[self._color]


class GlobalActor:
    def __init__(self, tl_settings_file, tl_file_dicts):
        self.tls = self.create_tl_managers(read_settings(tl_settings_file), tl_file_dicts)

    def __iter__(self) -> TrafficLightManager:
        yield from self.tls

    def __getitem__(self, item: str) -> TrafficLightManager:
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

    def re_initialize(self, ) -> None:
        """
        This functions reinitializes everything to its default values
        @return:
        """
        for tl_manager in self:
            tl_manager.re_initialize()

    @property
    def size(self, ) -> int:
        return {
            'state': [tl_manager.action_space_length for tl_manager in self],
            'color': [3 for _ in range(len(self.tls))],
            'last_time': len(self.tls)
        }

    @property
    def discrete_space_shape(self) -> int:
        return [tl_manager.action_space_length for tl_manager in self]

    @staticmethod
    def create_tl_managers(settings: dict, tl_files: dict) -> [
            TrafficLightManager,
    ]:
        return [
            TrafficLightManager(tl_id, tl_details, tl_files[tl_id])
            for tl_id, tl_details in settings['traffic_lights'].items()
        ]

    def update_lights(self, action_list: list, sim_time: float) -> None:
        _Timer.time = sim_time
        for action, tl_manager in zip(action_list, self):
            # if action < tl_manager.action_space_length:
            tl_manager.update_state(action, sim_time)
            # tl_manager.update_sumo()
        # return {tl_id: self.tls[tl_id].update_state(action) for tl_id, action in action_dict.items()}

    def get_current_state(self, ) -> [
            int,
    ]:
        """
        get the states of all the traffic lights in the network

        @return: list of int
        """
        states = []
        last_green_times = []
        light_head_colors = []

        for tl in self:
            states.append(tl.get_current_state())
            last_green_times.append(tl.get_last_green_time())
            light_head_colors.append(tl.get_light_head_color())

        return states, last_green_times, light_head_colors
