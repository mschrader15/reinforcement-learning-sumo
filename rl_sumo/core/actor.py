import enum
import json
import copy
from distutils.util import strtobool


def read_settings(settings_path):
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


class _TL_HEAD(enum.Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    YIELD = 3
    INACTIVE = 4


class _Timer:
    time = 0.

    @staticmethod
    def get_time():
        return copy.copy(_Timer.time)


class _Base:
    def __init__(self, ):
        # self.parent = parent
        self.init_state = copy.deepcopy(self.__dict__)

    def _re_initialize(self, ):
        for name, value in self.init_state.items():
            self.__dict__[name] = value

    def freeze(self, ):
        """
        This is gets called later than the init state. 
        """
        self.init_state = copy.deepcopy(self.__dict__)


class TrafficLightManager(_Base):

    def __init__(self, tl_id, tl_details):

        self.tl_id = tl_id
        self.current_state: list = [2, 6]
        self.potential_movements = list(map(int, tl_details['phase_order']))
        self.action_space, self.action_space_index_dict = self._create_states()
        self.action_space_length = len(self.action_space)
        # self.light_heads = self._compose_light_heads(tl_details)
        self._task_list = []
        self._last_light_string = ""
        self._last_green_time = 0
        self._transition_active = False
        super().__init__()
        self.traci_c = None

    # def re_initialize(self, ):
    #     self._re_initialize()
    #     for _, light_head in self.light_heads.items():
    #         light_head._re_initialize()

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
            # light_head = self.light_heads[phase]
            end_index = start_index + len(light_head.light_strings[_TL_HEAD.RED])
            substring = light_string[start_index:end_index]
            if 'G' in substring:
                # light_head.transition(_TL_HEAD.GREEN, force=True)
                actual_state.append(phase)
            # elif 'y' in substring:
            #     # light_head.transition(_TL_HEAD.YELLOW, force=True)
            # elif 'g' in substring:
            #     # light_head.transition(_TL_HEAD.YIELD, force=True)
            # else:
                # light_head.transition(_TL_HEAD.RED, force=True)
            # move the start_index along
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
        return self.action_space[action]

    def _create_states(self, ):
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

    # def _compose_light_heads(self, tl_details):
    #     linked_phases = []
    #     light_heads = {}
    #     for phase, phase_info in tl_details['phases'].items():
    #         light_heads[int(phase)] = LightHead(
    #             name="-".join([self.tl_id, phase]),
    #             phase=phase,
    #             phase_info=phase_info,
    #             initial_state=_TL_HEAD.GREEN if int(phase) in self.current_state else _TL_HEAD.RED)
    #         if light_heads[int(phase)].yield_allowed:
    #             linked_phases.append(int(phase))
    #     for yield_phase in linked_phases:
    #         phase_info = tl_details['phases'][str(yield_phase)]
    #         for paired_info in phase_info['yield_for'].items():
    #             paired_light = light_heads[int(paired_info[1])]
    #             light_heads[yield_phase].link_yield_observer(priority=paired_info[0], paired_phase=paired_light)

    #     # freezing the state again after initialization
    #     for head in light_heads.values():
    #         head.freeze()

    #     return light_heads

    def tasks_are_empty(self, ):
        return False if len(self._task_list) else True

    @staticmethod
    def compose_phase_name(c_phase, d_phase, color):
        return "-".join(["+".join(c_phase), "+".join(d_phase), color])


    def update_state(self, action):
        success = False
        desired_state = self._int_to_action(action)
        if (desired_state != self.current_state) and (desired_state in self.action_space):
            if self.tasks_are_empty():
                # set the transition to being active
                self._transition_active = True

                # yellow_to_red_states = [state for state in self.current_state if state not in desired_state]

                # state_progression = [[(self.light_heads[head].transition, light_state) for head in yellow_to_red_states] for light_state in [_TL_HEAD.YELLOW, _TL_HEAD.RED]] + \
                #                     [[(self.light_heads[head].transition, _TL_HEAD.GREEN) for head in desired_state]]

                yellow_name = self.compose_phase_name(self.current_state, desired_state, 'y')
                red_name = yellow_name[:-1] + 'r'

                state_progression = [[[self.set_light_state, yellow_name]], 
                                     [[self.set_light_state, red_name]],
                                     [[self.set_light_state, "+".join(desired_state)], [self._update_state, desired_state], [self._update_timer, ()]]
                                    ]

                # tell myself that I have changed
                # state_progression[-1] += [(self._update_state, desired_state)]

                # update the timer
                # state_progression[-1] += [(self._update_timer, ())]

                self._task_list.extend(state_progression)
                success = True
        light_heads_success = self._step()
        return success * light_heads_success

    def _update_state(self, state):
        self.current_state = state
        self._transition_active = False
        return True

    def _update_timer(self, *args, **kwargs):
        self._last_green_time = _Timer.get_time()
        # signifies a sucessful function completion
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

    def _generate_string(self, ):
        return "".join([self.light_heads[phase].get_string() for phase in self.potential_movements])

    def update_sumo(self, ):
        new_string = self._generate_string()
        if new_string not in self._last_light_string:
            self.traci_c.trafficlight.setRedYellowGreenState(self.tl_id, new_string)
            # print(self.tl_id, new_string)
            self._last_light_string = new_string

    def _check_timer(self, ):
            def _check_timer(self, ):
        return True if _Timer.time - self._timers[self.state] >= self._minimum_times[self.state] \
            else False


    def set_light_state(self, phase_name, color):
        if 



    def get_current_state(self, ):
        """
        Generates the enumerated state for an input to the RL algorithm

        state is the form (2, 6)

        the result will be 2261 if 2 is green and 6 is yellow

        @return: int <= 6383
        """
        # states = []
        # for phase in self.current_state:
        #     states.append(phase * 10 + self.light_heads[phase].state.value)
        # if len(states) > 1:
        #     return states[0] * 100 + states[1]
        # return states[0]
        return self.action_space_index_dict[tuple(self.current_state)]

    def get_last_green_time(self, ):
        return _Timer.time - self._last_green_time

    def get_light_head_colors(self, ):
        colors = [self.light_heads[phase].state.value for phase in self.current_state]
        if len(colors) < 2:
            colors.append(_TL_HEAD.INACTIVE.value)
        return colors


class GlobalActor:
    def __init__(
        self,
        tl_settings_file,
    ):
        self.tls = self.create_tl_managers(read_settings(tl_settings_file))

    def __iter__(self) -> TrafficLightManager:
        for item in self.tls:
            yield item

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
            'color': [_TL_HEAD.INACTIVE.value + 1 for _ in range(len(self.tls) * 2)],
            'last_time': len(self.tls)
        }

    @property
    def discrete_space_shape(self) -> int:
        return [tl_manager.action_space_length for tl_manager in self]

    @staticmethod
    def create_tl_managers(settings: dict) -> [
            TrafficLightManager,
    ]:
        return [TrafficLightManager(tl_id, tl_details) for tl_id, tl_details in settings['traffic_lights'].items()]

    def update_lights(self, action_list: list, sim_time: float) -> None:
        _Timer.time = sim_time
        for action, tl_manager in zip(action_list, self):
            # if action < tl_manager.action_space_length:
            tl_manager.update_state(action)
            tl_manager.update_sumo()
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
            light_head_colors.extend(tl.get_light_head_colors())

        return states, last_green_times, light_head_colors
