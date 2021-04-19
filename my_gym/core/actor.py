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


class Observable:
    """ From: https://stackoverflow.com/questions/44499490/alternate-ways-to-implement-observer-pattern-in-python
       This class is what allows the linked lights to communicate with one another. When an "observed light" aka phase 6
       changes color, say from r -> g, it notifies its observers.
       Its observer would be phase 1, and that phase will then do it's decision logic based on major/minor linked
       phases and the state it is already in
    """

    def __init__(self, ):
        self.observers = []
        self.deciders = []

    def register(self, fn):
        """
        register(fn) appends the observer function list
        :param fn: a function
        :return: None
        """
        self.observers.append(fn)

    def register_decision_logic(self, fn):
        """
        registers a decision logic function
        :param fn: a function
        :return: None
        """
        self.deciders.append(fn)

    def notify_observers(self, *args, **kwargs):
        """
        This function is called from the point of view of an observed light head and notifies the observers,
        and then calls their decision logic
        :param args:
        :param kwargs:
        :return:
        """
        for notify, decide in zip(self.observers, self.deciders):
            notify(*args, **kwargs)
            decide(*args, **kwargs)


class LightHead(Observable, _Base):

    STATES = [_TL_HEAD.GREEN, _TL_HEAD.YELLOW, _TL_HEAD.RED, ]
    ALLOWED_PROGRESSIONS = {_TL_HEAD.RED: [_TL_HEAD.GREEN, _TL_HEAD.YIELD, _TL_HEAD.RED],
                            _TL_HEAD.GREEN: [_TL_HEAD.YELLOW, _TL_HEAD.YIELD, _TL_HEAD.GREEN],
                            _TL_HEAD.YELLOW: [_TL_HEAD.RED, _TL_HEAD.YELLOW],
                            _TL_HEAD.YIELD: [_TL_HEAD.GREEN, _TL_HEAD.YELLOW, _TL_HEAD.YIELD],
                            }

    def __init__(self, name, phase, phase_info, initial_state):
        """
        This class represents a "light head", that is a string of letters that move together.
        i.e. phase 6 on 63069006 is 'GGGG' or 'srrr' (s meaning right on red)
        :param phase_name: the name of the phase ( 6 -> 'six')
        :param green_string: the string of letter representing a green
        :param yellow_string: ^^
        :param red_string: ^^
        """
        Observable.__init__(self)
        self.name = name
        self.phase_name = phase
        self.lane_num = int(phase_info['lane_num'])
        self.right_on_red = bool(strtobool(phase_info['right_on_red']))
        self.yield_allowed = False if phase_info['yield_for'] == "None" else True
        self.light_strings = self._compose_strings()
        self._paired_phase_actions = {}
        self._paired_priority = {}
        self.state = initial_state
        self._timers = {_TL_HEAD.RED: 0.0,
                        _TL_HEAD.YELLOW: 0.0,
                        _TL_HEAD.GREEN: 0.0,
                        _TL_HEAD.YIELD: 0.0
                        }
        self._minimum_times = {
            _TL_HEAD.RED: float(phase_info['min_red_time']),
            _TL_HEAD.YELLOW: float(phase_info['min_yellow_time']),
            _TL_HEAD.GREEN: float(phase_info['min_green_time']),
            _TL_HEAD.YIELD: 0.0,
        }
        _Base.__init__(self, )

    def _compose_strings(self, ):
        lane_num = self.lane_num + 1 if self.right_on_red else self.lane_num
        r = "".join(['s', (lane_num - 1) * 'r']) if self.right_on_red else 'r' * lane_num
        g = 'G' * lane_num
        y = 'y' * lane_num
        y_g = 'g' * lane_num if self.yield_allowed else 'g'
        return {_TL_HEAD.RED: r, _TL_HEAD.GREEN: g, _TL_HEAD.YELLOW: y, _TL_HEAD.YIELD: y_g}

    def _check_timer(self, ):
        return True if _Timer.time - self._timers[self.state] >= self._minimum_times[self.state] \
            else False

    def link_yield_observer(self, priority, paired_phase):
        """
        If a phase needs to observe another (1 needs to know 2 & 6 to decide to yield or not) this function is called
        to link the phase and the observed phase
        :param priority: the priority ('main' or 'secondary')
        :param yield_string: the string for yielding i.e. 'gggg'
        :param paired_phase: the paired phase (not the name but the actual object)
        :return: None
        """
        self._paired_priority[priority] = paired_phase.phase_name
        self._paired_phase_actions[paired_phase.phase_name] = ()
        paired_phase.register(self.yield_observer)
        paired_phase.register_decision_logic(self.process_observations)

    def transition(self, desired, ):
        """
        transition the light if it is an allowed move. See allowed_move
        :param desired: the desired color
        :param time: the simulation time
        :return: Boolean representing the success of the transition
        """
        if desired == self.state:
            return True
        if desired in self.ALLOWED_PROGRESSIONS[self.state]:
            if self._check_timer():
                self.state = desired
                self._timers[self.state] = _Timer.get_time()
                self.update_observers()
                return True
        return False

    def update_observers(self):
        """
        update the observers from the point of view of the observed. i.e. if 6 is observed by 1, then when 6 changes its
        state, see transition(), it will notify 1 of the change and 1 may change its state based on logic in
        process_observation()
        :return:
        """
        if self.observers:
            self.notify_observers(self.phase_name, self.state, self._timers[self.state])

    def get_string(self):
        """
        Get the light string for the specified color
        :return:
        """
        return self.light_strings[self.state]

    def yield_observer(self, phase_name, paired_phase_color, time):
        """
        yield_observer() is triggered by phase 1, for example, when phase 6 calls update_observers(). This function
        stores information about the paired phase' actions.
        For 1 to make a decision whether to yield or not, it needs to have the states and update times of both 6 and 2
        :param phase_name: name of the phase that I am storing
        :param paired_phase_color: the color of the phase that I am storing
        :param time: the time of the change
        :return:
        """
        self._paired_phase_actions[phase_name] = (paired_phase_color, time)

    def process_observations(self, *args, **kwargs):
        """
        This is the function called by 1 to decide what phase to go to when 2 and 6 are updated.
        This function is necessary because the SQL database doesn't report on yield actions
        :param args: same as yield_observer() inputs
        :param kwargs:
        :return:
        """
        main_name = self._paired_priority['main']
        main_color, main_time = _value_error_handler(self._paired_phase_actions[main_name], (_TL_HEAD.GREEN, 1e6))
        if len(self._paired_priority) < 2:
            if (main_color == _TL_HEAD.GREEN) and (self.state not in [_TL_HEAD.YIELD, _TL_HEAD.GREEN]):
                self.state = _TL_HEAD.YIELD
            elif main_color == _TL_HEAD.RED:
                self.state = main_color
            elif (main_color == _TL_HEAD.YELLOW) and self.state == _TL_HEAD.RED:
                # if 1 is red but 6 is yellow, do not put 1 to yellow. Stay red
                pass
        else:
            secondary_name = self._paired_priority['secondary']
            secondary_color, secondary_name = _value_error_handler(self._paired_phase_actions[secondary_name], (_TL_HEAD.GREEN, 1e6))
            if (secondary_color == _TL_HEAD.GREEN) and (main_color == _TL_HEAD.GREEN):
                self.state = _TL_HEAD.YIELD


class TrafficLightManager(_Base):

    def __init__(self, tl_id, tl_details):
        self.tl_id = tl_id
        self.traci_c = None
        self.current_state = (2, 6)
        self.potential_movements = list(map(int, tl_details['phase_order']))
        self.action_space = self._create_states()
        self.action_space_length = len(self.action_space)
        self.light_heads = self._compose_light_heads(tl_details)
        self._task_list = []
        self._last_light_string = ""
        super().__init__()

    def re_initialize(self, ):
        self._re_initialize()
        for _, light_head in self.light_heads.items():
            light_head._re_initialize()

    def set_traci(self, traci_c):
        self.traci_c = traci_c

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
        return possible_states

    def _compose_light_heads(self, tl_details):
        linked_phases = []
        light_heads = {}
        for phase, phase_info in tl_details['phases'].items():
            light_heads[int(phase)] = LightHead(name="-".join([self.tl_id, phase]), phase=phase, phase_info=phase_info,
                                                initial_state=_TL_HEAD.GREEN if int(phase) in self.current_state else _TL_HEAD.RED)
            if light_heads[int(phase)].yield_allowed:
                linked_phases.append(int(phase))
        for yield_phase in linked_phases:
            phase_info = tl_details['phases'][str(yield_phase)]
            for paired_info in phase_info['yield_for'].items():
                paired_light = light_heads[int(paired_info[1])]
                light_heads[yield_phase].link_yield_observer(priority=paired_info[0], paired_phase=paired_light)
        return light_heads

    def tasks_are_empty(self, ):
        return False if len(self._task_list) else True

    def update_state(self, action):
        success = False
        desired_state = self._int_to_action(action)
        if (desired_state != self.current_state) and (desired_state in self.action_space):
            if self.tasks_are_empty():
                yellow_to_red_states = [state for state in self.current_state if state not in desired_state]
                state_progression = [[(self.light_heads[head].transition, light_state) for head in yellow_to_red_states]
                                     for light_state in [_TL_HEAD.YELLOW, _TL_HEAD.RED]] + \
                                    [[(self.light_heads[head].transition, _TL_HEAD.GREEN) for head in desired_state]]
                state_progression[-1] += [(self._update_state, desired_state)]
                self._task_list.extend(state_progression)
                success = True
        light_heads_success = self._step()
        return success * light_heads_success

    def _update_state(self, state):
        self.current_state = state
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
            self.update_sumo()
        return True * result

    def _generate_string(self, ):
        return "".join([self.light_heads[phase].get_string() for phase in self.potential_movements])

    def update_sumo(self, ):
        new_string = self._generate_string()
        if new_string not in self._last_light_string:
            self.traci_c.trafficlight.setRedYellowGreenState(self.tl_id, new_string)
            print(self.tl_id, new_string)
            self._last_light_string = new_string

    def get_current_state(self, ):
        """
        Generates the enumerated state for an input to the RL algorithm

        state is the form (2, 6)

        the result will be 2261 if 2 is green and 6 is yellow

        @return: int <= 6383
        """
        states = []
        for phase in self.current_state:
            states.append(phase * 10 + self.light_heads[phase].state.value)
        if len(states) > 1:
            return states[0] * 100 + states[1]
        return states[0]


class GlobalActor:

    def __init__(self, tl_settings_file, ):
        self.tls = self.create_tl_managers(read_settings(tl_settings_file))

    def __iter__(self):
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
        return len(self.tls)

    @property
    def max_value(self) -> int:
        return max([tl_manager.action_space_length - 1 for tl_manager in self])

    @staticmethod
    def create_tl_managers(settings: dict) -> [TrafficLightManager, ]:
        return [TrafficLightManager(tl_id, tl_details) for tl_id, tl_details in settings['traffic_lights'].items()]

    def update_lights(self, action_list: list, sim_time: float) -> None:
        _Timer.time = sim_time

        for action, tl_manager in zip(action_list, self):
            if action < tl_manager.action_space_length:
                tl_manager.update_state(action)
                tl_manager.update_sumo()
        # return {tl_id: self.tls[tl_id].update_state(action) for tl_id, action in action_dict.items()}

    def get_current_state(self, ) -> [int, ]:
        """
        get the states of all the traffic lights in the network

        @return: list of int
        """
        return [tl.get_current_state() for tl in self]

