Edge_Lane = {
    '63082002': {
        2: ('gneE11', [0, 1]),
        5: ('gneE11', [2]),
        6: ('834845345#2', [0, 1]),
        4: ('gneE0.12', [0])
    },
    '63082003': {
        1: ('834845345#5.74.43', [2]),
        2: ('115872656#10.14', [0, 1]),
        5: ('115872656#10.14', [2]),
        6: ('834845345#5.74.43', [0, 1]),
        4: ('-638636924#1.9', [0]),
        7: ('-638636924#1.9', [1]),
        8: ('660891910#1.19', [0]),
        3: ('660891910#1.19', [1]),
    },
    '63082004': {
        1: ('834845345#8.108', [2]),
        2: ('gneE22.27', [0, 1]),
        5: ('gneE22.27', [2]),
        6: ('834845345#8.108', [0, 1]),
        4: ('gneE20', [0]),
        7: ('gneE20', [1]),
        8: ('gneE18', [0]),
        3: ('gneE18', [1]),
    }
}


class SimpleController:
    def __init__(self, tl_ids, action_space):
        self.tl_ids = tl_ids
        self.action_space = action_space

    def calc_action(self, counts, ):
        actions = {}
        for tl_id in self.tl_ids:
            local_counts = counts[tl_id]
            eq_data = Edge_Lane[tl_id]
            action_space = self.action_space[tl_id].action_space
            movement_counts = []
            for movements in action_space:
                count = 0
                for move in movements:
                    edge = eq_data[move][0]
                    lanes = ["_".join([edge, str(num)]) for num in eq_data[move][1]]
                    inner_count = 0
                    for lane in lanes:
                        inner_count += local_counts[edge][lane][0]
                    count += inner_count
                movement_counts.append([movements, count])
            actions[tl_id] = sorted(movement_counts, key=lambda x: x[1])[-1][0]
        return actions
