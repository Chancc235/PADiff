import copy
from enum import IntEnum
from queue import PriorityQueue


class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Node:
    def __init__(self, loc, g, h, parent=None, act_from_parent=None) -> None:
        self.loc = loc
        self.g = g
        self.h = h
        self.parent = parent
        self.act_from_parent = act_from_parent

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


def distance_func(loc1, loc2, mode="l1"):
    if mode == "l1":
        return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])  # l1 distance
    elif mode == "l2":
        return ((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2) ** 0.5   # l2 distance


class LBFRuleAgent:
    """ Simple A* Agent with extra ability to [wait for 1 step] if action failed (collision occurred)
        Has 4 policy types naming: ["NORTH", "SOUTH", "EAST", "WEST"]
        Each type means this agent always tries to eat the food at the [type_name]ernmost loction.

        currently, only support 2players and this agent controls [1]
    """

    def __init__(self, args) -> None:
        self.args = copy.deepcopy(args)
        env_info = args.env_info

        self.n_envs = args.batch_size_run
        self.field_size = env_info["field_size"]
        self.max_food = env_info["max_food"]
        self.n_env_agents = args.n_env_agents

        assert self.n_env_agents == 2  # only allow 2 players
        assert len(args.agent_ids) == 1  # rule agent should be at last
        self.agent_id = args.agent_ids[0]

        # ==== state
        # "object list" view of the map
        self.food_list = [None for _ in range(self.n_envs)]
        self.player_list = [None for _ in range(self.n_envs)]

        # ==== policy
        # shared by all env_id
        self.policy_type = None
        # each env has different
        self.goal_loc = [None for _ in range(self.n_envs)]  # current food that want to eat
        self.plan = [[] for _ in range(self.n_envs)]
        self.waiting = [False for _ in range(self.n_envs)]  # set this flag to avoid dead_lock
        self.env_finished = [False for _ in range(self.n_envs)]  # set this flag to avoid planning after having all food

    def setup(self, state, env_idx, first_set):
        assert state.shape[0] == (self.max_food + self.n_env_agents) * 3 + 3
        self.food_list[env_idx] = []
        self.player_list[env_idx] = []

        # setup food
        for i in range(self.max_food):
            y, x, level = state[3*i:3*i+3]
            if level > 0:
                self.food_list[env_idx].append([y, x, level])

        # setup players
        for i in range(self.max_food, self.max_food+self.n_env_agents):
            y, x, level = state[3*i:3*i+3]
            assert level > 0
            self.player_list[env_idx].append([y, x, level])

        # reset all goal and plan
        if first_set:
            self.clear_goal(env_idx)

    def switch_policy_type(self, policy_type):
        self.policy_type = policy_type

    def has_goal(self, env_idx) -> bool:
        return self.goal_loc[env_idx] is not None and self.plan[env_idx]

    def clear_goal(self, env_idx):
        self.goal_loc[env_idx] = None
        self.plan[env_idx] = []
        self.waiting[env_idx] = False

    def set_goal_and_plan(self, env_idx):
        def check_valid_action(loc, action):
            # sourcery skip: chain-compares, use-next
            new_loc = copy.deepcopy(loc)
            if action == Action.NORTH:
                new_loc[0] -= 1
            elif action == Action.SOUTH:
                new_loc[0] += 1
            elif action == Action.EAST:
                new_loc[1] += 1
            elif action == Action.WEST:
                new_loc[1] -= 1
            else:
                print("Invalid Action:", action)

            valid = [0, 0] <= new_loc and new_loc <= [self.field_size-1, self.field_size-1]
            for food in self.food_list[env_idx]:
                if food[:2] == new_loc:
                    valid = False
                    break
            for player in self.player_list[env_idx]:
                if player[:2] == new_loc:
                    valid = False
                    break

            return valid, new_loc

        def is_adjacent(loc1, loc2):
            return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1]) == 1

        # choose goal_food based on policy_type
        if self.policy_type == "NORTH":
            self.food_list[env_idx].sort(key=lambda x: x[0])
        elif self.policy_type == "SOUTH":
            self.food_list[env_idx].sort(key=lambda x: -x[0])
        elif self.policy_type == "EAST":
            self.food_list[env_idx].sort(key=lambda x: -x[1])
        elif self.policy_type == "WEST":
            self.food_list[env_idx].sort(key=lambda x: x[1])
        else:
            print("Invalid Policy:", self.policy_type)

        self.goal_loc[env_idx] = self.food_list[env_idx][0][:2]

        # ====== A* search for a plan to goal_loc ======
        open_list, close_list = PriorityQueue(), []

        current_loc = self.player_list[env_idx][self.agent_id][:2]
        goal_loc = self.goal_loc[env_idx]
        start_node = Node(current_loc, 0, distance_func(current_loc, goal_loc))
        open_list.put([start_node.f, start_node])

        while not open_list.empty():
            _, node = open_list.get()
            if is_adjacent(node.loc, goal_loc):
                back_trace_plan = []
                while node.parent is not None:
                    back_trace_plan.insert(0, int(node.act_from_parent))
                    node = node.parent
                back_trace_plan.append(int(Action.LOAD))
                self.plan[env_idx] = back_trace_plan
                break
            else:
                close_list.append(node)
                for action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]:
                    valid, new_loc = check_valid_action(node.loc, action)
                    for close_node in close_list:
                        if close_node.loc == new_loc:
                            valid = False
                            break
                    if valid:
                        new_node = Node(new_loc, node.g+1, distance_func(new_loc, goal_loc), node, action)
                        open_list.put([new_node.f, new_node])

    def act(self, state, env_idx, start_state=False):
        ''' state: state for one environment'''
        current_loc = state[(self.max_food+self.agent_id)*3: (self.max_food+self.agent_id)*3+2]
        if start_state:
            self.env_finished[env_idx] = False
            self.setup(state, env_idx, first_set=True)
        else:
            if self.env_finished[env_idx] == True:
                return int(Action.NONE)
            # check whether goal has been reached
            if self.goal_loc[env_idx] is not None:
                reached = all(self.goal_loc[env_idx] != food[:2] for food in self.food_list[env_idx])

                if reached:
                    self.clear_goal(env_idx)

            # check whether in dead_lock, and set "waiting" flag
            if self.plan[env_idx]:
                if self.waiting[env_idx]:
                    # has waitted for a step
                    self.waiting[env_idx] = False
                elif self.plan[env_idx][0] != Action.LOAD and (current_loc == self.player_list[env_idx][self.agent_id][:2]).all():
                    # action failed, wait for 1 step
                    self.waiting[env_idx] = True
                    self.clear_goal(env_idx)  # ! A little bit waste of time, but can avoid 2_step dead_lock
                else:
                    # confirm that the first action in plan has been operated successfully
                    self.plan[env_idx] = self.plan[env_idx][1:]

            # update state info
            self.setup(state, env_idx, first_set=False)
            self.env_finished[env_idx] = not self.food_list[env_idx]

        # setup goal and do planning
        if not self.env_finished[env_idx] and not self.has_goal(env_idx):
            self.set_goal_and_plan(env_idx)

        # finally, return action of the plan
        if self.env_finished[env_idx] or self.waiting[env_idx]:
            return int(Action.NONE)
        else:
            return self.plan[env_idx][0]
