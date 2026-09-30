from overcooked_ai_py.mdp.actions import Action, Direction
import queue
import os

from torch import layout

class Node:
    """
    Class for A* search algorithm.
    """
    def __init__(self, pos, last_node, cost_value):
        self.pos = pos
        self.last_node = last_node
        self.cost_v = cost_value

    def backtracking_path(self):
        """Backtrack the node and get the action list from start node to this node."""
        node, path = self, []
        while node.last_node is not None:
            direction = (node.pos[0] - node.last_node.pos[0], node.pos[1] - node.last_node.pos[1])
            path.append(Direction.DIRECTION_TO_INDEX[direction])
            node = node.last_node
        return path[::-1]
    
    def __lt__(self, other):
        return self.cost_v < other.cost_v

class OvercookedRuleAgent:
    """
    Rule Agent for designed environgment "surrounding".
    """
    def __init__(self, terrain_mtx, policy_type=None, agent_ego_idx=1, env_parallel=1, args=None):
        # Define some type-related informations
        layout_name = args.env_args["layout_name"]
        proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        layout_dir = os.path.join(proj_dir, "envs", "overcooked", "overcooked_ai", "overcooked_ai_py", "data", "layouts")
        layout_path = os.path.join(layout_dir, f"{layout_name}.layout")

        def load_layout(file_path):
            """Load layout file."""
            with open(file_path, 'r') as f:
                content = f.read()
            ret = eval(content)
            assert ret.__class__ is dict, f"Unsupported form of layout file: layout file {file_path}; file class {ret.__class__}"
            return ret

        layout_dict = load_layout(layout_path)
        self.type2goal = layout_dict["type2goal"]        
        self.orientation2action_idx = {
            (0, -1): 0,
            (0, 1):  1,
            (1, 0):  2,
            (-1, 0): 3,
        }
        # Define some position information
        self.pot_position = layout_dict["pot_position"]
        self.onion_position = layout_dict["onion_position"]
        self.dish_position = layout_dict["dish_position"]
        # Define some basic properties
        self.num_items_for_soup = layout_dict["num_items_for_soup"]
        self.cook_time = layout_dict["cook_time"]
        # Different parallel envs have the same terrain_mtx, policy_type and agent_ego_idx
        self.terrain_mtx = terrain_mtx
        self.policy_type = policy_type
        self.agent_ego_idx = agent_ego_idx          
        # Different parallel envs have different action plans
        self.env_parallel = env_parallel
        self.has_plan = [False for _ in range(self.env_parallel)]
        self.action_plan = [[] for _ in range(self.env_parallel)]
    
    def switch_policy_type(self, policy_type):
        """
        Switch the policy type of agent.
        """
        # switch the policy type and clear the state of agent.
        self.policy_type = policy_type
        self.clear_state()

    def clear_state(self):
        """
        Clear the state of agent.
        """
        self.has_plan = [False for _ in range(self.env_parallel)]
        self.action_plan = [[] for _ in range(self.env_parallel)]

    def act(self, state, env_idx=0, start_state=False):
        """151
        Give action according to set rules.
        """
        def hold_soup(player):
            """Check whether the player holds the soup."""
            return player.held_object and player.held_object.name == "soup"
        def hold_onion(player):
            """Check whether the player holds the onion."""
            return player.held_object and player.held_object.name == "onion"
        def hold_dish(player):
            """Check whether the player holds the dish."""
            return player.held_object and player.held_object.name == "dish"
        def soup_state(state):
            """
            Check the state of the soup in pot.
            Possible values of state:
                0: lack onions
                1: Onions is enough, but cook time is not enough.
                2: Soup is already ok.
            Check whether the pot is still lack of onions.
            """
            if not state.has_object(self.pot_position):
                return 0
            soup_obj = state.get_object(self.pot_position)
            _, num_items, cook_time = soup_obj.state
            if num_items < self.num_items_for_soup:
                return 0
            else:
                return 1 if cook_time < self.cook_time else 2 
        def is_adjacent(p_position, g_position):
            """
            Check whether p_position (player position) is adjacent to g_position (goal position).
            """
            return (abs(p_position[0] - g_position[0]) + abs(p_position[1] - g_position[1])) == 1
        def target_orientation(p_position, g_position):
            """Compute the expected orientation for player to interact with goal grid."""
            return (g_position[0] - p_position[0], g_position[1] - p_position[1])    
                        
        def A_star(p_position, g_position):
            """
            Use A* search algorithm to plan a path for player to fetch goal position.
            Node form: (pos, last_pos)
            """
            def heuristic(pos, g_pos):
                """Compute the heuristic function value of pos."""
                return abs(pos[0] - g_pos[0]) + abs(pos[1] - g_pos[1]) - 1
            # Init root node and priority queue
            root_node = Node(p_position, None, 0)
            root_hv = heuristic(p_position, g_position)
            open_list = queue.PriorityQueue()
            open_list.put([root_hv, root_node])
            # Declare final node
            final_node = None
            while not open_list.empty():
                _, node = open_list.get()
                current_pos = node.pos
                if self.terrain_mtx[current_pos[1]][current_pos[0]] != ' ':
                    # If node is not in accessible area, we pass this node.
                    continue
                if is_adjacent(current_pos, g_position):
                    # Already fetch one target node.
                    final_node = node
                    break
                for action in Direction.ALL_DIRECTIONS:
                    # Expand this node.
                    new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
                    new_node = Node(new_pos, node, node.cost_v+1)
                    node_hv = heuristic(new_pos, g_position)
                    open_list.put([node_hv+new_node.cost_v, new_node])
            # compute the path from the final_node
            return final_node.backtracking_path()

        if start_state:
            self.clear_state()

        if self.has_plan[env_idx]:
            # If we already have plan, do according to the plan.
            action = self.action_plan[env_idx].pop(0)
            if len(self.action_plan[env_idx]) == 0:
                self.has_plan[env_idx] = False
            return action
        else:
            # We don't have a plan.
            player = state.players[self.agent_ego_idx]
            if hold_soup(player):
                # The agent already holds soup.
                if not is_adjacent(player.position, self.type2goal[self.policy_type]):
                    path = A_star(player.position, self.type2goal[self.policy_type])
                    action = path.pop(0)
                    if len(path) > 0:
                        self.has_plan[env_idx] = True
                        self.action_plan[env_idx] = path
                    return action
                else:
                    # Check whether the orientation is correct.
                    t_orientation = target_orientation(player.position, self.type2goal[self.policy_type])
                    if player.orientation != t_orientation:
                        # If orientation is wrong, we correct the orientation.
                        return self.orientation2action_idx[t_orientation]
                    elif state.has_object(self.type2goal[self.policy_type]):
                        # The goal location currently has been occupied.
                        return Action.ACTION_TO_INDEX[Action.STAY]
                    else:
                        # The goal location currently is vacant.
                        return Action.ACTION_TO_INDEX[Action.INTERACT]
            else:
                if soup_state(state) == 0:
                    # Fetch onions and fill in the pot.
                    if hold_onion(player):
                        # The player already holds the onion; let player go to the pot.
                        if not is_adjacent(player.position, self.pot_position):
                            path = A_star(player.position, self.pot_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.pot_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Put the onion into the pot.
                                return Action.ACTION_TO_INDEX[Action.INTERACT]
                    else:
                        # The player hasn't held the onion; let player fetch onion.
                        if not is_adjacent(player.position, self.onion_position):
                            path = A_star(player.position, self.onion_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.onion_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Pick up the onion.
                                return Action.ACTION_TO_INDEX[Action.INTERACT]
                elif soup_state(state) == 1:
                    # We need to wait the soup to be ok.
                    # The behavior should be the same as that when soup_state is 2; the only difference 
                    # is that player will stay when soup_state is 1 and to interact when soup_state is 2.
                    if hold_dish(player):
                        # The player already holds the dish; let player go to the pot.
                        if not is_adjacent(player.position, self.pot_position):
                            path = A_star(player.position, self.pot_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.pot_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Wait the soup to be ok.
                                return Action.ACTION_TO_INDEX[Action.STAY]
                    else:
                        # The player hasn't held the dish; let player fetch dish.
                        if not is_adjacent(player.position, self.dish_position):
                            path = A_star(player.position, self.dish_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.dish_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Fetch the dish.
                                return Action.ACTION_TO_INDEX[Action.INTERACT]
                elif soup_state(state) == 2:
                    # We need to fetch the soup.
                    if hold_dish(player):
                        # The player already holds the dish; let player go to the pot.
                        if not is_adjacent(player.position, self.pot_position):
                            path = A_star(player.position, self.pot_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.pot_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Use dish to fetch the soup.
                                return Action.ACTION_TO_INDEX[Action.INTERACT]
                    else:
                        # The player hasn't held the dish; let player fetch dish.
                        if not is_adjacent(player.position, self.dish_position):
                            path = A_star(player.position, self.dish_position)
                            action = path.pop(0)
                            if len(path) > 0:
                                self.has_plan[env_idx] = True
                                self.action_plan[env_idx] = path
                            return action
                        else:
                            t_orientation = target_orientation(player.position, self.dish_position)
                            if player.orientation != t_orientation:
                                # If orientation is wrong, we correct the orientation.
                                return self.orientation2action_idx[t_orientation]
                            else:
                                # Fetch the dish.
                                return Action.ACTION_TO_INDEX[Action.INTERACT]
