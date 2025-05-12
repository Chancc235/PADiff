import gym
import numpy as np
import cv2
import os
from datetime import datetime
from .overcooked_ai.overcooked_ai_py.mdp.actions import Action, Direction
from .overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from .overcooked_ai.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from .overcooked_ai.overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from envs.multiagentenv import MultiAgentEnv


class OvercookedMultiEnv(MultiAgentEnv):
    def __init__(self,
                 layout_name="surrounding",
                 #  ego_agent_idx=0,
                 baselines=False,
                 seed=0,
                 render=False,
                 remote=True,
                 render_name="",
                 dense_reward=True,
                 reset_when_success=False,
                 ):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)

        np.random.seed(seed)

        self.reset_when_success = reset_when_success
        self.dense_reward = dense_reward

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)
        # self.ego_agent_idx = ego_agent_idx
        self.n_agents = 2
        self.episode_limit = DEFAULT_ENV_PARAMS["horizon"]

        # define property about render
        self.to_render = render
        self.is_remote = remote
        self.save_img = self.to_render and self.is_remote

        # make basic dirs
        file_path = os.path.abspath(__file__)
        # print(f"*****file_path:{file_path}")
        proj_dir = file_path[:file_path.index('marl_collector')+len('marl_collector')]

        self.base_img_dir = os.path.join(proj_dir, "imgs", render_name, datetime.now().strftime("%d-%H-%M-%S"))
        self.base_video_dir = os.path.join(proj_dir, "videos", render_name, datetime.now().strftime("%d-%H-%M-%S"))
        os.makedirs(self.base_img_dir, exist_ok=True)
        os.makedirs(self.base_video_dir, exist_ok=True)

        # try:
        # TODO: need better implementation
        self.render_init("src/envs/overcooked/overcooked_ai/overcooked_ai_js/assets")
        # except:
        # print("Warning: can't create visualization in current env setting.")

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, actions):
        """
        actions:
            [agent with index self.agent_idx action, other agent action]
            is an array with the joint action of the primary and secondary agents in index
            encoded as an int
        returns:
            reward, terminated, info.
        """
        # compute the joint action
        joint_action = (Action.INDEX_TO_ACTION[actions[0]], Action.INDEX_TO_ACTION[actions[1]])

        # step in the base_env
        _, sparse_reward, done, info = self.base_env.step(joint_action)
        reward = info['shaped_r'] if self.dense_reward else sparse_reward

        if self.reset_when_success:
            done = done or (sparse_reward > 0)

        if self.to_render:
            self.render(done)

        return reward, done, {}  # info = None

    def reset(self, arg_dict=None):
        """Reset the environment.
        """
        render_type = arg_dict.get("render_type", "")
        self.base_env.reset()   # reset base environment
        # self.ego_agent_idx = 0  # set ego_agent_idx to 0

        if self.save_img:
            # reset t_ep
            self.t_ep = 0
            # get the item number
            real_base_img_dir = os.path.join(self.base_img_dir, render_type)
            real_base_video_dir = os.path.join(self.base_video_dir, render_type)
            # make sure real base dir exists
            os.makedirs(real_base_img_dir, exist_ok=True)
            os.makedirs(real_base_video_dir, exist_ok=True)
            if len(os.listdir(real_base_img_dir)) == 0:
                item_num = 0
            else:
                item_num = int(max(os.listdir(real_base_img_dir), key=lambda x: int(x)))
            self.img_save_dir = os.path.join(real_base_img_dir, f"{item_num+1}")
            self.video_save_dir = os.path.join(real_base_video_dir, f"{item_num+1}")
            os.makedirs(self.img_save_dir, exist_ok=True)
            os.makedirs(self.video_save_dir, exist_ok=True)

        return self.get_obs(), self.get_state()

    def get_obs(self):
        """Return the observation information of agents.
        """
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        return [ob_p0, ob_p1]

    def get_obs_agent(self, agent_id):
        """Return observation for agent_id.
        """
        obs = self.featurize_fn(self.base_env.state)
        return obs[agent_id]

    def get_obs_size(self):
        """Return the size of observation.
        """
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].flatten().shape[0]
        return obs_shape

    def get_state(self):
        """Return the global state.
        """
        # TODO: another definition for state
        obs = self.featurize_fn(self.base_env.state)
        return [*obs[0], *obs[1]]

    def get_state_size(self):
        """Returns the size of the global state.
        """
        # TODO: another definition for state
        return self.get_obs_size()*2

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list.
        """
        # TODO: currently we allow all actions.
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id.
        """
        # TODO: currently we allow all actions.
        return [1] * self.lA

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take.
        """
        return self.lA

    def get_dynamic_env_info(self):
        """Returns the information that can help rule-based agents to do
           decisions.
        """
        dynamic_env_info = {
            "state": self.base_env.state,
        }
        return dynamic_env_info

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "terrain_mtx": self.base_env.mdp.terrain_mtx,
        }
        return env_info

    def get_stats(self):
        stats = {
        }
        return stats

    def render_init(self, base_dir):
        """Do initial work for rendering. Currently we don't support tomatoes
        """
        chefs_dir = os.path.join(base_dir, "chefs")
        objects_dir = os.path.join(base_dir, "objects")
        terrain_dir = os.path.join(base_dir, "terrain")

        def block_read(file_path):
            """
            read block image
            """
            return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        def build_chef(chef_dict, direction, color, background):
            """
            build information of chef
            """
            # load basic blocks
            chef_arr = block_read(os.path.join(chefs_dir, f"{direction}.png"))
            chef_onion = block_read(os.path.join(chefs_dir, f"{direction}-onion.png"))
            chef_dish = block_read(os.path.join(chefs_dir, f"{direction}-dish.png"))
            chef_sonion = block_read(os.path.join(chefs_dir, f"{direction}-soup-onion.png"))
            hat_arr = block_read(os.path.join(chefs_dir, f"{direction}-{color}hat.png"))
            # compute masks
            hat_mask = (hat_arr[:, :, -1] != 0)[:, :, None]
            chefs_mask = (chef_arr[:, :, -1] != 0)[:, :, None]

            def blocks_overlay(body_arr):
                """overy hat_block, chef_block and background"""
                return (hat_mask * hat_arr + (1 - hat_mask) * body_arr) * chefs_mask + (1 - chefs_mask) * background

            chef_dict[direction]["ept"] = blocks_overlay(chef_arr)
            chef_dict[direction]["onion"] = blocks_overlay(chef_onion)
            chef_dict[direction]["dish"] = blocks_overlay(chef_dish)
            chef_dict[direction]["sonion"] = blocks_overlay(chef_sonion)

        # basic terrain blocks
        counter_arr = block_read(os.path.join(terrain_dir, "counter.png"))
        floor_arr = block_read(os.path.join(terrain_dir, "floor.png"))
        onions_arr = block_read(os.path.join(terrain_dir, "onions.png"))
        dishes_arr = block_read(os.path.join(terrain_dir, "dishes.png"))
        pot_arr = block_read(os.path.join(terrain_dir, "pot.png"))
        serve_arr = block_read(os.path.join(terrain_dir, "serve.png"))
        self.counter_arr, self.pot_arr = counter_arr, pot_arr

        # define label2img
        label2img = {
            "X": counter_arr,
            " ": floor_arr,
            "O": onions_arr,
            "D": dishes_arr,
            "P": pot_arr,
            "S": serve_arr,
        }

        # define terrain array
        self.block_size = (15, 15, 4)
        self.block_h, self.block_w = self.block_size[0], self.block_size[1]
        H, W = len(self.mdp.terrain_mtx), len(self.mdp.terrain_mtx[0])
        self.terrain_arr = np.zeros((H*self.block_h, W*self.block_w, self.block_size[2]))
        self.terrain_arr[:, :] = [153, 178, 199, 255]
        for row_idx, row in enumerate(self.mdp.terrain_mtx):
            for col_idx, ele in enumerate(row):
                self.terrain_arr[row_idx*self.block_h:(row_idx+1)*self.block_h, col_idx *
                                 self.block_w:(col_idx+1)*self.block_w] = label2img[ele]

        # blocks relating to chefs
        self.blue_chef = {direction: {"ept": None, "onion": None, "dish": None, "sonion": None}
                          for direction in ["SOUTH", "NORTH", "EAST", "WEST"]}
        self.green_chef = {direction: {"ept": None, "onion": None, "dish": None, "sonion": None}
                           for direction in ["SOUTH", "NORTH", "EAST", "WEST"]}
        for direction in ["SOUTH", "NORTH", "EAST", "WEST"]:
            build_chef(self.blue_chef, direction, "blue", floor_arr)
            build_chef(self.green_chef, direction, "green", floor_arr)
        self.chefs = [self.blue_chef, self.green_chef]

        # get item blocks
        self.ob_dish_arr = block_read(os.path.join(objects_dir, "dish.png"))
        self.ob_onion_arr = block_read(os.path.join(objects_dir, "onion.png"))
        self.ob_pot_exp_arr = block_read(os.path.join(objects_dir, "pot-explosion.png"))
        self.ob_onion_1_arr = block_read(os.path.join(objects_dir, "soup-onion-1-cooking.png"))
        self.ob_onion_2_arr = block_read(os.path.join(objects_dir, "soup-onion-2-cooking.png"))
        self.ob_onion_3_arr = block_read(os.path.join(objects_dir, "soup-onion-3-cooking.png"))
        self.ob_onion_cooked_arr = block_read(os.path.join(objects_dir, "soup-onion-cooked.png"))
        self.ob_onion_dish_arr = block_read(os.path.join(objects_dir, "soup-onion-dish.png"))

        # Orientation
        self.tuple2direction = {
            Direction.NORTH: "NORTH",
            Direction.SOUTH: "SOUTH",
            Direction.EAST: "EAST",
            Direction.WEST: "WEST",
        }

        if not self.is_remote:
            # init/define viewer
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer(maxwidth=800)

    def render(self, done):
        """Function for the env's rendering.
        """
        def embed_arr(sub_arr, background_arr):
            """
            Embed sub_arr into the background_arr.
            """
            mask = (sub_arr[:, :, -1] != 0)[:, :, None]
            return mask * sub_arr + (1 - mask) * background_arr

        if not self.is_remote:
            cv2.destroyAllWindows()
        players_dict = {player.position: player for player in self.base_env.state.players}
        frame = self.terrain_arr.copy()
        for y, terrain_row in enumerate(self.mdp.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict:
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    direction_name = self.tuple2direction[orientation]
                    player_object = player.held_object
                    if player_object:
                        # TODO: how to deal with held objects
                        player_idx_lst = [i for i, p in enumerate(self.base_env.state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        if player_object.name == "onion":
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["onion"]
                        elif player_object.name == "dish":
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["dish"]
                        elif player_object.name == "soup":
                            soup_type, _, _ = player_object.state
                            assert soup_type == "onion", "Currently we only support the visualization of onion type."
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["sonion"]
                        else:
                            raise ValueError(f"Unsupported player_object.name {player_object.name}")
                    else:
                        player_idx_lst = [i for i, p in enumerate(self.base_env.state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1) *
                              self.block_w] = self.chefs[player_idx_lst[0]][direction_name]["ept"]
                else:
                    if element == "X" and self.base_env.state.has_object((x, y)):
                        counter_obj = self.base_env.state.get_object((x, y))
                        if counter_obj.name == "onion":
                            dynamic_arr = embed_arr(self.ob_onion_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif counter_obj.name == "dish":
                            dynamic_arr = embed_arr(self.ob_dish_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif counter_obj.name == "soup":
                            soup_type, _, _ = counter_obj.state
                            assert soup_type == "onion", "Currently we only support the visualization of onion type."
                            dynamic_arr = embed_arr(self.ob_onion_dish_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        else:
                            raise ValueError(f"Unsupported object name on counter {counter_obj.name}")
                    elif element == "P" and self.base_env.state.has_object((x, y)):
                        soup_obj = self.base_env.state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        assert soup_type == "onion", "Currently we only support the visualization of onion type."
                        if num_items == 1:
                            dynamic_arr = embed_arr(self.ob_onion_1_arr, self.pot_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif num_items == 2:
                            dynamic_arr = embed_arr(self.ob_onion_2_arr, self.pot_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif num_items >= 3:
                            if cook_time < 20:
                                dynamic_arr = embed_arr(self.ob_onion_3_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            else:
                                dynamic_arr = embed_arr(self.ob_onion_cooked_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        else:
                            raise ValueError(f"Invalid num_items for pot {num_items}")
        # self.viewer.imshow(frame[:, :, :-1]/255.)
        if not self.is_remote:
            cv2.imshow('window', frame[:, :, :-1]/255.)
            cv2.waitKey(1)
        else:
            cv2.imwrite(os.path.join(self.img_save_dir, f"{self.t_ep}.png"), frame[:, :, :-1])
            self.t_ep += 1
            if done:
                # transform imgs into video
                video_path = os.path.join(self.video_save_dir, "video0.avi")
                #self.img2video(self.img_save_dir, video_path)

    def img2video(self, img_dir, video_path):
        """
        Transform imgs into video
        """
        file_list = os.listdir(img_dir)
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))

        for item in file_list:
            if item.endswith('.png'):
                item = os.path.join(img_dir, item)
                img = cv2.imread(item)
                H, W, _ = img.shape
                break

        fps = 10
        size = (W, H)

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

        for item in file_list:
            if item.endswith('.png'):
                item = os.path.join(img_dir, item)
                img = cv2.imread(item)
                video.write(img)
        video.release()
        cv2.destroyAllWindows()

    def close(self):
        """Don't need to do anything.
        """
        pass
