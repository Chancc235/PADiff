import datetime
import os
import sys
from sys import stderr

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from envs.multiagentenv import MultiAgentEnv
from gym.envs.registration import register


class ForagingEnv(MultiAgentEnv):

    def __init__(self,
                 field_size: int,
                 players: int,
                 max_food: int,
                 force_coop: bool,
                 partiteammate_observe: bool,
                 is_print: bool,
                 seed: int,
                 need_render: bool,
                 sight: int = 0,
                 remote: bool = True,
                 render_output_path: str = ''):
        # record these as env_info
        self.field_size = field_size
        self.max_food = max_food

        if sight == 0:
            sight = field_size
        self.n_agents = players
        self.n_actions = 6
        self._total_steps = 0
        self._episode_steps = 0
        self.is_print = is_print
        self.need_render = need_render
        np.random.seed(seed)

        self.episode_limit = 50

        self.agent_score = np.zeros(players)

        env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(field_size, players, max_food,
                                                              "-coop" if force_coop else "",
                                                              "" if sight == field_size else f"-{sight}s")

        # ! only register used env in gym
        grid_obs = False
        register(
            id="Foraging{5}{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(field_size, players, max_food,
                                                               "-coop" if force_coop else "",
                                                               "" if sight == field_size else f"-{sight}s",
                                                               "-grid" if grid_obs else ""),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": players,
                "max_player_level": 3,
                "field_size": (field_size, field_size),
                "max_food": max_food,
                "sight": sight,
                "max_episode_steps": 50,
                "force_coop": force_coop,
                "grid_observation": grid_obs,
                "remote": remote
            },
        )

        if is_print:
            print('Env:', env_id, file=stderr)
        self.env = gym.make(env_id)
        self.env.seed(seed)

        if self.need_render:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%field_size")
            render_path = os.path.join(render_output_path, date)
            if not os.path.exists(render_path):
                os.makedirs(render_path, exist_ok=True)
            self.render_path = render_path

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = th.Tensor(actions) if type(actions) != th.Tensor else actions
        self._total_steps += 1
        self._episode_steps += 1

        if self.is_print:
            print(f'env step {self._episode_steps}', file=stderr)
            print('t_steps: %d' % self._episode_steps, file=stderr)
            print('current position: ', file=stderr)
            print(self.get_players_position(), file=stderr)
            print('choose actions: ', file=stderr)
            print(actions.cpu().numpy().tolist(), file=stderr)
            position_record = self.get_players_position()
            action_record = actions.cpu().numpy().tolist()
            env_info = {
                'position': position_record,
                'action': action_record
            }
            import pickle
            pickle.dump(env_info, open(os.path.join(self.render_path, f'info_step_{self._episode_steps}.pkl'), 'wb'))

        if self.need_render:
            fig = plt.figure()
            data = self.env.render(mode='rgb_array')
            plt.imshow(data)
            plt.axis('off')
            fig.savefig(os.path.join(self.render_path, f'image_step_{self._total_steps}.png'), bbox_inches='tight', dpi=600)
            plt.close('all')

        self.obs, rewards, dones, info = self.env.step(actions.cpu().numpy())

        self.agent_score += rewards

        reward = np.sum(rewards)
        terminated = np.all(dones)

        return reward, terminated, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.get_observation_space().shape[0]

    # TODO: currently, set 0th agent's ob as state, for full sight range and ordered agent mapping
    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        res = [0] * self.n_actions
        t = self.env.valid_actions[self.env.players[agent_id]]
        for i in range(len(t)):
            res[t[i].value] = 1
        return res

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self, arg_dict=None):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.agent_score = np.zeros(self.n_agents)
        self.obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """ use saved pictures to generate video
        """
        self.img2video(self.render_path, self.render_path+"/video0.avi")

    def get_env_info(self):
        env_info = {"field_size": self.field_size,
                    "max_food": self.max_food,
                    "state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "agent_score": self.agent_score,
        }
        return stats

    # ======== extra methods ========
    def get_players_position(self):
        return [player.position for player in self.env.players]

    def get_dynamic_env_info(self):
        """Returns the information that can help rule-based agents to do decisions.
        """

        return {"state": self.get_state()}

    def img2video(self, img_dir, video_path):
        """
        Transform imgs into video
        """
        file_list = os.listdir(img_dir)
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))

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
