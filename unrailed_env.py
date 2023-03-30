import datetime
import functools
import math

import Player
from game import UnrailedGame

import gymnasium
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pandas as pd


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAND = 4
PLACE_RAIL = 5
REMOVE_RAIL = 6

MOVES = ["LEFT", "RIGHT", "UP", "DOWN", "STAND", "PLACE_RAIL", "REMOVE_RAIL"]
NUM_ITERS = 600

def env(render_mode=None, episode_every=1, episode_after=0, max_cycles=1e10, observation_radius=2):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = UnrailedEnv(render_mode=internal_render_mode, episode_every=episode_every, episode_after=episode_after,
                      max_cycles=max_cycles, observation_radius=observation_radius)
    # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
    #     env = wrappers.CaptureStdoutWrapper(env)
    # # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # # Provides a wide vareity of helpful user errors
    # # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


class UnrailedEnv(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "UNRAILED!"}

    def __init__(self, low=np.inf, high=np.inf, render_mode=None, episode_every=1, episode_after=0,
                 max_cycles=900, observation_radius=2):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.GAME = UnrailedGame()

        self.record_path = ""
        self.low = low
        self.high = high
        self.max_cycles = max_cycles
        self.obs_radius = observation_radius

        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agents = self.possible_agents[:]
        self.observations = {agent: 0 for agent in self.agents}  # empty obs
        # self.possible_agents = [self.GAME.p0, self.GAME.p1]     # -> agents = possible agents
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # define what actions and how many of them are possible
        self._action_spaces = {agent: Discrete(7) for agent in self.possible_agents}
        # define what we want to observe
        self._observation_spaces = {
            agent: Box(self.low, self.high, shape=(66, )) for agent in self.possible_agents
        }

        self.render_mode = render_mode
        self.episode_every = episode_every
        self.episode_after = episode_after

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(self.low, self.high, shape=(66, ))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(7)

    # print game there
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        if self.render_mode == "human" and self.GAME.episode % self.episode_every == 0 and \
                self.GAME.episode >= self.episode_after:
            self.GAME.render_game([self._cumulative_rewards[agent] for agent in self.agents])
        # save image render sequence
        if self.render_mode == "human_record" and self.GAME.episode % self.episode_every == 0 and \
                (self.GAME.episode >= self.episode_after or self.GAME.episode == 0):
            self.GAME.render_game([self._cumulative_rewards[agent] for agent in self.agents])
            pg.image.save(self.GAME.screen, self.record_path + "\\" + '{:04d}'.format(self.num_frames) + ".jpeg")

    def calculate_observation(self, obs_radius, player: Player, ally_map):
        # region array vars
        x = player.x
        y = player.y
        add_left = 0
        add_right = 0
        add_up = 0
        add_down = 0
        # endregion
        if x - obs_radius >= 0:
            x_low = x - obs_radius
        else:
            x_low = 0
            add_left = abs(x - obs_radius)
        if x + obs_radius < self.GAME.field_x:
            x_high = x + obs_radius + 1
        else:
            x_high = self.GAME.field_x
            add_right = x + obs_radius - x_high + 1
        if y - obs_radius >= 0:
            y_low = y - obs_radius
        else:
            y_low = 0
            add_up = abs(y - obs_radius)
        if y + obs_radius < self.GAME.field_y:
            y_high = y + obs_radius + 1
        else:
            y_high = self.GAME.field_y
            add_down = y + obs_radius - y_high + 1

        walls = self.GAME.MAP_WALLS[y_low:y_high, x_low:x_high]
        # region get other maps
        steel = self.GAME.MAP_STEEL[y_low:y_high, x_low:x_high]
        trees = self.GAME.MAP_TREES[y_low:y_high, x_low:x_high]
        ally = ally_map[y_low:y_high, x_low:x_high]
        train = self.GAME.MAP_TRAIN[y_low:y_high, x_low:x_high]
        station = self.GAME.MAP_STATION[y_low:y_high, x_low:x_high]
        rails = self.GAME.MAP_RAILS[y_low:y_high, x_low:x_high]  # separately
        # endregion

        communication = [0, 0, 0]
        if ally.any():
            communication = player.ally.comm

        map_arr = [walls, steel, trees, ally, train, station]
        combined_map = np.zeros((len(walls), len(walls[0])))
        for i in range(len(map_arr)):
            combined_map += map_arr[i] * (i + 1)

        if add_left != 0:
            combined_map = np.insert(combined_map, 0, [[1] for i in range(add_left)], axis=1)
            rails = np.insert(rails, 0, [[0] for i in range(add_left)], axis=1)
        if add_right != 0:
            combined_map = np.insert(combined_map, len(combined_map[0]), [[1] for i in range(add_right)], axis=1)
            rails = np.insert(rails, len(rails[0]), [[0] for i in range(add_right)], axis=1)
        if add_up != 0:
            combined_map = np.insert(combined_map, 0, [[1] for i in range(add_up)], axis=0)
            rails = np.insert(rails, 0, [[0] for i in range(add_up)], axis=0)
        if add_down != 0:
            combined_map = np.insert(combined_map, len(combined_map), [[1] for i in range(add_down)], axis=0)
            rails = np.insert(rails, len(rails), [[0] for i in range(add_down)], axis=0)
        # region calculate last path rail coord
        x1 = self.GAME.rail_path[-1].rect.x // self.GAME.square_size if len(self.GAME.rail_path) > 0 \
            else self.GAME.used_rail_list[-1].rect.x // self.GAME.square_size
        y1 = self.GAME.rail_path[-1].rect.y // self.GAME.square_size if len(self.GAME.rail_path) > 0 \
            else self.GAME.used_rail_list[-1].rect.y // self.GAME.square_size
        # endregion
        obs = combined_map.flatten()
        obs = np.append(obs, rails.flatten())
        obs = np.append(obs, [self.GAME.collected_rails, self.GAME.collected_steel, self.GAME.collected_trees])
        obs = np.append(obs, [x, y])  # player coord
        obs = np.append(obs, [self.GAME.train.x, self.GAME.train.y])  # train coord
        obs = np.append(obs, [self.GAME.station_x, self.GAME.station_y])  # station coord
        obs = np.append(obs, [x1, y1])  # last path rail coord
        obs = np.append(obs, player.pl_dir['dir'])    # player direction
        obs = np.append(obs, communication)  # communication
        return obs

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        obs_radius = self.obs_radius
        obs = []
        # combined map contains walls, steel, trees, ally, train and station placement within observation radius
        # each object type is indicated by unique number:
        # 0 - nothing   4 - ally
        # 1 - walls     5 - train
        # 2 - steel     6 - station
        # 3 - trees
        # it is possible because by the game rules these objects cannot (or at least should not :) ) overlap
        if agent == "player_0":
            return self.calculate_observation(obs_radius, self.GAME.p0, self.GAME.MAP_PLAYER1)
        elif agent == "player_1":
            return self.calculate_observation(obs_radius, self.GAME.p1, self.GAME.MAP_PLAYER0)
        return obs

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """

        self.GAME.dsp.quit()
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.GAME.dsp.quit()
        self.GAME = UnrailedGame()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminate = False
        self.truncate = False
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}  # empty obs
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.num_frames = 0

        return self.observations, self.infos

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        agent = self.agent_selection
        for ag in self.agents:
            self.rewards[ag] = 0

        self.terminate, self.game_won, self.rewards[agent] = self.GAME.step()
        if self.GAME.game_over:
            for ag in self.agents:
                self.rewards[ag] = self.rewards[agent]
        else:
            if agent == "player_0":
                if not self.GAME.rail_path_completed:
                    self.rewards[agent] += self.GAME.env_action_update(action, self.GAME.p0, self.GAME.MAP_PLAYER0)
                self.GAME.tick_count += 1
            elif agent == "player_1" and not self.GAME.rail_path_completed:
                self.rewards[agent] += self.GAME.env_action_update(action, self.GAME.p1, self.GAME.MAP_PLAYER1)

        if not self.terminate:
            self.num_frames += 1
            self.truncate = self.num_frames >= self.max_cycles

        for ag in self.agents:
            self.terminations[ag] = self.terminate
            self.truncations[ag] = self.truncate
            self.infos[ag] = {}

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        self.render()

