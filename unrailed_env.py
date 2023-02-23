import functools

from game import UnrailedGame

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAND = 4
INTERACT = 5

MOVES = ["LEFT", "RIGHT", "UP", "DOWN", "STAND", "INTERACT"]
NUM_ITERS = 600
# REWARD_MAP = {
#     (ROCK, ROCK): (0, 0),
#     (ROCK, PAPER): (-1, 1),
#     (ROCK, SCISSORS): (1, -1),
#     (PAPER, ROCK): (1, -1),
#     (PAPER, PAPER): (0, 0),
#     (PAPER, SCISSORS): (-1, 1),
#     (SCISSORS, ROCK): (-1, 1),
#     (SCISSORS, PAPER): (1, -1),
#     (SCISSORS, SCISSORS): (0, 0),
# }


# <editor-fold desc="variables for the environment?">
screen_width = 600
screen_height = 500
tick_amount = 30
collected_trees = 0
collected_steel = 0
collected_rails = 0
trees_needed = 1
steel_needed = 1
info_panel_height = 100
bgclr = (255, 255, 255)
square_size = 20
square_color = 'grey'
game_over = False
wall_amount = 50
trees_amount = 50
steel_amount = 50


# </editor-fold>


def env(render_mode=None, max_cycles=900, observation_radius=2):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode, max_cycles=max_cycles, observation_radius=observation_radius)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "UNRAILED!"}

    def __init__(self, low=np.inf, high=np.inf, render_mode=None, max_cycles=900, observation_radius=2):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.GAME = UnrailedGame()

        self.low = low
        self.high = high
        self.max_cycles = max_cycles
        self.obs_radius = observation_radius

        self.possible_agents = ["player_" + str(r) for r in range(2)]
        # self.possible_agents = [self.GAME.p0, self.GAME.p1]     # -> agents = possible agents
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # define what actions and how many of them are possible
        self._action_spaces = {agent: Discrete(6) for agent in self.possible_agents}
        # define what we want to observe
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        self._observation_spaces = {
            # agent: MultiDiscrete([[len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)],
            #                            [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)],
            #                            [len(self.GAME.MAP_WALLS)], [1], [1], [1]]) for agent in self.possible_agents
            agent: Box(self.low, self.high, shape=(3,)) for agent in self.possible_agents
        }

        self.render_mode = render_mode

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return MultiDiscrete([[len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)],
        #                                [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)], [len(self.GAME.MAP_WALLS)],
        #                                [len(self.GAME.MAP_WALLS)],  [1], [1], [1]])
        return Box(self.low, self.high, shape=(3,))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6)

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

        self.GAME.render_game()
        # if len(self.agents) == 2:
        #     string = "Current state: Agent1: {} , Agent2: {}".format(
        #         MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
        #     )
        # else:
        #     string = "Game over"
        # print(string)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        obs_radius = self.obs_radius
        obs = []
        if agent == "player_0":
            x = self.GAME.p0.x
            y = self.GAME.p0.y
            x_low = x-obs_radius
            x_high = x+obs_radius+1
            y_low = y-obs_radius
            y_high = y+obs_radius+1
            walls = self.GAME.MAP_WALLS[y_low:y_high, x_low:x_high]
            steel = self.GAME.MAP_STEEL[y_low:y_high, x_low:x_high]
            trees = self.GAME.MAP_TREES[y_low:y_high, x_low:x_high]
            ally = self.GAME.MAP_PLAYER1[y_low:y_high, x_low:x_high]
            train = self.GAME.MAP_TRAIN     # [y_low:y_high, x_low:x_high]
            station = self.GAME.MAP_STATION     # [y_low:y_high, x_low:x_high]
            rails = self.GAME.MAP_RAILS[y_low:y_high, x_low:x_high]

            # observation of one agent is the surrounding box with info about ... ?
            # obs = [self.GAME.MAP_WALLS, self.GAME.MAP_STEEL, self.GAME.MAP_TREES,
            #        self.GAME.MAP_PLAYER1, self.GAME.MAP_TRAIN,
            #        self.GAME.MAP_STATION, self.GAME.MAP_RAILS,
            #        self.GAME.collected_rails, self.GAME.collected_steel, self.GAME.collected_trees]
            obs = np.array([{"walls": walls, "steel": steel, "trees": trees,
                             "ally": ally, "train": train,
                             "station": station, "rails": rails},
                            self.GAME.collected_rails, self.GAME.collected_steel, self.GAME.collected_trees, x, y])
        elif agent == "player_1":
            x = self.GAME.p1.x
            y = self.GAME.p1.y
            x_low = x-obs_radius
            x_high = x+obs_radius+1
            y_low = y-obs_radius
            y_high = y+obs_radius+1
            walls = self.GAME.MAP_WALLS[y_low:y_high, x_low:x_high]
            steel = self.GAME.MAP_STEEL[y_low:y_high, x_low:x_high]
            trees = self.GAME.MAP_TREES[y_low:y_high, x_low:x_high]
            ally = self.GAME.MAP_PLAYER0[y_low:y_high, x_low:x_high]
            train = self.GAME.MAP_TRAIN[y_low:y_high, x_low:x_high]
            station = self.GAME.MAP_STATION[y_low:y_high, x_low:x_high]
            rails = self.GAME.MAP_RAILS[y_low:y_high, x_low:x_high]

            # obs = [self.GAME.MAP_WALLS, self.GAME.MAP_STEEL, self.GAME.MAP_TREES,
            #        self.GAME.MAP_PLAYER0, self.GAME.MAP_TRAIN,
            #        self.GAME.MAP_STATION, self.GAME.MAP_RAILS,
            #        self.GAME.collected_rails, self.GAME.collected_steel, self.GAME.collected_trees]
            obs = np.array([{"walls": walls, "steel": steel, "trees": trees,
                             "ally": ally, "train": train,
                             "station": station, "rails": rails},
                            self.GAME.collected_rails, self.GAME.collected_steel, self.GAME.collected_trees, x, y])
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
        # self.state = {agent: NONE for agent in self.agents} # ???
        self.observations = {agent: 0 for agent in self.agents}  # empty obs
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.num_frames = 0

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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if agent == self.agents[0]:
        #     self.rewards = {a: 0 for a in self.agents}
        #     self.p0.update(self.area, action) # update player by action MAKE!!!
        # elif agent == self.agents[1]:
        #     self.p1.update(self.area, action) # update player by action MAKE!!!

        reward = 0
        self.terminate, reward = self.GAME.step()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        if agent == "player_0" and not self.GAME.rail_path_completed:
            reward += self.GAME.env_action_update(action, self.GAME.p0, self.GAME.MAP_PLAYER0)
            self.rewards[agent] += reward
        elif agent == "player_1" and not self.GAME.rail_path_completed:
            reward += self.GAME.env_action_update(action, self.GAME.p1, self.GAME.MAP_PLAYER1)
            self.rewards[agent] += reward
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        if self.terminate:
            reward = 0
            # self.close()  # ??????????????????????????????????????????????????????????????????????????????????
        if not self.terminate:
            self.num_frames += 1
            reward = 0
            self.truncate = self.num_frames >= self.max_cycles

        for ag in self.agents:
            # self.rewards[ag] = 0
            self.terminations[ag] = self.terminate
            self.truncations[ag] = self.truncate
            self.infos[ag] = {}
