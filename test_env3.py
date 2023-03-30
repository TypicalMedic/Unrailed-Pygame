from __future__ import annotations

import datetime
import queue
import random
import warnings

import numpy as np
import torch
import unrailed_env
from RL import REINFORCE
import csv
import json
import os
import о
from make_video import make_video


class TrainAgents:
    def __init__(self, train_every: int = 10, avg_freq: int = 100, total_num_episodes: int = int(1e4),
                 render_episode_every: int = 1, render_episode_after: int = 9900, render_mode="human",
                 learning_rate=1e-4, discount=0.33):
        """
        Class for training agents using RL algorythm and Unrailed environment

        :param train_every: after how many episodes' agents will update their policy weights
        :param avg_freq: after how many episodes' average reward is calculated
        :param total_num_episodes: how many episodes' agents will train
        :param render_episode_every: after how many episodes' game renders, 1 for every episode
        :param render_episode_after: after which episode game will try to render (using render_episode_every)
        """
        if type(render_episode_after) is not int:
            warnings.warn("Render threshold (render_episode_after) is not integer,"
                          " setting render_episode_after to 0...")
            render_episode_after = 0
        if type(total_num_episodes) is not int:
            warnings.warn("Episode amount (total_num_episodes) is not integer,"
                          " setting total_num_episodes to 100...")
            total_num_episodes = 100
        if total_num_episodes <= 0:
            warnings.warn("Episode amount (total_num_episodes) is less than 1,"
                          " setting total_num_episodes to 100...")
            total_num_episodes = 100
        if type(render_episode_every) is not int:
            warnings.warn("Render frequency (render_episode_every) is not integer,"
                          " setting render_episode_every to 1...")
            render_episode_every = 1
        elif render_episode_every <= 0:
            warnings.warn("Render frequency (render_episode_every) is less than 1,"
                          " setting render_episode_every to 1...")
            render_episode_every = 1
        if type(avg_freq) is not int:
            warnings.warn("Average calculation frequency (avg_freq) is not integer,"
                          " setting avg_freq to 1...")
            avg_freq = 1
        elif avg_freq <= 0:
            warnings.warn("Average calculation frequency (avg_freq) is less than 1,"
                          " setting avg_freq to 1...")
            avg_freq = 1
        if type(train_every) is not int:
            warnings.warn("Training frequency (train_every) is not integer, "
                          "setting train_every to 1...")
            train_every = 1
        elif train_every <= 0:
            warnings.warn("Training frequency (train_every) is less than 1, "
                          "setting train_every to 1...")
            train_every = 1

        self.learning_rate = learning_rate
        self.discount = discount
        self.train_every = train_every
        self.avg_freq = avg_freq
        self.total_num_episodes = total_num_episodes  # Total number of episodes

        # Create the environment
        self.env = unrailed_env.env(render_mode, episode_every=render_episode_every, episode_after=render_episode_after)
        parent_dir = os.path.abspath("Recordings")
        self.record_path = parent_dir + "\\" + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
        if self.env.render_mode == "human_record":
            os.mkdir(self.record_path)

        self.env.reset()
        self.render_episode_after = render_episode_after
        # Observation-space of env
        self.obs_space_dims = self.env.observation_space(self.env.agents[0]).shape[0]
        # Action-space of env
        self.action_space_dims = self.env.action_space(self.env.agents[0]).n
        self.communication_dims = 3
        self.export_data_header = ['episode', 'reward', 'rail length', 'game win']

    def train_agents(self, seeds: []):
        filename = ""
        for seed in seeds:  # Fibonacci seeds [1, 2, 3, 5, 8]
            data = []
            # set seed
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # Reinitialize agent every seed
            agents = {i: REINFORCE(self.obs_space_dims, self.action_space_dims, i, self.communication_dims,
                                   self.learning_rate, self.discount)
                      for i in self.env.agents}
            reward_last_n_episodes = queue.Queue(self.avg_freq)
            for episode in range(self.total_num_episodes):
                if self.env.render_mode == "human_record" and (episode >= self.render_episode_after or episode == 0):
                    self.env.record_path = self.record_path + "\\" + str(episode)
                    os.mkdir(self.env.record_path)
                # gymnasium v26 requires users to set seed while resetting the environment
                if episode == self.render_episode_after:
                    input('\033[92m' + "Episodes will render now. Press Enter to continue..." + '\033[0m')
                self.env.reset(seed=seed)
                self.env.GAME.episode = episode
                obs = self.env.observe(self.env.agents[0])
                for agent in self.env.agent_iter():
                    action, comm = agents[agent].sample_action(obs)

                    # if the episode is terminated, if the episode is truncated and
                    # additional info from the step
                    self.env.step(action)
                    self.env.GAME.agents[agent].comm = comm[0].tolist()
                    obs, reward, terminated, truncated, info = self.env.last()

                    # End the episode when either truncated or terminated is true
                    #  - truncated: The episode duration reaches max number of timesteps
                    #  - terminated: Any of the state space values is no longer finite.
                    if terminated or truncated:
                        # when the game is over both agents get game over reward
                        for i in self.env.agents:
                            if i == agent:
                                agents[i].rewards.append(self.env.rewards[i])
                            else:
                                agents[i].rewards[-1] += self.env.rewards[i]
                            # append rewards and probabilities for this episode and clear them before next episode
                            agents[i].episode_rewards.append(agents[i].rewards)
                            agents[i].episode_probs.append(agents[i].probs)
                            agents[i].rewards = []
                            agents[i].probs = []
                        break
                    # append reward after checking that the game is not over
                    agents[agent].rewards.append(self.env.rewards[agent])
                if reward_last_n_episodes.full():
                    reward_last_n_episodes.get()
                # append for avg calculation
                reward_last_n_episodes.put(
                    sum([sum(agents[agent].episode_rewards[-1]) for agent in self.env.agents]) / len(agents))

                print("Episode:", episode, "Reward:", reward_last_n_episodes.queue[-1],
                      "rail length:", len(self.env.GAME.used_rail_list), "ticks:", self.env.GAME.tick_count)
                # append data for csv exporting
                data.append([episode, reward_last_n_episodes.queue[-1],
                             len(self.env.GAME.used_rail_list), self.env.game_won])

                # calculate avg every n episodes
                if episode % self.avg_freq == 0 and episode != 0:
                    avg_reward = np.mean(reward_last_n_episodes.queue)
                    print("///////// Average Reward:", avg_reward, " /////////")

                # train both agents every n episodes using last n episodes data
                if episode % self.train_every == 0 and episode != 0:
                    print('\033[94m' + "training..." + '\033[0m')
                    for agent in agents:
                        agents[agent].update()
                if self.env.render_mode == "human_record" and (episode >= self.render_episode_after or episode == 0):
                    make_video(self.env.record_path)
            # save rewards and e.t.c. to csv
            filename = "Training rewards/" + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S_seed") + str(
                seed) + ".csv"

            f = open(filename, 'w+', encoding='UTF8', newline='')
            writer = csv.writer(f)
            writer.writerow(self.export_data_header)
            writer.writerows(data)
            f.close()
            input('\033[92m' + "Training is finished. The results are stored in the " + filename +
                  "\nPress Enter to exit the program..." + '\033[0m')
        о.show_plot(filename)


train = TrainAgents(train_every=10, avg_freq=10, total_num_episodes=int(1e2), render_episode_every=1,
                    render_episode_after=90, render_mode="human", learning_rate=1e-4, discount=0.33)
try:
    conf = open("RL_config.json", "r", encoding="UTF8", newline='')
    args = json.load(conf)
    train = TrainAgents(train_every=args["train_every"], avg_freq=args["avg_freq"], render_mode=args["render_mode"],
                        total_num_episodes=args["total_num_episodes"],learning_rate=args["learning_rate"],
                        render_episode_every=args["render_episode_every"], discount=args["discount"],
                        render_episode_after=args["render_episode_after"])
except FileNotFoundError:
    print("RL config file was not found. Launching with the default settings.")
train.train_agents([5])
