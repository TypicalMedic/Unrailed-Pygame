from __future__ import annotations

import datetime
import queue
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import gymnasium as gym
import unrailed_env
from RL import REINFORCE
import csv


def show_plot(data):
    """
    draws pyplot of the rewards from data

    :param data: csv parsed file with episode and reward columns
    :return:
    """
    df = pd.DataFrame(data)
    df["reward"].plot()

    z = np.polyfit(df["episode"], df["reward"], 1)
    p1 = np.poly1d(z)

    plt.plot(df["episode"], p1(df["episode"]))
    plt.show()


class TrainAgents:
    def __init__(self, train_every: int = 10, avg_freq: int = 100, total_num_episodes: int = int(1e4),
                 render_episode_every: int = 1, render_episode_after: int = 9900):
        """
        Class for training agents using RL algorythm and Unrailed environment

        :param train_every: after how many episodes' agents will update their policy weights
        :param avg_freq: after how many episodes' average reward is calculated
        :param total_num_episodes: how many episodes' agents will train
        :param render_episode_every: after how many episodes' game renders, 1 for every episode
        :param render_episode_after: after which episode game will try to render (using render_episode_every)
        """
        if render_episode_every <= 0:
            render_episode_every = 1
        if avg_freq <= 0:
            avg_freq = 1
        if train_every <= 0:
            train_every = 1

        self.train_every = train_every
        self.avg_freq = avg_freq
        self.total_num_episodes = total_num_episodes  # Total number of episodes

        # Create and wrap the environment
        # env = gym.make("InvertedPendulum-v4")
        self.env = unrailed_env.env("human", episode_every=render_episode_every, episode_after=render_episode_after)
        self.env.reset()
        self.render_episode_after = render_episode_after
        # Observation-space of env
        self.obs_space_dims = self.env.observation_space(self.env.agents[0]).shape[0]
        # Action-space of env
        self.action_space_dims = self.env.action_space(self.env.agents[0]).n
        self.communication_dims = 3
        self.reward_last_n_episodes = queue.Queue(avg_freq)

        self.export_data_header = ['episode', 'reward', 'rail length']

    def train_agents(self, seeds: []):
        rewards_over_seeds = []
        for seed in seeds:  # Fibonacci seeds [1, 2, 3, 5, 8]
            data = []
            # set seed
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # Reinitialize agent every seed
            agents = {i: REINFORCE(self.obs_space_dims, self.action_space_dims, i, self.communication_dims)
                      for i in self.env.agents}
            reward_over_episodes = []
            for episode in range(self.total_num_episodes):
                # gymnasium v26 requires users to set seed while resetting the environment
                if episode == self.render_episode_after:
                    input("Episodes will render now. Press Enter to continue...")
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

                reward_over_episodes.append(sum([sum(agents[agent].episode_rewards[-1])
                                                 for agent in self.env.agents]) / len(agents))
                if self.reward_last_n_episodes.full():
                    self.reward_last_n_episodes.get()
                # append for avg calculation
                self.reward_last_n_episodes.put(
                    sum([sum(agents[agent].episode_rewards[-1]) for agent in self.env.agents]) / len(agents))

                print("Episode:", episode, "Reward:", self.reward_last_n_episodes.queue[-1],
                      "rail length:", len(self.env.GAME.used_rail_list), "ticks:", self.env.GAME.tick_count)
                # append data for csv exporting
                data.append([episode, self.reward_last_n_episodes.queue[-1], len(self.env.GAME.used_rail_list)])

                # calculate avg every n episodes
                if episode % self.avg_freq == 0 and episode != 0:
                    avg_reward = np.mean(self.reward_last_n_episodes.queue)
                    print("///////// Average Reward:", avg_reward, " /////////")

                # train both agents every n episodes using last n episodes data
                if episode % self.train_every == 0 and episode != 0:
                    print("training...")
                    for agent in agents:
                        agents[agent].update()
            # save rewards and e.t.c. to csv
            filename = "Training rewards/" + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S_seed") + str(
                seed) + ".csv"

            f = open(filename, 'w+', encoding='UTF8', newline='')
            writer = csv.writer(f)
            writer.writerow(self.export_data_header)
            writer.writerows(data)
            f.close()
            rewards_over_seeds.append(reward_over_episodes)
            show_plot(pd.read_csv(filename))


train = TrainAgents(train_every=10, avg_freq=100, total_num_episodes=int(1e4),
                    render_episode_every=1, render_episode_after=9900)
train.train_agents([5])
