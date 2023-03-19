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
    df = pd.DataFrame(data)
    df["reward"].plot()

    z = np.polyfit(df["episode"], df["reward"], 1)
    p1 = np.poly1d(z)

    plt.plot(df["episode"], p1(df["episode"]))
    plt.show()


plt.rcParams["figure.figsize"] = (10, 5)

train_every = 10
avg_freq = 100
total_num_episodes = int(1e4)  # Total number of episodes

# Create and wrap the environment
# env = gym.make("InvertedPendulum-v4")
env = unrailed_env.env("human", episode_every=1, episode_after=9900)
env.reset()
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

# Observation-space of env
# obs_space_dims = [env.observation_space(agent).shape[0] for agent in env.agents]    # box?
obs_space_dims = env.observation_space(env.agents[0]).shape[0]     # box? can write only one because space is identical?
# Action-space of env
# action_space_dims = [env.action_space(agent).n for agent in env.agents]     # discrete
action_space_dims = env.action_space(env.agents[0]).n     # discrete can write only one because space is identical?
communication_dims = 3
rewards_over_seeds = []
reward_last_n_episodes = queue.Queue(avg_freq)

header = ['episode', 'reward', 'rail length']

for seed in [5]:  # Fibonacci seeds [1, 2, 3, 5, 8]
    data = []
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agents = {i: REINFORCE(obs_space_dims, action_space_dims, i, communication_dims) for i in env.agents}
    reward_over_episodes = []
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        if episode == 9900:
            print(episode)
        env.reset(seed=seed)
        env.GAME.episode = episode
        obs = env.observe(env.agents[0])
        for agent in env.agent_iter():
            action, comm = agents[agent].sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            env.step(action)
            env.GAME.agents[agent].comm = comm[0].tolist()
            obs, reward, terminated, truncated, info = env.last()
            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            if terminated or truncated:
                for i in env.agents:
                    if i == agent:
                        agents[i].rewards.append(env.rewards[i])
                    else:
                        agents[i].rewards[-1] += env.rewards[i]
                    agents[i].episode_rewards.append(agents[i].rewards)
                    agents[i].episode_probs.append(agents[i].probs)
                    agents[i].rewards = []
                    agents[i].probs = []
                break
            agents[agent].rewards.append(env.rewards[agent])
        # print(sum(agents["player_0"].episode_rewards[-1]))
        # print(sum(agents["player_1"].episode_rewards[-1]))
        reward_over_episodes.append(sum([sum(agents[agent].episode_rewards[-1]) for agent in env.agents])/len(agents))  # last episode reward (one number)
        if reward_last_n_episodes.full():
            reward_last_n_episodes.get()
        reward_last_n_episodes.put(sum([sum(agents[agent].episode_rewards[-1]) for agent in env.agents])/len(agents))

        print("Episode:", episode, "Reward:", reward_last_n_episodes.queue[-1],
              "rail length:", len(env.GAME.used_rail_list), "ticks:", env.GAME.tick_count)
        data.append([episode, reward_last_n_episodes.queue[-1], len(env.GAME.used_rail_list)])
        if episode % avg_freq == 0 and episode != 0:
            avg_reward = np.mean(reward_last_n_episodes.queue)
            print("///////// Average Reward:", avg_reward, " /////////")

        if episode % train_every == 0 and episode != 0:
            print("training...")
            for agent in agents:
                agents[agent].update()
    filename = "Training rewards/" + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S_seed") + str(seed) + ".csv"

    f = open(filename, 'w+', encoding='UTF8', newline='')
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    writer.writerows(data)
    f.close()
    rewards_over_seeds.append(reward_over_episodes)
    show_plot(pd.read_csv(filename))

rewards_to_plot = [[reward for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()


