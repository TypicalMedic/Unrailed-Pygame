from __future__ import annotations

import queue
import random

import matplotlib.pyplot as plt
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import unrailed_env


plt.rcParams["figure.figsize"] = (10, 5)


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, communication_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        #  //////////////////////////////BUZ/////////////////////
        self.fc1 = nn.Linear(obs_space_dims, hidden_space1)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_space1, hidden_space2)
        self.fc3 = nn.Linear(hidden_space2, action_space_dims)

        self.probs = nn.Softmax(dim=-1)
        #  //////////////////////////////BUZ/////////////////////

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, communication_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, communication_dims)
        )

    def forward(self, x: torch.Tensor):
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x[-3:].float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        res = self.fc1(x.float())
        res = self.relu(res)
        res = self.fc2(res)
        res = self.relu(res)
        res = self.fc3(res)
        res = self.probs(res)

        return res, action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, name: str, comm_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        self.name = name
        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims, comm_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> [float, torch.Tensor]:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        probs, action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action

        # action = distrib.sample()
        # prob = distrib.log_prob(action)
        #
        # action = action.numpy()
        dist = Categorical(probs)
        action = dist.sample().item()
        prob = probs[0, action]
        self.probs.append(prob)

        return action, action_means

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


# Create and wrap the environment
# env = gym.make("InvertedPendulum-v4")
env = unrailed_env.env("human")
env.reset()
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of env
# obs_space_dims = [env.observation_space(agent).shape[0] for agent in env.agents]    # box?
obs_space_dims = env.observation_space(env.agents[0]).shape[0]     # box? can write only one because space is identical?
# Action-space of env
# action_space_dims = [env.action_space(agent).n for agent in env.agents]     # discrete
action_space_dims = env.action_space(env.agents[0]).n     # discrete can write only one because space is identical?
communication_dims = 3
rewards_over_seeds = []
reward_last_n_episodes = queue.Queue(10)

for seed in [2, 3, 5, 8]:  # Fibonacci seeds [1, 2, 3, 5, 8]
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agents = {i: REINFORCE(obs_space_dims, action_space_dims, i, communication_dims) for i in env.agents}
    # agents = [REINFORCE(obs_space_dims, action_space_dims, i, communication_dims) for i in env.agents]
    # agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        if episode == 1:
            print(episode)
        env.reset(seed=seed)
        done = False
        i = 0
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            action, comm = agents[agent].sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            env.step(action)
            env.GAME.agents[agent].comm = comm[0].tolist()
            agents[agent].rewards.append(env.rewards[agent])

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            if terminated or truncated:
                break
        reward_over_episodes.append(sum([sum(agents[agent].rewards) for agent in env.agents])/len(agents))  # last episode reward (one number)
        if reward_last_n_episodes.full():
            reward_last_n_episodes.get()
        reward_last_n_episodes.put(sum([sum(agents[agent].rewards) for agent in env.agents])/len(agents))
        for agent in agents:
            agents[agent].update()

        if episode % 10 == 0:
            avg_reward = np.mean(reward_last_n_episodes.queue)
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()
