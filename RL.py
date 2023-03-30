from policy import PolicyNetwork
from torch.distributions import Categorical
import numpy as np
import torch


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, name: str, comm_dims: int,
                 learning_rate=1e-4, discount=0.33):
        """Initializes an agent that learns a policy via REINFORCE algorithm
        to solve the task at hand.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            name: Agent name
            comm_dims: Dimension of the communication
        """

        self.name = name
        # Hyper parameters
        self.learning_rate = learning_rate  # Learning rate for policy optimization
        self.gamma = discount  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.episode_rewards = []
        self.episode_probs = []

        self.net = PolicyNetwork(obs_space_dims, action_space_dims, comm_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> [float, torch.Tensor]:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed, communication array
        """
        state = torch.tensor(np.array([state]))
        probs, communication = self.net(state)
        dist = Categorical(probs)
        action = dist.sample().item()
        prob = probs[0, action]
        self.probs.append(prob)

        return action, communication

    def update(self):
        """Updates the policy network's weights."""
        loss = 0
        for i in range(len(self.episode_rewards)):
            running_g = 0
            gs = []

            # Discounted return (backwards) - [::-1] will return an array in reverse
            for R in self.episode_rewards[i][::-1]:
                running_g = R + self.gamma * running_g
                gs.insert(0, running_g)

            deltas = torch.tensor(gs)

            # minimize -1 * prob * reward obtained
            for log_prob, delta in zip(self.episode_probs[i], deltas):
                loss += torch.log(log_prob) * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.episode_probs = []
        self.episode_rewards = []
        print("loss: ", loss)
