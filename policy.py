import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, communication_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 1000  # Nothing special with 16, feel free to change
        hidden_space2 = 100  # Nothing special with 32, feel free to change
        hidden_space3 = 0  # Nothing special with 32, feel free to change -
        self.action_space_dims = action_space_dims
        #  //////////////////////////////BUZ/////////////////////
        self.fc1 = nn.Linear(obs_space_dims, hidden_space1)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_space1, action_space_dims + communication_dims)

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
        res = self.fc1(x.float())
        res = self.relu(res)
        res = self.fc2(res)
        obs = torch.clone(res)[0:1, self.action_space_dims:]
        res = torch.clone(res)[0:1, 0:self.action_space_dims]
        res = self.probs(res)

        return res, obs
