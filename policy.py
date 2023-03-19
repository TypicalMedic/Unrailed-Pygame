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
            communication_dims: Dimension of the communication
        """
        super().__init__()

        hidden_space1 = 1000    # how many spaces will be in the first layer
        self.action_space_dims = action_space_dims
        self.fc1 = nn.Linear(obs_space_dims, hidden_space1)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_space1, action_space_dims + communication_dims)

        self.probs = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            res: res: predicted probabilities of the agent's actions;
             communication: predicted numbers, which will be passed to another agent when possible
        """
        res = self.fc1(x.float())
        res = self.relu(res)
        res = self.fc2(res)
        communication = torch.clone(res)[0:1, self.action_space_dims:]
        res = torch.clone(res)[0:1, 0:self.action_space_dims]
        res = self.probs(res)

        return res, communication
