"""
This file contains the base class for all RL agents.
"""

import collections
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

Transition = collections.namedtuple("Transition", "s a r s1 done")


class ConvNet(nn.Module):
    """
    A convolutional neural network for RL agents.
    """

    def __init__(self, in_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = in_shape
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            n = self.feature(torch.zeros(1, c, h, w)).view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional neural network.

        Args:
            obs (torch.Tensor): The observation tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if obs.ndim == 3:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        else:
            obs = obs.permute(0, 3, 1, 2)

        obs = obs.float() / 255.0
        x = self.feature(obs)
        x = torch.flatten(x, 1)
        return self.head(x)


class Agent(ABC):
    """
    Abstract base class for RL agents.
    """

    def __init__(
            self,
            device: torch.device,
            in_shape: tuple[int, int, int],
            n_actions: int,
            gamma: float,
            lr: float,
            tau: float):
        self.device = device
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

    @abstractmethod
    @torch.no_grad()
    def act(self, obs: torch.Tensor, eps: float) -> int:
        """Return an action given a single observation (C,H,W) and epsilon.

        Args:
            obs (torch.Tensor): The observation tensor.
            eps (float): The epsilon value.

        Raises:
            NotImplementedError: The agent must implement this method.

        Returns:
            int: The action to take.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Sequence[Transition]) -> None:
        """Update the agent from a mini-batch of transitions.

        Args:
            batch (Sequence[Transition]): The batch of transitions.

        Raises:
            NotImplementedError: The agent must implement this method.
        """
        raise NotImplementedError
