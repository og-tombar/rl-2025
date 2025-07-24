"""
This file contains the DQN agent.
"""

import random
from collections.abc import Sequence

import torch
from torch import nn, optim

from models.base_agent import Agent, ConvNet, Transition


class DQNAgent(Agent):
    """
    A DQN agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = ConvNet(self.in_shape, self.n_actions).to(self.device)
        self.q_tgt = ConvNet(self.in_shape, self.n_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), self.lr)

    # epsilon greedy
    @torch.no_grad()
    def act(self, obs: torch.Tensor, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(self.q(obs.unsqueeze(0).to(self.device)).argmax().cpu())

    def update(self, batch: Sequence[Transition]) -> None:
        s, a, r, s1, done = map(torch.stack, zip(*batch))

        s = s.to(self.device).float()
        s1 = s1.to(self.device).float()
        a = a.to(self.device).long().unsqueeze(1)
        r = r.to(self.device).float()
        done = done.to(self.device).float()

        q_sa = self.q(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            tgt_max = self.q_tgt(s1).max(1).values
            q_target = r + self.gamma * tgt_max * (1 - done)

        loss = nn.functional.smooth_l1_loss(q_sa, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # update target network
        for tgt, src in zip(self.q_tgt.parameters(), self.q.parameters()):
            tgt.data.mul_(1 - self.tau).add_(self.tau * src.data)
