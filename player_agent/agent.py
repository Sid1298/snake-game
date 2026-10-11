"""Deep Q-Network agent that learns to play snake from reward alone."""
import random
from collections import deque

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Double DQN with experience replay and a periodically synced target network."""

    def __init__(
        self,
        n_inputs,
        n_actions,
        gamma=0.95,
        lr=1e-3,
        batch_size=256,
        buffer_size=100_000,
        warmup_steps=1_000,
        target_sync_steps=1_000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=50_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_sync_steps = target_sync_steps
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps

        self.policy = QNetwork(n_inputs, n_actions)
        self.target = QNetwork(n_inputs, n_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.total_steps = 0

    @property
    def epsilon(self):
        progress = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q_values = self.policy(torch.from_numpy(state).unsqueeze(0))
        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def learn(self):
        """One gradient step on a random batch of past experience. Returns the loss, or None."""
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: the policy net picks the next action, the target net scores it
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        if self.total_steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

    def save(self, path, **extra):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                **extra,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        return checkpoint
