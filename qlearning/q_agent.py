from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable
import random


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float
    gamma: float
    epsilon: float
    min_eps: float
    eps_decay: float
    episodes: int
    max_steps: int
    greedy_steps: int

    @staticmethod
    def from_dict(d: dict) -> "QLearningConfig":
        return QLearningConfig(
            alpha=float(d["alpha"]),
            gamma=float(d["gamma"]),
            epsilon=float(d["epsilon"]),
            min_eps=float(d["min_eps"]),
            eps_decay=float(d["eps_decay"]),
            episodes=int(d["episodes"]),
            max_steps=int(d["max_steps"]),
            greedy_steps=int(d["greedy_steps"]),
        )


class QLearningAgent:
    """Tabular Q-learning agent (epsilon-greedy).

    Designed for RL-guided SPC: each decision (state -> choose system k) is treated
    as a one-step transition by default (done=True), i.e. contextual bandit.
    """

    def __init__(self, num_actions: int, config: QLearningConfig, seed: int | None = None):
        assert num_actions >= 1
        self.num_actions = num_actions
        self.cfg = config
        self._rng = random.Random(seed)

        # Q-table: state -> list[Q(a)] length = num_actions
        self.q: dict[Hashable, list[float]] = {}

        self.epsilon = float(config.epsilon)
        self.total_steps = 0
        self.episode = 0

    def start_episode(self) -> None:
        self.episode += 1

    def _ensure_state(self, state: Hashable) -> list[float]:
        if state not in self.q:
            self.q[state] = [0.0] * self.num_actions
        return self.q[state]

    def select_action(self, state: Hashable, valid_actions: Iterable[int] | None = None) -> int:
        """Return action in [1..num_actions]."""
        q_row = self._ensure_state(state)

        if valid_actions is None:
            valid = list(range(1, self.num_actions + 1))
        else:
            valid = list(valid_actions)
            if not valid:
                valid = list(range(1, self.num_actions + 1))

        # After max_steps, switch to greedy policy.
        exploring_allowed = self.total_steps < self.cfg.max_steps

        if exploring_allowed and self._rng.random() < self.epsilon:
            return self._rng.choice(valid)

        # Greedy
        best_a = valid[0]
        best_q = q_row[best_a - 1]
        for a in valid[1:]:
            v = q_row[a - 1]
            if v > best_q:
                best_q = v
                best_a = a
        return best_a

    def update(self, state: Hashable, action: int, reward: float, next_state: Hashable | None = None, done: bool = True) -> None:
        """Standard Q-learning update."""
        assert 1 <= action <= self.num_actions

        q_row = self._ensure_state(state)
        q_sa = q_row[action - 1]

        if done or next_state is None:
            target = reward
        else:
            next_row = self._ensure_state(next_state)
            target = reward + self.cfg.gamma * max(next_row)

        q_row[action - 1] = (1.0 - self.cfg.alpha) * q_sa + self.cfg.alpha * target

        # Step bookkeeping + epsilon decay
        self.total_steps += 1
        if self.total_steps < self.cfg.max_steps:
            self.epsilon = max(self.cfg.min_eps, self.epsilon * self.cfg.eps_decay)
        else:
            # Greedy-only phase
            self.epsilon = 0.0
