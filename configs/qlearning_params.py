from __future__ import annotations

# Q-learning configuration for RL-guided Segment-Preserving Crossover (SPC).

QLEARNING_CONFIG = {
    "alpha": 0.10,           # Learning rate
    "gamma": 0.00,           # Discount factor (0.0 = contextual bandit / one-step)
    "epsilon": 1.00,         # Initial exploration rate
    "min_eps": 0.05,         # Minimum exploration rate
    "eps_decay": 0.995,      # Epsilon decay rate (applied per update step)
    "episodes": 1,           # Number of training episodes (interpreted as GA/HGA runs)
    "max_steps": 200_000,    # Maximum Q-updates with exploration before greedy-only
    "greedy_steps": 0,       # Optional greedy-only evaluation steps after max_steps
}


def get_qlearning_config() -> dict:
    """Return a shallow copy of QLEARNING_CONFIG."""
    return dict(QLEARNING_CONFIG)
