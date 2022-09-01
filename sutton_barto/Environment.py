
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Environment(ABC):
    """Abstract base class for k-armed bandit problem environments."""

    def __init__(self, k: int, baseline: float = 0.0):
        self.num_actions = k
        self.reward_means = self._generate_reward_means(baseline)
        self._opt_action = int(np.max(self._reward_means))