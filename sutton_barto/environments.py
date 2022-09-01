
from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import Optional


class Environment(ABC):
    """Abstract base class for k-armed bandit problem environments."""

    def __init__(self, k: int, seed: int = None, baseline: float = 0.0):
        """Initialize an Environment.
        
        Args:
            k (int): Number of possible actions to be taken in this environment.
            seed (Optional[int]): Seed for generating reward means.
            baseline (Optional[float]): Base reward to add to every randomly-generated
                action reward mean.
        """
        self.num_actions = k # number of arms
        self.reward_means = self._generate_reward_means(seed, baseline) # means for each arm
        self.opt_action = int(np.argmax(self.reward_means)) # optimal action

    def __call__(self, action: int) -> float:
        """Take an action and output a reward.
        
        Args:
            action (int): Integer index for the selected action taken against the environment.

        Returns:
            float: reward signal.
        """
        return self._map_action_to_reward(action=action)

    @abstractmethod
    def _map_action_to_reward(self, action: int) -> float:
        """The problem-specific mapping of actions to rewards.
        May be deterministic or stochastic. May change over time.
        """
        pass

    def _generate_reward_means(self, seed: int = None, baseline: float = 0.0) -> np.ndarray:
        """Initialization helper function for randomly generating reward means from a unit
        normal distribution, with an optional baseline reward.

        Args:
            seed (Optional[int]): Seed for generating reward means.
            baseline (Optional[float]): Baseline reward to add to all actions' reward means.

        Returns:
            np.ndarray: Array of randomly-generated reward means.
        """
        if seed is not None:
            np.random.seed(seed=seed)
        return scipy.stats.norm(loc=0, scale=1).rvs(self.num_actions) + baseline



class KArmedBandit(Environment):
    """General environment for the k-armed bandit problem as described in Sutton
    & Barto Ch2.
    """
    def __init__(self, k: int, seed: int = None, baseline: float = 0.0):
        super().__init__(k=k, seed=seed, baseline=baseline)
        self.reward_stds: np.ndarray = np.ones(k) # reward standard deviations

    def _map_action_to_reward(self, action: int) -> float:
        """Rewards are drawn randomly from normal distributions whose parameters are specified
        (but unknown to the agent).
        
        Args:
            action (int): Integer index for the selected action taken against the KArmedBandit.
        """
        return self.reward_means[action] + self.reward_stds[action] * np.random.randn()

    def plot_reward_dists(self, ax: matplotlib.axes.Axes) -> None:
        """Plot the actual distributions of rewards for each 'lever' of this KArmedBandit.
        
        Args:
            ax (matplotlib.axes.Axes): Axes object upon which to draw distributions.
        """
        for idx in range(self.num_actions):
            rwd_mean, rwd_std = self.reward_means[idx], self.reward_stds[idx]
            y_p = scipy.stats.norm(loc=rwd_mean, scale=rwd_std).rvs(5_000)
            parts = ax.violinplot(
                y_p,
                positions=[idx+1],
                showmeans=False,
                showextrema=False,
                points=10_000,
            )
            for pc in parts['bodies']:
                pc.set_facecolor('#000000')
                pc.set_alpha(0.5)
            _width = 0.30
            ax.hlines(xmin=idx+1-_width, xmax=idx+1+_width, y=rwd_mean, color='k', lw=0.5)

        ax.plot([0,self.num_actions+1], [0, 0], color='k', linestyle='--', lw=1, alpha=0.5)
        ax.set(
            xlim=[0, self.num_actions+1],
            xticks=range(1, self.num_actions+1),
            xticklabels=[str(k) for k in range(1, self.num_actions + 1)],
            ylim=[-3.5,3.5],
            yticks=range(-3,4),
            yticklabels=[str(k) for k in range(-3, 4)],
            )
        ax.set_xlabel(xlabel='Action')
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.set_ylabel(ylabel='Reward\ndistribution', rotation=0) #, labelpad=35)
        ax.yaxis.set_label_coords(-0.15, 0.45)



if __name__ == "__main__":
    pass