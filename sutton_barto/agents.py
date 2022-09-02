
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from environments import Environment


class Agent(ABC):
    """Base class for solving the k-armed bandit problem via
    action-value estimation.
    """

    def __init__(self, k: int, q_init: float = 0.0):
        """Instantiate an agent.
        
        Args:
            k (int): Number of possible actions to take.
            q_init (int): Initial baseline action-value estimate. Used to
                set optimistic initial values.
        """
        self.num_actions = k
        self.q = q_init * np.ones(self.num_actions) # hold current action-value estimates

    
    @abstractmethod
    def select_action(self) -> int:
        """Use current action-value estimates to select an action.

        Returns:
            int: Integer index of the action selected to be taken.
        """
        pass

    @abstractmethod
    def update_action_values(self, reward_obs: float) -> None:
        """Update current action-value estimates to account
        for a new observation.
        
        Args:
            reward_obs (float): Newly-observed reward.
        """
        pass


class EpsilonGreedyAgent(Agent):

    def __init__(self, k: int, epsilon: float = 1.0, q_init: float = 0, update_step_size: float = None):
        super().__init__(k=k, q_init=q_init)
        self.epsilon = epsilon # greediness parameter
        self.total_reward_accum: float = 0.0
        self.num_selections_accum: np.ndarray = np.zeros(self.num_actions)
        self.update_step_size = update_step_size

    def reset(self, q_init: float):
        self.total_reward_accum = 0.0
        self.num_selections_accum = np.zeros(self.num_actions)

    def select_action(self) -> Tuple[int, bool]:
        """Use current action-value estimates to select an action. Uses epsilon-greedy
        action selection: take a non-greedy (exploratory) action with probability epsilon.

        Returns:
            int: Integer index of the action selected to be taken.
            bool: Flag indicating if action was taken greedily.
        """
        if np.random.rand() > self.epsilon:     # select greedy action
            argmaxes = np.argwhere(self.q == np.amax(self.q).flatten()).flatten()
            action = np.random.choice(argmaxes)
            greedy_flag = True
        else:                 
            greedy_flag = False                  # select exploratory action
            argnmaxes = np.argwhere(self.q != np.amax(self.q).flatten()).flatten()
            if argnmaxes.size > 0:
                action = np.random.choice(argnmaxes)
            else:
                action = np.random.choice(range(self.num_actions))

        self.num_selections_accum[action] += 1
        return action, greedy_flag
            
    def update_action_values(self, action: int, reward_obs: float) -> None:
        """Update the action-value estimates. 
        
        If self.update_step_size is None, updates by sample-averaging.
        Otherwise, updates with a constant step size. (recency-weighted averaging)

        Recall that sample-averaging means the effect of newer observations is lesser than
        for older observations!

        Args:
            action (int): Integer index for the action taken.
            reward_obs (float): Observed reward after taking action `action`.
        """
        step_size = self.update_step_size or 1 / self.num_selections_accum[action]
        self.q[action] += step_size * (reward_obs - self.q[action])



class GradientBandit(Agent):
    """Solves the k-armed bandit problem by the gradient bandit algorithm, i.e., by
    performing stochastic gradient ascent on the space of the agent's 'preferences'
    for the various actions.

    The agent maintains a preference vector `h` which is used to select actions based on
    a softmax distribution. While receiving rewards, the preference vector is updated. The
    update gradient step involves the average reward observed thus far, so this must be
    tracked by the agent (it is incrementally updated).
    """
    def __init__(self, k: int, baseline: float = 0.0, alpha: float = 0.5):
        super().__init__(k=k, q_init=0.0)
        self.h: np.ndarray = baseline * np.ones(k)
        self.reward_running_average: float = 0.0
        self.alpha: float = alpha
        self.num_obs: int = 0

    def select_action(self) -> Tuple[int, bool]:
        """Select an action by softmaxing the preference vector to get a probability
        distribution; then sample an action from this distribution.
        """
        pi_ = self._softmax(self.h)
        return np.random.choice(range(self.num_actions), p=pi_), False

    @staticmethod
    def _softmax(v: np.ndarray) -> np.ndarray:
        pi_ = np.exp(v)
        return pi_ / pi_.sum()

    def update_action_values(self, action: int, reward_obs: float) -> None:
        # update reward running average with new observation
        self.num_obs += 1
        self.reward_running_average += (reward_obs - self.reward_running_average) / self.num_obs

        pi_ = self._softmax(self.h)

        resid = reward_obs - self.reward_running_average
        for action_idx in range(self.num_actions):
            if action_idx == action: # the action that was taken
                self.h[action_idx] += self.alpha * resid * (1 - pi_[action_idx])
            else:
                self.h[action_idx] -= self.alpha * resid * pi_[action_idx]
