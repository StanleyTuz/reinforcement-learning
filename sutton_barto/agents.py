
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from environments import Environment


class Agent(ABC):
    """Base class for solving the k-armed bandit problem via
    action-value estimation.

    - Uses epsilon-greedy method to select actions.
    - Updates action-value estimate using sample-averages.
    """

    def __init__(self, k: int):
        """Instantiate an agent.
        
        Args:
            k (int): Number of possible actions to take.
        """
        self.num_actions = k
        self.q = np.zeros(self.num_actions) # hold current action-value estimates

    
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

    def __init__(self, k: int, epsilon: float = 1.0):
        super().__init__(k=k)
        self.epsilon = epsilon # greediness parameter
        self.total_reward_accum: float = 0.0
        self.num_selections_accum: np.ndarray = np.zeros(self.num_actions)
        self.reset()

    def reset(self):
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
        """Update the action-value estimates by sample-averaging. 

        Recall that sample-averaging means the effect of newer observations is lesser than
        for older observations!

        Args:
            action (int): Integer index for the action taken.
            reward_obs (float): Observed reward after taking action `action`.
        """
        self.q[action] += (reward_obs - self.q[action]) / self.num_selections_accum[action]