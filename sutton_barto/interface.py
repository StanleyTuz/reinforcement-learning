import agents
import environments

import numpy as np
from typing import Tuple


def run(agent: agents.Agent, environment: environments.Environment, num_steps: int = 1) -> Tuple[np.ndarray, ...]:
    """Allow agent to take actions in environment and update its action-value estimates.

    Returns:
        np.ndarray: Array of actions indices in the order they were taken.
        np.ndarray: Array of rewards observed in the order they were observed.
        np.ndarray: Binary array indicating if each action was taken greedily or not.
    """
    actions_taken = np.zeros(num_steps)
    rewards_observed = np.zeros(num_steps)
    greedy_indic = np.zeros(num_steps)

    for step_idx in range(num_steps):
        # agent: select action
        action, greedy_flg = agent.select_action()

        # env: return reward
        reward_obs = environment(action)

        # agent: update action-values
        agent.update_action_values(action, reward_obs)

        actions_taken[step_idx] = action
        rewards_observed[step_idx] = reward_obs
        greedy_indic[step_idx] = int(greedy_flg)

    return actions_taken, rewards_observed, greedy_indic



if __name__ == "__main__":

    k = 5
    env = environments.KArmedBandit(k=k)
    agent = agents.EpsilonGreedyAgent(k=k, epsilon=0.5)

    print(f"Before running: {agent.q}")
    at, ro, gf = run(agent, env, 10)
    print(f"After running:  {agent.q}")

    # print(f"Actions selected: {agent.num_selections_accum}")
    print(f"Actions trajectory: {at}")
    print(f"Observed rewards trajectory: {ro}")
    print(f"Actual reward means: {env.reward_means}")
    print(f"Actual optimal action: {np.argmax(env.reward_means)}")

    print(f"Greediness trajectory: {gf}")
