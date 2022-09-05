import agents
import environments

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Sequence


def run(agent: agents.Agent, environment: environments.Environment, num_steps: int = 1, stationary: bool = True) -> Tuple[np.ndarray, ...]:
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

        if not stationary:
            environment.evolve()

    return actions_taken, rewards_observed, greedy_indic


def plot_results(agent: agents.Agent, env: environments.Environment, ax: matplotlib.axes.Axes):
    ax.bar(range(1, agent.num_actions+1), agent.q, color='black', alpha=0.5)
    ax.set(
        title=f'Estimated action-vaues (epsilon = {agent.epsilon})',
        ylabel=r'$q\left(a\right)$',
        xlabel=r'$a$',
        xticks=range(1, agent.num_actions+1),
        );
    _width = 0.50
    for idx, rwm in enumerate(env.reward_means):
        ax.hlines(xmin=idx+1-_width, xmax=idx+1+_width, y=rwm, color='r', lw=2)

def cumavg(x: np.ndarray):
    """Take the cumulative average of a vector of numbers."""
    if x.ndim != 1:
        raise ValueError('Input array must be a vector.')
    return np.cumsum(x) / np.arange(1, len(x)+1, 1)


def run_testbed(
    epsilons: Sequence[float],
    num_actions: int,
    num_runs_per_epsilon: int,
    num_steps_per_run: int,
    q_inits: Sequence[float] = None,
    update_step_sizes: Sequence[float] = None,
    stationaries: Sequence[bool] = None,
    verbose: bool = False,
    ) -> dict:
    """
    Args:
        epsilons (Sequence[float]):
        num_actions (int): Number of possible actions for each agent/environment.
        num_runs_per_epsilon (int): Number of agents to run for each epsilon.
        num_steps_per_run (int): Number of observation steps to make for each run.
        q_inits(Optional[Sequence[float]]): Vector of optimistic initial values to use
            for each epsilon's simulations. Defaults to all zeros.
        update_step_sizes (Optional[Sequence[float]]): Vector of step sizes to use
            in action-value updates. If this is none, all agents will use sample-averaging
            to update their action-value estimates.

    Returns
        dict: A dict mapping a (zero-based) test index to a dictionary containing
            the keys 'avg_rewards' and 'frac_opt_action' which map to np.ndarrays
            containing the average reward trajectory and the average fraction of times 
            the actual optimal action was selected, respectively.
    """

    trajectories = { idx: { 'epsilon': eps } for idx, eps in enumerate(epsilons) }
    q_inits = q_inits or np.zeros(len(epsilons))
    stationaries = stationaries or [True] * len(epsilons)
    update_step_sizes = update_step_sizes or [None] * len(epsilons)

    for idx, eps in enumerate(epsilons):
        if verbose:
            print(f"Starting simulation for epsilon = {eps}")
        # initialize result accumulators
        rewards_accum = np.zeros(num_steps_per_run)
        opt_action_accum = np.zeros(num_steps_per_run)

        for n_iter in range(num_runs_per_epsilon):
            if verbose and (n_iter+1) % 100 == 0:
                print(f"  iter {n_iter+1:4}")
            # instantiate an env
            env = environments.KArmedBandit(k=num_actions)
            # instantiate an agent
            agent = agents.EpsilonGreedyAgent(
                k=num_actions,
                epsilon=eps,
                q_init=q_inits[idx],
                update_step_size=update_step_sizes[idx],
                )

            # run a learning simulation
            at, ro, _ = run(
                agent=agent,
                environment=env,
                num_steps=num_steps_per_run,
                stationary=stationaries[idx],
                )
        
            # record the results of this run
            rewards_accum += ro
            opt_action_accum += (at == env.opt_action).astype(int)

        # average the results across all runs
        rewards_accum = rewards_accum / num_runs_per_epsilon
        opt_action_accum = opt_action_accum / num_runs_per_epsilon
        
        # store results for this test
        trajectories[idx]['avg_rewards'] = rewards_accum
        trajectories[idx]['frac_opt_action'] = opt_action_accum
        
    return trajectories


def run_testbed_gradient(
    alphas: Sequence[float],
    num_actions: int,
    num_runs_per_alpha: int,
    num_steps_per_run: int,
    baselines: Sequence[float] = None,
    stationaries: Sequence[bool] = None,
    verbose: bool = False,
    ) -> dict:
    """
    Args:
        alphas (Sequence[float]):
        num_actions (int): Number of possible actions for each agent/environment.
        num_runs_per_alpha (int): Number of agents to run for each epsilon.
        num_steps_per_run (int): Number of observation steps to make for each run.
        baselines(Optional[Sequence[float]]): Vector of optimistic initial values to use
            for each epsilon's simulations. Defaults to all zeros.
        update_step_sizes (Optional[Sequence[float]]): Vector of step sizes to use
            in action-value updates. If this is none, all agents will use sample-averaging
            to update their action-value estimates.

    Returns
        dict: A dict mapping a (zero-based) test index to a dictionary containing
            the keys 'avg_rewards' and 'frac_opt_action' which map to np.ndarrays
            containing the average reward trajectory and the average fraction of times 
            the actual optimal action was selected, respectively.
    """

    trajectories = { idx: { 'alpha': alpha } for idx, alpha in enumerate(alphas) }
    baselines = baselines or np.zeros(len(alphas))
    stationaries = stationaries or [True] * len(alphas)

    for idx, alpha in enumerate(alphas):
        if verbose:
            print(f"Starting simulation for alpha = {alpha}")
        # initialize result accumulators
        rewards_accum = np.zeros(num_steps_per_run)
        opt_action_accum = np.zeros(num_steps_per_run)

        for n_iter in range(num_runs_per_alpha):
            if verbose and (n_iter+1) % 100 == 0:
                print(f"  iter {n_iter+1:4}")
            # instantiate an env
            env = environments.KArmedBandit(k=num_actions)
            # instantiate an agent
            agent = agents.GradientBandit(
                k=num_actions,
                baseline=baselines[idx],
                alpha=alpha,
                )

            # run a learning simulation
            at, ro, _ = run(
                agent=agent,
                environment=env,
                num_steps=num_steps_per_run,
                stationary=stationaries[idx],
                )
        
            # record the results of this run
            rewards_accum += ro
            opt_action_accum += (at == env.opt_action).astype(int)

        # average the results across all runs
        rewards_accum = rewards_accum / num_runs_per_alpha
        opt_action_accum = opt_action_accum / num_runs_per_alpha
        
        # store results for this test
        trajectories[idx]['avg_rewards'] = rewards_accum
        trajectories[idx]['frac_opt_action'] = opt_action_accum
        
    return trajectories


def plot_trajectories_testbed(trajectories: dict, ax: matplotlib.axes.Axes = None, param_name: str = None):
    """Plotting helper for K-armed Testbed simulations. Attempts to replicate
    Figure 2.2 from Sutton & Barto (2018).

    Each element of `trajectories` corresponds to a particular simulation, i.e., the 
    aggregate results of running multiple instances of identically parametrized agents
    against similar environments.   

    Specifically, `trajectories` must be a dict mapping a (zero-based) simulation index
    to a dictionary containing the keys 'avg_rewards' and 'frac_opt_action' which map to
    np.ndarrays containing the average reward trajectory and the average fraction of times 
    the actual optimal action was selected, respectively.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    param_name = param_name or 'epsilon'
    
    xticks_ = np.arange(0, 1000+1, 250)

    colors = ('green', 'red', 'blue')
    for idx, traj_dict in trajectories.items():
        label_ = "param = " + str(traj_dict[param_name])
        ax[0].plot(traj_dict['avg_rewards'], color=colors[idx], label=label_)
        ax[1].plot(traj_dict['frac_opt_action'], color=colors[idx], label=label_)

    ax[0].set(ylim=[0,1.5], yticks=[0, 0.5, 1.0, 1.5], yticklabels=[0, 0.5, 1, 1.5], xticks=xticks_, xticklabels=xticks_)
    ax[0].set_xlabel(xlabel="Steps", fontsize=15)
    ax[0].set_ylabel(ylabel="Average\nreward", rotation=0, fontsize=15)
    ax[0].yaxis.set_label_coords(-.15, .5)
    ax[0].legend()


    yticks_ = np.arange(0, 1.0+.1, .20)
    yticklabels_ = [ f"{int(100*tick)}%" for tick in yticks_]
    ax[1].set(ylim=[0,1.0], yticks=yticks_, yticklabels=yticklabels_, xticks=xticks_, xticklabels=xticks_)
    ax[1].set_xlabel(xlabel="Steps", fontsize=15)
    ax[1].set_ylabel(ylabel="%\nOptimal\naction", rotation=0, fontsize=15)
    ax[1].yaxis.set_label_coords(-.15, .5)
    ax[1].legend()

    plt.show()