import numpy as np
import matplotlib.pyplot as plt
import sympy


def get_dynamics() -> np.ndarray:
    """
    The dynamics is a mapping 
        p: S' x R x S x A -> [0, 1]
    
    For our purposes, we would like to treat this as a function
    mapping (s,a) to a distribution over (s',r). For finite MDP,
    including Gridworld, this distribution is a finite, discrete
    set.
    
    Thus, the dynamics should be the function
        s, a |-> {(s', r, p)}
    """
    rewards = {0: -1.0, 1: 0.0, 2: 5.0, 3: 10.0}
    dynamics = np.zeros((25, 4, 25, 4)) # s, a, s', r -> p
    for s in range(25):
        for a in range(4):
            if a == 0:
                if s - 5 < 0:
                    dynamics[s, a, s, 0] = 1.0
                else:
                    dynamics[s, a, s-5, 1] = 1.0

            if a == 1:
                if s + 5 >= 25:
                    dynamics[s, a, s, 0] = 1.0
                else:
                    dynamics[s, a, s+5, 1] = 1.0

            if a == 2:
                if s % 5 == 0:
                    dynamics[s, a, s, 0] = 1.0
                else:
                    dynamics[s, a, s-1, 1] = 1.0

            if a == 3:
                if (s+1) % 5 == 0:
                    dynamics[s, a, s, 0] = 1.0
                else:
                    dynamics[s, a, s+1, 1] = 1.0

                    
    # Add in the arbitrary dynamics
    dynamics[1, :, :, :] = 0.0 
    dynamics[1, :, 21, 3] = 1.0

    dynamics[3, :, :, :] = 0.0 
    dynamics[3, :, 13, 2] = 1.0
    
    return dynamics


def get_policy_epr() -> np.ndarray:
    """
    Equi-probable random policy.
    """
    return np.stack(
        [np.array([0.25, 0.25, 0.25, 0.25])]*25
    )


def plot_v(v: np.ndarray, ax = None):
    if ax is None:
        _, ax = plt.subplots()

    ax.set(xlim=[-0.5,4.5], ylim=[-0.5,4.5])
    ax.set_xticks(np.arange(-0.5, 4.5, 1))
    ax.set_yticks(np.arange(-0.5, 4.5, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid()

    
    def state_to_txt_coord(s: int):
        y_coord = 5 - ( s // 5) - 1
        x_coord = s % 5
        return x_coord, y_coord

    for s in range(len(v)):
        x_t, y_t = state_to_txt_coord(s)
        ax.text(x=x_t, y=y_t, s=f"{v[s]:.1f}")
    ax.set_aspect('equal')
    # plt.show()




def calc_sv_exact(
    policy: np.ndarray,
    dynamics: np.ndarray,
    gamma: float = 0.5,
):
    """Calculate the exact state-value function for a policy
    in an environment by solving the Bellman equation as a
    linear system.
    
    This is feasible only for small, discrete problems (tabular case)
    in which the dynamics are exactly known.

    Args:
        policy (np.ndarray): Array indexed by (state, action) and containing
            the probabilities of taking an action in a particular state.
        dynamics (np.ndarray): Array indexed by 
            (initial_state, action, final_state, reward_index) and containing
            the probabilities of observing the reward indexed by reward_index
            and the resultant state final_state after taking action action
            in state initial_state. This is the environment dynamics in a
            tabular, np.array representation.
        gamma (float): discount factor used to calculate the return. Must be <1
            in order for return to converge to something finite. Smaller gamma
            means we care less about future rewards; gamma = 0 is immediate
            short-term rewards.
    """

    S = np.zeros((25, 25))
    b = np.zeros((25, 1))

    states = list(range(25))
    actions = list(range(4))
    rewards = {0: -1.0, 1: 0.0, 2: 5.0, 3: 10.0}

    # constant terms
    for s in states:
        actions = policy[s] # (a, p)
        for a, p_a_s in enumerate(actions):
            results = dynamics[s, a] # (sp, r, p)
            for sp in states:
                for r_idx in rewards.keys():
                    b[s] += rewards[r_idx] * results[sp, r_idx] * p_a_s

    # linear terms
    for s in states:
        actions = policy[s] # (a, p)
        for a, p_a_s in enumerate(actions):
            results = dynamics[s,a] # (sp, r, p)
            for sp in states:
                for r_idx in rewards.keys():
                    S[s, sp] += p_a_s * results[sp, r_idx]

    A = np.eye(25) - gamma * S

    v = np.linalg.inv(A) @ b
    return v.flatten()


def calc_q_from_v(v: np.ndarray, dynamics: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Given a state-value function, v, calculate the corresponding
    state-action-value function, q using the known problem dynamics.
    """
    q = np.zeros((25, 4))

    states = list(range(25))
    actions = list(range(4))
    rewards = {0: -1.0, 1: 0.0, 2: 5.0, 3: 10.0}

    for s in states:
        for a in actions:
            q[s, a] = 0.0
            for sp in states:
                for ri, r in rewards.items():
                    q[s, a] += ( r + gamma * v[sp] )* dynamics[s, a, sp, ri]
    return q


def argmax_q(q):
    policy_new = np.zeros((25, 4))

    for i in range(q.shape[0]):
        row = q[i, :]
        idxs_max = np.argwhere(row == np.max(row)).flatten()
        p = 1.0 / len(idxs_max)
        for idx in idxs_max:
            policy_new[i, idx] = p
    return policy_new


def policy_improvement(policy: np.ndarray, dynamics: np.ndarray, gamma: float):
    
    # get state-value function for this policy
    v = calc_sv_exact(policy, dynamics, gamma)

    # get action-value function for this policy
    q = calc_q_from_v(v, dynamics, gamma)

    # get improved policy by acting greedily
    policy_new = argmax_q(q)

    return policy_new


def calc_sv_optimal_exact(
    dynamics: np.ndarray,
    gamma: float = 0.5,
):
    """Calculate the exact *optimal* state-value function
    in an environment by solving the Bellman optimality
    equation as a non-linear system. sympy is used for the
    actual nonlinear solving.
    
    This is feasible only for small, discrete problems (tabular case)
    in which the dynamics are exactly known.

    Args:
        dynamics (np.ndarray): Array indexed by 
            (initial_state, action, final_state, reward_index) and containing
            the probabilities of observing the reward indexed by reward_index
            and the resultant state final_state after taking action action
            in state initial_state. This is the environment dynamics in a
            tabular, np.array representation.
        gamma (float): discount factor used to calculate the return. Must be <1
            in order for return to converge to something finite. Smaller gamma
            means we care less about future rewards; gamma = 0 is immediate
            short-term rewards.
    """
    from sympy.functions.elementary.miscellaneous import Max

    gamma = sympy.S(gamma)

    states = list(range(25))
    actions = list(range(4))

    rewards = { i: sympy.S(x) for i,x in enumerate([-1.0, 0.0, 5.0, 10.0])}
    sym = [sympy.Symbol(f"v({s})", real=True) for s in states]

    eqn_list = []
    for s in states:
        terms = []
        for a in actions:
            sum_ = sympy.S(0)  # init to zero
            for sp in states:
                for ri, r in rewards.items():
                    sum_ += sympy.S(dynamics[s, a, sp, ri]) * (r + gamma * sym[sp])
            terms.append(sum_)

        eqn = sym[s] - Max(*terms)
        eqn_list.append(eqn)

    soln = sympy.nsolve(eqn_list, sym, [1]*25)
    return np.array(soln).flatten()


