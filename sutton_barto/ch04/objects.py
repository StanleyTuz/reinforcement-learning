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


def get_dynamics4_1() -> np.ndarray:
    # Dynamics for the Gridworld in Example 4.1
    rewards = {0: -1.0, 1: 0.0}
    dynamics = np.zeros((16, 4, 16, 2)) # s, a, s', r -> p

    for s in range(16):
        # terminal states s=0, s=15
        # all actions stay in place, return zero reward
        if s == 0 or s == 15:
            for a in range(4):
                dynamics[s, a, s, 1] = 1.0
        else: # non-terminal states
            for a in range(4):
                if a == 0: # up
                    sp = s if s - 4 < 0 else s - 4
                    dynamics[s, a, sp, 0] = 1.0

                if a == 1: # down
                    sp = s if s + 4 > 15 else s + 4
                    dynamics[s, a, sp, 0] = 1.0

                if a == 2: # left
                    sp = s if s % 4 == 0 else s - 1
                    dynamics[s, a, sp, 0] = 1.0

                if a == 3: # right
                    sp = s if ( s + 1 ) % 4 == 0 else s + 1
                    dynamics[s, a, sp, 0] = 1.0

    return dynamics


def get_policy_epr(n_states: int = 25, n_actions: int = 4) -> np.ndarray:
    """
    Equi-probable random policy.
    """
    p = 1/n_actions
    return np.stack(
        [np.array([p]*n_actions)]*n_states
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


def plot_policy(policy: np.ndarray, ax = None):
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

    arrows = {0: [0,1], 1: [0, -1], 2: [-1, 0], 3: [1, 0]}
    scale = 0.40
    eps = 1e-5
    for s in range(policy.shape[0]):
        p_s = policy[s]
        x_t, y_t = state_to_txt_coord(s)
        for idx, arr in arrows.items():
            arrow = scale * np.array(arr) * p_s[idx]
            if any(abs(arrow) > eps):
                ax.arrow(
                    x_t,
                    y_t,
                    arrow[0],
                    arrow[1],
                    head_width=0.05,
                    head_length=0.1,
                    fc='k',
                    ec='k',
                    )
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


def calc_q_from_v(
    v: np.ndarray,
    dynamics: np.ndarray,
    rewards_dict: dict,
    gamma: float = 0.5,
    ) -> np.ndarray:
    """Given a state-value function, v, calculate the corresponding
    state-action-value function, q using the known problem dynamics.
    """
    n_states = dynamics.shape[0]
    n_actions = dynamics.shape[1]
    q = np.zeros((n_states, n_actions))

    states = list(range(n_states))
    actions = list(range(n_actions))
    rewards = rewards_dict

    for s in states:
        for a in actions:
            q[s, a] = 0.0
            for sp in states:
                for ri, r in rewards.items():
                    q[s, a] += ( r + gamma * v[sp] )* dynamics[s, a, sp, ri]
    return q


def argmax_q(q):
    policy_new = np.zeros_like(q)

    for i in range(q.shape[0]):
        row = q[i, :]
        idxs_max = np.argwhere(row == np.max(row)).flatten()
        p = 1.0 / len(idxs_max)
        for idx in idxs_max:
            policy_new[i, idx] = p
    return policy_new


def calc_greedy_policy_from_v(v, dynamics, rewards_dict, gamma):
    # calculate action-value function
    q = calc_q_from_v(v, dynamics, rewards_dict, gamma)

    # take argmax
    return argmax_q(q)


def policy_improvement(
    dynamics: np.ndarray,
    rewards_dict: dict,
    policy: np.ndarray = None,
    v: np.ndarray = None,
    gamma: float = 1.0,
    ):
    
    if v is None and policy is None:
        raise ValueError("Must pass one of [policy, v]")

    if v is None:
        # get state-value function for this policy
        v = calc_sv_exact(policy, dynamics, gamma)

    # get action-value function for this policy
    q = calc_q_from_v(v, dynamics, rewards_dict, gamma)

    # get improved policy by acting greedily
    policy_new = argmax_q(q)

    return policy_new


def calc_sv_optimal_exact(
    dynamics: np.ndarray,
    rewards_dict: dict,
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

    sz = dynamics.shape
    states = list(range(sz[0]))
    actions = list(range(sz[1]))
    rewards = { i: sympy.S(x) for i,x in rewards_dict.items()}
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

    soln = sympy.nsolve(eqn_list, sym, [0]*sz[0])
    return np.array(soln).flatten()



def policy_evaluation(
    policy: np.ndarray,
    dynamics: np.ndarray,
    rewards_dict: dict,
    gamma: float,
    theta: float = None,
    max_iters: int = 100,
    v_init: np.ndarray = None,
    ) -> np.ndarray:
    """Perform iterative policy evaluation to approximate
    the state-value function v, as described in Sutton &
    Barto 4.1."""

    n_states = policy.shape[0]
    
    V = np.zeros(n_states) if v_init is None else v_init
    
    states = range(n_states)
    actions = range(4)
    converged = False

    for _ in range(max_iters):

        delta = 0
        V_new = np.zeros_like(V)

        for s in states:
            accum = 0
            for a in actions:
                for sp in states:
                    for ri, r in rewards_dict.items():
                        accum += policy[s, a] * dynamics[s, a, sp, ri] * (r + gamma * V[sp])
            delta = np.max([delta, np.abs(accum - V[s])])
        
            V_new[s] = accum

        V = V_new.copy()

        if theta is not None and delta < theta:
            converged = True
            break

    if theta is not None and not converged:
        print('failed to converge')
    
    return V


    



def plot_v2(v: np.ndarray, ax = None):    
    if ax is None:
        _, ax = plt.subplots()

    gs = int(np.sqrt(len(v))) # grid size

    ax.set(xlim=[-0.5,-0.5 + gs], ylim=[-0.5,-0.5 + gs])
    ax.set_xticks(np.arange(-0.5, -0.5+gs, 1))
    ax.set_yticks(np.arange(-0.5, -0.5+gs, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid()

    
    def state_to_txt_coord(s: int):
        y_coord = gs - ( s // gs) - 1
        x_coord = s % gs
        return x_coord, y_coord

    for s in range(len(v)):
        x_t, y_t = state_to_txt_coord(s)
        ax.text(x=x_t, y=y_t, s=f"{v[s]:.1f}")
    ax.set_aspect('equal')


def plot_policy2(policy: np.ndarray, ax = None):
    if ax is None:
        _, ax = plt.subplots()
    
    n_states = policy.shape[0]
    n_actions = policy.shape[1]

    gs = int(np.sqrt(n_states)) # grid size

    ax.set(xlim=[-0.5,-0.5 + gs], ylim=[-0.5,-0.5 + gs])
    ax.set_xticks(np.arange(-0.5, -0.5+gs, 1))
    ax.set_yticks(np.arange(-0.5, -0.5+gs, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid()

    def state_to_txt_coord(s: int):
        y_coord = gs - ( s // gs) - 1
        x_coord = s % gs
        return x_coord, y_coord

    arrows = {0: [0,1], 1: [0, -1], 2: [-1, 0], 3: [1, 0]}
    scale = 0.40
    eps = 1e-5
    for s in range(n_states):
        p_s = policy[s]
        x_t, y_t = state_to_txt_coord(s)
        for idx, arr in arrows.items():
            arrow = scale * np.array(arr) * p_s[idx]
            if any(abs(arrow) > eps):
                ax.arrow(
                    x_t,
                    y_t,
                    arrow[0],
                    arrow[1],
                    head_width=0.05,
                    head_length=0.1,
                    fc='k',
                    ec='k',
                    )
    ax.set_aspect('equal')