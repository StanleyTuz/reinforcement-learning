
import matplotlib.pyplot as plt
import numpy as np
import objects

dynamics = objects.get_dynamics4_1()
policy = objects.get_policy_epr(16, 4)

theta = 0.0000001

gamma = 1.0
V = np.zeros(16)
V_new = V.copy()

states = range(16)
actions = range(4)
rewards_dict = {0: -1.0, 1: 0.0}

print(V_new.reshape(4,4))
for iter_ in range(1,11):
    
    delta = 0
    V_new = np.zeros_like(V)

    for s in states:
        accum = 0
        for a in actions:
            for sp in states:
                for ri,r in rewards_dict.items():
                    accum += policy[s, a] * dynamics[s, a, sp, ri] * (r + gamma * V[sp])
        delta = np.max([delta, np.abs(accum - V[s])])
        
        V_new[s] = accum
    
    # update approximation V
    V = V_new.copy()
    
    if iter_ in [1, 2, 3, 10]:
        print(V_new.reshape(4,4))


