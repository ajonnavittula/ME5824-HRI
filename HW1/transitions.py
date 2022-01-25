import numpy as np

def transitions(s, a, S):
    n_states = len(S)
    P = np.zeros(n_states)
    s_index = np.where(S==s)

    s1 = s + a
    if np.where(S==s1):
        P[np.where(S==s1)] = 0.8
    else:
        P[s_index] = 0.8

    a1 = np.array([[0, -1], [1, 0]]) * a.T
    s1 = s + a1
    if np.where(S==s1):
        P[np.where(S==s1)] = 0.1
    else:
        P[s_index] += 0.1

    a1 = np.array([[0, 1], [-1, 0]]) * a.T
    s1 = s + a1
    if np.where(S==s1):
        P[np.where(S==s1)] = 0.1
    else:
        P[s_index] += 0.1    

    return P
