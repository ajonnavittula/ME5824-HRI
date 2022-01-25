import numpy as np

S = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]])

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

gamma = 1
r_empty = -0.04
R = np.ones(11)*r_empty
R[10] = 1
R[6] = -1

V = np.zeros(len(S))
V1 = np.zeros(len(S))

for interaction in range (10000):

    for s_idx in range (len(S)):
        if s_idx == 6 or s_idx == 10:
            V1[s_idx] = R[s_idx]
            continue

        next_state_V = np.zeros(len(A))
        for a_idx in range (len(A)):
            next_state_V[a_idx] = np.sum(np.multiply(transition(S[s_idx,:], A[a_idx,:], S)))

        V1[s_idx] = R[s_idx] + gamma * max(next_state_V)

    V = V1


