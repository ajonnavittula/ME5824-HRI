import numpy as np
from transitions import transitions
import sys
S = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]])

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

gamma = 1
r_empty = -1.0
R = np.ones(11)*r_empty
R[10] = 1
R[6] = -1


def main():
    V = np.zeros(len(S))
    V1 = np.zeros(len(S))

    for interaction in range (100):

        for s_idx in range (len(S)):
            if s_idx == 6 or s_idx == 10:
                V1[s_idx] = R[s_idx]
                continue

            next_state_V = np.zeros(len(A))
            for a_idx in range (len(A)):
                transition = transitions(S[s_idx,:], A[a_idx,:], S)
                # sys.exit()
                next_state_V[a_idx] = np.sum(np.multiply(transition, V))

            V1[s_idx] = R[s_idx] + gamma * np.max(next_state_V)

        V = np.copy(V1)
    print(V)

if __name__ == "__main__":
    main()