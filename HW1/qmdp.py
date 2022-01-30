import numpy as np
from mdp import S, A, value_iteration, transitions

# QMDP with given beliefs and rewards  
def qmdp(gamma, b, r_empties):
    n_beliefs = len(b)
    Q_final = np.zeros((len(S), len(A)))
    for i in range(n_beliefs):
        Q = value_iteration(gamma, r_empties[i])[1] 
        Q_final += b[i] * Q
    policy = np.zeros((len(S), 4))
    for s_idx in range(len(S)):
        if s_idx == 6 or s_idx == 10:
            policy[s_idx, :] = np.array([S[s_idx,0], S[s_idx,1], 0, 0])
            continue
        a_idx = np.argmax(Q_final[s_idx])
        policy[s_idx, :] = np.array([S[s_idx,0], S[s_idx,1], A[a_idx,0], A[a_idx,1]])
    return policy

def main():
    gamma = 1.0
    b = [0.1, 0.2, 0.7]
    r_empties = [-2, -0.5, 0.5]
    policy = qmdp(gamma, b, r_empties)
    print("policy")
    print(policy)

if __name__ == "__main__":
    main()