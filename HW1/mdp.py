import numpy as np


# Define the set of all possible states
S = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]])
# Define the set of all possible actions
A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

# System dynamics
def transitions(s, a):
    n_states = len(S)
    P = np.zeros(n_states)
    s_index = np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0]
    
    s1 = s + a 
    if np.size(np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]):
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.8
    else:
        P[s_index] = 0.8

    a1 = np.array([[0, -1], [1, 0]]).dot(a)
    s1 = s + a1
    if np.size(np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]):
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1

    a1 = np.array([[0, 1], [-1, 0]]).dot(a)
    s1 = s + a1
    if np.size(np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]):
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1    

    return P

# Value iteration algorithm
def value_iteration(gamma, r_empty):
    n_states = len(S)
    n_actions = len(A)
    R = np.ones(n_states) * r_empty
    R[10] = 1
    R[6] = -1

    V = np.zeros(n_states)
    V1 = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    policy = np.zeros((n_states, 4))
    converged = False
    while not converged:
        V1 = np.copy(V)
        for s_idx in range(n_states):
            # Terminal Conditions
            if s_idx == 6 or s_idx == 10:
                V[s_idx] = R[s_idx]
                # Take no action in terminal states
                policy[s_idx, :] = np.array([S[s_idx,0], S[s_idx,1], 0, 0])
                Q[s_idx] = R[s_idx]
                continue
            next_state_V = np.zeros(n_actions)
            for a_idx in range(n_actions):
                T = transitions(S[s_idx], A[a_idx])
                next_state_V[a_idx] = np.sum(np.multiply(T, V1))
                Q[s_idx, a_idx] = R[s_idx] + gamma * next_state_V[a_idx]

            max_v_idx = np.argmax(next_state_V)
            V[s_idx] = R[s_idx] + gamma * next_state_V[max_v_idx]
            policy[s_idx, :] = np.array([S[s_idx,0], S[s_idx,1], A[max_v_idx,0], A[max_v_idx,1]])
        # Converge if value function settles or if it is only adding positive reward from r_empty
        R_positive = np.ones(n_states) * r_empty
        R_positive[10] = 0
        R_positive[6] = 0
        if np.linalg.norm(V-V1) < 1e-5 or (gamma==1. and r_empty > 0. and np.linalg.norm(V-V1-R_positive) < 1e-5):
            converged = True

    return V, Q, policy

def main():
    gamma = 1.0
    r_empty = -0.5
    V, Q, policy = value_iteration(gamma, r_empty)
    print("Value function")
    print(V)
    print("Q function")
    print(Q)
    print("policy")
    print(policy)

if __name__ == "__main__":
    main()