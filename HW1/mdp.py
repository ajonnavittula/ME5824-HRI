import numpy as np
import sys
S = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]])

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def transitions(s, a, S):

    n_states = len(S)
    P = np.zeros(n_states)
    s_index = np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0]
    
    s1 = s + a 
    # if s[0] == 2 and s[1] == 1 and a[0] == -1 and a[1] ==0:
    #     print(np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0])

    if np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]:
        if s[0] == 2 and s[1] == 1 and a[0] == -1 and a[1] ==0:
            print (s1)
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.8
    else:
        P[s_index] = 0.8

    a1 = np.array([[0, -1], [1, 0]]).dot(a)

    s1 = s + a1
    # if s[0] == 2 and s[1] == 1 and a[0] == -1 and a[1] ==0:
    #     print(s1)

    if np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]:
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1

    a1 = np.array([[0, 1], [-1, 0]]).dot(a)
    s1 = s + a1
    # if s[0] == 2 and s[1] == 1 and a[0] == -1 and a[1] ==0:
        # print(a)

    if np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]:
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1    

    return P


def main():
    gamma = 1
    r_empty = -0.04
    R = np.ones(11)*r_empty
    R[10] = 1
    R[6] = -1

    V = np.zeros(len(S))
    V1 = np.zeros(len(S))
    policy = np.zeros((11,4))
    done = False
    iter = 0

    # for interaction in range (1000):
    while not done:
        V_prime = np.copy(V)
        for s_idx in range (len(S)):
            
            if s_idx == 6 or s_idx == 10:
                V1[s_idx] = R[s_idx]
                policy[s_idx,:] = np.array([S[s_idx,0], S[s_idx,1], 0., 0.])
                continue

            next_state_V = np.zeros(len(A))
            for a_idx in range (len(A)):
                # print(A[a_idx,:])
                # print(a_idx)
                transition = transitions(S[s_idx,:], A[a_idx,:], S)
                next_state_V[a_idx] = np.sum(np.multiply(transition, V))
                # print("Transiotion = {}, A = {}".format(transition,A[a_idx,:]))

            V1[s_idx] = R[s_idx] + gamma * np.max(next_state_V)
            # print(V1)
            # print(iter)
            max_V_id = np.where(next_state_V == np.max(next_state_V))
            max_V_idx = max_V_id[0][0]
            policy[s_idx,:] = np.array([S[s_idx,0], S[s_idx,1], A[max_V_idx,0], A[max_V_idx, 1]])


        V = np.copy(V1)
        if np.linalg.norm(V-V_prime) < 1e-6:
            done  = True 
    
    # print(V)
    # print(policy)
        iter+=1

if __name__ == "__main__":
    main()