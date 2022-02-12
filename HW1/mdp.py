import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

S = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [4, 3]])

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def transitions(s, a, S):

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


def QMDP():
    b = [0.1, 0.2, 0.7]
    Q_mat_1 = np.load('Q_mat_-2.0.npy')
    Q_mat_2 = np.load('Q_mat_0.5.npy')
    Q_mat_3 = np.load('Q_mat_-0.5.npy')
    policy_Q = np.zeros((11,4))

    for s_idx in range (len(S)):
        if s_idx == 6 or s_idx == 10:
            policy_Q[s_idx, :] = np.array([S[s_idx,0], S[s_idx,1], 0., 0.])
            continue

        next_state_Q = np.zeros(len(A))

        for a_idx in range (len(A)):
            next_state_Q[a_idx] = b[0] * Q_mat_1[s_idx,a_idx] + b[1] * Q_mat_2[s_idx,a_idx] + b[2] * Q_mat_3[s_idx,a_idx]
        print(np.max(next_state_Q))
        
        max_Q_id = np.where(next_state_Q == np.max(next_state_Q))
        max_Q_idx = max_Q_id[0][0]

        policy_Q[s_idx,:] = np.array([S[s_idx,0], S[s_idx,1], A[max_Q_idx,0], A[max_Q_idx, 1]])

    print(policy_Q)
            




def main():
    gamma = 0.7
    r_empty = 0.1
    R = np.ones(11)*r_empty
    R[10] = 1
    R[6] = -1

    V = np.zeros(len(S))
    V1 = np.zeros(len(S))
    Q = np.zeros((len(S), len(A)))
    policy = np.zeros((11,4))
    done = False
    iter = 0
    save_name = "Q_mat_" + str(r_empty)

    while not done:
        V_prime = np.copy(V)
        for s_idx in range (len(S)):
            
            if s_idx == 6 or s_idx == 10:
                V1[s_idx] = R[s_idx]
                policy[s_idx,:] = np.array([S[s_idx,0], S[s_idx,1], 0., 0.])
                Q[s_idx] = R[s_idx]
                continue

            next_state_V = np.zeros(len(A))
            for a_idx in range (len(A)):
                transition = transitions(S[s_idx,:], A[a_idx,:], S)
                next_state_V[a_idx] = np.sum(np.multiply(transition, V))

                Q[s_idx,a_idx] = R[s_idx] + gamma * next_state_V[a_idx]

            V1[s_idx] = R[s_idx] + gamma * np.max(next_state_V)
            max_V_id = np.where(next_state_V == np.max(next_state_V))
            max_V_idx = max_V_id[0][0]
            policy[s_idx,:] = np.array([S[s_idx,0], S[s_idx,1], A[max_V_idx,0], A[max_V_idx, 1]])


        V = np.copy(V1)
        if np.linalg.norm(V-V_prime) < 1e-6:
            done  = True 

        iter+=1
    
    print("The Value function is", V)
    print("The Q matrix is ", Q)
    print("policy for r_empty = {}, is {}".format(r_empty,policy))
    # np.save(save_name, Q)
    # QMDP()


    




if __name__ == "__main__":
    main()