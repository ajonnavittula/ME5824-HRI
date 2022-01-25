import numpy as np

def transitions(s, a, S):
    n_states = len(S)
    P = np.zeros(n_states)
    s_index = np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0]

    s1 = s + a
    # print(np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0])
    if np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0]:
        P[np.where((S[:,0] == s[0]) & (S[:,1]==s[1]))[0]] = 0.8
    else:
        P[s_index] = 0.8

    a1 = np.array([[0, -1], [1, 0]]).dot(a)

    s1 = s + a1
    if np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]:
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1

    a1 = np.array([[0, 1], [-1, 0]]).dot(a)
    s1 = s + a1
    if np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]:
        P[np.where((S[:,0] == s1[0]) & (S[:,1]==s1[1]))[0]] = 0.1
    else:
        P[s_index] += 0.1    
    # print(P)
    return P
