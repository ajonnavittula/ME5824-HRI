import numpy as np


def Question_1():
    beta = 1.0

    #Locations of goals
    CUP = [-1, 0]
    PLATE = [0, 1]
    BOWL = [-0.5, -0.5]

    GOALS = [[-1, 0], [0, 1], [-0.5, -0.5]]

    # Prior beliefs over goals
    b = [0.4, 0.4, 0.2]
    b_update = []

    # Action Space with N actions
    N = 10000
    A = []
    for _ in range (N):
        A.append([1-2*np.random.rand(), 1-2*np.random.rand()])

    A.append([-0.2, 0.6])

    # Belief Update using Boltzmann Rational Model
    for goal in range (len(GOALS)):
            reward_all_actions = np.sum(np.exp(-np.linalg.norm(np.array(action) - np.array(GOALS[goal]))) for action in A)
            b_update.append(np.exp(-np.linalg.norm(np.array(A[N]) - np.array(GOALS[goal])))/reward_all_actions*b[goal])

    norm_const = np.sum(b_update)
    for idx in range (len(b_update)):
        b_update[idx] = b_update[idx]/norm_const
    print(b_update)

def Question_3(beta):
    
    # Q = {'q1': [40, 60], 'q2': [30, 70], 'q3': [20, 80], 'q4': [10, 90], 'q5': [0, 100]}
    Q = [[40, 60], [30, 70], [20, 80], [10, 90], [0, 100]]
    THETA = np.linspace(0,100,101)
    Q_star = []
    temp = 0

    def Prob(q_idx, question, theta):
        # print("q")
        probability = np.exp(-beta*abs(theta - question[q_idx]))/(np.exp(-beta*abs(theta - question[0])) + np.exp(-beta*abs(theta - question[1])))
        # print(probability)
        return probability

    for question in Q:
        # print(question)
        # print(question)
        for q_idx in range (len(question)):
            den = np.sum(Prob(q_idx, question, theta_p) for theta_p in THETA)
            # print("denom", den)
            for theta in THETA:
            
                num = Prob(q_idx, question, theta)
                temp = temp + num* np.log2(101*num/den)
        
        Q_star.append(temp)
        
        temp = 0
    Q_optimal = Q[np.where(Q_star == np.max(Q_star))[0][0]]
    print("For BETA value =", beta)
    print(Q_star)
    print(Q_optimal)



if __name__ == '__main__':
    Question_1()
    betas = [0.01, 0.05, 0.1, 0.5, 1.0,]
    for beta in betas:
        Question_3(beta)