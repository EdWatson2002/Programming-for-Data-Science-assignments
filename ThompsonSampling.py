import numpy as np

def bandit(arm):
    mu1 = 0.6
    mu2 = 0.4
    if arm == 1:
        return np.random.binomial(1, mu1)
    elif arm == 2:
        return np.random.binomial(1, mu2)
    
def Thompson_sampling(n):
    alpha = 1
    beta = 1

    arms = np.zeros(n, int)
    rewards = np.zeros(n, int)

    s1 = 0; f1 = 0; s2 = 0; f2 = 0

    for i in range(n):
        m1 = np.random.beta(alpha + s1, beta + f1)
        m2 = np.random.beta(alpha + s2, beta + f2)
        if m1 >= m2:
            arms[i] = 1
            r = bandit(1)
            rewards[i] = r
            if r == 1: 
                s1 += 1
            else: f1 += 1
        else:
            arms[i] = 2
            r = bandit(2)
            rewards[i] = r
            if r == 1: 
                s2 += 1
            else: f2 += 1

    return(arms, rewards)

n = 100
print(sum(Thompson_sampling(n)[1])/n)