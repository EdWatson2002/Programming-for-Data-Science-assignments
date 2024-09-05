import numpy as np

def bandit(arm):
    mu1 = 0.6
    mu2 = 0.4
    if arm == 1:
        return np.random.binomial(1, mu1)
    elif arm == 2:
        return np.random.binomial(1, mu2)



def epsilon_decreasing(n, epsilon_sequence):

    def choose_arm(epsilon, wins_1, plays_1, wins_2, plays_2):

        if np.random.binomial(1, epsilon) == 1:
            arm = np.random.binomial(1, 0.5) + 1
            return arm
        else:
            win_rate_1 = wins_1/plays_1
            win_rate_2 = wins_2/plays_2
            if win_rate_1 > win_rate_2:
                return 1
            elif win_rate_2 > win_rate_1:
                return 2
            else:
                return np.random.binomial(1,0.5) + 1
        
    arms = np.zeros(n, int)
    rewards = np.zeros(n, int)
    
    arms[0] = 1
    rewards[0] = bandit(1)

    arms[1] = 2
    rewards[1] = bandit(2)

    plays_1 = 1; plays_2 = 1
    wins_1 = rewards[0]; wins_2 = rewards[1]

    for i in range(2, n):
        
        arm = choose_arm(epsilon_sequence[i], wins_1, plays_1, wins_2, plays_2)
        
        if arm == 1:
            arms[i] = 1
            plays_1 += 1
            r = bandit(1)
            rewards[i] = r
            wins_1 += r
        else:
            arms[i] = 2
            plays_2 += 1
            r = bandit(2)
            rewards[i] = r
            wins_2 += r
    
    return (arms, rewards)

    
n = 100
C = 1            #low value of C
seq = []
for i in range(1,n+1):
    seq.append(min(1, C/(i**2)))
    
for i in range(20):
    print(sum(epsilon_decreasing(n, seq)[1])/n)