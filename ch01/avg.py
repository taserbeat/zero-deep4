import numpy as np

try_count = 10  # 試行回数

np.random.seed(0)  # シードを固定
rewards = []

for n in range(1, try_count + 1):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)

print('---')

np.random.seed(0)
Q = 0

for n in range(1, try_count + 1):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    print(Q)
