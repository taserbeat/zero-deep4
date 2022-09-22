import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


runs = 200  # 実験回数
steps = 1000  # 1回の実験で試行する回数
epsilon = 0.1
all_rates = np.zeros((runs, steps))  # (2000, 1000)

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates  # 1. 報酬の結果を記録

avg_rates = np.average(all_rates, axis=0)  # 2. 各ステップにおける平均を求める

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()
