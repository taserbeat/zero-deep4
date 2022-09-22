import numpy as np
import matplotlib.pyplot as plt

import typing as t


class Bandit:
    def __init__(self, arms=10) -> None:
        """slot machines

        Args:
            arms (int, optional): number of slot machines. Defaults to 10.
        """

        self.rates = np.random.rand(arms)

    def play(self, arm: t.Union[int, np.ndarray, np.intp]) -> int:
        """play slot machine

        Args:
            arm (int | np.ndarray): select slot number

        Returns:
            int: Win: 1, Lose: 0
        """

        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


bandit = Bandit()


class Agent:
    def __init__(self, epsilon: float, action_size=10) -> None:
        """コンストラクタ

        Args:
            epsilon (float): ε-greedy法におけるランダムに行動する確率
                            epsilon=0.1の場合、10%の確率でランダムな行動を取る

            action_size (int, optional): エージェントが選択できる行動の数
        """

        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

        return

    def update(self, action, reward):
        """行動と報酬から学習する
        """

        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

        return

    def get_action(self):
        """行動を選択する
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))

        return np.argmax(self.Qs)


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()  # 1. 行動を選ぶ
        reward = bandit.play(action)  # 2. 実際にプレイして報酬を得る
        agent.update(action, reward)  # 3. 行動と報酬から学ぶ
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print("total_reward: {}".format(total_reward))

    plt.ylabel('Total reward (= win count)')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Win Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()

    exit(0)
