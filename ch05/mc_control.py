if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs

import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld
# from common.utils import greedy_probs


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    """
    行動を決定する確率分布を返す
    """

    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()  # 環境の初期化
    agent.reset()  # エージェントの初期化

    while True:
        action = agent.get_action(state)  # 行動を選択
        next_state, reward, done = env.step(action)  # 行動を起こし、報酬を得て次の状態に遷移

        agent.add(state, action, reward)  # 状態、行動、報酬を履歴に追加
        if done:
            agent.update()  # 1エピソード終了時にQ関数を更新
            break

        state = next_state

env.render_q(agent.Q)
