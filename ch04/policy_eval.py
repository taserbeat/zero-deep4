if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi: defaultdict, V: defaultdict, env: GridWorld, gamma=0.9):
    """反復方策の1ステップの評価

    Args:
        pi (defaultdict): 方策
        V (defaultdict): 価値関数
        env (GridWorld): 環境
        gamma (float, optional): 割引率 (Defaults to 0.9)

    Returns:
        更新後の価値関数
    """

    for state in env.states():  # 1. 各状態へアクセス
        if state == env.goal_state:  # 2. ゴールの価値関数は常に0
            V[state] = 0
            continue

        action_probs: dict = pi[state]  # probsはprobabilitiesの略
        new_V = 0

        # 3. 各行動へアクセス
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 4. 新しい価値関数
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi: defaultdict, V: defaultdict, env: GridWorld, gamma: float, threshold=0.001):
    """反復方策で方策評価を行う

    Args:
        pi (defaultdict): 方策
        V (defaultdict): 価値関数
        env (GridWorld): 環境
        gamma (float, optional): 割引率
        threshould (float, optional): 更新量の閾値 この閾値を下回るまで反復方策を繰り返す

    Returns:
        _type_: _description_
    """

    while True:
        old_V = V.copy()  # 更新前の価値関数
        V = eval_onestep(pi, V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 閾値との比較
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # ランダムな方策
    V = defaultdict(lambda: 0)  # 価値関数

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
