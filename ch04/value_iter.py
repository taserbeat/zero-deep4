if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy


def value_iter_onestep(V: defaultdict, env: GridWorld, gamma: float):
    """価値反復法の1回あたりの更新処理

    Args:
        V (defaultdict): 価値関数
        env (GridWorld): 環境
        gamma (float): 割引率

    Returns:
        _type_: 更新後の価値関数
    """

    for state in env.states():  # 1. すべての状態にアクセス
        if state == env.goal_state:
            # ゴールの価値関数は常に0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():  # 2. すべての行動にアクセス
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  # 3. 新しい価値関数
            action_values.append(value)

        V[state] = max(action_values)  # 4. 最大値を取り出す
    return V


def value_iter(V: defaultdict, env: GridWorld, gamma: float, threshold=0.001, is_render=True):
    """価値反復法を繰り返す

    Args:
        V (defaultdict): 価値関数
        env (GridWorld): 環境
        gamma (float): 割引率
        threshold (float, optional): 価値関数の更新をストップするための閾値
        is_render (bool, optional): 評価・改善を行う過程を描画するかを表すフラグ

    Returns:
        _type_: _description_
    """

    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
