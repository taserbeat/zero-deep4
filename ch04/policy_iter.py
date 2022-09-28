if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval

import typing as t


def argmax(d: t.Dict[int, int]):
    """[int, int]な辞書から値が最大のキーを返す
    """
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V: defaultdict, env: GridWorld, gamma: float):
    """greedy化した方策を返す

    Args:
        V (defaultdict): 価値関数
        env (GridWorld): 環境
        gamma (float): 割引率

    Returns:
        dict: greedy化した方策
    """
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi


def policy_iter(env: GridWorld, gamma: float, threshold=0.001, is_render=True):
    """方策反復法で「評価」と「改善」を繰り返す

    Args:
        env (GridWorld):環境
        gamma (float): 割引率
        threshold (float, optional): 方策評価を行うときの更新をストップする閾値
        is_render (bool, optional): 方策の評価・過程を描画するかどうかのフラグ

    Returns:
        _type_: _description_
    """

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        # 1. 評価
        V = policy_eval(pi, V, env, gamma, threshold)  # type: ignore

        # 2. 改善
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        # 3. 更新チェック
        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
