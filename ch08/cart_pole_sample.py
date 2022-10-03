import gym

env = gym.make("CartPole-v1")

state = env.reset()
print(state)  # 初期状態
# 状態は4つの要素を持つ配列であり、順に以下が格納されている
# カートの位置座標、カートの速度、棒の角度、棒の角速度

action_space = env.action_space
print(action_space)  # 行動の次元数
# 行動は左に移動と右に移動の2種類がある
# 左に移動: 0、右に移動: 1

action = 0
next_state, reward, done, info, _ = env.step(action=action)
print(next_state)
