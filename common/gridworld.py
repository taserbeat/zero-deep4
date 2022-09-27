import numpy as np
from common.gridworld_render import Renderer


class GridWorld:
    """
    グリッドワールドのクラス
    """

    def __init__(self):
        # 行動の候補
        self.action_space = [0, 1, 2, 3]

        # 行動の内容説明
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # 各マスを踏んだときに得られる報酬マップ
        self.reward_map = np.array(
            [
                [0, 0, 0, 1.0],
                [0, None, 0, -1.0],
                [0, 0, 0, 0]
            ]
        )

        self.goal_state = (0, 3)  # ゴールのマス
        self.wall_state = (1, 1)  # 壁のマス (移動できない)
        self.start_state = (2, 0)  # スタート地点のマス
        self.agent_state = self.start_state  # エージェントのスタート位置を設定

    @property
    def height(self):
        """グリッドワールドの縦サイズ
        """
        return len(self.reward_map)

    @property
    def width(self):
        """グリッドワールドの横サイズ
        """
        return len(self.reward_map[0])

    @property
    def shape(self):
        """グリッドワールドのshape
        """
        return self.reward_map.shape

    def actions(self):
        """エージェントが行動する選択肢
        """
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        # 1. 移動先のの場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 移動先がグリッドワールドの枠の外か、それとも移動先が壁か?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            # 移動先がグリッドワールドの外の場合、移動前のマスに留まる
            next_state = state
        elif next_state == self.wall_state:
            # 移動先が壁の場合、移動前のマスに留まる
            next_state = state

        # 3. 次の状態を返す
        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_q(q, print_value)
