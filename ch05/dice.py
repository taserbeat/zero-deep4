import numpy as np


def sample(dices=2) -> int:
    """ダイスをn回振って出た目の合計を返す

    Args:
        dices (int, optional): ダイスを振る回数

    Returns:
        int: 合計値
    """
    x = 0
    for _ in range(dices):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x


trial = 1000
V, n = 0, 0

for i, _ in enumerate(range(trial), start=1):
    s = sample()
    n += 1
    V += (s - V) / n
    print(f"#{i}: {V}")
