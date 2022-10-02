import numpy as np
from dezero import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    """
    ローゼンブロック関数
    この関数は(x0, x1) = (1, 1)のとき、y = 0で最小値となる。
    """
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2  # type: ignore
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001
iters = 10000

for i in range(iters):
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad.data  # type: ignore
    x1.data -= lr * x1.grad.data  # type: ignore

print(x0, x1)
