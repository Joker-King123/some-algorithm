"""
优化器
=====
优化器负责根据梯度更新模型参数。

梯度下降的基本公式:
  w_new = w_old - lr * dL/dw

不同的优化器在这个基本公式上添加了"助推器"(momentum)和
"自适应学习率"等改进。
"""

import numpy as np


class SGD:
    """
    带动量的随机梯度下降 (Stochastic Gradient Descent with Momentum)

    基本 SGD:
      w = w - lr * dw

    问题:
    - 在某些方向上震荡（如峡谷地形）
    - 收敛速度受学习率限制

    动量 (Momentum) 的改进:
      v = momentum * v - lr * dw    ← 速度更新（指数移动平均）
      w = w + v                      ← 位置更新

    物理直觉:
    想象一个球在山谷中滚:
    - 梯度是重力（指向最陡下降方向）
    - 速度累积之前的运动方向
    - 球不会立即转向，而是保持一定的"惯性"
    - 这有助于: 1) 加速收敛 2) 逃离局部最小值 3) 减少震荡

    参数:
        lr: 学习率 (learning rate) —— 控制每一步的步长
        momentum: 动量系数 (0~1) —— 越大，惯性越强
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}  # 为每个层存储速度

    def update(self, layers):
        """
        更新所有层的参数

        参数:
            layers: 模型的所有层（包括激活层，它们不会更新）
        """
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'W'):
                continue  # 跳过没有权重的层（激活、池化、展平）

            # 为该层的每个参数初始化速度
            if i not in self.velocities:
                self.velocities[i] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b),
                }

            # ---- 动量更新 ----
            # 速度 = 动量 * 旧速度 - 学习率 * 梯度
            self.velocities[i]['W'] = (
                self.momentum * self.velocities[i]['W']
                - self.lr * layer.dW
            )
            self.velocities[i]['b'] = (
                self.momentum * self.velocities[i]['b']
                - self.lr * layer.db
            )

            # 权重 = 权重 + 速度
            layer.W += self.velocities[i]['W']
            layer.b += self.velocities[i]['b']


class Adam:
    """
    Adam (Adaptive Moment Estimation) —— 最流行的优化器之一

    Adam 结合了两种思想:
    1. Momentum: 用梯度的指数移动平均(EMA)来平滑更新方向
    2. RMSprop: 用梯度平方的EMA来自适应调整每个参数的学习率

    算法步骤:
    m = β1 * m + (1-β1) * dw      ← 一阶矩(均值)的EMA
    v = β2 * v + (1-β2) * dw^2    ← 二阶矩(方差)的EMA
    m_hat = m / (1-β1^t)          ← 偏差校正
    v_hat = v / (1-β2^t)          ← 偏差校正
    w = w - lr * m_hat / (√v_hat + ε)  ← 参数更新

    为什么需要偏差校正？
    初始时 m=0, v=0，它们的估计偏向0。
    随着 t 增大，1-β^t → 1，校正因子趋近于1，影响减小。

    直觉:
    - m: "梯度指向哪个方向？"（一阶矩——均值）
    - v: "梯度在这个方向上有多不确定？"（二阶矩——方差）
    - 如果梯度一直很大(m大)且稳定(v小) → 大步伐
    - 如果梯度震荡很大(v大) → 小步伐，谨慎前进

    参数:
        lr: 学习率 (通常 0.001)
        beta1: 一阶矩衰减率 (默认 0.9)
        beta2: 二阶矩衰减率 (默认 0.999)
        eps: 防止除零的小常数 (默认 1e-8)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}      # 一阶矩
        self.v = {}      # 二阶矩
        self.t = {}      # 时间步计数器
        self.layers = None

    def update(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'W'):
                continue

            # 初始化
            if i not in self.m:
                self.m[i] = {'W': np.zeros_like(layer.W),
                             'b': np.zeros_like(layer.b)}
                self.v[i] = {'W': np.zeros_like(layer.W),
                             'b': np.zeros_like(layer.b)}
                self.t[i] = 0

            self.t[i] += 1
            t = self.t[i]
            beta1, beta2, eps = self.beta1, self.beta2, self.eps

            for param_name in ['W', 'b']:
                grad = layer.dW if param_name == 'W' else layer.db

                # 更新一阶矩: 梯度的 EMA
                self.m[i][param_name] = (
                    beta1 * self.m[i][param_name] + (1 - beta1) * grad
                )
                # 更新二阶矩: 梯度平方的 EMA
                self.v[i][param_name] = (
                    beta2 * self.v[i][param_name] + (1 - beta2) * grad**2
                )

                # 偏差校正
                m_hat = self.m[i][param_name] / (1 - beta1**t)
                v_hat = self.v[i][param_name] / (1 - beta2**t)

                # 参数更新
                if param_name == 'W':
                    layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                else:
                    layer.b -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
