"""
激活函数
======
激活函数为神经网络引入非线性，这是深度学习的关键。

没有激活函数（或用线性激活），多层网络等价于单层：
  如果 f(x) = x, 则 W2(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)
  这仍然是线性的！多层失去意义。

有了非线性激活，网络才能学习复杂的非线性映射。
"""

import numpy as np


class ReLU:
    """
    ReLU (Rectified Linear Unit) —— 最常用的激活函数

    公式: f(x) = max(0, x)

    为什么 ReLU 好用？
    1. 计算简单: 只需一次比较操作
    2. 缓解梯度消失: 正半区梯度恒为1，不会像 sigmoid/tanh 那样饱和
    3. 稀疏激活: 负半区输出为0，网络自动学习稀疏表示
    4. 生物学合理性: 类似真实神经元的"全或无"发放特性

    缺点 —— "Dying ReLU" 问题:
    如果某个神经元总是输出负值，它的梯度恒为0，权重永不更新。
    解决方案: Leaky ReLU, PReLU, ELU 等变体。

    反向传播:
    dL/dx = dL/dy * (1 if x > 0 else 0)
    即: 梯度只在 x > 0 时通过，否则被切断
    """

    def __init__(self):
        self.cache = {}

    def forward(self, x):
        """
        ReLU 前向传播: 将所有负值置为0

        参数:
            x: 任意形状的 numpy 数组

        返回:
            max(0, x)，同形状
        """
        self.cache['x'] = x
        return np.maximum(0, x)

    def backward(self, d_out):
        """
        ReLU 反向传播

        梯度乘以 (x > 0) 的指示函数:
        - x > 0: 梯度原样通过
        - x <= 0: 梯度被截断为 0（Dying ReLU 的根源）
        """
        x = self.cache['x']
        return d_out * (x > 0)

    def update(self, lr):
        """ReLU 没有可学习参数"""
        pass


class Softmax:
    """
    Softmax —— 将任意实数向量映射为概率分布

    公式: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    性质:
    1. 输出值在 (0, 1) 之间
    2. 所有输出之和 = 1（形成有效的概率分布）
    3. 保持输入的大小顺序（单调性）
    4. 差值越大 → softmax 输出越接近 one-hot（越大越接近1）

    数值稳定性技巧:
    直接计算 exp(x_i) 可能溢出（x_i 很大时）。
    解决方法: 减去最大值再计算
      softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    这不会改变结果（分子分母同除 exp(max(x))），但数值稳定。

    注意: 本类只实现 softmax 的前向传播。
    反向传播通常与交叉熵损失合并计算，在 loss.py 中处理。
    因为 softmax + cross-entropy 的联合梯度非常简单:
      dL/dx = y_pred - y_true  （惊人的简洁！）
    """

    def __init__(self):
        self.cache = {}

    def forward(self, x):
        """
        Softmax 前向传播

        参数:
            x: 输入 logits，形状 (N, num_classes)

        返回:
            概率分布，形状 (N, num_classes)，每行和为1
        """
        # 数值稳定性: 每行减去该行最大值
        x_shifted = x - np.max(x, axis=1, keepdims=True)

        # 计算指数
        exp_x = np.exp(x_shifted)

        # 归一化: 除以每行之和
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.cache['out'] = out
        return out

    def backward(self, d_out):
        """
        Softmax 反向传播（独立版本）

        注意: 当 softmax 后面紧跟交叉熵损失时，
        应该使用 CrossEntropyLoss 中的合并梯度计算，
        因为合并计算更稳定且更高效。

        这里提供独立版本的梯度计算供学习参考。
        Softmax 的雅可比矩阵:
        ds_i/dx_j = s_i * (δ_ij - s_j)
        其中 δ_ij 是克罗内克δ (i=j时为1，否则为0)

        雅可比矩阵是 (N, C, C) —— 每对输出-输入之间都有梯度联系！
        这是因为改变一个输入会影响所有输出（归一化分母变了）。
        """
        s = self.cache['out']  # (N, C)
        N, C = s.shape

        # 创建雅可比矩阵: (N, C, C)
        # ds_i/dx_j = -s_i * s_j (for i != j)
        # ds_i/dx_i = s_i * (1 - s_i) (for i == j)
        # = s_i * (δ_ij - s_j)

        # δ_ij (N, C, C): 对角线为1的单位矩阵
        dX = np.zeros((N, C, C))

        for i in range(C):
            for j in range(C):
                if i == j:
                    dX[:, i, j] = s[:, i] * (1 - s[:, i])
                else:
                    dX[:, i, j] = -s[:, i] * s[:, j]

        # dL/dx_k = Σ_i dL/ds_i * ds_i/dx_k
        # (N, C) @ (N, C, C) → (N, C)
        out = np.einsum('ni,nij->nj', d_out, dX)

        return out

    def update(self, lr):
        """Softmax 没有可学习参数"""
        pass
