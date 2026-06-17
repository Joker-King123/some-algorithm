"""
损失函数
======
损失函数衡量模型预测与真实标签之间的差距。
训练的目标就是最小化这个差距。

对于分类任务，交叉熵损失(Cross-Entropy Loss)是标准选择。
"""

import numpy as np


class CrossEntropyLoss:
    """
    交叉熵损失 (Cross-Entropy Loss)

    公式:
    L = -Σ_c y_true_c * log(y_pred_c)

    其中:
    - y_true: one-hot 编码的真实标签，如 [0, 1, 0, 0, 0]
    - y_pred: 模型预测的概率分布，如 [0.05, 0.85, 0.03, 0.05, 0.02]

    直觉理解:
    交叉熵衡量两个概率分布之间的"距离"。
    - 当预测完全正确时（预测概率=1 在正确类别上），loss = 0
    - 预测越偏离真实分布，loss 越大
    - 预测完全错误（预测概率≈0 在正确类别上），loss → ∞

    为什么用交叉熵而不是均方误差(MSE)？
    1. 交叉熵配合 softmax 产生的梯度是 y_pred - y_true
       （线性！梯度不会饱和）
    2. MSE 配合 softmax 产生的梯度包含 y_pred*(1-y_pred) 因子，
       当 y_pred 接近 0 或 1 时梯度接近 0（梯度消失）

    Softmax + CrossEntropy 的联合梯度推导 (重要！):
    ======================================================
    令 s_i = softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    L = -Σ_k y_k * log(s_k)

    dL/dx_i = Σ_k dL/ds_k * ds_k/dx_i

    其中:
    dL/ds_k = -y_k / s_k
    ds_k/dx_i = s_k * (δ_ki - s_i)  [softmax梯度]

    代入:
    dL/dx_i = Σ_k (-y_k / s_k) * s_k * (δ_ki - s_i)
            = Σ_k -y_k * (δ_ki - s_i)
            = -y_i + s_i * Σ_k y_k
            = s_i - y_i   (因为 Σ_k y_k = 1, y是one-hot)

    所以: dL/dx = y_pred - y_true
    惊人地简洁！这就是为什么 softmax+crossentropy 是黄金组合。
    """

    def __init__(self):
        self.cache = {}

    def forward(self, logits, y_true):
        """
        计算交叉熵损失

        为了数值稳定性，我们在计算 softmax 时减去最大值。
        这不会改变 softmax 结果，但防止 exp 溢出。

        参数:
            logits: 模型原始输出 (未经过 softmax)，形状 (N, num_classes)
            y_true: 真实标签 (one-hot 编码)，形状 (N, num_classes)

        返回:
            loss: 标量，平均交叉熵损失
        """
        N = logits.shape[0]

        # 数值稳定的 softmax 计算
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        softmax_out = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 计算交叉熵: L = -Σ y_true * log(y_pred)
        # 加上一个极小值 epsilon 防止 log(0) = -inf
        eps = 1e-15
        loss = -np.sum(y_true * np.log(softmax_out + eps)) / N

        # 缓存 softmax 输出和真实标签供反向传播使用
        self.cache['softmax_out'] = softmax_out
        self.cache['y_true'] = y_true
        self.cache['N'] = N

        return loss

    def backward(self):
        """
        交叉熵损失的梯度 (与 softmax 合并计算)

        如前推导: dL/d(logits) = (softmax_out - y_true) / N

        除以 N 是因为我们计算的是平均损失。
        """
        softmax_out = self.cache['softmax_out']  # (N, C)
        y_true = self.cache['y_true']  # (N, C)
        N = self.cache['N']

        # 这就是那神奇简洁的公式！
        return (softmax_out - y_true) / N


class MSELoss:
    """
    均方误差损失 (Mean Squared Error)

    公式: L = (1/N) * Σ (y_pred - y_true)^2

    主要用于回归任务。这里提供但手势识别不会用到。
    """

    def __init__(self):
        self.cache = {}

    def forward(self, y_pred, y_true):
        N = y_pred.shape[0]
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.cache['N'] = N
        return np.sum((y_pred - y_true) ** 2) / N

    def backward(self):
        y_pred = self.cache['y_pred']
        y_true = self.cache['y_true']
        N = self.cache['N']
        return 2 * (y_pred - y_true) / N
