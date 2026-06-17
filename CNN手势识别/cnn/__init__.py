"""
CNN 从零实现库
==============
这个库从零实现了卷积神经网络(CNN)的核心组件，
仅使用 NumPy 进行矩阵运算，不使用 TensorFlow/PyTorch 等高层框架。

通过阅读这些代码和注释，你可以深入理解 CNN 的数学原理和实现细节。

包含的模块:
- layers: 卷积层、池化层、展平层、全连接层 (含前向传播和反向传播)
- activations: ReLU、Softmax 激活函数
- loss: 交叉熵损失函数
- optimizer: SGD、Adam 优化器
- model: Sequential 模型组装器
"""

from .layers import Conv2D, MaxPool2D, Flatten, Dense
from .activations import ReLU, Softmax
from .loss import CrossEntropyLoss
from .optimizer import SGD, Adam
from .model import Sequential

__all__ = [
    'Conv2D', 'MaxPool2D', 'Flatten', 'Dense',
    'ReLU', 'Softmax',
    'CrossEntropyLoss',
    'SGD', 'Adam',
    'Sequential',
]
