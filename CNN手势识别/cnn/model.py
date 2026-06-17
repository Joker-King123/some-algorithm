"""
Sequential 模型
==============
将各个层串联成一个完整的神经网络模型。

Sequential 模型表示一个"线性管道"：
输入 → 层1 → 层2 → ... → 层N → 输出

数据依次流过每一层，每层的输出成为下一层的输入。
"""

import numpy as np
import pickle
import os


class Sequential:
    """
    顺序模型 —— 按顺序堆叠层的容器

    使用方式:
        model = Sequential([
            Conv2D(1, 8, 3),
            ReLU(),
            MaxPool2D(2),
            Conv2D(8, 16, 3),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Dense(3136, 128),
            ReLU(),
            Dense(128, 5),
            Softmax(),
        ])
        loss = model.train_step(x_batch, y_batch, optimizer, loss_fn)
    """

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """添加一层到模型末尾"""
        self.layers.append(layer)
        return self  # 支持链式调用

    def forward(self, x):
        """
        前向传播: 数据依次流过每一层

        参数:
            x: 原始输入数据

        返回:
            模型输出
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_out):
        """
        反向传播: 梯度从最后一层反向流到第一层

        链式法则:
        dL/dx_{i} = dL/dy_{i} * dy_{i}/dx_{i}
        其中 dL/dy_{i} 是从上一层传来的梯度

        参数:
            d_out: 损失函数对模型输出的梯度
        """
        grad = d_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, x_batch, y_batch, optimizer, loss_fn):
        """
        一次完整的训练步骤: 前向 → 计算损失 → 反向 → 更新参数

        参数:
            x_batch: 一批输入数据
            y_batch: 对应的真实标签
            optimizer: 优化器实例
            loss_fn: 损失函数实例

        返回:
            loss: 这批数据的损失值
            accuracy: 这批数据的准确率
        """
        # 步骤1: 前向传播 —— 得到预测
        y_pred = self.forward(x_batch)

        # 步骤2: 计算损失
        loss = loss_fn.forward(y_pred, y_batch)

        # 步骤3: 反向传播 —— 计算所有参数的梯度
        d_loss = loss_fn.backward()
        self.backward(d_loss)

        # 步骤4: 参数更新 —— 优化器根据梯度调整权重
        optimizer.update(self.layers)

        # 步骤5: 计算准确率
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        accuracy = np.mean(pred_labels == true_labels)

        return loss, accuracy

    def predict(self, x):
        """
        预测: 只做前向传播，不计算梯度

        参数:
            x: 输入数据

        返回:
            预测的类别标签
        """
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, x):
        """返回预测的概率分布"""
        return self.forward(x)

    def save(self, filepath):
        """
        保存模型参数到文件

        只保存可学习参数（权重和偏置），
        不保存层结构（加载时需要重新构建模型）。
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[i] = {'W': layer.W, 'b': layer.b}

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"[模型] 参数已保存至: {filepath}")

    def load(self, filepath):
        """
        从文件加载模型参数

        注意: 加载前模型需要先构建好相同的层结构。
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)

        for i, layer in enumerate(self.layers):
            if i in params and hasattr(layer, 'W'):
                # 验证形状匹配
                if layer.W.shape == params[i]['W'].shape:
                    layer.W = params[i]['W']
                    layer.b = params[i]['b']
                else:
                    print(f"[警告] 层 {i} 参数形状不匹配，跳过加载")

        print(f"[模型] 参数已加载自: {filepath}")

    def summary(self):
        """Print model structure summary"""
        print("\n" + "=" * 70)
        print(f"{'Layer':<25} {'Output Shape':<25} {'Params':<10}")
        print("=" * 70)

        total_params = 0
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            n_params = 0
            out_shape = "?"

            if hasattr(layer, 'W'):
                n_params = layer.W.size + layer.b.size
                if hasattr(layer, 'out_channels'):
                    out_shape = f"(N, {layer.out_channels}, H, W)"
                else:
                    out_shape = f"(N, {layer.out_features})"
            elif name == 'MaxPool2D':
                out_shape = "(N, C, H/2, W/2)"
            elif name == 'ReLU':
                out_shape = "(same)"
            elif name == 'Softmax':
                out_shape = "(N, num_classes)"
            elif name == 'Flatten':
                out_shape = "(N, C*H*W)"

            total_params += n_params
            param_str = f"{n_params:,}" if n_params > 0 else "-"
            print(f"{i}: {name:<23} {out_shape:<25} {param_str:<10}")

        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print("=" * 70 + "\n")
