"""
CNN 核心层实现
=============
本文件实现了卷积神经网络的核心层，每一层都包含：
1. __init__(): 初始化权重和偏置
2. forward(): 前向传播 —— 从输入计算输出
3. backward(): 反向传播 —— 计算损失函数对输入和参数的梯度

核心概念：im2col (Image to Column)
---------------------------------
卷积操作本质上是一个滑动窗口操作。直接实现需要用多重循环，
效率很低。im2col 技巧将卷积转化为矩阵乘法，既高效又容易理解。

im2col 的思想：
  输入图像 → 将每个卷积窗口展开成一列 → 得到一个大矩阵(列矩阵)
  卷积核 → 将每个滤波器的权重展开成一行 → 得到权重矩阵
  卷积操作 = 权重矩阵 × 列矩阵 → 矩阵乘法！

反向传播时，col2im 将列矩阵的梯度重新映射回图像形状。

数据格式说明：
  本实现使用 (N, C, H, W) 格式:
  - N: batch size (批次大小)
  - C: channels (通道数，灰度图=1, RGB=3)
  - H: height (高度)
  - W: width (宽度)
"""

import numpy as np


# ============================================================================
#                           im2col / col2im 辅助函数
# ============================================================================

def im2col(x, kernel_h, kernel_w, stride, pad):
    """
    将图像转换为列矩阵 (Image to Column)

    这是 CNN 中最关键的技巧之一。让我用一个具体例子说明：

    假设输入是一张 4×4 的单通道图像:
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10,11,12],
         [13,14,15,16]]

    使用 3×3 卷积核 (stride=1, pad=0):
    输出大小 = (4-3)/1 + 1 = 2×2

    卷积核在图像上滑动了 4 个位置，im2col 将每个位置
    对应的 3×3 区域展平成一列:

    位置(0,0): [1,2,3,5,6,7,9,10,11]^T
    位置(0,1): [2,3,4,6,7,8,10,11,12]^T
    位置(1,0): [5,6,7,9,10,11,13,14,15]^T
    位置(1,1): [6,7,8,10,11,12,14,15,16]^T

    结果矩阵形状: (1*9, 4) = (9, 4)
    卷积核展平: (1, 9)
    矩阵乘法: (1, 9) × (9, 4) = (1, 4) → reshape → (2, 2) ✓

    参数:
        x: 输入数据，形状 (N, C, H, W)
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        stride: 步长
        pad: 填充大小

    返回:
        col: 列矩阵，形状 (N, C*KH*KW, OH*OW)
        OH: 输出高度
        OW: 输出宽度
    """
    N, C, H, W = x.shape

    # 步骤1: 如果需要，对输入进行零填充(zero-padding)
    # padding 的作用：控制输出尺寸，保留边缘信息
    if pad > 0:
        # 在 H 和 W 维度两侧各填充 pad 个零
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode='constant'
        )
    else:
        x_padded = x

    H_padded = H + 2 * pad
    W_padded = W + 2 * pad

    # 步骤2: 计算输出尺寸
    # 公式: OH = (H + 2*pad - KH) / stride + 1
    OH = (H_padded - kernel_h) // stride + 1
    OW = (W_padded - kernel_w) // stride + 1

    # 步骤3: 使用广播(broadcasting)创建索引矩阵
    # 这避开了 Python 循环，利用 NumPy 的 C 级循环加速

    # i0: 卷积核内的高度偏移 [0,0,0, 1,1,1, 2,2,2] (对于 3×3 核)
    # 每个高度值重复 kernel_w 次
    i0 = np.repeat(np.arange(kernel_h), kernel_w)  # 形状: (KH*KW,)
    i0 = np.tile(i0, C)  # 对每个通道重复 → (C*KH*KW,)

    # i1: 每个输出行对应的输入起始位置
    # [0,0, 1,1] * stride (对于 OH=2, OW=2, stride=1)
    i1 = stride * np.repeat(np.arange(OH), OW)  # 形状: (OH*OW,)

    # 完整的高度索引: i0的每个元素 + i1的每个元素
    # 形状: (C*KH*KW, OH*OW)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    # j0: 卷积核内的宽度偏移 [0,1,2, 0,1,2, 0,1,2] (对于 3×3 核)
    j0 = np.tile(np.arange(kernel_w), kernel_h * C)  # 形状: (C*KH*KW,)

    # j1: 每个输出列对应的输入起始位置
    j1 = stride * np.tile(np.arange(OW), OH)  # 形状: (OH*OW,)

    # 完整的宽度索引
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # 形状: (C*KH*KW, OH*OW)

    # c: 通道索引，每个通道对应 kernel_h*kernel_w 行
    c = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)
    # 形状: (C*KH*KW, 1)

    # 步骤4: 使用高级索引一次性提取所有列
    # x_padded[:, c, i, j] 的形状: (N, C*KH*KW, OH*OW)
    # 这就是我们要的列矩阵！
    col = x_padded[:, c, i, j]

    return col, OH, OW


def col2im(col, input_shape, kernel_h, kernel_w, stride, pad):
    """
    将列矩阵的梯度转换回图像形状 (Column to Image)

    这是 im2col 的逆操作，用于反向传播。
    因为每个输入像素可能被多个卷积窗口共用，
    所以需要将梯度累加回原位置。

    参数:
        col: 列矩阵梯度，形状 (N, C*KH*KW, OH*OW)
        input_shape: 原始输入形状 (N, C, H, W)
        kernel_h, kernel_w: 卷积核尺寸
        stride: 步长
        pad: 填充大小

    返回:
        图像形状的梯度，形状 (N, C, H, W)
    """
    N, C, H, W = input_shape

    H_padded = H + 2 * pad
    W_padded = W + 2 * pad

    OH = (H_padded - kernel_h) // stride + 1
    OW = (W_padded - kernel_w) // stride + 1

    # 创建零填充的图像梯度
    x_padded_grad = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)

    # 使用与 im2col 相同的索引
    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(OH), OW)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(kernel_w), kernel_h * C)
    j1 = stride * np.tile(np.arange(OW), OH)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    c = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)

    # np.add.at: 无缓冲的原地累加
    # 因为同一像素可能被多个窗口包含，所以需要累加
    # 例如：中心像素属于所有 9 个卷积窗口（对于 3×3 核）
    np.add.at(x_padded_grad, (slice(None), c, i, j), col)

    # 去掉填充
    if pad > 0:
        return x_padded_grad[:, :, pad:-pad, pad:-pad]
    return x_padded_grad


# ============================================================================
#                           Conv2D: 二维卷积层
# ============================================================================

class Conv2D:
    """
    二维卷积层 —— CNN 的核心！

    卷积层的作用是检测图像中的局部模式（边缘、纹理、形状等）。

    工作原理:
    1. 一个小的滤波器(卷积核)在输入图像上滑动
    2. 每个位置，滤波器与图像局部区域做元素乘法后求和
    3. 不同的滤波器检测不同的特征

    例如，下面是一个 3×3 边缘检测滤波器:
        [[-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]]
    当它滑过图像时，在边缘处会产生大的响应值。

    参数说明:
        in_channels: 输入通道数（第一层灰度图=1, RGB=3）
        out_channels: 输出通道数（=滤波器数量，每个滤波器学习一种特征）
        kernel_size: 卷积核尺寸（如 3 表示 3×3）
        stride: 滑动步长（默认1，大于1会下采样）
        padding: 边缘填充（默认0，"valid"卷积；填1为"same"卷积的近似）
        lr: 该层的学习率（可覆盖全局学习率）
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, lr=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) \
            else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.lr = lr  # 如果为 None，则使用优化器的学习率

        KH, KW = self.kernel_size

        # ================================================================
        # He 初始化 (Kaiming Initialization)
        #
        # 为什么需要特殊初始化？
        #   如果权重初始化为0，所有神经元输出相同，无法学习
        #   如果权重随机太大，激活值会爆炸 → 梯度爆炸
        #   如果权重随机太小，激活值会消失 → 梯度消失
        #
        # He 初始化的公式:
        #   W ~ N(0, sqrt(2 / fan_in))
        #   其中 fan_in = in_channels * KH * KW
        #
        # 这个公式保证了前向传播时各层输出的方差不变，
        # 对于 ReLU 激活函数特别有效。
        # ================================================================
        fan_in = in_channels * KH * KW
        std = np.sqrt(2.0 / fan_in)  # He 初始化的标准差

        # 权重形状: (out_channels, in_channels, KH, KW)
        # 每个输出通道有 in_channels*KH*KW 个权重
        self.W = np.random.randn(out_channels, in_channels, KH, KW) * std

        # 偏置初始化为0，每个输出通道一个偏置
        self.b = np.zeros(out_channels)

        # 缓存前向传播的中间结果，供反向传播使用
        self.cache = {}

    def forward(self, x):
        """
        卷积层前向传播

        计算步骤:
        1. im2col: 将输入图像转为列矩阵
        2. 将卷积核权重展平为矩阵
        3. 矩阵乘法: output = W_flat @ col
        4. 加上偏置并 reshape

        数学公式:
        output[n, oc, oh, ow] = Σ_{ic, kh, kw}
            W[oc, ic, kh, kw] * x[n, ic, oh*stride+kh, ow*stride+kw]
            + b[oc]

        参数:
            x: 输入，形状 (N, C_in, H, W)

        返回:
            输出，形状 (N, C_out, H_out, W_out)
        """
        self.cache['x_shape'] = x.shape

        N, C, H, W = x.shape
        KH, KW = self.kernel_size

        # 步骤1: im2col —— 将滑窗操作转为矩阵乘法
        col, OH, OW = im2col(x, KH, KW, self.stride, self.padding)
        # col 形状: (N, C*KH*KW, OH*OW)

        # 步骤2: 将权重展平为 (out_channels, C*KH*KW)
        W_flat = self.W.reshape(self.out_channels, -1)

        # 步骤3: 矩阵乘法 —— 这就是卷积！
        # W_flat @ col: (OC, K) @ (N, K, P) → (N, OC, P)
        # 其中 K = C*KH*KW, P = OH*OW
        # 使用爱因斯坦求和约定进行批量矩阵乘法
        # out[n, o, p] = Σ_k W_flat[o, k] * col[n, k, p]
        out = np.einsum('ok,nkp->nop', W_flat, col)

        # 步骤4: 重塑输出并加上偏置
        # out 当前形状: (N, OC, OH*OW)
        out = out.reshape(N, self.out_channels, OH, OW)
        # 加上偏置: b 形状 (OC,) 自动广播到 (N, OC, OH, OW)
        out += self.b.reshape(1, -1, 1, 1)

        # 缓存中间结果供反向传播使用
        self.cache['col'] = col
        self.cache['W_flat'] = W_flat
        self.cache['OH'] = OH
        self.cache['OW'] = OW

        return out

    def backward(self, d_out):
        """
        卷积层反向传播

        需要计算三个梯度:
        1. dW: 损失对权重的梯度 → 用于更新权重
        2. db: 损失对偏置的梯度 → 用于更新偏置
        3. dX: 损失对输入的梯度 → 传给前一层

        数学推导:
        --------
        令 L 为损失函数, Y 为卷积输出

        (1) 对权重的梯度 dL/dW:
        Y = W * X (简化表示，* 是卷积操作)
        dL/dW = dL/dY * X^T (卷积变相关)

        在 im2col 框架下:
        Y_flat = W_flat @ col
        dL/dW_flat = dL/dY_flat @ col^T

        (2) 对偏置的梯度 dL/db:
        Y = W*X + b
        dL/db = Σ dL/dY (在 N, H, W 维度求和)

        (3) 对输入的梯度 dL/dX:
        Y = W*X
        dL/dX = W_rotated * dL/dY (卷积核旋转180°的相关操作)

        在 im2col 框架下:
        dL/d(col) = W_flat^T @ dL/dY_flat
        dL/dX = col2im(dL/d(col))

        参数:
            d_out: 损失对输出的梯度，形状 (N, OC, OH, OW)

        返回:
            dX: 损失对输入的梯度，形状 (N, C, H, W)
        """
        N, OC, OH, OW = d_out.shape

        col = self.cache['col']
        W_flat = self.cache['W_flat']

        # (1) 计算 dW: 损失对权重的梯度
        # ----------------------------------------------------------------
        # d_out 展平: (N, OC, OH*OW)
        # col 转置:   (N, OH*OW, C*KH*KW)
        # dW = d_out_flat @ col^T → 对 batch 求和
        # 形状: (OC, C*KH*KW) → reshape → (OC, C, KH, KW)
        d_out_flat = d_out.reshape(N, OC, -1)  # (N, OC, OH*OW)

        # 爱因斯坦求和: dW[o, k] = Σ_{n, p} d_out_flat[n, o, p] * col[n, k, p]
        self.dW = np.einsum('nop,nkp->ok', d_out_flat, col)
        self.dW = self.dW.reshape(self.W.shape)

        # (2) 计算 db: 损失对偏置的梯度
        # ----------------------------------------------------------------
        # 偏置加到每个输出通道的每个位置
        # db[oc] = Σ_{n, oh, ow} d_out[n, oc, oh, ow]
        self.db = np.sum(d_out, axis=(0, 2, 3))  # 形状: (OC,)

        # (3) 计算 dX: 损失对输入的梯度
        # ----------------------------------------------------------------
        # d_col = W_flat^T @ d_out_flat
        # d_col[n, k, p] = Σ_{o} W_flat[o, k] * d_out_flat[n, o, p]
        d_col = np.einsum('ok,nop->nkp', W_flat, d_out_flat)
        # d_col 形状: (N, C*KH*KW, OH*OW)

        # col2im: 将列梯度映射回图像空间
        KH, KW = self.kernel_size
        dX = col2im(d_col, self.cache['x_shape'],
                    KH, KW, self.stride, self.padding)

        return dX

    def update(self, lr):
        """使用梯度下降更新权重和偏置"""
        actual_lr = self.lr if self.lr is not None else lr
        self.W -= actual_lr * self.dW
        self.b -= actual_lr * self.db


# ============================================================================
#                        MaxPool2D: 最大池化层
# ============================================================================

class MaxPool2D:
    """
    最大池化层 —— 降采样，提取主要特征

    池化层的作用:
    1. 降采样(Downsampling): 减小特征图尺寸，降低计算量
    2. 平移不变性: 小位移不影响池化结果
    3. 扩大感受野: 后续层能看到更大的输入区域

    最大池化 vs 平均池化:
    - 最大池化: 取窗口中的最大值 —— 保留最显著的特征
    - 平均池化: 取窗口中的平均值 —— 保留整体信息

    最大池化更常用，因为它保留了"是否有某特征存在"
    这个关键信息，而不是"特征有多强"。

    参数:
        pool_size: 池化窗口大小（2 表示 2×2）
        stride: 步长（默认等于 pool_size，即不重叠的窗口）
    """

    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) \
            else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.cache = {}

    def forward(self, x):
        """
        最大池化前向传播

        对于每个池化窗口:
        output = max(窗口内的所有值)

        同时记录最大值的位置(索引)，供反向传播使用。
        因为反向传播时，梯度只流向最大值位置。

        参数:
            x: 输入，形状 (N, C, H, W)

        返回:
            输出，形状 (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        PH, PW = self.pool_size
        stride_h, stride_w = self.stride

        # 计算输出尺寸
        OH = (H - PH) // stride_h + 1
        OW = (W - PW) // stride_w + 1

        # 创建索引 —— 与 im2col 类似的思想
        # i: 池化窗口内的高度偏移 + 窗口起始位置
        i0 = np.repeat(np.arange(PH), PW)  # [0,0, 1,1] for 2x2
        i1 = stride_h * np.repeat(np.arange(OH), OW)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # (PH*PW, OH*OW)

        j0 = np.tile(np.arange(PW), PH)  # [0,1, 0,1] for 2x2
        j1 = stride_w * np.tile(np.arange(OW), OH)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # (PH*PW, OH*OW)

        # 提取所有池化窗口的值
        # patches 形状: (N, C, PH*PW, OH*OW)
        patches = x[:, :, i, j]

        # 对每个窗口取最大值
        out = np.max(patches, axis=2)  # 沿窗口维度取max → (N, C, OH*OW)
        out = out.reshape(N, C, OH, OW)

        # 记录每个窗口中最大值的位置(one-hot形式)
        # 用于反向传播时确定梯度去向
        max_mask = (patches == np.max(patches, axis=2, keepdims=True))
        # 如果有多个最大值，只取第一个（用 argmax）
        argmax = np.argmax(patches, axis=2)  # (N, C, OH*OW) - 每个窗口最大值的位置索引

        self.cache['x_shape'] = x.shape
        self.cache['i'] = i
        self.cache['j'] = j
        self.cache['argmax'] = argmax
        self.cache['OH'] = OH
        self.cache['OW'] = OW

        return out

    def backward(self, d_out):
        """
        最大池化反向传播

        关键理解：梯度只流向池化窗口中最大值的位置！

        为什么？
        因为只有最大值参与了最终输出，改变非最大值
        不会影响输出，所以它们对损失的梯度为0。

        这是最大池化的一个重要特性：
        梯度信号非常稀疏，只有最大值位置能收到梯度。

        参数:
            d_out: 损失对池化输出的梯度，形状 (N, C, OH, OW)

        返回:
            dX: 损失对池化输入的梯度，形状 (N, C, H, W)
        """
        N, C, H, W = self.cache['x_shape']
        PH, PW = self.pool_size

        d_out_flat = d_out.reshape(N, C, -1)  # (N, C, OH*OW)

        # 创建一个零梯度数组
        dX_col = np.zeros((N, C, PH * PW, d_out_flat.shape[-1]),
                          dtype=d_out.dtype)

        # 将 d_out 的值放到 argmax 位置（每个窗口的最大值位置）
        # 创建批量索引
        n_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
        c_idx = np.arange(C)[None, :, None]  # (1, C, 1)
        p_idx = np.arange(d_out_flat.shape[-1])[None, None, :]  # (1, 1, OH*OW)

        # argmax 形状: (N, C, OH*OW)
        # 使用高级索引放置梯度
        n_idx_bc = np.broadcast_to(n_idx, (N, C, d_out_flat.shape[-1]))
        c_idx_bc = np.broadcast_to(c_idx, (N, C, d_out_flat.shape[-1]))

        dX_col[n_idx_bc, c_idx_bc, self.cache['argmax'], p_idx] = d_out_flat

        # col2im: 将列梯度映射回图像空间
        dX = np.zeros((N, C, H, W), dtype=d_out.dtype)
        np.add.at(dX, (slice(None), slice(None),
                       self.cache['i'], self.cache['j']), dX_col)

        return dX

    def update(self, lr):
        """池化层没有可学习参数，此方法为空"""
        pass


# ============================================================================
#                        Flatten: 展平层
# ============================================================================

class Flatten:
    """
    展平层 —— 将多维特征图变为一维向量

    为什么需要展平？
    CNN 的卷积层输出是多维的 (N, C, H, W)，
    但全连接层(Dense)需要一维输入。
    Flatten 层架起了 CNN 特征提取器和分类器之间的桥梁。

    例如:
    输入: (32, 16, 14, 14)  # 32个样本, 16通道, 14×14特征图
    输出: (32, 3136)        # 32个样本, 每个3136维向量
           ↑ 3136 = 16 × 14 × 14
    """

    def __init__(self):
        self.cache = {}

    def forward(self, x):
        """
        展平前向传播

        保持 batch 维度不变，将其他维度展平。
        reshape 操作不改变数据，只改变"视角"。
        """
        self.cache['input_shape'] = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)  # (N, C*H*W)

    def backward(self, d_out):
        """
        展平反向传播

        反向操作：将一维梯度 reshape 回原始多维形状。
        因为 reshape 没有改变数据排列，反向传播只需恢复形状。
        """
        return d_out.reshape(self.cache['input_shape'])

    def update(self, lr):
        """展平层没有可学习参数"""
        pass


# ============================================================================
#                        Dense: 全连接层
# ============================================================================

class Dense:
    """
    全连接层 (Fully Connected / Linear Layer)

    最经典的神经网络层：每个输入神经元连接到每个输出神经元。

    数学公式:
    y = x @ W + b

    其中:
    - x: 输入向量，形状 (N, D_in)
    - W: 权重矩阵，形状 (D_in, D_out)
    - b: 偏置向量，形状 (D_out,)
    - y: 输出向量，形状 (N, D_out)

    反向传播推导:
    -----------------
    y = x @ W + b

    (1) dL/dW = x^T @ dL/dy
        因为: dy/dW = x (从 y_i = Σ_j x_j * W_{ji} 推导)
        所以: dL/dW_{ji} = Σ_n x_{nj} * dL/dy_{ni}
        矩阵形式: dL/dW = x^T @ dL/dy

    (2) dL/db = Σ_n dL/dy_n (沿 batch 维度求和)
        因为每个样本都加了同一个 b

    (3) dL/dx = dL/dy @ W^T
        因为: dy/dx = W (从 y_i = Σ_j x_j * W_{ji} 推导)
        所以: dL/dx_j = Σ_i dL/dy_i * W_{ji}
        矩阵形式: dL/dx = dL/dy @ W^T
    """

    def __init__(self, in_features, out_features, lr=None):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        # He 初始化: 保证前向传播方差不变，适配 ReLU
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * std
        self.b = np.zeros(out_features)

        self.cache = {}

    def forward(self, x):
        """
        全连接层前向传播: y = x @ W + b

        参数:
            x: 输入，形状 (N, D_in)

        返回:
            输出，形状 (N, D_out)
        """
        self.cache['x'] = x

        # x @ W: 矩阵乘法
        # (N, D_in) @ (D_in, D_out) → (N, D_out)
        out = x @ self.W + self.b

        return out

    def backward(self, d_out):
        """
        全连接层反向传播

        计算 dW, db, dX 三个梯度
        """
        x = self.cache['x']  # (N, D_in)

        # (1) dW = x^T @ d_out
        # (N, D_in)^T @ (N, D_out) → (D_in, D_out)
        self.dW = x.T @ d_out

        # (2) db = Σ d_out 沿 batch 维度求和
        # (N, D_out) → (D_out,)
        self.db = np.sum(d_out, axis=0)

        # (3) dX = d_out @ W^T
        # (N, D_out) @ (D_out, D_in) → (N, D_in)
        dX = d_out @ self.W.T

        return dX

    def update(self, lr):
        """使用梯度下降更新权重和偏置"""
        actual_lr = self.lr if self.lr is not None else lr
        self.W -= actual_lr * self.dW
        self.b -= actual_lr * self.db
