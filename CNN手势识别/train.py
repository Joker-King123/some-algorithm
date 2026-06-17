"""
CNN 手势识别模型 —— 训练脚本
=============================
使用从零实现的 CNN 对手势数据进行训练。

训练流程:
1. 加载采集的手势图像数据
2. 数据预处理（形状调整、one-hot编码）
3. 构建 CNN 模型
4. Mini-batch 梯度下降训练
5. 评估模型并保存权重

CNN架构（详细）:
  Input: (N, 1, 64, 64)     — 灰度手部图像
  Conv2D(1→8, 3×3):          — 8个3×3卷积核，检测低级特征（边缘、角点）
    → (N, 8, 62, 62)
  ReLU:                       — 非线性激活
  MaxPool2D(2×2):            — 降采样一半
    → (N, 8, 31, 31)
  Conv2D(8→16, 3×3):        — 16个卷积核，检测中级特征（形状、纹理）
    → (N, 16, 29, 29)
  ReLU:
  MaxPool2D(2×2):
    → (N, 16, 14, 14)
  Flatten:                    — 展平
    → (N, 3136)
  Dense(3136→128):           — 全连接，高级特征组合
  ReLU:
  Dense(128→5):              — 输出5个类别的logits
  Softmax:                    — 转为概率分布
"""

import numpy as np
import os
import sys
import time

# 导入我们的 CNN 库
sys.path.insert(0, os.path.dirname(__file__))
from cnn import (
    Conv2D, MaxPool2D, Flatten, Dense,
    ReLU, Softmax,
    CrossEntropyLoss,
    SGD, Adam,
    Sequential,
)


# ============================================================================
#                              配置参数
# ============================================================================

# 数据参数
DATA_DIR = 'gesture_data'
IMG_SIZE = 64
NUM_CLASSES = 5

# 训练参数
BATCH_SIZE = 16         # 每批样本数
EPOCHS = 30             # 训练轮数
LEARNING_RATE = 0.001   # 学习率
MOMENTUM = 0.9          # SGD 动量
TRAIN_SPLIT = 0.8       # 训练集比例

# 模型保存路径
MODEL_PATH = 'gesture_model.pkl'

# 手势名称
GESTURE_NAMES = ['palm', 'fist', 'peace', 'thumbs_up', 'ok']


def load_data():
    """
    加载并预处理手势数据

    数据格式转换:
    原始: (N, 64, 64) 灰度图像, (N,) 标签
    → CNN输入: (N, 1, 64, 64) 增加通道维度
    → 标签: (N, 5) one-hot 编码

    One-hot 编码示例:
    标签 2 (peace手势) →
    [0, 0, 1, 0, 0]  ← 只有第2个位置是1
    """
    X_path = os.path.join(DATA_DIR, 'X.npy')
    y_path = os.path.join(DATA_DIR, 'y.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"[错误] 未找到训练数据！")
        print(f"请先运行 collect_data.py 采集手势数据。")
        print(f"预期路径: {X_path}, {y_path}")
        sys.exit(1)

    print(f"[加载] 数据文件: {DATA_DIR}/")
    X = np.load(X_path)  # 形状: (N, 64, 64)
    y = np.load(y_path)  # 形状: (N,)

    print(f"  图像数据: {X.shape}, 值域 [{X.min():.3f}, {X.max():.3f}]")
    print(f"  标签数据: {y.shape}, 类别: {np.unique(y)}")

    # 统计每个类别的样本数
    print(f"\n  各类别样本分布:")
    for i, name in enumerate(GESTURE_NAMES):
        count = np.sum(y == i)
        print(f"    [{i}] {name:12s}: {count} 张")

    # ---- 数据预处理 ----

    # (1) 增加通道维度: (N, 64, 64) → (N, 1, 64, 64)
    # CNN 的 Conv2D 期��输入格式为 (N, C, H, W)
    X = X[:, np.newaxis, :, :]

    # (2) 标签 One-hot 编码
    # 例如: 标签 2 → [0, 0, 1, 0, 0]
    y_onehot = np.zeros((len(y), NUM_CLASSES))
    y_onehot[np.arange(len(y)), y] = 1

    return X, y_onehot, y


def split_data(X, y_onehot, y_raw, train_ratio=0.8):
    """
    划分训练集和测试集

    使用随机排列而非顺序划分，确保各类别在
    训练集和测试集中的比例大致相同。
    """
    N = len(X)
    indices = np.random.permutation(N)
    split_idx = int(N * train_ratio)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return (
        X[train_idx], y_onehot[train_idx], y_raw[train_idx],
        X[test_idx], y_onehot[test_idx], y_raw[test_idx],
    )


def create_model():
    """
    构建 CNN 模型

    层设计说明:
    -----------
    为什么第一层用8个卷积核？
    - 第一层检测简单的低级特征（边缘、角点、线段）
    - 8个核足够覆盖基本的方向和频率
    - 核数太多容易过拟合（尤其是小数据集）

    为什么用2次Conv+Pool组合？
    - Conv提取特征 → Pool降采样
    - 每次Pool后，特征图的"语义层次"提升一级
    - 第一组: 检测局部边缘/纹理
    - 第二组: 组合低级特征为中级特征（手指轮廓、指缝等）

    为什么全连接层用128个神经元？
    - 需要在表达能力（太多→过拟合）和效率之间平衡
    - 128对于5分类任务足够
    """
    KH, KW = 3, 3  # 卷积核尺寸

    model = Sequential([
        # ---- 第一个卷积块 ----
        # 输入: (N, 1, 64, 64)
        Conv2D(in_channels=1, out_channels=8, kernel_size=(KH, KW),
               stride=1, padding=0),
        # 输出: (N, 8, 62, 62)
        ReLU(),
        # 输出不变: (N, 8, 62, 62)
        MaxPool2D(pool_size=2, stride=2),
        # 输出: (N, 8, 31, 31)  — 尺寸减半

        # ---- 第二个卷积块 ----
        Conv2D(in_channels=8, out_channels=16, kernel_size=(KH, KW),
               stride=1, padding=0),
        # 输出: (N, 16, 29, 29)
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        # 输出: (N, 16, 14, 14)  — 再减半

        # ---- 分类器 ----
        Flatten(),
        # 输出: (N, 16*14*14) = (N, 3136)

        Dense(in_features=16 * 14 * 14, out_features=128),
        # 输出: (N, 128)
        ReLU(),

        Dense(in_features=128, out_features=NUM_CLASSES),
        # 输出: (N, 5) — logits

        Softmax(),
        # 输出: (N, 5) — 概率分布
    ])

    return model


def evaluate(model, X, y_onehot, y_raw, loss_fn):
    """评估模型在给定数据上的性能"""
    y_pred = model.forward(X)
    loss = loss_fn.forward(y_pred, y_onehot)

    pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_labels == y_raw)

    # 计算每类准确率
    class_acc = {}
    for i, name in enumerate(GESTURE_NAMES):
        mask = y_raw == i
        if np.sum(mask) > 0:
            class_acc[name] = np.mean(pred_labels[mask] == i)
        else:
            class_acc[name] = 0.0

    return loss, accuracy, class_acc


def train():
    """主训练函数"""
    print("=" * 60)
    print("     从零实现的 CNN —— 手势识别模型训练")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    X, y_onehot, y_raw = load_data()
    print(f"\n[数据] 总样本数: {len(X)}")

    # ---- 2. 划分训练/测试集 ----
    X_train, y_train_oh, y_train_raw, \
        X_test, y_test_oh, y_test_raw = split_data(X, y_onehot, y_raw, TRAIN_SPLIT)

    print(f"[划分] 训练集: {len(X_train)} 张, 测试集: {len(X_test)} 张")

    # ---- 3. 构建模型 ----
    model = create_model()
    model.summary()

    # ---- 4. 初始化损失函数和优化器 ----
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=LEARNING_RATE)  # Adam 通常比 SGD 收敛更快

    # ---- 5. 训练循环 ----
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    N_train = len(X_train)
    n_batches = max(1, N_train // BATCH_SIZE)

    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---- 每个 epoch 打乱数据顺序 ----
        # 这很重要！防止模型"记住"数据顺序，提高泛化能力
        perm = np.random.permutation(N_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train_oh[perm]

        # ---- Mini-batch 训练 ----
        total_loss = 0
        total_acc = 0

        for b in range(n_batches):
            # 取一批数据
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # 一个训练步骤: forward → loss → backward → update
            loss, acc = model.train_step(X_batch, y_batch, optimizer, loss_fn)

            total_loss += loss
            total_acc += acc

        # 计算本 epoch 的平均指标
        avg_train_loss = total_loss / n_batches
        avg_train_acc = total_acc / n_batches

        # ---- 评估测试集 ----
        test_loss, test_acc, class_acc = evaluate(
            model, X_test, y_test_oh, y_test_raw, loss_fn
        )

        # ---- 记录历史 ----
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # ---- 保存最佳模型 ----
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(MODEL_PATH)
            best_marker = " ★ (best)"
        else:
            best_marker = ""

        # ---- 打印进度 ----
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.2%} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2%} | "
              f"Time: {epoch_time:.1f}s{best_marker}")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time:.1f}s")
    print(f"最佳测试准确率: {best_test_acc:.2%}")

    # ---- 6. 最终评估 ----
    print("\n" + "=" * 60)
    print("最终评估 — 各类别准确率")
    print("-" * 60)
    _, _, final_class_acc = evaluate(model, X_test, y_test_oh, y_test_raw, loss_fn)
    for name, acc in final_class_acc.items():
        bar_len = int(acc * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"  {name:12s}: {bar} {acc:.1%}")
    print("=" * 60)

    # ---- 7. 训练曲线（文本版） ----
    print("\n训练过程 (每10步的Loss):")
    steps_to_show = min(len(history['train_loss']), EPOCHS)
    step = max(1, steps_to_show // 15)
    for i in range(0, steps_to_show, step):
        marker = " ← best" if history['test_acc'][i] == best_test_acc else ""
        print(f"  Epoch {i+1:3d}: "
              f"train_loss={history['train_loss'][i]:.4f}, "
              f"test_acc={history['test_acc'][i]:.2%}{marker}")

    print(f"\n模型已保存至: {MODEL_PATH}")
    print("可以运行 recognize.py 进行实时手势识别！")


if __name__ == '__main__':
    train()
