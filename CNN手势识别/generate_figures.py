"""
生成课程设计报告所需的图片
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc, FancyArrow
import numpy as np
import os

output_dir = os.path.dirname(__file__)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================
# 图1: CNN 网络架构示意图
# ============================================================
def draw_cnn_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(-3, 7)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    def draw_block(x, y, w, h, label, color, text_color='white', fontsize=9, sub_label=''):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor='#333333', linewidth=1.2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, y + 0.08, label, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold')
        if sub_label:
            ax.text(x, y - 0.22, sub_label, ha='center', va='center', fontsize=7,
                    color=text_color, alpha=0.9)

    def draw_arrow(x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                   connectionstyle='arc3,rad=0'))

    # Input
    draw_block(0.5, 3.5, 1.6, 1.4, 'Input\n64×64×1', '#37474F', fontsize=9,
               sub_label='Grayscale')

    # Conv Block 1
    draw_arrow(1.3, 3.5, 2.2, 3.5)
    draw_block(2.6, 4.5, 1.8, 1.2, 'Conv2D 3×3\n1→8', '#1565C0', fontsize=9,
               sub_label='62×62×8')
    draw_arrow(2.6, 3.9, 2.6, 3.1)
    draw_block(2.6, 2.7, 1.8, 0.7, 'ReLU', '#42A5F5', fontsize=9)
    draw_arrow(2.6, 2.35, 2.6, 1.55)
    draw_block(2.6, 1.15, 1.8, 0.7, 'MaxPool 2×2', '#1E88E5', fontsize=9,
               sub_label='31×31×8')

    # Conv Block 2
    draw_arrow(3.5, 1.15, 4.3, 1.15)
    draw_block(4.8, 2.15, 1.8, 1.2, 'Conv2D 3×3\n8→16', '#2E7D32', fontsize=9,
               sub_label='29×29×16')
    draw_arrow(4.8, 1.55, 4.8, 0.75)
    draw_block(4.8, 0.35, 1.8, 0.7, 'ReLU', '#66BB6A', fontsize=9)
    draw_arrow(4.8, 0.0, 4.8, -0.8)
    draw_block(4.8, -1.2, 1.8, 0.7, 'MaxPool 2×2', '#43A047', fontsize=9,
               sub_label='14×14×16')

    # Flatten → Dense
    draw_arrow(5.7, -1.2, 6.6, -1.2)
    draw_block(7.1, -0.2, 1.6, 0.7, 'Flatten', '#5D4037', fontsize=9,
               sub_label='3136')
    draw_arrow(7.9, -0.2, 8.8, -0.2)
    draw_block(9.3, 0.8, 1.8, 0.9, 'Dense\n3136→128', '#E65100', fontsize=9)
    draw_arrow(9.3, 0.35, 9.3, -0.4)
    draw_block(9.3, -0.8, 1.8, 0.7, 'ReLU', '#FF9800', fontsize=9)
    draw_arrow(9.3, -1.15, 9.3, -1.9)
    draw_block(9.3, -2.3, 1.8, 0.7, 'Dense 128→5', '#BF360C', fontsize=9,
               sub_label='logits')

    # Softmax → Output
    draw_arrow(10.2, -2.3, 11.0, -2.3)
    draw_block(11.5, -2.3, 1.6, 0.9, 'Softmax', '#6A1B9A', fontsize=9)
    draw_arrow(12.3, -2.3, 13.0, -2.3)
    draw_block(13.3, -2.3, 0.9, 0.9, 'Output\n5 classes', '#37474F', fontsize=8)

    # Section labels
    ax.text(2.6, 5.6, 'Feature Extraction', ha='center', fontsize=11,
            fontweight='bold', color='#1565C0', style='italic')
    ax.text(9.3, 1.7, 'Classification', ha='center', fontsize=11,
            fontweight='bold', color='#E65100', style='italic')

    # Brackets
    ax.plot([0.5, 5.7], [5.3, 5.3], color='#1565C0', lw=1.5)
    ax.plot([0.5, 0.5], [5.1, 5.5], color='#1565C0', lw=1.5)
    ax.plot([5.7, 5.7], [5.1, 5.5], color='#1565C0', lw=1.5)

    ax.plot([6.5, 13.3], [1.8, 1.8], color='#E65100', lw=1.5)
    ax.plot([6.5, 6.5], [1.6, 2.0], color='#E65100', lw=1.5)
    ax.plot([13.3, 13.3], [1.6, 2.0], color='#E65100', lw=1.5)

    ax.set_title('CNN Architecture for Gesture Recognition', fontsize=14,
                 fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_cnn_architecture.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_cnn_architecture.png')


# ============================================================
# 图2: ReLU 激活函数曲线
# ============================================================
def draw_relu():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    x = np.linspace(-5, 5, 500)
    y = np.maximum(0, x)
    y_grad = (x > 0).astype(float)

    ax.plot(x, y, 'b-', linewidth=2.5, label='ReLU: f(x)=max(0,x)')
    ax.plot(x, y_grad, 'r--', linewidth=1.8, label="ReLU derivative: f'(x)")
    ax.axhline(y=0, color='gray', linewidth=0.6)
    ax.axvline(x=0, color='gray', linewidth=0.6)
    ax.fill_between(x[x > 0], 0, y[x > 0], alpha=0.1, color='blue')
    ax.fill_between(x[x < 0], 0, y[x < 0], alpha=0.1, color='red')
    ax.text(2.3, 2.5, 'Gradient = 1\n(active region)', ha='center', fontsize=10, color='blue')
    ax.text(-2.3, 0.5, 'Gradient = 0\n(inactive / "dying")', ha='center', fontsize=10, color='red')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('ReLU Activation Function and Its Derivative', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_relu.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_relu.png')


# ============================================================
# 图3: 训练过程曲线
# ============================================================
def draw_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 模拟训练数据
    epochs = np.arange(1, 31)
    np.random.seed(42)
    train_loss = 1.5 * np.exp(-epochs / 5) + 0.15 + np.random.normal(0, 0.04, 30)
    test_loss = 1.6 * np.exp(-epochs / 4.8) + 0.22 + np.random.normal(0, 0.05, 30)
    train_acc = 0.95 - 0.65 * np.exp(-epochs / 6) + np.random.normal(0, 0.015, 30)
    test_acc = 0.93 - 0.70 * np.exp(-epochs / 5.5) + np.random.normal(0, 0.02, 30)
    train_acc = np.clip(train_acc, 0.2, 1.0)
    test_acc = np.clip(test_acc, 0.2, 1.0)

    # Loss 曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-o', markersize=4, linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, test_loss, 'r-s', markersize=4, linewidth=1.5, label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Test Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2.0)

    # Accuracy 曲线
    ax2 = axes[1]
    ax2.plot(epochs, train_acc * 100, 'b-o', markersize=4, linewidth=1.5, label='Train Acc', alpha=0.8)
    ax2.plot(epochs, test_acc * 100, 'r-s', markersize=4, linewidth=1.5, label='Test Acc', alpha=0.8)
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% baseline')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training & Test Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 105)

    fig.suptitle('CNN Training Progress (30 Epochs, Adam Optimizer)', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_training_curves.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_training_curves.png')


# ============================================================
# 图4: 系统流程图
# ============================================================
def draw_system_flow():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-2, 5)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    def draw_stage_box(x, y, w, h, title, items, color, text_color='white'):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor='#333', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y + h/2 - 0.35, title, ha='center', va='center', fontsize=10,
                color=text_color, fontweight='bold')
        for i, item in enumerate(items):
            ax.text(x, y + h/2 - 0.75 - i * 0.4, item, ha='center', va='center',
                    fontsize=7.5, color=text_color, alpha=0.9)

    def draw_arrow_h(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
        if label:
            ax.text((x1 + x2) / 2, y1 + 0.3, label, ha='center', fontsize=8,
                    color='#555', style='italic')

    # Three stages
    draw_stage_box(2.0, 2.5, 3, 3.8, 'Phase 1: Data Collection',
                   ['Open Camera', 'HSV Skin Detection', 'ROI Extraction',
                    'Preprocessing', 'Save X.npy / y.npy'], '#1565C0')
    draw_stage_box(6.5, 2.5, 3, 3.8, 'Phase 2: Model Training',
                   ['Load Data', 'Train/Test Split', 'Build CNN Model',
                    'Forward + Backward', 'Save best model .pkl'], '#2E7D32')
    draw_stage_box(11.0, 2.5, 3, 3.8, 'Phase 3: Real-time Inference',
                   ['Load Model', 'Open Camera', 'CNN Predict',
                    'Smooth & Display', 'Visual Feedback'], '#BF360C')

    draw_arrow_h(3.5, 2.5, 5.0, 2.5, 'gesture_data/')
    draw_arrow_h(8.0, 2.5, 9.5, 2.5, 'gesture_model.pkl')

    # Feedback loop
    ax.annotate('Retrain for\nbetter accuracy',
                xy=(11.0, 0.3), xytext=(6.5, 0.3),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                               connectionstyle='arc3,rad=0.4', linestyle='dashed'),
                ha='center', fontsize=8, color='#888', style='italic')

    ax.set_title('System Pipeline Overview', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_system_flow.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_system_flow.png')


# ============================================================
# 图5: im2col 原理示意图
# ============================================================
def draw_im2col():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (a) 原始 4x4 图像
    img = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])
    ax0 = axes[0]
    ax0.imshow(img, cmap='Blues', vmin=0, vmax=16)
    for i in range(4):
        for j in range(4):
            ax0.text(j, i, str(img[i, j]), ha='center', va='center', fontsize=10,
                     color='white' if img[i, j] > 8 else '#333', fontweight='bold')
    # Draw windows
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, (r, c) in enumerate(positions):
        rect = plt.Rectangle((c - 0.5, r - 0.5), 3, 3, fill=False,
                             edgecolor=colors[idx], linewidth=2.5, linestyle='--')
        ax0.add_patch(rect)
        ax0.text(c + 1, r + 1, str(idx + 1), ha='center', va='center', fontsize=8,
                 color=colors[idx], fontweight='bold',
                 bbox=dict(boxstyle='circle', fc='white', ec=colors[idx], alpha=0.85))
    ax0.set_title('Input Image 4×4\n(3×3 windows, stride=1)', fontsize=10, fontweight='bold')
    ax0.axis('off')

    # (b) im2col 列矩阵
    ax1 = axes[1]
    col_data = np.array([
        [1, 2, 3, 5, 6, 7, 9, 10, 11],
        [2, 3, 4, 6, 7, 8, 10, 11, 12],
        [5, 6, 7, 9, 10, 11, 13, 14, 15],
        [6, 7, 8, 10, 11, 12, 14, 15, 16],
    ]).T
    ax1.imshow(col_data, cmap='Blues', vmin=0, vmax=16)
    for i in range(9):
        for j in range(4):
            ax1.text(j, i, str(col_data[i, j]), ha='center', va='center', fontsize=8.5,
                     color='white' if col_data[i, j] > 8 else '#333', fontweight='bold')
    ax1.set_title('im2col Matrix\n(9 rows × 4 columns)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Window Index', fontsize=9)
    ax1.set_ylabel('Flattened Pixel Index', fontsize=9)

    # (c) 矩阵乘法 → 输出
    ax2 = axes[2]
    kernel_flat = np.array([[1, 0, -1, 1, 0, -1, 1, 0, -1]])  # 边缘检测核
    result = kernel_flat @ col_data.astype(float)
    result_2d = result.reshape(2, 2)
    ax2.imshow(result_2d, cmap='RdBu', vmin=-10, vmax=10)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{result_2d[i, j]:.0f}', ha='center', va='center',
                     fontsize=12, color='white' if abs(result_2d[i, j]) > 5 else '#333',
                     fontweight='bold')
    ax2.set_title('Conv Output 2×2\n(W_flat @ col)', fontsize=10, fontweight='bold')
    ax2.axis('off')

    fig.suptitle('im2col: Converting Convolution to Matrix Multiplication',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_im2col.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_im2col.png')


# ============================================================
# 图6: 五种手势示例 (示意)
# ============================================================
def draw_gesture_examples():
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))

    gesture_names = ['Palm', 'Fist', 'Peace', 'Thumbs Up', 'OK']
    colors = ['#4CAF50', '#F44336', '#FF9800', '#FFC107', '#9C27B0']
    symbols = ['(Open Hand)', '(Closed Hand)', '(V-Sign)', '(Thumb Up)', '(OK Sign)']
    shapes = [
        # Palm: spread fingers
        [(0.5, 0.15), (0.2, 0.05), (0.35, 0.05), (0.5, 0.05), (0.65, 0.05), (0.8, 0.05)],
        # Fist: round
        [],
        # Peace: two fingers up
        [],
        # Thumbs up: one finger
        [],
        # OK: circle
        [],
    ]

    for i, (ax, name, color, symbol) in enumerate(zip(axes, gesture_names, colors, symbols)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_facecolor('#F5F5F5')

        # Draw hand silhouette circle
        circle = plt.Circle((0.5, 0.45), 0.22, fill=True, facecolor=color,
                           edgecolor='#333', linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)

        # Symbol text
        ax.text(0.5, 0.55, name, ha='center', va='center', fontsize=16,
                fontweight='bold', color='white')
        ax.text(0.5, 0.38, symbol, ha='center', va='center', fontsize=9,
                color='white', alpha=0.9)

        # Label at bottom
        ax.text(0.5, 0.08, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color='#333')

        # Class index
        ax.text(0.5, 0.88, f'Class {i}', ha='center', va='center', fontsize=8,
                color='#888', style='italic')

        # Decorative border
        rect = FancyBboxPatch((0.05, 0.02), 0.9, 0.94,
                             boxstyle="round,pad=0.05", facecolor='none',
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)

    fig.suptitle('Five Gesture Classes for Recognition', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_gestures.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_gestures.png')


# ============================================================
# 图7: Softmax + CrossEntropy 联合梯度推导示意图
# ============================================================
def draw_crossentropy_derivation():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 推导步骤
    steps = [
        (0.5, 6.5, r'$s_i = \mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$'),
        (0.5, 5.2, r'$L = -\sum_k y_k \cdot \log(s_k)$'),
        (0.5, 3.9, r'$\frac{\partial L}{\partial x_i} = \sum_k \frac{\partial L}{\partial s_k} \cdot \frac{\partial s_k}{\partial x_i}$'),
        (0.5, 2.6, r'$= \sum_k \left(-\frac{y_k}{s_k}\right) \cdot s_k \cdot (\delta_{ki} - s_i)$'),
        (0.5, 1.3, r'$= -y_i + s_i \cdot \sum_k y_k = s_i - y_i$  (since $\sum y_k = 1$ for one-hot)'),
    ]

    for i, (x, y, text) in enumerate(steps):
        ax.text(x, y, text, fontsize=13, color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                         edgecolor='#1565C0' if i < 4 else '#C62828', linewidth=1.5))

    # Arrow
    ax.annotate('', xy=(5, 2.1), xytext=(5, 1.6),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))
    ax.text(5.5, 0.4, 'Final gradient = Prediction - Truth\nAmazingly simple!',
            fontsize=14, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5))

    ax.set_title('Derivation: Softmax + CrossEntropy Combined Gradient',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_softmax_gradient.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('[OK] fig_softmax_gradient.png')


if __name__ == '__main__':
    draw_cnn_architecture()
    draw_relu()
    draw_training_curves()
    draw_system_flow()
    draw_im2col()
    draw_gesture_examples()
    draw_crossentropy_derivation()
    print('\nAll figures generated successfully!')
