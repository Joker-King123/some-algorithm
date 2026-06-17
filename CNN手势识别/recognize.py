"""
实时手势识别（简化版）
====================
使用训练好的CNN模型实时识别笔记本电脑摄像头中的手势，
简洁的UI交互，支持中文显示。
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

# 导入 CNN 库
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

IMG_SIZE = 64
ROI_SIZE = 250
NUM_CLASSES = 5

# HSV 肤色范围
SKIN_LOWER = np.array([0, 20, 40], dtype=np.uint8)
SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)

MODEL_PATH = 'gesture_model.pkl'

# 手势信息（使用中文名称）
GESTURE_INFO = {
    0: {'name': '张开手掌', 'color': (0, 255, 0)},
    1: {'name': '握拳',     'color': (0, 0, 255)},
    2: {'name': '剪刀手',   'color': (255, 80, 0)},
    3: {'name': '竖大拇指', 'color': (0, 215, 255)},
    4: {'name': 'OK手势',   'color': (255, 0, 255)},
}


# ============================================================================
#                          中文字体工具
# ============================================================================

def get_chinese_font(size=24):
    """获取中文字体，优先使用系统字体"""
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',       # 黑体
        'C:/Windows/Fonts/msyh.ttc',          # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',        # 宋体
        'C:/Windows/Fonts/simkai.ttf',        # 楷体
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    # 回退：使用默认字体（可能不支持中文）
    return ImageFont.load_default()


def put_chinese_text(img, text, pos, color, font_size=28):
    """在OpenCV图像上绘制中文文字（使用PIL）"""
    if img is None or not text:
        return
    font = get_chinese_font(font_size)
    # OpenCV BGR → PIL RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 文字阴影
    shadow_pos = (pos[0] + 2, pos[1] + 2)
    draw.text(shadow_pos, text, font=font, fill=(0, 0, 0))

    # 文字主体（PIL用RGB，OpenCV用BGR，所以需要转换颜色）
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)

    # PIL RGB → OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[:] = img_bgr


def put_text_simple(img, text, pos, color, scale=0.7, thickness=2):
    """绘制英文/数字文字（OpenCV原生）"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


# ============================================================================
#                          预处理函数
# ============================================================================

def preprocess_hand(roi_bgr):
    """预处理ROI手部图像"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hand_only = cv2.bitwise_and(gray, gray, mask=skin_mask)
    hand_resized = cv2.resize(hand_only, (IMG_SIZE, IMG_SIZE))
    hand_normalized = hand_resized.astype(np.float32) / 255.0

    return hand_normalized, skin_mask


def create_model():
    """构建与训练时相同的CNN模型结构"""
    model = Sequential([
        Conv2D(in_channels=1, out_channels=8, kernel_size=(3, 3),
               stride=1, padding=0),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        Conv2D(in_channels=8, out_channels=16, kernel_size=(3, 3),
               stride=1, padding=0),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        Flatten(),
        Dense(in_features=16 * 14 * 14, out_features=128),
        ReLU(),
        Dense(in_features=128, out_features=NUM_CLASSES),
        Softmax(),
    ])
    return model


# ============================================================================
#                          简洁绘制函数
# ============================================================================

def draw_roi_box(frame, x, y, size, color, thickness=2):
    """绘制简单的ROI识别框"""
    cv2.rectangle(frame, (x, y), (x + size, y + size), color, thickness)
    # 四角标记
    corner_len = 20
    for cx, cy, dx, dy in [
        (x, y, 1, 1), (x + size, y, -1, 1),
        (x, y + size, 1, -1), (x + size, y + size, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 2)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 2)


def draw_gesture_label(frame, gesture_name, confidence, center, color):
    """在ROI上方显示识别结果（中文）"""
    if confidence < 0.4:
        text = '等待手势...'
        clr = (180, 180, 180)
    else:
        text = f'{gesture_name}  ({confidence:.0%})'
        clr = color

    # 计算文字位置（居中显示在ROI上方）
    cx, cy = center
    font = get_chinese_font(24)
    # 用PIL测量文字大小
    pil_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = cx - text_w // 2
    text_y = cy - ROI_SIZE // 2 - text_h - 15

    put_chinese_text(frame, text, (text_x, text_y), clr, 24)


def draw_confidence_bar(frame, confidence, x, y, width, color):
    """绘制简洁的置信度进度条"""
    bar_h = 8
    # 背景
    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (100, 100, 100), 1)
    # 进度
    bar_w = int(confidence * width)
    if bar_w > 0:
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), color, -1)


def draw_info_panel(frame, gesture_id, confidence, x, y):
    """绘制简洁的信息面板"""
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 280, y + 85), (30, 30, 35), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 280, y + 85), (70, 70, 80), 1)

    put_text_simple(frame, 'CNN Gesture Recognition', (x + 10, y + 22),
                    (200, 200, 200), 0.5, 1)

    cv2.line(frame, (x + 10, y + 28), (x + 270, y + 28), (60, 60, 70), 1)

    if gesture_id >= 0 and confidence > 0.4:
        info = GESTURE_INFO[gesture_id]
        put_chinese_text(frame, info['name'], (x + 10, y + 38),
                         info['color'], 28)
        draw_confidence_bar(frame, confidence, x + 10, y + 70, 200, info['color'])
        put_text_simple(frame, f'{confidence:.0%}', (x + 215, y + 70),
                        info['color'], 0.45, 1)
    else:
        put_chinese_text(frame, '等待手势...', (x + 10, y + 38),
                         (150, 150, 150), 22)


# ============================================================================
#                              主程序
# ============================================================================

def main():
    """主函数：简化版实时手势识别"""
    print('=' * 55)
    print('  CNN 实时手势识别（简化版）')
    print('=' * 55)

    # ---- 1. 加载模型 ----
    if not os.path.exists(MODEL_PATH):
        print(f'\n[错误] 未找到训练好的模型: {MODEL_PATH}')
        print('请先运行 train.py 训练模型。')
        sys.exit(1)

    print(f'\n[加载] 模型: {MODEL_PATH}')
    model = create_model()
    model.load(MODEL_PATH)
    print('[就绪] 模型已加载')

    # ---- 2. 打开摄像头 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头！')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x = (frame_width - ROI_SIZE) // 2
    roi_y = (frame_height - ROI_SIZE) // 2
    roi_center = (frame_width // 2, frame_height // 2)

    print(f'\n[摄像头] {frame_width}x{frame_height}')
    print('[操作] 将手放在画面中央的框中')
    print('[操作] 按 Q 退出 | 按 D 切换调试模式\n')

    # ---- 3. 初始化 ----
    smooth_buffer = []
    SMOOTH_WINDOW = 5

    start_time = time.time()
    frame_count = 0
    current_gesture = -1
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        elapsed = time.time() - start_time

        # ---- 提取ROI并预处理 ----
        roi = frame[roi_y:roi_y + ROI_SIZE, roi_x:roi_x + ROI_SIZE]
        processed, skin_mask = preprocess_hand(roi)

        # ---- CNN预测 ----
        cnn_input = processed[np.newaxis, np.newaxis, :, :]
        probabilities = model.predict_proba(cnn_input)
        gesture_id = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities, axis=1)[0])

        # ---- 平滑预测 ----
        smooth_buffer.append(gesture_id)
        if len(smooth_buffer) > SMOOTH_WINDOW:
            smooth_buffer.pop(0)

        most_common = Counter(smooth_buffer).most_common(1)[0]
        stable_gesture = most_common[0]
        stable_count = most_common[1]

        if stable_count >= SMOOTH_WINDOW * 0.6:
            if stable_gesture != current_gesture:
                current_gesture = stable_gesture
                info = GESTURE_INFO[stable_gesture]
                print(f'  [{elapsed:5.1f}s] 识别: {info["name"]} '
                      f'(置信度: {confidence:.1%})')

        has_hand = confidence > 0.4 and current_gesture >= 0

        # ---- 绘制 ----
        # ROI框
        if has_hand:
            roi_color = GESTURE_INFO[current_gesture]['color']
        else:
            roi_color = (0, 220, 0)
        draw_roi_box(frame, roi_x, roi_y, ROI_SIZE, roi_color, 2)

        # 手势标签（在ROI上方显示中文）
        if has_hand:
            info = GESTURE_INFO[current_gesture]
            draw_gesture_label(frame, info['name'], confidence,
                              roi_center, info['color'])
        else:
            draw_gesture_label(frame, '', 0, roi_center, (180, 180, 180))

        # 左上角信息面板
        draw_info_panel(frame, current_gesture if has_hand else -1,
                       confidence, 10, 10)

        # FPS
        fps = frame_count / elapsed if elapsed > 0 else 0
        fps_text = f'FPS: {fps:.0f}'
        put_text_simple(frame, fps_text,
                       (frame_width - 100, frame_height - 15),
                       (0, 255, 0) if fps > 15 else (0, 200, 255),
                       0.5, 1)

        # 调试模式
        if debug_mode:
            debug_x, debug_y = frame_width - 170, 10
            # CNN输入预览
            debug_img = (processed * 255).astype(np.uint8)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.resize(debug_img, (150, 150))
            cv2.putText(debug_img, 'CNN Input', (5, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frame[debug_y:debug_y + 150, debug_x:debug_x + 150] = debug_img
            # 肤色掩码预览
            skin_disp = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            skin_disp = cv2.resize(skin_disp, (150, 150))
            cv2.putText(skin_disp, 'Skin Mask', (5, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frame[debug_y + 160:debug_y + 310, debug_x:debug_x + 150] = skin_disp

        # ---- 显示 ----
        cv2.imshow('CNN Gesture Recognition', frame)

        # 键盘处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f'  [调试] 调试模式: {"ON" if debug_mode else "OFF"}')

    # ---- 清理 ----
    cap.release()
    cv2.destroyAllWindows()
    print(f'\n[退出] 共处理 {frame_count} 帧, 运行 {elapsed:.1f}s')
    print(f'[平均] FPS: {frame_count / elapsed:.1f}')
    print('再见！')


if __name__ == '__main__':
    main()
