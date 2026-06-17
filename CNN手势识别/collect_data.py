"""
手势数据采集工具
===============
使用笔记本电脑摄像头采集手势图像，用于训练CNN模型。

支持的手势类别（共5种）:
  按数字键 0: 张开手掌 (Palm)       —— 五指张开
  按数字键 1: 握拳 (Fist)          —— 五指握紧
  按数字键 2: 剪刀手 (Peace)       —— 食指和中指竖起
  按数字键 3: 竖大拇指 (ThumbsUp)   —— 大拇指竖起
  按数字键 4: OK手势 (OK)          —— 拇指和食指圈起

操作说明:
  - 将手放在绿色ROI框内
  - 按数字键 0-4 采集对应手势（每按一次采集一张）
  - 按 's' 键保存当前采集的所有数据
  - 按 'q' 键退出（不保存）

肤色检测原理:
  将BGR图像转换到HSV色彩空间，利用肤色在HSV空间中的
  特定范围进行分割。相比RGB，HSV对光照变化更鲁棒。
"""

import cv2
import numpy as np
import os
import time

# ============================================================================
#                              配置参数
# ============================================================================

# 图像预处理参数
IMG_SIZE = 64          # CNN输入尺寸 (64×64)
ROI_SIZE = 250         # 采集框大小（像素）

# HSV 肤色范围
# 为什么用HSV？在HSV空间中，颜色(H)、饱和度(S)、亮度(V)分离，
# 肤色在H和S上有较稳定的范围，对光照变化的容忍度比RGB好得多。
SKIN_LOWER = np.array([0, 20, 40], dtype=np.uint8)     # H_min, S_min, V_min
SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)  # H_max, S_max, V_max

# 手势类别名称
GESTURE_NAMES = {
    0: 'palm',       # 张开手掌
    1: 'fist',       # 握拳
    2: 'peace',      # 剪刀手
    3: 'thumbs_up',  # 竖大拇指
    4: 'ok',         # OK手势
}

# 每种手势的目标采集数量
TARGET_PER_CLASS = 200

# 保存路径
DATA_DIR = 'gesture_data'


def preprocess_hand(roi_bgr):
    """
    预处理ROI区域中的手部图像

    处理流程:
    1. BGR → HSV: 转换色彩空间便于肤色检测
    2. 肤色阈值: 创建二值掩码(肤色=255, 背景=0)
    3. 形态学操作: 去除噪声，填充空洞
    4. 灰度化 + 高斯模糊: 转为灰度并平滑
    5. 应用掩码: 保留手部区域，背景置黑
    6. 缩放: 统一尺寸为 64×64
    7. 归一化: [0, 255] → [0, 1]

    为什么需要高斯模糊？
    平滑图像可以:
    - 减少皮肤纹理的细微变化（这些对识别无益）
    - 降低噪声
    - 帮助CNN聚焦于手部整体形状而非纹理细节
    """
    # 步骤1: BGR → HSV
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # 步骤2: 创建肤色掩码
    # inRange: 在范围内的像素置255，范围外置0
    skin_mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)

    # 步骤3: 形态学操作
    # 开运算(先腐蚀后膨胀): 去除孤立的小噪声点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 闭运算(先膨胀后腐蚀): 填充手部区域中的小空洞
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 步骤4: 灰度化
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 步骤5: 高斯模糊 (5×5核，σ=0自动计算)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 步骤6: 应用肤色掩码 —— 只保留手部
    hand_only = cv2.bitwise_and(gray, gray, mask=skin_mask)

    # 步骤7: 缩放到统一尺寸
    hand_resized = cv2.resize(hand_only, (IMG_SIZE, IMG_SIZE))

    # 步骤8: 归一化到 [0, 1]
    hand_normalized = hand_resized.astype(np.float32) / 255.0

    return hand_normalized, skin_mask


def main():
    """主函数：打开摄像头，采集手势数据"""
    print("=" * 60)
    print("          CNN 手势识别 —— 数据采集工具")
    print("=" * 60)
    print(f"\n目标: 每种手势采集 {TARGET_PER_CLASS} 张图像")
    print("\n操作说明:")
    for key, name in GESTURE_NAMES.items():
        print(f"  按 [{key}] 采集: {name}")
    print("  按 [s] 保存数据")
    print("  按 [q] 退出\n")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头！请检查摄像头是否被其他程序占用。")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化数据存储
    data = {i: [] for i in range(5)}  # {gesture_id: [images]}

    # 计算ROI在画面中的位置（居中）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x = (frame_width - ROI_SIZE) // 2
    roi_y = (frame_height - ROI_SIZE) // 2

    print(f"摄像头分辨率: {frame_width}×{frame_height}")
    print(f"ROI 采集框: {ROI_SIZE}×{ROI_SIZE}, 位于画面中央")
    print("-" * 60)

    last_capture_time = 0
    capture_cooldown = 0.3  # 两次采集之间的冷却时间（秒），防止连拍

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("[错误] 读取摄像头帧失败")
            break

        # 水平翻转（镜像效果，更自然）
        frame = cv2.flip(frame, 1)

        # 绘制ROI采集框（绿色）
        cv2.rectangle(frame,
                      (roi_x, roi_y),
                      (roi_x + ROI_SIZE, roi_y + ROI_SIZE),
                      (0, 255, 0), 2)

        # 提取ROI区域
        roi = frame[roi_y:roi_y + ROI_SIZE, roi_x:roi_x + ROI_SIZE]

        # 预处理ROI中的手部图像
        processed, skin_mask = preprocess_hand(roi)

        # ---- 显示信息 ----
        # 在画面左上角显示各类手势已采集的数量
        y_pos = 30
        cv2.putText(frame, f"Collected:", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for gesture_id, name in GESTURE_NAMES.items():
            y_pos += 30
            count = len(data[gesture_id])
            color = (0, 255, 0) if count >= TARGET_PER_CLASS else (0, 165, 255)
            cv2.putText(frame,
                        f"[{gesture_id}] {name}: {count}/{TARGET_PER_CLASS}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        # 显示ROI内部的肤色掩码（小窗口预览）
        skin_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        skin_display = cv2.resize(skin_display, (150, 150))
        cv2.putText(skin_display, "Skin Mask", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 将肤色掩码预览放在主画面右上角
        frame[10:160, frame_width - 160:frame_width - 10] = skin_display

        # 显示处理后的手部图像（CNN实际看到的）
        hand_display = (processed * 255).astype(np.uint8)
        hand_display = cv2.cvtColor(hand_display, cv2.COLOR_GRAY2BGR)
        hand_display = cv2.resize(hand_display, (150, 150))
        cv2.putText(hand_display, "CNN Input", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        frame[170:320, frame_width - 160:frame_width - 10] = hand_display

        # 显示主画面
        cv2.imshow('Gesture Data Collection - Press 0-4 to capture, S to save, Q to quit', frame)

        # 键盘处理
        key = cv2.waitKey(1) & 0xFF

        current_time = time.time()
        # 处理0-5数字键采集
        if ord('0') <= key <= ord('4'):
            gesture_id = key - ord('0')
            # 冷却时间检查，防止连拍
            if current_time - last_capture_time < capture_cooldown:
                continue

            last_capture_time = current_time
            data[gesture_id].append(processed)
            count = len(data[gesture_id])
            print(f"  ✓ 采集 {GESTURE_NAMES[gesture_id]:12s} — "
                  f"第 {count:3d}/{TARGET_PER_CLASS} 张")

            # 采集完成提示
            if count >= TARGET_PER_CLASS:
                print(f"    └─ {GESTURE_NAMES[gesture_id]} 已达到目标数量！")

        # 's' 键保存
        elif key == ord('s'):
            save_data(data)
            print("[完成] 数据已保存。按 Q 退出或继续采集。")

        # 'q' 键退出
        elif key == ord('q'):
            total = sum(len(v) for v in data.values())
            if total > 0:
                print(f"\n共采集 {total} 张图像，是否保存？")
                print("按 'y' 保存并退出，按 'n' 直接退出，按其他键继续...")
                confirm = cv2.waitKey(0) & 0xFF
                if confirm == ord('y'):
                    save_data(data)
                    break
                elif confirm == ord('n'):
                    break
                else:
                    continue
            else:
                print("\n未采集任何数据，直接退出。")
                break

    cap.release()
    cv2.destroyAllWindows()


def save_data(data):
    """
    保存采集的数据到磁盘

    保存格式:
    - X.npy: 图像数据，形状 (N, 64, 64)
    - y.npy: 标签数据，形状 (N,) — 每个元素是 0-4 的类别编号
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    X_list = []
    y_list = []

    for gesture_id, images in data.items():
        if len(images) > 0:
            X_list.append(np.array(images))
            y_list.append(np.full(len(images), gesture_id))

    if X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        # 打乱数据顺序
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]

        save_path_X = os.path.join(DATA_DIR, 'X.npy')
        save_path_y = os.path.join(DATA_DIR, 'y.npy')

        np.save(save_path_X, X_all)
        np.save(save_path_y, y_all)

        print(f"\n{'=' * 60}")
        print(f"数据已保存至 {DATA_DIR}/")
        print(f"  图像数据: X.npy — 形状 {X_all.shape}, 类型 {X_all.dtype}")
        print(f"  标签数据: y.npy — 形状 {y_all.shape}, 类型 {y_all.dtype}")
        print(f"\n各类别样本数:")

        for gesture_id, name in GESTURE_NAMES.items():
            count = len(data.get(gesture_id, []))
            bar = '█' * (count // 5)  # 简易柱状图
            print(f"  [{gesture_id}] {name:12s}: {count:3d} 张 {bar}")

        print(f"  {'总计':14s}: {len(X_all)} 张")
        print(f"{'=' * 60}")
    else:
        print("[警告] 没有数据可保存！")


if __name__ == '__main__':
    main()
