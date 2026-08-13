import cv2
import mss
import numpy as np
import time
import pyautogui

pyautogui.FAILSAFE = True

# ========== 1. 颜色范围 ==========
BACKGROUND_LOWER = np.array([100, 120, 150], dtype=np.uint8)
BACKGROUND_UPPER = np.array([160, 180, 210], dtype=np.uint8)

SCAN_REGION = (320, 100, 1700, 900)

# ========== 2. 面积过滤 ==========
MIN_AREA = 3000
MAX_AREA = 250000


def capture_region(region):
    with mss.mss() as sct:
        monitor = {"top": region[1], "left": region[0],
                   "width": region[2] - region[0], "height": region[3] - region[1]}
        img = sct.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def debug_dynamic_split():
    print("====== 动态比例检测 + 左侧画布自动扩充测试 ======")
    print("3秒后开始截图，请切回游戏界面...")
    time.sleep(3)

    print("正在截图...")
    original = capture_region(SCAN_REGION)

    # Step 1: 颜色过滤与形态学黏合
    mask = cv2.inRange(original, BACKGROUND_LOWER, BACKGROUND_UPPER)
    kernel_size = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: 寻找所有候选轮廓
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_rects = []
    max_width = 0
    max_left_pad = 0  # 记录所有色块中，向左扩充的最大像素量，用于调整整个测试画布大小

    # 【第一次遍历】提取轮廓 -> 检测 20:9 宽高比 -> 计算向左补全量
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)
            diff_w = 0

            # 单格标准 20:9 ≈ 2.22，阈值取 1.8。低于 1.8 说明左侧被遮挡/被截屏框裁切
            if aspect_ratio < 1.8:
                # 依据精确比例 20/9，根据当前高度 h 反推标准单格宽度
                expected_single_w = h * (20.0 / 9.0)

                # 如果原本是单格或连格被遮挡，确保补全到标准的几何宽度
                needed_w = max(w, int(expected_single_w))
                diff_w = needed_w - w

                print(
                    f"🔧 发现残缺色块(宽高比 {aspect_ratio:.2f}) | 原始W: {w}px -> 补全W: {needed_w}px | 向左扩充: {diff_w}px")

            # 记录最大的向左扩充需求
            if diff_w > max_left_pad:
                max_left_pad = diff_w

            processed_rects.append({'x': x, 'y': y, 'w': w, 'h': h, 'diff_w': diff_w})

    if not processed_rects:
        print("❌ 未检测到任何符合面积的色块！")
        return []

    # 找出修补补全后的【最大宽度】，作为基准动态反推单格宽度
    for item in processed_rects:
        full_w = item['w'] + item['diff_w']
        if full_w > max_width:
            max_width = full_w

    dynamic_single_width = max_width / 3.0

    print(f"\n📊 动态基准分析：")
    print(f"  - 修复补全后最大宽度 W_max = {max_width} px")
    print(f"  - 动态估算单格标准宽度 W_std = {dynamic_single_width:.1f} px\n")

    # ================= 💡 核心：为图像添加左侧 Padding 扩充画布 =================
    # 如果有色块需要向左扩充，我们在图像左边拼接一块黑框画布，让画图可视化能够完美展示！
    pad_w = max_left_pad + 20  # 多加 20px 留出安全边缘

    if pad_w > 0:
        # 给原始图和 Mask 图的左侧添加黑色内边距
        canvas_result = cv2.copyMakeBorder(original, 0, 0, pad_w, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        canvas_mask = cv2.copyMakeBorder(mask_closed, 0, 0, pad_w, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # 画一条紫色的虚线标记出原本 SCAN_REGION 的左边界
        cv2.line(canvas_result, (pad_w, 0), (pad_w, canvas_result.shape[0]), (255, 0, 255), 2)
    else:
        canvas_result = original.copy()
        canvas_mask = mask_closed.copy()

    click_targets = []

    # 【第二次遍历】在扩充后的画布上精准绘制 & 计算绝对屏幕坐标
    for item in processed_rects:
        diff_w = item['diff_w']

        # 补全后的实际物理宽度
        real_w = item['w'] + diff_w
        # 在扩充画布上的左上角 X 坐标：原本的 x 加上 左侧 Padding，再减去向左修补的 diff_w
        canvas_x = item['x'] + pad_w - diff_w
        canvas_y = item['y']
        h = item['h']

        ratio = real_w / dynamic_single_width
        grid_count = max(1, round(ratio))

        print(f"色块修正后总宽: {real_w}px | 比例: {ratio:.2f} -> 判定为 {grid_count} 连格")

        # 1. 在扩充画布上画红色外包络框（包住被补全的左侧区域）
        cv2.rectangle(canvas_result, (canvas_x, canvas_y), (canvas_x + real_w, canvas_y + h), (0, 0, 255), 2)

        # 2. 均匀切分
        sub_w = real_w / grid_count
        for i in range(grid_count):
            sub_center_x = canvas_x + int((i + 0.5) * sub_w)
            sub_center_y = canvas_y + h // 2

            # 【应用层面绝对坐标换算】
            # 屏幕 X = 原本ROI相对坐标 - 向左补全量 + SCAN_REGION起点X
            screen_x = (item['x'] - diff_w) + int((i + 0.5) * sub_w) + SCAN_REGION[0]
            screen_y = item['y'] + h // 2 + SCAN_REGION[1]
            click_targets.append((screen_x, screen_y))

            # 在扩充画布上画中心红点
            cv2.circle(canvas_result, (sub_center_x, sub_center_y), 5, (0, 0, 255), -1)

            # 画黄色切分线
            if i > 0:
                split_x = canvas_x + int(i * sub_w)
                cv2.line(canvas_result, (split_x, canvas_y), (split_x, canvas_y + h), (0, 255, 255), 1,
                         lineType=cv2.LINE_AA)

    print(f"\n✅ 识别与修补完毕！共获得 {len(click_targets)} 个可点击坐标点。")

    # 显示窗口
    cv2.namedWindow('1. 原始截图', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2. 闭运算(左侧扩充画布)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3. 动态拆分与左侧补全结果', cv2.WINDOW_NORMAL)

    cv2.moveWindow('1. 原始截图', 0, 0)
    cv2.moveWindow('2. 闭运算(左侧扩充画布)', 600, 0)
    cv2.moveWindow('3. 动态拆分与左侧补全结果', 1200, 0)

    cv2.imshow('1. 原始截图', original)
    cv2.imshow('2. 闭运算(左侧扩充画布)', canvas_mask)
    cv2.imshow('3. 动态拆分与左侧补全结果', canvas_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return click_targets


if __name__ == "__main__":
    debug_dynamic_split()