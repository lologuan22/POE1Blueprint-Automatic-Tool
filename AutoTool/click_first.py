from multiprocessing.spawn import spawn_main
from random import sample

import cv2
import mss
import numpy as np
import time
import pyautogui
import keyboard

# ====================== 全局控制 ======================
pyautogui.FAILSAFE = True
running = False
stop = False
restart = False

# ====================== 完美分辨率自适应 ======================
BASE_W = 1920
BASE_H = 1080
screen_w, screen_h = pyautogui.size()
scale_w = screen_w / BASE_W
scale_h = screen_h / BASE_H
scale_area = scale_w * scale_h


def s(x, y=None):
    if y is None:
        return int(x * scale_w)
    return int(x * scale_w), int(y * scale_h)


# ====================== 延迟参数 ======================
CLICK_DELAY = 0.03
WAIT_DELAY = 0.2
STEP = 0.01


# ====================== 工具函数 ======================
def show_speed():
    print(f"\r✅ CLICK_DELAY={CLICK_DELAY:.2f}s | WAIT_DELAY={WAIT_DELAY:.2f}s | ↑加快 ↓减慢 ", end="")


# ====================== 快捷键 ======================
def check_keys():
    global running, stop, restart, CLICK_DELAY, WAIT_DELAY
    if keyboard.is_pressed('f10'):
        running = not running
        print("\n▶ 运行" if running else "\n⏸ 暂停")
        time.sleep(0.3)
    if keyboard.is_pressed('f11'):
        stop = True
        print("\n🛑 已停止")
    if keyboard.is_pressed('f12'):
        restart = True
        running = False
        print("\n🔄 重来")
        time.sleep(0.3)
    if keyboard.is_pressed('up'):
        CLICK_DELAY = max(0.01, CLICK_DELAY - STEP)
        WAIT_DELAY = max(0.05, WAIT_DELAY - STEP)
        show_speed()
        time.sleep(0.15)
    if keyboard.is_pressed('down'):
        CLICK_DELAY += STEP
        WAIT_DELAY += STEP
        show_speed()
        time.sleep(0.15)


def wait_continue():
    while not running and not stop and not restart:
        check_keys()
        time.sleep(0.01)


# ====================== 所有坐标+面积 统一完美适配 ======================
BACKGROUND_LOWER = np.array([100, 120, 150], np.uint8)
BACKGROUND_UPPER = np.array([160, 180, 210], np.uint8)

# 闭运算会把散落色块黏合，故大幅度提高判定面积的自适应上下限
MIN_AREA = int(3000 * scale_area)
MAX_AREA = int(150000 * scale_area)

SCAN_REGION = (s(320), s(100), s(1700), s(900))
SECOND_CLICK_POS = s(860, 540)
BACKPACK_START = s(1296, 616)
GRID_COLS = 12
GRID_ROWS = 5
GRID_STEP = s(51, 51)
MARK_POS = s(956, 930)
C1_POS = s(960, 980)


# ====================== 更宽松的红点检测 ======================
def has_red_mark(pos, size=15):  # 1. 检测区域改大：15x15
    try:
        x, y = pos
        left = x - size // 2
        top = y - size // 2
        with mss.mss() as sct:
            img = sct.grab({"top": top, "left": left, "width": size, "height": size})
            img_np = np.array(img)
            r, g, b = img_np[:, :, 2], img_np[:, :, 1], img_np[:, :, 0]

            # 2. 降低红色阈值：R不用那么高，G/B可以更高
            # 3. 减少需要的像素数：有1个像素就算（或者你改成2）
            print(np.sum((r > 120) & (g < 130) & (b < 130)))
            return np.sum((r > 120) & (g < 130) & (b < 130)) >= 1
    except:
        return False


# ====================== 截图 ======================
def capture_region(region):
    with mss.mss() as sct:
        mon = {"top": region[1], "left": region[0],
               "width": region[2] - region[0], "height": region[3] - region[1]}
        return cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)


# ====================== 【全新升级】自适应膨胀识别坐标 ======================
# def get_block_centers():
#     print("\n正在识别...")
#     img = capture_region(SCAN_REGION)
#
#     # 1. 颜色提取
#     mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)
#
#     # 2. 形态学闭运算（将碎块强行黏合）。核大小同样适配屏幕分辨率
#     kernel_size = max(5, int(21 * scale_w))
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # 3. 寻找所有候选轮廓
#     contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     valid_rects = []
#     max_width = 0
#
#     # 【第一轮筛选】过滤噪点并找出最大宽度 (作为三连格的黄金基准)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if MIN_AREA < area < MAX_AREA:
#             x, y, w, h = cv2.boundingRect(cnt)
#             valid_rects.append((x, y, w, h))
#             if w > max_width:
#                 max_width = w
#
#     # 如果什么都没有识别到，直接安全返回空列表
#     if not valid_rects:
#         print("⚠️ 未检测到有效的大方格区域")
#         return []
#
#     # 【自适应计算】通过当前画面下的最大宽度，动态反推单格子的精准宽度
#     dynamic_single_width = max_width / 3.0
#
#     blocks = []
#     rx, ry = SCAN_REGION[0], SCAN_REGION[1]
#
#     # 【第二轮切分】根据自适应比例，对每一个大色块进行安全分拆
#     for x, y, w, h in valid_rects:
#         # 计算该大色块是单格宽度的多少倍，并四舍五入得出包含的格子数
#         ratio = w / dynamic_single_width
#         grid_count = max(1, round(ratio))
#
#         # 均匀计算格子切分点，确保在残缺情况下红点也在框内部
#         sub_w = w / grid_count
#         for i in range(grid_count):
#             cx = rx + x + int((i + 0.5) * sub_w)
#             cy = ry + y + h // 2
#             blocks.append((cx, cy))
#
#     print(f"✅ 动态识别成功！检测到 {len(valid_rects)} 个色块，共自适应切分出 {len(blocks)} 个可点击点")
#     return blocks


def get_block_centers():
    """
    自适应获取所有方格区域的中心点击坐标（绝对屏幕坐标）
    包含 20:9 几何宽高比判定与左侧遮挡/裁切自动补全逻辑
    """
    # 1. 抓取设定区域
    img = capture_region(SCAN_REGION)

    # 2. 颜色提取与形态学闭运算黏合
    mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)
    kernel_size = 21  # 可按需乘以 scale_w
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 寻找所有候选轮廓
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_rects = []
    max_width = 0

    # 【第一轮遍历】轮廓筛选 -> 20:9 宽高比检测 -> 几何反推左侧缺失补全量
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)
            diff_w = 0

            # 单格标准 20:9 ≈ 2.22，阈值取 1.8。低于 1.8 说明左侧被遮挡/裁切
            if aspect_ratio < 1.8:
                # 依据精确比例 20/9，根据无遮挡的高度 h 反推标准的单格宽度
                expected_single_w = h * (20.0 / 9.0)

                # 确保补全到标准几何宽度
                needed_w = max(w, int(expected_single_w))
                diff_w = needed_w - w

            processed_rects.append({'x': x, 'y': y, 'w': w, 'h': h, 'diff_w': diff_w})

    # 未检测到有效区域，安全返回空列表
    if not processed_rects:
        return []

    # 找出修补后的【最大宽度】，作为基准动态反推单格标准宽度
    for item in processed_rects:
        full_w = item['w'] + item['diff_w']
        if full_w > max_width:
            max_width = full_w

    dynamic_single_width = max_width / 3.0
    blocks = []

    # 【第二轮切分】计算补全后的连格数，并换算绝对屏幕坐标
    for item in processed_rects:
        x, y, w, h = item['x'], item['y'], item['w'], item['h']
        diff_w = item['diff_w']

        # 补全后的实际几何总宽
        real_w = w + diff_w

        # 计算连格数
        ratio = real_w / dynamic_single_width
        grid_count = max(1, round(ratio))

        # 均匀切分每个子格子，并计算中心点
        sub_w = real_w / grid_count
        for i in range(grid_count):
            # 核心换算：绝对屏幕 X 坐标 = 原本ROI相对X - 左侧补全量diff_w + 子格相对偏移 + SCAN_REGION起点X
            screen_x = (x - diff_w) + int((i + 0.5) * sub_w) + SCAN_REGION[0]
            screen_y = y + (h // 2) + SCAN_REGION[1]

            blocks.append((screen_x, screen_y))

    return blocks
# ====================== 点击流程 ======================
def auto_click_all(centers):
    print(f"\n{WAIT_DELAY / 2} 秒后开始点击")
    time.sleep(WAIT_DELAY / 2)

    for i, (cx, cy) in enumerate(centers, 1):
        check_keys()
        wait_continue()
        if stop or restart:
            return
        print(f"[{i}/{len(centers)}] 点击 ({cx},{cy})")
        pyautogui.click(cx, cy)
        time.sleep(CLICK_DELAY)
        pyautogui.click(*SECOND_CLICK_POS)
        time.sleep(CLICK_DELAY)

    pyautogui.click(*SECOND_CLICK_POS)
    print("\n✅ 所有块点击完成")
    time.sleep(CLICK_DELAY)

    # ========== 3次重试 → 暂停 ==========
    max_retry = 3
    retry = 0

    while retry < max_retry:
        check_keys()
        wait_continue()
        pyautogui.click(*C1_POS)
        pyautogui.click(*C1_POS)
        time.sleep(CLICK_DELAY)
        if has_red_mark(MARK_POS):
            print("✅ 红点已出现")
            pyautogui.moveTo(MARK_POS[0] - 5 * scale_w, MARK_POS[1] - 20 * scale_h)
            pyautogui.click()
            if has_red_mark(pyautogui.position()):
                print("检查到已经吸附")
                break
        retry += 1
        print(f"重试 {retry}/{max_retry}")

    if retry >= max_retry:
        print("\n⏸️ 3次失败 → 自动暂停")
        print("按 F10 继续")
        running = False
        while not running and not stop:
            check_keys()
            time.sleep(0.05)
        pyautogui.click(*C1_POS)
        time.sleep(CLICK_DELAY)
        pyautogui.click(*MARK_POS)
        time.sleep(CLICK_DELAY)


# ====================== 背包格子 ======================
def get_backpack_positions():
    ps = []
    for col in range(GRID_COLS):
        rows = range(GRID_ROWS) if col % 2 == 0 else reversed(range(GRID_ROWS))
        for row in rows:
            x = BACKPACK_START[0] + col * GRID_STEP[0]
            y = BACKPACK_START[1] + row * GRID_STEP[1]
            ps.append((x, y))
    return ps


# ====================== 主循环 ======================
def process_all_backpack():
    global running
    while not stop:
        running = False
        restart = False
        ps = get_backpack_positions()
        print(f"\n✅ 就绪 {len(ps)} 格 | F10 开始")
        print(f"✅ 当前分辨率：{screen_w}x{screen_h} | 面积缩放：{scale_area:.2f}x")
        show_speed()

        for idx, (bx, by) in enumerate(ps, 1):
            check_keys()
            if stop or restart:
                break
            wait_continue()

            print(f"\n===== 第 {idx}/{len(ps)} 格 =====")

            # ====================== 简化版 Ctrl 逻辑 ======================
            # 1. 去掉复杂检测，直接快速按下
            # 2. 确保所有地方都有 check_keys()
            time.sleep(0.05)  # 稍微等一下，确保按下

            # 执行点击
            # pyautogui.click()
            pyautogui.moveTo(bx, by)
            pyautogui.click()
            time.sleep(WAIT_DELAY)
            pyautogui.moveTo(screen_w // 4 + 150 * scale_w, screen_h // 2 - 50 * scale_h)
            pyautogui.click()

            time.sleep(WAIT_DELAY)
            # =================================================================

            cs = get_block_centers()
            if not cs:
                continue
            auto_click_all(cs)
            time.sleep(WAIT_DELAY)


# ====================== 启动 ======================
if __name__ == "__main__":
    print("=" * 70)
    print(" F10 运行/暂停 | F11 停止 | F12 重来 | ↑↓ 调节速度")
    print(" F3 设置点击间隔 | F4 设置等待间隔")
    print(f"✅ 完美自适应分辨率：{screen_w}x{screen_h}")
    print("✅ 已简化 Ctrl 逻辑，快捷键随时响应")
    print("=" * 70)
    process_all_backpack()