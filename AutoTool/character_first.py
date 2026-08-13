import os
import time
from itertools import product

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

# ====================== 1. 全局控制与路径配置 ======================
pyautogui.FAILSAFE = True
running = False
stop = False
restart = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "name_templates")

# ====================== 2. 分辨率自适应 ======================
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


# ====================== 3. 延迟参数 ======================
CLICK_DELAY = 0.03
WAIT_DELAY = 0.2
STEP = 0.01

# ====================== 4. 遮挡区与关闭按钮坐标 ======================
MODAL_BAN_ZONE = (s(680), s(400), s(1220), s(680))
CLOSE_BTN_X, CLOSE_BTN_Y = s(1240, 410)

# ====================== 5. 人物优先级权重字典 ======================
CHARACTER_PRIORITY = {
    "Vinderi": 1000,  # 温德利
    "Karst": 700,     # 卡斯特
    "Nenet": 500,     # 奈尼特
    "Gianna": 100,    # 工具人
    "Huck": 100,
    "Tullina": 100,
    "Isla": 100,
    "Tibbs": 100,
    "Niles": 100,
    "UNKNOWN": -1
}

BASE_TEMPLATE_W, BASE_TEMPLATE_H = 60, 25
NAME_OFFSET_Y = 21

FIVE_GOLD_ANCHORS_1K = [
    (825, 567),   # Pos 0: 3人最左
    (891, 567),   # Pos 1: 2人左
    (956, 567),   # Pos 2: 3人中
    (1021, 567),  # Pos 3: 2人右
    (1086, 567)   # Pos 4: 3人最右
]

TARGET_OFFSET_Y = s(70)

# ====================== 6. 统一坐标与检测范围 ======================
BACKGROUND_LOWER = np.array([100, 120, 150], np.uint8)
BACKGROUND_UPPER = np.array([160, 180, 210], np.uint8)

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


# ====================== 7. 工具与快捷键函数 ======================
def show_speed():
    print(f"\r✅ CLICK_DELAY={CLICK_DELAY:.2f}s | WAIT_DELAY={WAIT_DELAY:.2f}s | ↑加快 ↓减慢 ", end="")


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


def capture_region(region):
    with mss.mss() as sct:
        mon = {"top": region[1], "left": region[0],
               "width": region[2] - region[0], "height": region[3] - region[1]}
        return cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)


def capture_full_screen():
    with mss.mss() as sct:
        mon = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        return cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)


def has_red_mark(pos, size=15):
    try:
        x, y = pos
        left = x - size // 2
        top = y - size // 2
        with mss.mss() as sct:
            img = sct.grab({"top": top, "left": left, "width": size, "height": size})
            img_np = np.array(img)
            r, g, b = img_np[:, :, 2], img_np[:, :, 1], img_np[:, :, 0]
            return np.sum((r > 120) & (g < 130) & (b < 130)) >= 1
    except:
        return False


# ====================== 8. 遮挡防护与图像模板匹配 ======================
def is_in_ban_zone(x, y):
    x1, y1, x2, y2 = MODAL_BAN_ZONE
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def close_modal():
    pyautogui.click(CLOSE_BTN_X, CLOSE_BTN_Y)
    time.sleep(0.12)


def is_golden_pixel(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_gold = np.array([15, 120, 150])
    upper_gold = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    return np.sum(mask > 0) >= 4


def match_character_name(roi_1k_img):
    best_val = -1.0
    best_char = "UNKNOWN"
    gray_roi = cv2.cvtColor(roi_1k_img, cv2.COLOR_BGR2GRAY)

    for char_id in CHARACTER_PRIORITY.keys():
        if char_id == "UNKNOWN":
            continue
        template_path = os.path.join(TEMPLATE_DIR, f"{char_id}.png")
        if not os.path.exists(template_path):
            continue

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        res = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_val and max_val > 0.35:
            best_val = max_val
            best_char = char_id

    return best_char, best_val


def scan_slot_candidates_at_current_screen():
    full_screenshot = capture_full_screen()
    candidates = []
    for pos_idx, (gx, gy) in enumerate(FIVE_GOLD_ANCHORS_1K):
        rgx, rgy = s(gx, gy)
        gold_patch = full_screenshot[rgy - 3:rgy + 4, rgx - 3:rgx + 4]

        if not is_golden_pixel(gold_patch):
            continue

        cur_name_w = int(BASE_TEMPLATE_W * scale_w)
        cur_name_h = int(BASE_TEMPLATE_H * scale_h)
        cur_name_x = rgx - (cur_name_w // 2)
        cur_name_y = rgy + int(NAME_OFFSET_Y * scale_h)

        raw_roi = full_screenshot[cur_name_y: cur_name_y + cur_name_h, cur_name_x: cur_name_x + cur_name_w]
        roi_1k = cv2.resize(raw_roi, (BASE_TEMPLATE_W, BASE_TEMPLATE_H), interpolation=cv2.INTER_AREA)

        char_name, score = match_character_name(roi_1k)

        if char_name != "UNKNOWN":
            candidates.append({
                'char_name': char_name,
                'pos_idx': pos_idx,
                'score': score,
                'gold_pos': (rgx, rgy)
            })

    return candidates


# ====================== 9. 色块分组与梯队算法 ======================
def get_all_big_blocks():
    img = capture_region(SCAN_REGION)
    mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)
    kernel_size = max(5, int(21 * scale_w))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_rects = []
    max_width = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            diff_w = 0
            if aspect_ratio < 1.8:
                expected_single_w = h * (20.0 / 9.0)
                needed_w = max(w, int(expected_single_w))
                diff_w = needed_w - w

            processed_rects.append({'x': x, 'y': y, 'w': w, 'h': h, 'diff_w': diff_w})

    if not processed_rects:
        return []

    for item in processed_rects:
        full_w = item['w'] + item['diff_w']
        if full_w > max_width:
            max_width = full_w

    dynamic_single_width = max_width / 3.0
    big_blocks = []

    for item in processed_rects:
        real_w = item['w'] + item['diff_w']
        grid_count = max(1, round(real_w / dynamic_single_width))
        sub_w = real_w / grid_count

        sub_slots = []
        for i in range(grid_count):
            screen_x = (item['x'] - item['diff_w']) + int((i + 0.5) * sub_w) + SCAN_REGION[0]
            screen_y = item['y'] + (item['h'] // 2) + SCAN_REGION[1]
            sub_slots.append((screen_x, screen_y))

        big_blocks.append(sub_slots)

    big_blocks.sort(key=lambda block: block[0][0])
    return big_blocks


def get_position_bonus(pos_idx):
    if pos_idx == 0:
        return 100
    elif pos_idx in (1, 2):
        return 50
    return 0


def solve_best_combination(slot_candidates_list):
    if not slot_candidates_list:
        return []

    all_combinations = list(product(*slot_candidates_list))
    best_combo = None
    best_score = -999999

    for combo in all_combinations:
        chosen_chars = [c['char_name'] for c in combo]
        unique_count = len(set(chosen_chars))
        duplicate_penalty = (len(chosen_chars) - unique_count) * 10000

        score = -duplicate_penalty

        for choice in combo:
            char_prio = CHARACTER_PRIORITY.get(choice['char_name'], 100)
            pos_bonus = get_position_bonus(choice['pos_idx'])
            score += (char_prio + pos_bonus)

        if score > best_score:
            best_score = score
            best_combo = combo

    return list(best_combo) if best_combo else []


# ====================== 10. 全图先识别，后统一分配点击 ======================
def process_and_execute_smart_plan():
    big_blocks = get_all_big_blocks()
    if not big_blocks:
        print("⚠️ 未检测到有效的大方格区域")
        return

    print(f"✅ 动态识别成功！检测到 {len(big_blocks)} 个色块，开始【阶段一：全图纯识别】...")

    # 存储全局所有色块算出来的最佳方案：[ (block_slots, best_plan), ... ]
    all_blocks_plans = []

    # ==================== 🔍 阶段一：纯扫描与方案计算 ====================
    for block_idx, target_block_slots in enumerate(big_blocks):
        check_keys()
        wait_continue()
        if stop or restart:
            return

        slot_candidates_list = []
        num_slots = len(target_block_slots)

        for slot_idx in range(num_slots):
            check_keys()
            wait_continue()
            if stop or restart:
                return

            sx, sy = target_block_slots[slot_idx]
            pyautogui.click(sx, sy)
            time.sleep(0.15)

            candidates = scan_slot_candidates_at_current_screen()
            slot_candidates_list.append(candidates)

            # 遮挡预判
            if slot_idx + 1 < num_slots:
                next_sx, next_sy = target_block_slots[slot_idx + 1]
                if is_in_ban_zone(next_sx, next_sy):
                    close_modal()

        # 扫描完当前色块，计算最优解存入列表
        valid_candidates_list = [c for c in slot_candidates_list if len(c) > 0]
        if valid_candidates_list:
            best_plan = solve_best_combination(valid_candidates_list)
            all_blocks_plans.append((target_block_slots, best_plan))
            print(f"  └─ 色块 [{block_idx + 1}] 识别完毕，已规划出方案: {[c['char_name'] for c in best_plan]}")

        # 当前色块扫描完毕，清理弹窗准备扫描下一个色块
        close_modal()

    # ==================== 🎯 阶段二：统一纯粹执行点击 ====================
    print("\n🚀 全图识别完毕！开始【阶段二：全图统一上阵点击】...")

    for block_idx, (target_block_slots, best_plan) in enumerate(all_blocks_plans):
        print(f"📦 执行色块 [{block_idx + 1}] 上阵...")
        for slot_i, choice in enumerate(best_plan):
            check_keys()
            wait_continue()
            if stop or restart:
                return

            # 1. 点击区域槽位弹窗
            sx, sy = target_block_slots[slot_i]
            pyautogui.click(sx, sy)
            time.sleep(CLICK_DELAY)

            # 2. 点击对应金圈 Y - 70 像素
            gx, gy = choice['gold_pos']
            target_x = gx
            target_y = gy - TARGET_OFFSET_Y

            pyautogui.moveTo(target_x, target_y)
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(CLICK_DELAY)
            print(f"  └─ Slot [{slot_i + 1}] -> 点击坐标: ({target_x}, {target_y}) 上阵 [{choice['char_name']}]")

    pyautogui.click(*SECOND_CLICK_POS)
    print("\n✅ 所有色块统一点击与选人完成")
    time.sleep(CLICK_DELAY)

    # ========== 3次重试 → 暂停 ==========
    max_retry = 3
    retry = 0

    while retry < max_retry:
        check_keys()
        wait_continue()
        if stop or restart:
            return

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
        global running
        running = False
        while not running and not stop:
            check_keys()
            time.sleep(0.05)
        pyautogui.click(*C1_POS)
        time.sleep(CLICK_DELAY)
        pyautogui.click(*MARK_POS)
        time.sleep(CLICK_DELAY)


# ====================== 11. 背包网格与主循环 ======================
def get_backpack_positions():
    ps = []
    for col in range(GRID_COLS):
        rows = range(GRID_ROWS) if col % 2 == 0 else reversed(range(GRID_ROWS))
        for row in rows:
            x = BACKPACK_START[0] + col * GRID_STEP[0]
            y = BACKPACK_START[1] + row * GRID_STEP[1]
            ps.append((x, y))
    return ps


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
            time.sleep(0.05)

            pyautogui.moveTo(bx, by)
            pyautogui.click()
            time.sleep(WAIT_DELAY)

            pyautogui.moveTo(screen_w // 4 + 150 * scale_w, screen_h // 2 - 50 * scale_h)
            pyautogui.click()
            time.sleep(WAIT_DELAY)

            process_and_execute_smart_plan()
            time.sleep(WAIT_DELAY)


# ====================== 12. 启动 ======================
if __name__ == "__main__":
    print("=" * 70)
    print(" F10 运行/暂停 | F11 停止 | F12 重来 | ↑↓ 调节速度")
    print(f"✅ 完美自适应分辨率：{screen_w}x{screen_h}")
    print("✅ 已更新为【全图先纯识别，后统一上阵点击】模式！")
    print("=" * 70)
    process_all_backpack()