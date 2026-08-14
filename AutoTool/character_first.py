import os
import sys
import time
from itertools import product

import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard as pynput_keyboard

# ====================== 1. 全局控制与 PyInstaller 路径自适应 ======================
pyautogui.FAILSAFE = True
running = False
stop = False
restart = False
USE_EXTENDED_BACKPACK = False  # ⚡ F12 开关：是否将处理完的图纸顺手放回扩展背包

# ⚡⚡⚡ 兼容 PyInstaller 打包（临时解压目录）与源码运行 ⚡⚡⚡
if getattr(sys, 'frozen', False):
    CURRENT_DIR = sys._MEIPASS
else:
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
    "Tullina": 300,
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
MAX_AREA = int(250000 * scale_area)

SCAN_REGION = (s(320), s(100), s(1700), s(900))
SECOND_CLICK_POS = s(860, 540)

# 主背包坐标 (12列 x 5行 = 60格)
BACKPACK_START_X = 1301.5 * scale_w
BACKPACK_START_Y = 616.0 * scale_h
GRID_COLS = 12
GRID_ROWS = 5
GRID_STEP_X = 52.3 * scale_w
GRID_STEP_Y = 52.3 * scale_h
GRID_SIZE = int(50 * scale_w)

# 🚩 修正：扩展背包配置 (6列 x 5行 = 30格)
EXT_COLS = 6
EXT_ROWS = 5
EXT_GAP = 17.0 * scale_w  # 主背包与扩展背包的间隔

# 🚩 核心修正：低于 10 个像素的垃圾红点直接丢弃！
MIN_RED_PIXELS = 10

MARK_POS = s(956, 930)
C1_POS = s(960, 980)


# ====================== 7. pynput 后台异步按键监听器（零延迟） ======================
def show_speed():
    print(f"\r✅ CLICK_DELAY={CLICK_DELAY:.2f}s | WAIT_DELAY={WAIT_DELAY:.2f}s | ↑加快 ↓减慢 ", end="")


def on_press(key):
    global running, stop, restart, USE_EXTENDED_BACKPACK, CLICK_DELAY, WAIT_DELAY
    try:
        if key == pynput_keyboard.Key.f10:
            running = not running
            print("\n▶ 运行" if running else "\n⏸ 暂停")
        elif key == pynput_keyboard.Key.f11:
            stop = True
            print("\n🛑 已停止")
        elif key == pynput_keyboard.Key.f12:
            USE_EXTENDED_BACKPACK = not USE_EXTENDED_BACKPACK
            status = f"[开启] -> 放回左侧扩展背包(横{EXT_COLS}x竖{EXT_ROWS})" if USE_EXTENDED_BACKPACK else "[关闭] -> 放回主背包原位"
            print(f"\n🔄 扩展背包放回模式切换为: {status}")
        elif key == pynput_keyboard.Key.up:
            CLICK_DELAY = max(0.01, CLICK_DELAY - STEP)
            WAIT_DELAY = max(0.05, WAIT_DELAY - STEP)
            show_speed()
        elif key == pynput_keyboard.Key.down:
            CLICK_DELAY += STEP
            WAIT_DELAY += STEP
            show_speed()
    except Exception:
        pass


# 开启独立线程监听
listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()


def check_keys():
    pass


def wait_continue():
    """等待暂停恢复"""
    while not running and not stop and not restart:
        time.sleep(0.01)


def safe_click(x, y):
    """拟真点击：先 moveTo 建立 UI Hover，再 click"""
    pyautogui.moveTo(x, y)
    pyautogui.click()


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
    except Exception:
        return False


# ====================== 8. 遮挡防护与图像模板匹配 ======================
def is_in_ban_zone(x, y):
    x1, y1, x2, y2 = MODAL_BAN_ZONE
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def close_modal():
    safe_click(CLOSE_BTN_X, CLOSE_BTN_Y)
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

        try:
            th, tw = template.shape[:2]
            rh, rw = gray_roi.shape[:2]

            if rh < th or rw < tw:
                gray_roi_eval = cv2.resize(gray_roi, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                gray_roi_eval = gray_roi

            res = cv2.matchTemplate(gray_roi_eval, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_val and max_val > 0.30:
                best_val = max_val
                best_char = char_id
        except Exception:
            continue

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


# ====================== 9. 色块切死角 + 死锁 3 槽位算法 ======================
def crop_out_protrusions(cnt, mask_closed):
    x, y, w, h = cv2.boundingRect(cnt)
    roi = mask_closed[y:y + h, x:x + w]

    row_density = np.sum(roi > 0, axis=1) / float(w)
    valid_rows = np.where(row_density > 0.15)[0]
    if len(valid_rows) > 0:
        y1 = y + valid_rows[0]
        y2 = y + valid_rows[-1]
        h = y2 - y1 + 1
        y = y1

    col_density = np.sum(roi > 0, axis=0) / float(h)
    valid_cols = np.where(col_density > 0.15)[0]
    if len(valid_cols) > 0:
        x1 = x + valid_cols[0]
        x2 = x + valid_cols[-1]
        w = x2 - x1 + 1
        x = x1

    return x, y, w, h


def get_all_big_blocks():
    img = capture_region(SCAN_REGION)
    mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)

    kw = max(25, int(45 * scale_w))
    kh = max(3, int(5 * scale_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_blocks = []

    LARGE_BLOCK_MIN_W = int(120 * scale_w)
    LARGE_BLOCK_MIN_H = int(40 * scale_h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, _, w_raw, h_raw = cv2.boundingRect(cnt)

        is_normal_area = (MIN_AREA < area < MAX_AREA)
        is_large_rescued_block = (w_raw >= LARGE_BLOCK_MIN_W or h_raw >= LARGE_BLOCK_MIN_H)

        if is_normal_area or is_large_rescued_block:
            cx, cy, cw, ch = crop_out_protrusions(cnt, mask_closed)

            if cw > 20 and ch > 10:
                ideal_12_5_w = ch * (12.0 / 5.0)
                x_right = cx + cw
                corrected_cx = x_right - ideal_12_5_w

                sub_slots = []
                for i in range(3):
                    slot_center_x = corrected_cx + (i + 0.5) * (ideal_12_5_w / 3.0) + SCAN_REGION[0]
                    slot_center_y = cy + (ch / 2.0) + SCAN_REGION[1]
                    sub_slots.append((int(slot_center_x), int(slot_center_y)))

                big_blocks.append(sub_slots)

    if not big_blocks:
        return []

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


# ====================== 10. 全图先识别，后统一分配点击（含详细日志） ======================
def process_and_execute_smart_plan():
    big_blocks = get_all_big_blocks()
    if not big_blocks:
        print("⚠️ 未检测到有效的大方格区域")
        return

    print(f"✅ 动态识别成功！检测到 {len(big_blocks)} 个色块，开始【阶段一：全图纯识别】...")

    all_blocks_plans = []

    # ==================== 🔍 阶段一：纯扫描与方案计算 ====================
    for block_idx, target_block_slots in enumerate(big_blocks):
        wait_continue()
        if stop or restart:
            return

        print(f"  ------------------- 📦 色块 [{block_idx + 1}/{len(big_blocks)}] 识别中 -------------------")
        slot_candidates_list = []
        num_slots = len(target_block_slots)

        for slot_idx in range(num_slots):
            wait_continue()
            if stop or restart:
                return

            sx, sy = target_block_slots[slot_idx]
            safe_click(sx, sy)
            time.sleep(0.12)

            candidates = scan_slot_candidates_at_current_screen()
            slot_candidates_list.append(candidates)

            # ⚡ 打印当前 Slot 提取到的人名列表
            if candidates:
                cand_str = ", ".join([f"{c['char_name']}(匹配度:{c['score']:.2f})" for c in candidates])
                print(f"     📍 Slot {slot_idx + 1} ({sx}, {sy}) 识别到: [ {cand_str} ]")
            else:
                print(f"     📍 Slot {slot_idx + 1} ({sx}, {sy}) 未识别到有效人物")

            if slot_idx + 1 < num_slots:
                next_sx, next_sy = target_block_slots[slot_idx + 1]
                if is_in_ban_zone(next_sx, next_sy):
                    close_modal()

        valid_candidates_list = [c for c in slot_candidates_list if len(c) > 0]
        if valid_candidates_list:
            best_plan = solve_best_combination(valid_candidates_list)
            all_blocks_plans.append((target_block_slots, best_plan))
            # ⚡ 打印最终为此色块规划的算法最优解
            plan_str = " -> ".join([c['char_name'] for c in best_plan])
            print(f"     🎯 色块 [{block_idx + 1}] 决策阵容: {plan_str}\n")

        close_modal()

    # ==================== 🎯 阶段二：统一纯粹执行点击 ====================
    print("🚀 全图识别完毕！开始【阶段二：全图统一上阵点击】...")

    for block_idx, (target_block_slots, best_plan) in enumerate(all_blocks_plans):
        print(f"📦 执行色块 [{block_idx + 1}] 上阵...")
        for slot_i, choice in enumerate(best_plan):
            wait_continue()
            if stop or restart:
                return

            sx, sy = target_block_slots[slot_i]
            safe_click(sx, sy)
            time.sleep(CLICK_DELAY)

            gx, gy = choice['gold_pos']
            target_x = gx
            target_y = gy - TARGET_OFFSET_Y

            safe_click(target_x, target_y)
            time.sleep(CLICK_DELAY)
            print(f"  └─ Slot [{slot_i + 1}] -> 点击坐标: ({target_x}, {target_y}) 上阵 [{choice['char_name']}]")

    safe_click(*SECOND_CLICK_POS)
    print("✅ 所有色块统一点击与选人完成\n")
    time.sleep(CLICK_DELAY)

    # ========== 3次重试 → 暂停 ==========
    max_retry = 3
    retry = 0

    while retry < max_retry:
        wait_continue()
        if stop or restart:
            return

        safe_click(*C1_POS)
        safe_click(*C1_POS)
        time.sleep(CLICK_DELAY)

        if has_red_mark(MARK_POS):
            print("✅ 红点已出现")
            safe_click(MARK_POS[0] - int(5 * scale_w), MARK_POS[1] - int(20 * scale_h))
            if has_red_mark(pyautogui.position()):
                print("检查到已经吸附")
                break
        retry += 1
        print(f"重试 {retry}/{max_retry}")

    if retry >= max_retry:
        print("\n⏸️ 3次失败 → 自动暂停，按 F10 继续")
        global running
        running = False
        wait_continue()
        safe_click(*C1_POS)
        time.sleep(CLICK_DELAY)
        safe_click(*MARK_POS)
        time.sleep(CLICK_DELAY)


# ====================== 11. 智能背包扫描与坐标计算 ======================
def get_extended_fill_positions():
    """ 生成扩展背包 30 格坐标 (S 型排列: 6列 x 5行) """
    ps = []
    ext_start_x = BACKPACK_START_X - (EXT_COLS * GRID_STEP_X) - EXT_GAP
    ext_start_y = BACKPACK_START_Y

    for col in range(EXT_COLS):
        rows = range(EXT_ROWS) if col % 2 == 0 else reversed(range(EXT_ROWS))
        for row in rows:
            cx = int(ext_start_x + col * GRID_STEP_X)
            cy = int(ext_start_y + row * GRID_STEP_Y)
            ps.append((cx, cy))
    return ps


def scan_backpack_for_targets():
    """ 智能扫描背包 60 格 """
    print("\n📸 正在智能扫描背包...")
    screen_img = capture_full_screen()
    target_coords = []

    already_done_count = 0
    ignored_count = 0

    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            grid_idx = col * GRID_ROWS + row + 1

            center_x = int(BACKPACK_START_X + col * GRID_STEP_X)
            center_y = int(BACKPACK_START_Y + row * GRID_STEP_Y)

            half_box = GRID_SIZE // 2
            x1, y1 = max(0, center_x - half_box), max(0, center_y - half_box)
            x2, y2 = min(screen_w, center_x + half_box), min(screen_h, center_y + half_box)

            roi = screen_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

            # 红色标记检测
            red_roi_mask = (r >= 140) & \
                           ((r.astype(int) - g.astype(int)) >= 45) & \
                           ((r.astype(int) - b.astype(int)) >= 45)

            has_red = np.sum(red_roi_mask) >= MIN_RED_PIXELS

            # 羊皮纸图纸特征检测
            map_bg_roi_mask = (r >= 70) & (r <= 190) & \
                              (g >= 60) & (g <= 180) & \
                              (b >= 50) & (b <= 170) & \
                              (r.astype(int) >= g.astype(int)) & \
                              (g.astype(int) >= b.astype(int))

            bg_pixel_count = np.sum(map_bg_roi_mask)
            is_map = bg_pixel_count >= int(180 * scale_area)

            if has_red:
                already_done_count += 1
            elif is_map:
                target_coords.append((center_x, center_y))
                print(f"  └─ 🎯 [格子 {grid_idx:02d}] 锁定待处理图纸，坐标: ({center_x}, {center_y})")
            else:
                ignored_count += 1

    print("\n📊 扫描分析结果：")
    print(f"  🔴 已标记(红色像素>={MIN_RED_PIXELS})：{already_done_count} 个")
    print(f"  ⚪ 杂物 / 石头 / 垃圾点跳过    ：{ignored_count} 个")
    print(f"  🟡 本轮待处理目标图纸         ：{len(target_coords)} 个\n")

    return target_coords


def process_all_backpack():
    global running, restart
    while not stop:
        running = False
        restart = False
        ext_status = f"[开启 - 处理后放入扩展背包(横{EXT_COLS}x竖{EXT_ROWS})]" if USE_EXTENDED_BACKPACK else "[关闭 - 处理后放回主背包原位]"
        print(f"\n✅ 自动化脚本准备就绪 | F10 运行 | F12 扩展模式: {ext_status}")
        print(f"✅ 当前分辨率：{screen_w}x{screen_h} | 缩放比例：{scale_area:.2f}x")
        show_speed()

        wait_continue()
        if stop:
            break

        target_positions = scan_backpack_for_targets()

        if not target_positions:
            print("💡 背包中没有检测到需要处理的图纸，程序自动暂停。")
            continue

        ext_fill_positions = get_extended_fill_positions()
        ext_put_idx = 0

        total_targets = len(target_positions)
        print(f"🚀 锁定 {total_targets} 张图纸，开始精确处理...")

        for idx, (bx, by) in enumerate(target_positions, 1):
            if stop or restart:
                break
            wait_continue()

            print(f"\n===== 处理第 [{idx}/{total_targets}] 张图纸 (主背包源坐标: {bx}, {by}) =====")
            time.sleep(0.05)

            safe_click(bx, by)
            time.sleep(WAIT_DELAY)

            safe_click(screen_w // 4 + int(150 * scale_w), screen_h // 2 - int(50 * scale_h))
            time.sleep(WAIT_DELAY)

            process_and_execute_smart_plan()
            time.sleep(WAIT_DELAY)

            if USE_EXTENDED_BACKPACK and ext_put_idx < len(ext_fill_positions):
                rx, ry = ext_fill_positions[ext_put_idx]
                safe_click(rx, ry)
                print(f"  └─ 📥 图纸已放入扩展背包 第 [{ext_put_idx + 1}/30] 格 (坐标: {rx}, {ry})")
                ext_put_idx += 1
            else:
                safe_click(bx, by)
                print(f"  └─ ↩️ 图纸已放回主背包原位 (坐标: {bx}, {by})")

            time.sleep(WAIT_DELAY)


# ====================== 12. 启动入口 ======================
if __name__ == "__main__":
    print("=" * 70)
    print(" F10 运行/暂停 | F11 结束 | F12 切换扩展背包放回(6x5) | ↑↓ 调节速度")
    print(f"✅ 完美自适应分辨率：{screen_w}x{screen_h}")
    print(f"✅ 强效红点过滤：必须 >= {MIN_RED_PIXELS} 个红色像素才算做已标记，彻底杜绝噪点干扰！")
    print("=" * 70)
    process_all_backpack()