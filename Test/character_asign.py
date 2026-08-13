import cv2
import mss
import numpy as np
import time
import pyautogui
import os
from itertools import product

pyautogui.FAILSAFE = True

# ====================== 1. 基础配置与绝对路径 ======================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "name_templates")

BASE_W, BASE_H = 1920, 1080
screen_w, screen_h = pyautogui.size()
scale_w = screen_w / BASE_W
scale_h = screen_h / BASE_H

SCAN_REGION = (320, 100, 1700, 900)
BACKGROUND_LOWER = np.array([100, 120, 150], dtype=np.uint8)
BACKGROUND_UPPER = np.array([160, 180, 210], dtype=np.uint8)

MIN_AREA, MAX_AREA = 3000, 250000

# ====================== 2. 遮挡区与关闭按钮坐标 ======================
# 选人弹窗在屏幕中央覆盖的矩形区域: (左X, 上Y, 右X, 下Y) —— 绝对屏幕坐标
MODAL_BAN_ZONE = (680, 400, 1220, 680)

# 选人弹窗右上角关闭(X)按钮的屏幕绝对坐标
CLOSE_BTN_X = 1240
CLOSE_BTN_Y = 410

# ====================== 3. 人物优先级权重字典 ======================
CHARACTER_PRIORITY = {
    "Vinderi": 1000,  # 温德利 (最高优先)
    "Karst": 700,     # 卡斯特
    "Nenet": 500,     # 奈尼特
    "Gianna": 100,    # 吉安娜
    "Huck": 70,       # 哈克
    "Tullina": 60,    # 图林娜
    "Isla": 50,       # 伊斯拉
    "Tibbs": 40,      # 特卜斯
    "Niles": 20,      # 奈尔斯
    "UNKNOWN": -1     # 未识别
}

BASE_TEMPLATE_W, BASE_TEMPLATE_H = 60, 25
NAME_OFFSET_Y = 21

# 5 个金圈相对坐标（1K基准）
FIVE_GOLD_ANCHORS_1K = [
    (825, 567),  # Pos 0: 3人模式 - 最左
    (891, 567),  # Pos 1: 2人模式 - 左
    (956, 567),  # Pos 2: 3人模式 - 中
    (1021, 567), # Pos 3: 2人模式 - 右
    (1086, 567)  # Pos 4: 3人模式 - 最右
]


# ====================== 4. 底层工具与遮挡判断 ======================
def capture_region(region):
    with mss.mss() as sct:
        monitor = {"top": region[1], "left": region[0],
                   "width": region[2] - region[0], "height": region[3] - region[1]}
        img = sct.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def capture_full_screen():
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        img = sct.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


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

        if max_val > best_val and max_val > 0.65:
            best_val = max_val
            best_char = char_id

    return best_char, best_val


def is_in_ban_zone(x, y):
    """判断坐标 (x, y) 是否位于中央遮挡危险区内"""
    x1, y1, x2, y2 = MODAL_BAN_ZONE
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def close_modal():
    """安全清屏：点击弹窗右上角叉号关闭当前打开的弹窗"""
    print(f"  🧹 关窗防护：点击叉号 ({CLOSE_BTN_X}, {CLOSE_BTN_Y}) 清理弹窗视野...")
    pyautogui.click(CLOSE_BTN_X, CLOSE_BTN_Y)
    time.sleep(0.12)  # 等待关窗动画完毕


# ====================== 5. 带左侧修复的全图多色块检测 ======================
def get_all_block_regions():
    img = capture_region(SCAN_REGION)
    mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
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
            if aspect_ratio < 1.8:  # 左侧遮挡补全
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

    # 按从左到右整体排序
    big_blocks.sort(key=lambda block: block[0][0])
    return big_blocks


def scan_slot_candidates_at_current_screen():
    full_screenshot = capture_full_screen()

    candidates = []
    for pos_idx, (gx, gy) in enumerate(FIVE_GOLD_ANCHORS_1K):
        rgx, rgy = int(gx * scale_w), int(gy * scale_h)
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
                'pos_idx': pos_idx,      # 位置索引 (0最左)
                'score': score,
                'click_pos': (rgx, rgy)  # 金圈真实坐标
            })

    return candidates


# ====================== 6. 单色块求解算法 ======================
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
            char_prio = CHARACTER_PRIORITY.get(choice['char_name'], 0)
            left_bonus = (5 - choice['pos_idx']) * 0.1
            score += (char_prio + left_bonus)

        if score > best_score:
            best_score = score
            best_combo = combo

    return list(best_combo) if best_combo else []


# ====================== 7. 主流程 (正确的遮挡防呆逻辑) ======================
def test_all_blocks_decision():
    print("====== 🚀【全色块识别与无冲突策略规划】启动 ======")
    print("请切回游戏界面，3秒后开始扫描...")
    time.sleep(3)

    # 1. 扫描所有色块
    big_blocks = get_all_block_regions()
    if not big_blocks:
        print("❌ 未检测到任何色块，测试终止！")
        return

    print(f"\n✅ 全图成功检测到 {len(big_blocks)} 个色块！开始依次识别...")

    all_blocks_results = []

    # 2. 遍历所有色块
    for block_idx, target_block_slots in enumerate(big_blocks):
        print(f"\n" + "=" * 60)
        print(f"📦 正在处理 第 [{block_idx + 1}/{len(big_blocks)}] 个色块 (包含 {len(target_block_slots)} 个槽位)...")
        print("=" * 60)

        slot_candidates_list = []
        num_slots = len(target_block_slots)

        # 遍历当前色块内部的所有槽位
        for slot_idx in range(num_slots):
            sx, sy = target_block_slots[slot_idx]

            print(f"\n👉 [色块{block_idx + 1}] 点击展开 Slot [{slot_idx + 1}] (坐标: {sx}, {sy})...")
            pyautogui.click(sx, sy)
            time.sleep(0.15)  # 等待弹窗展开

            # 识别当前槽位展开的候选人
            candidates = scan_slot_candidates_at_current_screen()

            print(f"  └─ Slot [{slot_idx + 1}] 识别出的候选人:")
            for cand in candidates:
                prio = CHARACTER_PRIORITY.get(cand['char_name'], 0)
                print(
                    f"      • 人物: {cand['char_name']:<10} | 位置: Pos_{cand['pos_idx']} | 权重: {prio} | 匹配度: {cand['score']:.2f}")

            if not candidates:
                print(f"  ⚠️ 警告：Slot [{slot_idx + 1}] 未识别到候选人！")

            slot_candidates_list.append(candidates)

            # 💡【核心修复逻辑】：检查下一个槽位
            # 如果还有下一个槽位，且下一个槽位落在了遮挡危险区内，说明当前展开的这个弹窗会挡住下一个槽位！
            # 必须现在就把当前的弹窗关掉，确保下一个槽位暴露出来！
            if slot_idx + 1 < num_slots:
                next_sx, next_sy = target_block_slots[slot_idx + 1]
                if is_in_ban_zone(next_sx, next_sy):
                    print(f"  ⚠️ 预判到下一个 Slot [{slot_idx + 2}] 位于遮挡区！提前关闭当前弹窗...")
                    close_modal()
            else:
                # 最后一个槽位处理完毕后，如果它本身在遮挡区（或者为了全局干净），可以关掉弹窗
                if is_in_ban_zone(sx, sy):
                    close_modal()

        # 计算当前色块内部的最佳无冲突方案
        valid_candidates_list = [c for c in slot_candidates_list if len(c) > 0]
        if valid_candidates_list:
            best_plan = solve_best_combination(valid_candidates_list)
            all_blocks_results.append({
                'block_index': block_idx + 1,
                'plan': best_plan
            })
        else:
            print(f"⚠️ 色块 [{block_idx + 1}] 未搜集到有效人物数据，跳过组合计算。")

    # 3. 汇总打印
    print("\n" + "★" * 60)
    print("🎉 🎉 🎉 全图所有色块扫描完毕！各色块最终算法决策结果汇总：")
    print("★" * 60)

    for result in all_blocks_results:
        b_num = result['block_index']
        plan = result['plan']
        print(f"\n📦 【色块 {b_num}】的最优分配策略：")
        for idx, choice in enumerate(plan):
            print(
                f"  └─ Slot [{idx + 1}] -> 选择: [{choice['char_name']:<10}] | 位置: Pos_{choice['pos_idx']} | 对应金圈: {choice['click_pos']}")

    print("\n" + "★" * 60)
    print("💡 规划完毕！")


if __name__ == "__main__":
    test_all_blocks_decision()