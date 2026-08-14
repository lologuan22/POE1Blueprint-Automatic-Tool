import os
import time
from itertools import product

import cv2
import mss
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True

# ====================== 1. 基础配置与绝对路径 ======================
BASE_W = 1920
BASE_H = 1080
screen_w, screen_h = pyautogui.size()

scale_w = screen_w / BASE_W
scale_h = screen_h / BASE_H


def s(x, y=None):
    if y is None:
        return int(x * scale_w)
    return int(x * scale_w), int(y * scale_h)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "name_templates")

# 💡 全局唯一 mss 实例（解决 Deprecation 警告并提升截图帧率）
sct = mss.mss()

# 模板尺寸与偏移
BASE_TEMPLATE_W = 60
BASE_TEMPLATE_H = 25
NAME_OFFSET_Y = 21

# 扫描区域与轮廓检测参数
SCAN_REGION = (s(320), s(100), s(1700), s(900))
BACKGROUND_LOWER = np.array([100, 120, 150], dtype=np.uint8)
BACKGROUND_UPPER = np.array([160, 180, 210], dtype=np.uint8)
MIN_AREA, MAX_AREA = int(3000 * scale_w * scale_h), int(250000 * scale_w * scale_h)

# 遮挡危险区与叉号按钮
MODAL_BAN_ZONE = (s(680), s(400), s(1220), s(680))
CLOSE_BTN_X, CLOSE_BTN_Y = s(1240, 410)

# ====================== 2. 映射表与人物优先级权重 ======================
CHARACTER_MAP = {
    "Vinderi": "Vinderi",
    "Karst": "Karst",
    "Nenet": "Nenet",
    "Gianna": "Gianna",
    "Huck": "Huck",
    "Tullina": "Tullina",
    "Isla": "Isla",
    "Tibbs": "Tibbs",
    "Niles": "Niles",
}

CHARACTER_PRIORITY = {
    "Vinderi": 1000,
    "Karst": 700,
    "Nenet": 500,
    "Gianna": 100,
    "Huck": 70,
    "Tullina": 60,
    "Isla": 50,
    "Tibbs": 40,
    "Niles": 20,
    "UNKNOWN": -1
}

# 1K 基准下的 5 个金圈锚点
FIVE_GOLD_ANCHORS_1K = [
    (825, 567),  # Pos 0: 3人最左
    (891, 567),  # Pos 1: 2人左
    (956, 567),  # Pos 2: 3人中
    (1021, 567),  # Pos 3: 2人右
    (1086, 567)  # Pos 4: 3人最右
]


# ====================== 3. 基础工具与防遮挡 ======================
def capture_region(region):
    monitor = {"top": region[1], "left": region[0],
               "width": region[2] - region[0], "height": region[3] - region[1]}
    img = sct.grab(monitor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def capture_full_screen():
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
    best_alias_name = "UNKNOWN"
    gray_roi = cv2.cvtColor(roi_1k_img, cv2.COLOR_BGR2GRAY)

    for template_filename, show_name in CHARACTER_MAP.items():
        template_path = os.path.join(TEMPLATE_DIR, f"{template_filename}.png")
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
                best_alias_name = template_filename
        except Exception:
            continue

    if best_alias_name in CHARACTER_MAP:
        return CHARACTER_MAP[best_alias_name], best_val
    return "UNKNOWN", 0.0


def is_in_ban_zone(x, y):
    x1, y1, x2, y2 = MODAL_BAN_ZONE
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def close_modal():
    print(f"  🧹 关窗防护：点击叉号 ({CLOSE_BTN_X}, {CLOSE_BTN_Y}) 清理弹窗视野...")
    pyautogui.moveTo(CLOSE_BTN_X, CLOSE_BTN_Y)
    pyautogui.click()
    time.sleep(0.12)


# ====================== 4. 色块识别与严谨 12:5 补齐算法 ======================
def crop_out_protrusions(cnt, mask_closed):
    """
    【精确裁剪凸角】：切掉膨胀或粘连产生的微小死角
    """
    x, y, w, h = cv2.boundingRect(cnt)
    roi = mask_closed[y:y + h, x:x + w]

    # 1. 垂直方向（切上下凸角）
    row_density = np.sum(roi > 0, axis=1) / float(w)
    valid_rows = np.where(row_density > 0.15)[0]
    if len(valid_rows) > 0:
        y1 = y + valid_rows[0]
        y2 = y + valid_rows[-1]
        h = y2 - y1 + 1
        y = y1

    # 2. 水平方向（切左右凸角）
    col_density = np.sum(roi > 0, axis=0) / float(h)
    valid_cols = np.where(col_density > 0.15)[0]
    if len(valid_cols) > 0:
        x1 = x + valid_cols[0]
        x2 = x + valid_cols[-1]
        w = x2 - x1 + 1
        x = x1

    return x, y, w, h


def get_all_block_regions():
    """
    【优化后的图像识别与切分算法】：
    1. 包含大色块保底救援逻辑（被遮挡时防漏掉）。
    2. 切除黏连死角。
    3. 以右边缘为锚点，强行按 12:5 (2.4:1) 向左扩充还原被遮挡的全尺寸色块。
    4. 均等输出 3 个绝对槽位点击坐标。
    """
    img = capture_region(SCAN_REGION)
    mask = cv2.inRange(img, BACKGROUND_LOWER, BACKGROUND_UPPER)
    kernel_size = max(5, int(15 * scale_w))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_blocks = []
    processed_rects = []

    LARGE_BLOCK_MIN_W = int(120 * scale_w)
    LARGE_BLOCK_MIN_H = int(40 * scale_h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, _, w_raw, h_raw = cv2.boundingRect(cnt)

        is_normal_area = (MIN_AREA < area < MAX_AREA)
        is_large_rescued_block = (w_raw >= LARGE_BLOCK_MIN_W or h_raw >= LARGE_BLOCK_MIN_H)

        if is_normal_area or is_large_rescued_block:
            # 1. 切除死角凸角
            cx, cy, cw, ch = crop_out_protrusions(cnt, mask_closed)

            if cw > 20 and ch > 10:
                # 2. 按 12:5 (2.4:1) 比例以右边缘为基准向左扩展补齐
                ideal_12_5_w = ch * (12.0 / 5.0)

                x_right = cx + cw
                corrected_cx = x_right - ideal_12_5_w

                # 记录补齐后的全框坐标（供可视化 debug 画图使用）
                processed_rects.append({
                    'x': int(corrected_cx),
                    'y': cy,
                    'w': int(ideal_12_5_w),
                    'h': ch,
                    'diff_w': 0
                })

                # 3. 绝对 3 等分计算 3 个槽位点击中心
                sub_slots = []
                for i in range(3):
                    slot_center_x = corrected_cx + (i + 0.5) * (ideal_12_5_w / 3.0) + SCAN_REGION[0]
                    slot_center_y = cy + (ch / 2.0) + SCAN_REGION[1]
                    sub_slots.append((int(slot_center_x), int(slot_center_y)))

                big_blocks.append(sub_slots)

    if not big_blocks:
        return [], []

    # 将色块由左至右排序
    # 同步对 big_blocks 和 processed_rects 排序
    combined = list(zip(big_blocks, processed_rects))
    combined.sort(key=lambda item: item[0][0][0])

    big_blocks = [item[0] for item in combined]
    processed_rects = [item[1] for item in combined]

    return big_blocks, processed_rects


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

        # 降采样回 1K 进行匹配
        if scale_w != 1.0 or scale_h != 1.0:
            roi_1k = cv2.resize(raw_roi, (BASE_TEMPLATE_W, BASE_TEMPLATE_H), interpolation=cv2.INTER_AREA)
        else:
            roi_1k = raw_roi

        char_name, score = match_character_name(roi_1k)

        if char_name != "UNKNOWN":
            candidates.append({
                'char_name': char_name,
                'pos_idx': pos_idx,
                'score': score,
                'click_pos': (rgx, rgy)
            })

    return candidates


# ====================== 5. 组合最优解计算 ======================
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


# ====================== 🎨 6. 最终分配图绘制输出 ======================
def draw_and_show_allocation_result(full_bg_img, raw_rects, all_blocks_results):
    canvas = full_bg_img.copy()

    # 1. 绘制遮挡危险区（半透明红色区域）
    x1, y1, x2, y2 = MODAL_BAN_ZONE
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(canvas, "BAN ZONE (MODAL COVER)", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 2. 绘制识别/还原后的完整色块边缘（绿色方框）
    for item in raw_rects:
        rx = item['x'] + SCAN_REGION[0]
        ry = item['y'] + SCAN_REGION[1]
        rw = item['w']
        rh = item['h']
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    # 3. 绘制每个色块对应槽位的选人结果
    for result in all_blocks_results:
        b_num = result['block_index']
        plan = result['plan']
        slots = result['slots']

        for idx, (choice, (sx, sy)) in enumerate(zip(plan, slots)):
            # 槽位中心标记点
            cv2.circle(canvas, (sx, sy), 8, (0, 0, 255), -1)
            cv2.circle(canvas, (sx, sy), 10, (255, 255, 255), 2)

            # 在槽位上方标出分配人物名字
            label = f"B{b_num}-Slot{idx + 1}: {choice['char_name']}"
            txt_x, txt_y = sx - 45, sy - 15

            # 黑底边框 + 黄色文字
            cv2.putText(canvas, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 窗口适应屏幕大小
    display_w = int(screen_w * 0.75)
    display_h = int(screen_h * 0.75)
    resized_canvas = cv2.resize(canvas, (display_w, display_h))

    print("\n🎨 [色块分配可视化结果图生成成功！]")
    cv2.imshow("Full Map Allocation Debug Window", resized_canvas)
    print("👉 请在弹出的图像窗口上按【任意键】或【ESC】关闭测试。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================== 7. 主控制流程 ======================
def run_full_allocation_test():
    print("====== 🚀 色块识别 & 全局人物分配测试 ======")
    print(f"屏幕分辨率: {screen_w}x{screen_h} (缩放比: {scale_w:.2f})")
    print("请迅速切回游戏界面，3秒后开始全局扫描...")
    time.sleep(3)

    bg_screenshot = capture_full_screen()

    # 1. 扫描色块
    big_blocks, raw_rects = get_all_block_regions()
    if not big_blocks:
        print("❌ 未检测到任何有效色块！")
        return

    print(f"\n✅ 成功扫描到 {len(big_blocks)} 个色块！开始逐个展开并识别槽位...")

    all_blocks_results = []

    # 2. 遍历色块与内部槽位
    for block_idx, target_block_slots in enumerate(big_blocks):
        print(f"\n" + "=" * 50)
        print(f"📦 正在扫描 色块 [{block_idx + 1}/{len(big_blocks)}] ({len(target_block_slots)} 个槽位)...")

        slot_candidates_list = []
        num_slots = len(target_block_slots)

        for slot_idx in range(num_slots):
            sx, sy = target_block_slots[slot_idx]
            print(f"  👉 点击展开 Slot [{slot_idx + 1}] -> 坐标: ({sx}, {sy})")
            pyautogui.moveTo(sx, sy)
            #time.sleep(0.02)
            pyautogui.click()
            time.sleep(0.05)

            candidates = scan_slot_candidates_at_current_screen()
            print(f"      └─ 识别结果: {[c['char_name'] for c in candidates]}")

            slot_candidates_list.append(candidates)

            # 防遮挡检查
            if slot_idx + 1 < num_slots:
                next_sx, next_sy = target_block_slots[slot_idx + 1]
                if is_in_ban_zone(next_sx, next_sy):
                    close_modal()
            else:
                close_modal()

        # 计算当前色块最优解
        valid_candidates_list = [c for c in slot_candidates_list if len(c) > 0]
        if valid_candidates_list:
            best_plan = solve_best_combination(valid_candidates_list)
            all_blocks_results.append({
                'block_index': block_idx + 1,
                'plan': best_plan,
                'slots': target_block_slots
            })

    # 3. 打印分配策略汇总
    print("\n" + "★" * 50)
    print("🎉 全图匹配规划完成！分配方案如下：")
    for result in all_blocks_results:
        b_num = result['block_index']
        print(f"  📦 色块 {b_num}:")
        for idx, choice in enumerate(result['plan']):
            print(f"      └─ Slot [{idx + 1}] -> 指派人物: [{choice['char_name']}]")

    # 4. 弹出画图窗口
    draw_and_show_allocation_result(bg_screenshot, raw_rects, all_blocks_results)


if __name__ == "__main__":
    run_full_allocation_test()