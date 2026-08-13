import cv2
import os
import numpy as np
import mss
import time
import pyautogui

# ====================== 1. 基础分辨率自适应配置 ======================
BASE_W = 1920
BASE_H = 1080
screen_w, screen_h = pyautogui.size()

# 计算全局分辨率缩放比例（以 1K 为基准 1.0）
scale_w = screen_w / BASE_W
scale_h = screen_h / BASE_H


def s(x, y=None):
    if y is None:
        return int(x * scale_w)
    return int(x * scale_w), int(y * scale_h)


# ====================== 2. 静态配置：全英文映射表 ======================
CHARACTER_MAP = {
    # "模板文件名(不含.png)": "展示的英文名"
    "Huck": "Huck",
    "Karst": "Karst",
    "Tullina": "Tullina",
    "Isla": "Isla",
    "Vinderi": "Vinderi",
    "Tibbs": "Tibbs",
    "Nenet": "Nenet",
    "Gianna": "Gianna",
    "Niles": "Niles",
}

# 动态获取当前脚本所在的绝对物理路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "name_templates")

# 【黄金基准】1080P(1K) 下名字框的标准大小，所有的模板图均为此尺寸
BASE_TEMPLATE_W = 60
BASE_TEMPLATE_H = 25

# 【1K基准参数】金圈中心点到下方名字中心点的垂直偏移像素
NAME_OFFSET_Y = 21


# ====================== 3. HSV 金圈亮起检测 ======================
def is_golden_pixel(img_bgr):
    """
    检测 区域内是否为高亮金色
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 金色 HSV 过滤范围
    lower_gold = np.array([15, 120, 150])
    upper_gold = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    gold_pixel_count = np.sum(mask > 0)
    return gold_pixel_count >= 4  # 区域内有4个及以上符合金色即可


# ====================== 4. 精准模板匹配 (1K 标准基准) ======================
def match_character_name(roi_1k_img):
    """
    传入的 roi_1k_img 保证是已经被降采样回 1K 尺寸 (60x25) 的图
    直接与 1K 原生模板比对，零拉伸失真！
    """
    best_val = -1.0
    best_alias_name = "UNKNOWN"

    # 转换为灰度图
    gray_roi = cv2.cvtColor(roi_1k_img, cv2.COLOR_BGR2GRAY)

    for template_filename, show_name in CHARACTER_MAP.items():
        template_path = os.path.join(TEMPLATE_DIR, f"{template_filename}.png")
        if not os.path.exists(template_path):
            continue

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        try:
            # 核心优化：不再对 template 进行 resize！
            # 如果截图缩小后的 ROI 和模板像素微小的 1 像素偏差，自动对齐
            th, tw = template.shape[:2]
            rh, rw = gray_roi.shape[:2]

            # 确保尺寸匹配（防越界）
            if rh < th or rw < tw:
                gray_roi_eval = cv2.resize(gray_roi, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                gray_roi_eval = gray_roi

            # 进行 1:1 精准模板匹配
            res = cv2.matchTemplate(gray_roi_eval, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            # 降采样匹配精度大幅提升，阈值可提升到 0.30防止UNKNOWN
            if max_val > best_val and max_val > 0.30:
                best_val = max_val
                best_alias_name = template_filename
        except Exception as e:
            continue

    if best_alias_name in CHARACTER_MAP:
        return CHARACTER_MAP[best_alias_name], best_val
    return "UNKNOWN", 0.0


# ====================== 5. 核心测试主函数 ======================
def run_visual_test():
    print("====== 选人界面 CV 识别测试 (1K 降采样优化版) ======")
    print(f"当前屏幕分辨率: {screen_w}x{screen_h} (缩放比: {scale_w:.2f})")
    print("3秒后开始抓取屏幕，请迅速切到游戏选人界面...")
    time.sleep(3)

    # 1080P(1920x1080) 坐标系下的金圈中心点
    gold_ring_anchors = [
        (825, 567),  # 左边候选人
        (956, 567),  # 中间候选人
        (1086, 567), # 右边候选人
        (890, 567),
        (1020, 567)
    ]

    # 截取全屏
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        screenshot = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)

    result_img = screenshot.copy()

    print("\n开始扫描候选位置...")

    for idx, (gx, gy) in enumerate(gold_ring_anchors):
        # 1. 自动定位当前分辨率下的绝对金圈坐标
        rgx, rgy = s(gx, gy)

        # 2. 截取金圈检测微区 (7x7)
        gold_patch = screenshot[rgy - 3:rgy + 4, rgx - 3:rgx + 4]

        # 判断金圈状态
        is_active = is_golden_pixel(gold_patch)

        # 在结果图上标出金圈检测点（蓝色圆圈）
        cv2.circle(result_img, (rgx, rgy), 6, (255, 0, 0), -1)

        if not is_active:
            print(f"位置 {idx}: ❌ 金圈未激活")
            continue

        print(f"位置 {idx}: 🔸 检测到金圈亮起！开始提取名字...")

        # 3. 计算当前屏幕分辨率下的名字 ROI 真实坐标
        cur_name_w = int(BASE_TEMPLATE_W * scale_w)
        cur_name_h = int(BASE_TEMPLATE_H * scale_h)
        cur_name_x = rgx - (cur_name_w // 2)
        cur_name_y = rgy + int(NAME_OFFSET_Y * scale_h)

        # 防止坐标越界
        cur_name_x = max(0, min(screen_w - cur_name_w, cur_name_x))
        cur_name_y = max(0, min(screen_h - cur_name_h, cur_name_y))

        # 4. 从当前屏幕截取高清的名字 ROI
        raw_name_roi = screenshot[cur_name_y: cur_name_y + cur_name_h, cur_name_x: cur_name_x + cur_name_w]

        # 【核心黑科技：降采样！】
        # 如果当前不是 1K 屏幕，把高清 ROI 使用抗锯齿算法强行缩小回 1K 的 (60, 25) 尺寸
        if scale_w != 1.0 or scale_h != 1.0:
            roi_1k = cv2.resize(raw_name_roi, (BASE_TEMPLATE_W, BASE_TEMPLATE_H), interpolation=cv2.INTER_AREA)
        else:
            roi_1k = raw_name_roi

        # 5. 进行 1K 降采样比对
        char_name, score = match_character_name(roi_1k)

        # 6. 【可视化绘制】在 2K/当前全屏底图上画红色矩形框
        cv2.rectangle(result_img, (cur_name_x, cur_name_y),
                      (cur_name_x + cur_name_w, cur_name_y + cur_name_h), (0, 0, 255), 2)

        display_text = f"{char_name} ({score:.2f})" if char_name != "UNKNOWN" else "UNKNOWN"
        print(f"位置 {idx}: 🎯 识别结果: {display_text}")

        # 在红框上方打出英文识别文本
        cv2.putText(result_img, f"{display_text}", (cur_name_x, cur_name_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 展示结果窗口
    cv2.namedWindow('CV_Test_Result', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Test_Result', result_img)
    print("\n🔍 识别可视化窗口已弹出，查看完毕后请在窗口上按【任意键】或【ESC】关闭。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_visual_test()