import os
import json
import time
import cv2
import numpy as np
from constants import BALL_COLOR, FONT_SCALE, LINE_THICKNESS, MIN_POLY, SETTINGS_FILE,\
      TARGET_COLOR, TEXT_MARGIN, TRACE_RADIUS, TRAJECTORY_SPLIT_INTERVAL


# 从元信息中读取参数
score_list = []
num_target = 0
if os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, 'r') as file:
        settings = json.load(file)
    score_list = settings["target_score"]
    num_target = len(score_list)

# 根据检测结果绘制网球目标框
def draw_ball_boxes(frame, ball_positions):
    """
    根据存储的网球框位置信息，在帧上绘制框。
    """
    for box in ball_positions:
        top, left, right, bottom = [int(_b) for _b in box]
        # 绘制矩形框，绿色，线宽3
        cv2.rectangle(frame, (top, left), (right, bottom), BALL_COLOR, LINE_THICKNESS)
        # 添加标签文字
        label = "Ball"
        text_position = (top, left - 6)
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BALL_COLOR, LINE_THICKNESS)

    return frame

# 判断轮廓是否为圆形
def is_circle(contour):
    """
    根据轮廓点判断轮廓是否为类圆形
    """
    if len(contour) < MIN_POLY:
        return False

    # 拟合椭圆
    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)

    # 计算圆度（长轴和短轴的比例）
    circularity = minor_axis / major_axis if major_axis != 0 else 0

    # 圆度接近1且面积大于一定阈值
    return circularity > 0.8 and cv2.contourArea(contour) > 100

# 检测目标轮廓
def detect_target(frame):
    """
    检测 frame 中的椭圆轮廓，并选择面积最大的 6 个椭圆作为目标。
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 用于存储检测到的轮廓及其面积
    detected_contours = []

    # 遍历轮廓
    detected_contours = [contour for contour in contours if is_circle(contour) and cv2.contourArea(contour) > 100]

    # 如果有检测到的轮廓，选择面积最大的 TARGET_NUM 个作为结果类
    result_contours = []
    if len(detected_contours) > 0:
        # 将检测到的轮廓按面积从大到小排序
        sorted_contours = sorted(detected_contours, key=lambda c: cv2.contourArea(c), reverse=True)

        # 选择面积最大的 TARGET_NUM 个轮廓作为结果类
        if len(detected_contours) > num_target:
            result_contours = sorted_contours[:num_target]
        else:
            result_contours = sorted_contours[:len(detected_contours)]

    # 返回结果
    result = {score: [] for score in score_list}

    for i in range(len(result_contours)):
        score = score_list[i]
        result[score].append(result_contours[i])

    return result

# 根据检测结果绘制标靶目标框
def draw_target_boxes(frame, config):
    """
    根据检出的标靶轮廓绘制目标框和椭圆
    """
    for key, value in config.items():
        cls = value["cls"]                      # 甲方说要显示序号而不是分数（保留）
        x = int(value["center_x"])              # 椭圆中心点 x 坐标
        y = int(value["center_y"])              # 椭圆中心点 y 坐标
        major_axis = int(value["major_axis"])   # 椭圆的长轴长度
        minor_axis = int(value["minor_axis"])   # 椭圆的短轴长度
        label = "Target_" + str(key)            # 标签
        text_position = (x + int(minor_axis / 2) - TEXT_MARGIN, y - int(major_axis / 2) + TEXT_MARGIN)

        # 绘制椭圆
        cv2.ellipse(frame, (x, y), (minor_axis // 2, major_axis // 2), 0, 0, 360, TARGET_COLOR, LINE_THICKNESS)

        # 在框旁边显示标签
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TARGET_COLOR, LINE_THICKNESS)

    return frame

# 构建靶标内网球识别状态
def build_target_status(config):
    """
    根据config参数构建状态列表
    """
    target_status = {}
    for key, value in config.items():
        target_center = (int(value["center_x"]), int(value["center_y"]))
        target_width = int(value["minor_axis"])
        target_height = int(value["major_axis"])
        target_status[key] = {
            "center": target_center,
            "width": target_width,
            "height": target_height,
            "score": 0 if value["cls"] == "undef" else int(value["cls"]),
            "trajectory": [],
            "size": [],
            "has_ball": False,
            "last_update_time": time.time(),
        }

    return target_status

# 更新靶标内网球识别状态
def update_target_status(target_status, ball_center, ball_size):
    """
    根据网球位置更新状态列表
    """
    for key, value in target_status.items():
        status = target_status[key]
        target_center = status["center"]
        target_width = status["width"]      # 短轴
        target_height = status["height"]    # 长轴
        a = target_width / 2                # 半短轴
        b = target_height / 2               # 半长轴

        # 使用椭圆方程判断球是否在椭圆范围内
        if ((ball_center[0] - target_center[0]) ** 2) / a ** 2 + ((ball_center[1] - target_center[1]) ** 2) / b ** 2 <= 1:
            status["last_update_time"] = time.time()
            status["trajectory"].append(ball_center)
            status["size"].append(ball_size)
            status["has_ball"] = True

    return target_status

# 检查每个靶标内的网球轨迹状态
def check_target_status(target_status, frame):
    """
    检查状态列表中是否有网球在靶标区域
    """
    for key, value in target_status.items():
        status = target_status[key]
        global index
        if time.time() - status["last_update_time"] > TRAJECTORY_SPLIT_INTERVAL and status["has_ball"]:
                is_collided = trajectory_fitting(np.array(status["trajectory"]), np.array(status["size"]), frame)
                status["trajectory"] = []
                status["size"] = []
                status["has_ball"] = False
                if is_collided:
                    return True, value["score"], key

    return False, 0, None

def has_minimum(size):
    """
    判断size数组中是否存在极小值
    """
    if len(size) < 3:
        return False  # 数组长度小于3，无法形成极小值

    for i in range(1, len(size) - 1):
        if size[i] < size[i-1] and size[i] < size[i+1]:
            return True  # 找到极小值

    return False  # 没有找到极小值

SIZE_FILE = "size_data.txt"  # 存储size数组的文件

def trajectory_fitting(trajectory, size, frame):
    """
    根据trajectory轨迹判断是否发生碰撞
    angle_threshold: 角度变化的阈值，单位为度
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # 绘制轨迹点
    print(f"sizes: {size}")
    for xi, yi in zip(x, y):
        cv2.circle(frame, (xi, yi), radius=TRACE_RADIUS, color=BALL_COLOR, thickness=-1)

    size = smooth_size(size, 3)

    # 将size数组存储到本地文件
    save_size_to_file(size)

    # 检测size数组中是否存在极小值
    if has_minimum(size):
        return True

    return False

def smooth_size(size, window_size=3):
    """
    对size数组进行平滑操作（使用移动平均法）
    :param size: 原始size数组
    :param window_size: 移动平均的窗口大小
    :return: 平滑后的size数组
    """
    if len(size) < window_size:
        return size  # 数组长度小于窗口大小，无法平滑

    # 使用移动平均法进行平滑
    smoothed_size = np.convolve(size, np.ones(window_size)/window_size, mode='valid')
    return smoothed_size

def save_size_to_file(size):
    """
    将size数组存储到本地文件
    """
    with open(SIZE_FILE, "a") as file:
        # 将size数组转换为字符串，并用逗号分隔
        size_str = ",".join(map(str, size))
        file.write(size_str + "\n")  # 每个size数组占一行

# 对输入的 mask 进行形态学优化操作
def refine_mask(mask, ksize):
    """
    对输入的图像进行形态学操作
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))  # 调整核大小
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去噪点

    return mask

# 判断目标结果集合是否符合设定
def is_target_result_valid(target_result, num_target):
    """
    判断靶标识别结果是否符合settings中的设定
    """
    # 统计所有轮廓的总数
    total_contours = 0
    for score, contours in target_result.items():
        total_contours += len(contours)

    # 判断总数是否等于 num_target
    return total_contours == num_target
