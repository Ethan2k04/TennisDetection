import os
import json
import time
import cv2
import numpy as np
import threading
from sklearn.cluster import KMeans
from yolo11 import setup_model, post_process
from py_utils.coco_utils import COCO_test_helper
from tools import get_trackbar_values_filter, get_trackbar_values_confidence, \
    get_trackbar_values_morphology, log_with_timestamp
from constants import ANGLE_THRESHOLD, AREA_THRESHOLD_PERCENTAGE, BALL_COLOR, ERODE_ITER, FONT_SCALE, IMG_SIZE, \
    LINE_THICKNESS, LOWER_BLACK, MIN_DETECTION_SAMPLE, MIN_POLY, NONLINEAR_THRESHOLD, PERI_BIAS, RANDOM_STATE, \
    SETTINGS_FILE, TARGET_COLOR, TARGET_MODEL_PATH, TENNIS_MODEL_PATH, TEXT_MARGIN, TRACE_RADIUS, TRAJECTORY_SPLIT_INTERVAL, \
    UPPER_WHITE, VELOCITY_RATIO_THRESHOLD


# 用于过滤小面积噪音
area_threshold = 0

# 使用yolo11检测网球（核心）
# def detect_balls(frame):
#     """
#     使用 YOLO 检测网球，并根据检测结果绘制目标框。
#     """
#     # 获取检测结果
#     ball_conf, _ = get_trackbar_values_confidence()
#     img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.expand_dims(img, axis=0)
#     outputs = model_tennis.run([img])
#     boxes = []
#     if outputs is not None:
#         boxes, _, scores = post_process(outputs)

#     # 用于存储所有网球框的位置信息
#     ball_positions = []

#     # 遍历检测结果并提取位置信息
#     if boxes is not None:
#         boxes = co_helper.get_real_box(boxes)
#         for i, box in enumerate(boxes):
#             if scores[i] > ball_conf: 
#                 top, left, right, bottom = box
#                 ball_positions.append((int(top), int(left), int(right), int(bottom)))

#     return ball_positions


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
        text_position = (top, left - 6)  # 文字位置设置在圆的右侧
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BALL_COLOR, LINE_THICKNESS)

    return frame


# Gamma校正
def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# 判断轮廓是否为圆形
def is_circle(contour):
    """
    根据轮廓点判断轮廓是否为类圆形
    """
    if len(contour) < MIN_POLY:
        return False

    # 拟合椭圆
    ellipse = cv2.fitEllipse(contour)
    (x, y), (major_axis, minor_axis), angle = ellipse

    # 计算圆度（长轴和短轴的比例）
    circularity = minor_axis / major_axis if major_axis != 0 else 0

    # 圆度接近1且面积大于一定阈值
    return circularity > 0.8 and cv2.contourArea(contour) > 100

# 检测目标轮廓
def detect_target(frame, model_digit, co_helper, debug=False):
    """
    检测 frame 中的圆形轮廓，并识别中间的数字
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选圆形轮廓
    circles = [contour for contour in contours if is_circle(contour)]

    # 用于保存符合条件的轮廓
    valid_circles = []

    # 遍历圆形轮廓并进行数字识别
    for contour in circles:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_region = frame[y: y + h, x: x + w]

        # 使用YOLO模型进行数字识别
        img = co_helper.letter_box(im=cropped_region.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        outputs = model_digit.run([img])
        if outputs is not None:
            boxes, _, scores = post_process(outputs)

        # 判断检测是否有结果，如果有结果则保留该轮廓
        if boxes is not None and len(boxes) > 0 and max(scores) > 0.5:  # 置信度阈值
            valid_circles.append(contour)

    # 返回结果
    result = {"targets": valid_circles}
    return result

# 聚类分析
def cluster_target(data, n_clusters):
    """
    对检出的标靶大小进行聚类，根据聚类结果判定标靶类型
    """
    data = np.array(data).reshape(-1, 1)
    clusters = {}
    if len(data) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        clusters = {i: [] for i in range(n_clusters)}
        for value, label in zip(data.flatten(), labels):
            clusters[label].append(value)

        sorted_clusters = sorted(clusters.items(), key=lambda x: centers[x[0]])
        clusters = {i: sorted(values) for i, (_, values) in enumerate(sorted_clusters)}

    return clusters


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
            "has_ball": False,
            "last_update_time": time.time(),
        }

    return target_status


# 更新靶标内网球识别状态
def update_target_status(target_status, ball_center):
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
                is_collided = trajectory_fitting(np.array(status["trajectory"]), frame)
                status["trajectory"] = []
                status["has_ball"] = False
                if is_collided:
                    return True, value["score"], key

    return False, 0, None


# 计算点与点之间的欧几里得距离
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# 计算两个向量的夹角（返回角度）
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


# 根据网球踪迹判断是否碰撞
# def trajectory_fitting(trajectory, frame):
#     """
#     根据trajectory轨迹判断是否发生碰撞
#     """
#     x = trajectory[:, 0]
#     y = trajectory[:, 1]

#     # 绘制轨迹点
#     print(f"traj: {trajectory}")
#     for xi, yi in zip(x, y):
#         cv2.circle(frame, (xi, yi), radius=TRACE_RADIUS, color=BALL_COLOR, thickness=-1)

#      # 若没有足够的点进行计算（使用线性拟合）
#     if len(trajectory) <= MIN_DETECTION_SAMPLE:
#         coeffs = np.polyfit(x, y, 1)
#         residuals = np.sum((np.polyval(coeffs, x) - y) ** 2) / len(x)
#         # print(f"res: {residuals}")
#         if residuals > NONLINEAR_THRESHOLD:
#             return True
#     # 若有足够的点进行计算（使用突变点检测）
#     else:
#         # 计算速度方向变化的突变点
#         velocity_change_points = []
#         for i in range(1, len(trajectory) - 1):
#             v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
#             v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
#             v1_norm = np.linalg.norm(v1)
#             v2_norm = np.linalg.norm(v2)
#             if v1_norm > 0 and v2_norm > 0:
#                 ratio = v1_norm / v2_norm
#                 if ratio > VELOCITY_RATIO_THRESHOLD:
#                     velocity_change_points.append(i)

#         # 如果没有速度突变点，则返回False
#         if not velocity_change_points:
#             return False

#         # 对每个速度突变点之前和之后的点序列进行距离计算
#         for change_point in velocity_change_points:
#             before_point = trajectory[change_point - 1]
#             after_point = trajectory[change_point + 1]
#             v_before = np.array(trajectory[change_point]) - np.array(before_point)
#             v_after = np.array(after_point) - np.array(trajectory[change_point])
#             angle = calculate_angle(v_before, v_after)
#             if angle > ANGLE_THRESHOLD and angle < 180 - ANGLE_THRESHOLD:
#                 return True

#     return False

ACCELERATION_THRESHOLD = 0.98

# 根据网球踪迹判断是否碰撞
def trajectory_fitting(trajectory, frame):
    """
    根据trajectory轨迹判断是否发生碰撞
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # 绘制轨迹点
    print(f"traj: {trajectory}")
    for xi, yi in zip(x, y):
        cv2.circle(frame, (xi, yi), radius=TRACE_RADIUS, color=BALL_COLOR, thickness=-1)

    # 计算一阶差分（速度）和二阶差分（加速度）
    velocities = []
    accelerations = []

    for i in range(1, len(trajectory) - 1):
        # 计算速度（两点之间的位移）
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])  # 速度 v1
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])  # 速度 v2
        velocity = v2  # 速度变化量
        velocities.append(velocity)

        # 计算加速度（速度的变化）
        if i > 1:
            acceleration = v2 - v1  # 加速度是速度的变化
            accelerations.append(acceleration)

    # 打印速度和加速度数组，方便调试
    velocities_magnitude = [np.linalg.norm(v) for v in velocities]
    accelerations_magnitude = [np.linalg.norm(a) for a in accelerations]

    print(f"Velocities: {velocities_magnitude}")
    print(f"Accelerations: {accelerations_magnitude}")

    # 判断加速度是否发生突变
    for i in range(1, len(accelerations_magnitude)):
        if abs(accelerations_magnitude[i] - accelerations_magnitude[i-1]) > ACCELERATION_THRESHOLD:
            return True  # 碰撞发生

    return False


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
    total_length = sum(len(v) for k, v in target_result.items() if k != "undef")
    return total_length == num_target and all(
        len(v) > 0 for k, v in target_result.items() if k != "undef")


# 释放资源
def destruct():
    model_tennis.release()
    model_digit.release()
