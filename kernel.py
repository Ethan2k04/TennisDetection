import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from yolo11 import setup_model, post_process
from py_utils.coco_utils import COCO_test_helper
from tools import get_trackbar_values_filter, get_trackbar_values_confidence, \
    get_trackbar_values_morphology
from constants import TENNIS_MODEL_PATH, TARGET_MODEL_PATH, BALL_COLOR, LINE_THICKNESS, FONT_SCALE, LOWER_BLACK, \
                       UPPER_WHITE, RANDOM_STATE, MIN_POLY, AREA_THRESHOLD_PERCENTAGE, PERI_BIAS, SETTINGS_FILE, \
                       TARGET_COLOR, NONLINEAR_THRESHOLD, IMG_SIZE


# yolo11 模型初始化
model_tennis = setup_model(TENNIS_MODEL_PATH)
model_digit = setup_model(TARGET_MODEL_PATH)
co_helper = COCO_test_helper(enable_letter_box=True)

# 用于过滤小面积噪音
area_threshold = 0


# 使用yolo11检测网球
def detect_balls(frame):
    """
    使用 YOLO 检测网球，并根据检测结果绘制目标框。
    """
    # 获取检测结果
    ball_conf, _ = get_trackbar_values_confidence()
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    outputs = model_tennis.run([img])
    boxes = []
    if outputs is not None:
        boxes, _, _ = post_process(outputs)

    # 用于存储所有网球框的位置信息
    ball_positions = []

    # 遍历检测结果并提取位置信息
    if boxes is not None:
        boxes = co_helper.get_real_box(boxes)
        for box in boxes:
            top, left, right, bottom = box
            ball_positions.append((int(top), int(left), int(right), int(bottom)))

    return ball_positions


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


# 筛选出标靶颜色
def filter_target_color(frame):
    """
    通过动态更新的 HSV 范围筛选黑色和白色。
    """
    # 获取当前滑块的值，更新 V 范围
    upper_black, lower_white, = get_trackbar_values_filter()

    # 转换为 HSV 色彩范围
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 筛选黑色和白色像素
    mask_black = cv2.inRange(hsv_frame, LOWER_BLACK, upper_black)
    mask_white = cv2.inRange(hsv_frame, lower_white, UPPER_WHITE)

    # 合并黑色和白色的掩码
    mask = cv2.bitwise_or(mask_black, mask_white)
    return mask


# 对检测到的标靶进行聚类分析
def cluster_target(data, n_clusters):
    """
    对检出的标靶大小进行聚类，根据聚类结果判定标靶类型
    """
    # 转为 NumPy 数组
    data = np.array(data).reshape(-1, 1)

    # 使用 K-Means 聚类
    clusters = {}
    if len(data) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 将数据和聚类结果对应
        clusters = {i: [] for i in range(n_clusters)}
        for value, label in zip(data.flatten(), labels):
            clusters[label].append(value)

        # 按中心值对簇排序
        sorted_clusters = sorted(clusters.items(), key=lambda x: centers[x[0]])

        # 重新分配 keys，并对每个簇的 values 排序
        clusters = {i: sorted(values) for i, (_, values) in enumerate(sorted_clusters)}

    # 返回结果
    return clusters


# 判断轮廓是不是椭圆形
def is_ellipse(contour):
    """
    根据轮廓点判断轮廓是否为类椭圆形
    """
    # 如果轮廓点数小于 MIN_POLY，则不可能是椭圆
    if len(contour) < MIN_POLY:
        return False

    # 拟合椭圆并计算相关属性
    ellipse = cv2.fitEllipse(contour)
    peri = cv2.arcLength(contour, True)  # 计算轮廓的周长
    (x, y), (major_axis, minor_axis), angle = ellipse
    area = cv2.contourArea(contour)

    # 如果面积为零，直接返回 False
    if area == 0:
        return False

    # 使用 Ramanujan 的第二公式计算椭圆的周长
    a = major_axis / 2  # 长轴半径
    b = minor_axis / 2  # 短轴半径
    ellipse_peri = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

    # 计算周长比率，接近1说明轮廓较规则
    peri_ratio = peri / ellipse_peri

    # 使用多边形逼近轮廓，排除常见的非椭圆形状（如三角形、四边形等）
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # 筛选椭圆条件
    is_valid_ellipse = abs(peri_ratio - 1) < PERI_BIAS and len(approx) > MIN_POLY

    return is_valid_ellipse


# 检测目标轮廓
def detect_target(frame):
    """
    检测 frame 中的标靶轮廓，返回字典结果
    """
    # 获取 trackbar 设定的参数
    refine_ksize, erode_ksize, erode_iter = get_trackbar_values_morphology()

    # 过滤目标颜色并精细化掩膜
    mask = filter_target_color(frame)
    mask = refine_mask(mask, refine_ksize)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.MORPH_ELLIPSE, cv2.CHAIN_APPROX_NONE)

    # 创建一个空白掩膜，用于存放选中的轮廓区域
    new_mask = np.zeros_like(mask)  # 初始掩膜全为0

    # 遍历所有轮廓，并填充相应区域
    for contour in contours:
        # 在每个轮廓区域填充1
        cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)  # 填充轮廓区域

    # 使用腐蚀操作进行降噪
    kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
    mask = cv2.erode(new_mask, kernel, iterations=erode_iter)

    # =================================================== #
    # cv2.imwrite("target_mask.png", mask)  # 调试用
    # cv2.imshow("target_mask", mask)       # 调试用
    # cv2.moveWindow("target_mask", CENTER_X, CENTER_Y)
    # =================================================== #

    contours, _ = cv2.findContours(mask, cv2.MORPH_ELLIPSE, cv2.CHAIN_APPROX_NONE)

    # 识别符合条件的椭圆
    ellipses = [contour for contour in contours if is_ellipse(contour)]

    # 用于保存符合 YOLO 检测条件的轮廓
    valid_ellipses = []

    # print(f"eliipses:{len(ellipses)}")    # 调试用

    # 遍历椭圆轮廓并进行 YOLO 检测
    for contour in ellipses:
        global area_threshold
        if cv2.contourArea(contour) > area_threshold:
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            cropped_region = frame[y: y + h, x: x + w]  # 截取对应的区域
            # cv2.imshow("cropped_region", cropped_region)  # 调试用
            # 使用YOLO模型进行检测
            _, target_conf = get_trackbar_values_confidence()
            img = co_helper.letter_box(im=cropped_region.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            outputs = model_digit.run([img])
            if outputs is not None:
                boxes, _, _ = post_process(outputs)

            # 判断检测是否有结果，如果有结果则保留该轮廓
            if boxes is not None and len(boxes) > 0:
                valid_ellipses.append(contour)  # 如果检测到目标，保留该轮廓

    # 计算所有有效椭圆的面积并排序
    area_list = sorted([cv2.contourArea(contour) for contour in valid_ellipses])
    if len(area_list) > 0:
        area_threshold = max(area_list) / AREA_THRESHOLD_PERCENTAGE
    # print(f"valid_ellipses: {len(valid_ellipses)}")   # 调试用

    # 识别的标靶结果
    result = {"undef": []}

    # 从元信息中读取参数
    num_cls = 0
    num_target = 0
    score_list = []
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as file:
            settings = json.load(file)

        num_cls = settings["num_cls"]
        score_list = settings["score_list"]
        num_target = settings["num_target"]

    # 聚类目标面积
    cluster = cluster_target(area_list, num_cls)

    # 识别的标靶结果
    for score in score_list:
        result[str(score)] = []

    if is_target_result_valid(cluster, num_target):
        for contour in valid_ellipses:
            contour_area = cv2.contourArea(contour)
            for i in range(len(score_list)):
                if len(cluster[i]) > 0:
                    if min(cluster[i]) <= contour_area <= max(cluster[i]):
                        result[str(score_list[i])].append(contour)

        return result
    else:
        result["undef"] = valid_ellipses
        return result


# 根据检测结果绘制标靶目标框
def draw_target_boxes(frame, config):
    """
    根据检出的标靶轮廓绘制目标框
    """
    for key, value in config.items():
        cls = value["cls"]
        x = int(value["center_x"])
        y = int(value["center_y"])
        major_axis = int(value["major_axis"])
        minor_axis = int(value["minor_axis"])
        label = "Target_" + str(cls)
        text_position = (x + int(minor_axis / 2) + 10, y - int(major_axis / 2))
        cv2.rectangle(frame, (x - int(minor_axis / 2), y - int(major_axis / 2)),
                      (x + int(minor_axis / 2), y + int(major_axis / 2)), TARGET_COLOR, LINE_THICKNESS)
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TARGET_COLOR,
                    LINE_THICKNESS)

    return frame


import time
import numpy as np
import cv2

# 根据网球踪迹拟合直线并判断是否碰撞
def update_ball_status(ball_id, ball_center, config, ball_status, frame, last_detection_time):
    """
    更新网球状态并判断是否发生碰撞。
    :param ball_id: 当前网球的唯一标识
    :param ball_center: 网球中心点坐标 (x, y)
    :param config: 靶子区域配置
    :param ball_status: 字典，记录每个网球的状态和轨迹
    :param frame: 当前帧图像
    :param last_detection_time: 上次识别网球的时间
    :return: (是否发生碰撞, 得分, 是否离开靶子区域)
    """
    if ball_id not in ball_status:
        ball_status[ball_id] = {
            "in_target": False,
            "trajectory": [],
            "last_target_score": 0,  # 记录最后在的靶子分数
            "last_detection_time": time.time(),  # 记录上次检测时间
        }

    status = ball_status[ball_id]
    current_time = time.time()

    # 检测是否在靶子区域内
    in_target = False
    for key, value in config.items():
        target_center = (int(value["center_x"]), int(value["center_y"]))
        target_width = int(value["minor_axis"])
        target_height = int(value["major_axis"])

        if (target_center[0] - target_width // 2 <= ball_center[0] <= target_center[0] + target_width // 2) and \
                (target_center[1] - target_height // 2 <= ball_center[1] <= target_center[1] + target_height // 2):
            in_target = True
            status["trajectory"].append(ball_center)
            status["last_detection_time"] = current_time
            status["last_target_score"] = 0 if value["cls"] == "undef" else int(value["cls"])  # 记录靶子分数
            break

    # 当没有网球在靶子内时，等待1秒钟
    if not in_target:
        if current_time - status["last_detection_time"] > 1:  # 1秒没有网球时进行结算
            trajectory = np.array(status["trajectory"])
            if len(trajectory) > 0:
                x = trajectory[:, 0]
                y = trajectory[:, 1]
                print(f"sample num: {len(x)}")
                for xi, yi in zip(x, y):
                    cv2.circle(frame, (xi, yi), radius=5, color=(0, 255, 0), thickness=-1)  # Green dots
                coeffs_1 = np.polyfit(x, y, 1)
                residuals_1 = np.sum((np.polyval(coeffs_1, x) - y) ** 2) / len(x)
                coeffs_2 = np.polyfit(x, y, 2)
                residuals_2 = np.sum((np.polyval(coeffs_2, x) - y) ** 2) / len(x)

                # 碰撞判断：残差大于阈值
                collision_detected = False
                if len(x) > 10:
                    if residuals_1 > 100 and residuals_2 > 100:  # 根据实际场景调整阈值
                        collision_detected = True
                
                sc = status["last_target_score"]
                print(f"residual_1: {residuals_1}, residual_2: {residuals_2}, score:{sc}\n")
                # 结算后清除轨迹记录
                status["trajectory"] = []
                return collision_detected, sc, True  # 碰撞检测，返回得分和离开靶子区域标识

    # 更新状态
    status["in_target"] = in_target
    status["last_detection_time"] = current_time  # 更新检测时间

    return False, 0, False


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
    total_length = sum(len(v) for k, v in target_result.items() if k != "undef")
    return total_length == num_target and all(
        len(v) > 0 for k, v in target_result.items() if k != "undef")


# 释放资源
def destruct():
    model_tennis.release()
    model_digit.release()
