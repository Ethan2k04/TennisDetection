import os
import json
import time
import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
from constants import (
    BALL_COLOR, FONT_SCALE, LINE_THICKNESS, MIN_POLY, SETTINGS_FILE,
    TARGET_COLOR, TEXT_MARGIN, TRACE_RADIUS, TRAJECTORY_SPLIT_INTERVAL,
    PERI_BIAS
)


# will come from config.json
# target canvas scope
min_x = 80
min_y = 63 
max_x = 601
max_y = 403

# tenis ball - area - points count
min_area = 20 
max_area = 500

# background bright : remove background black
# range: 0 to 255
min_value = 80
max_value = 200

# range: 0 to 179, green 60
# tennis ball color: can distinguish from canvas background
min_hue = 30
max_hue = 50

# remove background white
max_saturation = 40 

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
        cv2.putText(
            frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, BALL_COLOR, LINE_THICKNESS
        )

    return frame

# 判断轮廓是否为圆形
def is_circle(contour):
    """
    根据轮廓点判断轮廓是否为类圆形。
    """
    if len(contour) < MIN_POLY:
        return False

    # 拟合椭圆
    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)

    # 计算圆度（长轴和短轴的比例）
    circularity = minor_axis / major_axis if major_axis != 0 else 0

    # 圆度接近1且面积大于一定阈值
    return circularity > 1.0 - PERI_BIAS and cv2.contourArea(contour) > 100

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
    根据检出的标靶轮廓绘制目标框和椭圆。
    """
    for key, value in config.items():
        cls = value["cls"]  # 甲方说要显示序号而不是分数（保留）
        x = int(value["center_x"])  # 椭圆中心点 x 坐标
        y = int(value["center_y"])  # 椭圆中心点 y 坐标
        major_axis = int(value["major_axis"])  # 椭圆的长轴长度
        minor_axis = int(value["minor_axis"])  # 椭圆的短轴长度
        label = "Target_" + str(key)  # 标签
        text_position = (x + int(minor_axis / 2) - TEXT_MARGIN, y - int(major_axis / 2) + TEXT_MARGIN)

        # 绘制椭圆
        cv2.ellipse(frame, (x, y), (minor_axis // 2, major_axis // 2), 0, 0, 360, TARGET_COLOR, LINE_THICKNESS)

        # 在框旁边显示标签
        cv2.putText(
            frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, TARGET_COLOR, LINE_THICKNESS
        )

    return frame


# 构建靶标内网球识别状态
def build_target_status(config):
    """
    根据 config 参数构建状态列表。
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
    根据网球位置更新状态列表。
    """
    for key, value in target_status.items():
        status = target_status[key]
        target_center = status["center"]
        target_width = status["width"]  # 短轴
        target_height = status["height"]  # 长轴
        a = target_width / 2  # 半短轴
        b = target_height / 2  # 半长轴

        # 使用椭圆方程判断球是否在椭圆范围内
        if ((ball_center[0] - target_center[0]) ** 2) / a ** 2 + ((ball_center[1] - target_center[1]) ** 2) / b ** 2 <= 1:
            status["last_update_time"] = time.time()
            status["trajectory"].append(ball_center)
            status["has_ball"] = True

    return target_status

# 检查每个靶标内的网球轨迹状态
def check_target_status(target_status, frame):
    """
    检查状态列表中是否有网球在靶标区域。
    """
    for key, value in target_status.items():
        status = target_status[key]
        if time.time() - status["last_update_time"] > TRAJECTORY_SPLIT_INTERVAL and status["has_ball"]:
            is_collided = trajectory_fitting(np.array(status["trajectory"]), frame)
            status["trajectory"] = []
            status["has_ball"] = False
            if is_collided:
                return True, value["score"], key

    return False, 0, None

def trajectory_fitting(trajectory, frame):
    """
    根据 trajectory 数组判断是否发生碰撞
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # 绘制轨迹点
    for xi, yi in zip(x, y):
        cv2.circle(frame, (xi, yi), radius=TRACE_RADIUS, color=BALL_COLOR, thickness=-1)

    # TODO: 碰撞检测逻辑

    return False

# 判断目标结果集合是否符合设定
def is_target_result_valid(target_result, num_target):
    """
    判断靶标识别结果是否符合 settings 中的设定。
    """
    # 统计所有轮廓的总数
    total_contours = sum(len(contours) for contours in target_result.values())

    # 判断总数是否等于 num_target
    return total_contours == num_target

def make_binary_bitmap_from_frame(frame: np.ndarray, rect: List) -> np.ndarray:
      # Convert the frame to HSV format
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV frame into its channels
    h, s, v = cv2.split(hsv_frame)

    # Apply a threshold to the value channel to create a binary bitmap
    # _, binary_bitmap = cv2.threshold(v, 128, 1, cv2.THRESH_BINARY)
    return make_binary_bitmap(h, s, v, rect)

def make_binary_bitmap(h: np.ndarray, s: np.ndarray, v: np.ndarray, rect: List) -> np.ndarray:
    """
    Create a binary bitmap based on hue and value thresholds.

    Parameters:
    h (np.ndarray): Hue channel of the image.
    s (np.ndarray) : staturation channel of image
    v (np.ndarray): Value channel of the image.
    rect (List): (left, top, right, bottom) range of the targets
   
    Returns:
    np.ndarray: Binary bitmap.
    """

    # please aution to use ai to optimize, I try several time but failure
    binary_bitmap = np.ones_like(v)

    # get the range of the targets
    min_x, min_y, max_x, max_y = rect

    for (i, j), value in np.ndenumerate(v):
        if i < min_y or i > max_y :
            binary_bitmap[i,j] = 0
            continue 

        if j < min_x or j > max_x :
            binary_bitmap[i,j] = 0
            continue
        
        h_value = h[i, j]
        s_value = s[i,j]

        # white
        if s_value < max_saturation :
            binary_bitmap[i,j] = 0
            continue

        if h_value > max_hue or h_value < min_hue:
            binary_bitmap[i, j] = 0
        elif value > max_value or value < min_value:
            binary_bitmap[i, j] = 0

    # Set binary_bitmap to 1 where the conditions are met
    # binary_bitmap[(v >= min_value) & (v <= max_value) & (h >= min_hue) & (h <= max_hue)] = 1
    # open : erode and dilate, remove noise 
    # kernel = np.ones((5,5), np.uint8)
    kernel = np.ones((3,3), np.uint8)
    opened_bitmap = cv2.morphologyEx(binary_bitmap, cv2.MORPH_OPEN, kernel)

    return opened_bitmap


# record tennis ball info
class TenisBall:
    def __init__(self, centerx: int, centery: int, width: int, height: int, area: int, step_count:int):
        self.centerx = centerx
        self.centery = centery
        self.width = width
        self.height = height
        self.area = area
        self.step_count = step_count
        self.v_x = 0.0 
        self.v_y = 0.0 
        self.a_x = 0.0
        self.a_y = 0.0

    def __repr__(self):
        return f"TenisBall(centerx={self.centerx}, centery={self.centery}, width={self.width}, height={self.height}, area={self.area} ,step_count={self.step_count})"

    def get_center(self):
        return (self.centerx, self.centery)

    def get_dimensions(self):
        return (self.width, self.height)

    def get_area(self):
        return self.area
    
    def get_setup_count(self):
        return self.step_count

    def calculate_v_a(self, prev_ball: 'TenisBall'):
        step = self.step_count - prev_ball.step_count
        if step == 0:
            return
        self.v_x = (self.centerx - prev_ball.centerx) / step
        self.v_y = (self.centery - prev_ball.centery) / step
        self.a_x = self.v_x - prev_ball.v_x
        self.a_y = self.v_y - prev_ball.v_y

    def show_v_a(self):
        a = math.sqrt( self.a_x * self.a_x + self.a_y * self.a_y)
        return f"TenisBall(centerx={self.centerx}, centery={self.centery}, v.x = {self.v_x}, v.y = {self.v_y},a.x = {self.a_x} ,a.y= {self.a_y} ,a= {a}"


def add_point(list:List[Tuple[int,int]], point:Tuple[int,int])-> bool:
    if len(list) == 0 :
        list.append(point)
        return True
    
    for p in list:
        if abs(point[0] - p[0]) + abs(point[1] - p[1]) < 4:
            # neighbour point 
            list.append(point)
            return True
        
    return False
    
def find_tenis_ball(binary_image: np.ndarray):
    """
    Find the tennis ball in a binary image.

    Parameters:
    binary_image (np.ndarray): Binary image where the tennis ball is to be found.

    Returns:
    Tuple[bool, Optional[TenisBall]]: A tuple containing a boolean indicating if a tennis ball was found,
    and an instance of TenisBall if found, otherwise None.
    """
    # aggreation points
    points_list : List[List[Tuple[int,int]]] = []

    # all points that value = 1
    points =  np.argwhere(binary_image == 1)
    area = len(points)

    # not found ball 
    if area < min_area :
         return False, None
    
    # classify the indices
    for p in points:
        is_add = False
        for list in points_list:
            ret = add_point(list,p)
            if ret:
                is_add = True
                break

        if not is_add:
           l = [p]
           points_list.append(l) 

    for lst in points_list:
        area = len(lst)
        if area > min_area and area < max_area:
            # Split the list of tuples into two lists
            y_coords, x_coords = zip(*lst)

            # Calculate the center and dimensions of the detected object
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            height = max_y - min_y

            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            width = max_x - min_x

            # Create a TenisBall instance
            tennis_ball = [int(center_x - width // 2), int(center_y - height // 2),
                           int(center_x + width // 2), int(center_y + height // 2)]
            return True, tennis_ball
    else:
        return False, []

def tennis_ball_hit_test(lst:List[TenisBall])->bool:
    """
    list  of tenis ball detect
    return : hit : true, false : no hit
    """
    # too short to test
    if len(lst) < 4 :
        return False
    
    # too long , ball hit target on last short time
    if len(lst) > 40 :
        return False
    
    last_2 = lst[-2]
    last_1 = lst[-1]

    change = last_1.v_x * last_2.v_x 

    return change < -1
