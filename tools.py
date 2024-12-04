import cv2
import json

from constants import *


def refine_mask(mask, ksize):
    """
    对输入的图像进行形态学操作
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))  # 调整核大小
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去噪点

    return mask


# 定义保存目标框信息的函数
def save_target_to_config(target_data):
    # 清空 config 内容
    config = {}

    # 将新的目标框信息添加到配置文件中
    config.update(target_data)

    # 保存更新后的信息
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file, indent=4)


# 创建滑块更新 V 范围
def nothing(x):
    pass


def create_trackbar():
    # 创建一个窗口，便于添加滑块
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    # 只为黑色和白色的 V 分量创建滑块
    cv2.createTrackbar("UP_V_BLACK", "Trackbars", UPPER_BLACK[2], 255, nothing)
    cv2.createTrackbar("LOW_V_WHITE", "Trackbars", LOWER_WHITE[2], 255, nothing)
    cv2.createTrackbar("BALL_CONF", "Trackbars", int(BALL_CONF * 100), 100, nothing)
    cv2.createTrackbar("TARGET_CONF", "Trackbars", int(TARGET_CONF * 100), 100, nothing)
    cv2.createTrackbar("BALL_HIT_WAIT_SEC", "Trackbars", int(BALL_HIT_WAIT_SEC * 100), 100, nothing)
    cv2.createTrackbar("RETARGET_WAIT_SEC", "Trackbars", RETARGET_WAIT_SEC, 100, nothing)
    cv2.createTrackbar("REFINE_KSIZE", "Trackbars", REFINE_KSIZE, 30, nothing)
    cv2.createTrackbar("ERODE_KSIZE", "Trackbars", ERODE_KSIZE, 30, nothing)
    cv2.createTrackbar("ERODE_ITER", "Trackbars", ERODE_ITER, 10, nothing)


def get_trackbar_values_filter():
    # 获取滑块的当前值
    up_v_black = cv2.getTrackbarPos("UP_V_BLACK", "Trackbars")
    low_v_white = cv2.getTrackbarPos("LOW_V_WHITE", "Trackbars")

    # 返回更新后的黑色和白色的 V 范围
    return (
        np.array([UPPER_BLACK[0], UPPER_BLACK[1], up_v_black]),  # 只修改 V 值
        np.array([LOWER_WHITE[0], LOWER_WHITE[1], low_v_white]), # 只修改 V 值
    )


def get_trackbar_values_confidence():
    ball_conf = cv2.getTrackbarPos("BALL_CONF", "Trackbars")
    target_conf = cv2.getTrackbarPos("TARGET_CONF", "Trackbars")

    return float(ball_conf / 100), float(target_conf / 100)


def get_trackbar_values_wait_sec():
    ball_hit = cv2.getTrackbarPos("BALL_HIT_WAIT_SEC", "Trackbars")
    retarget = cv2.getTrackbarPos("RETARGET_WAIT_SEC", "Trackbars")

    return float(ball_hit / 100), float(retarget)

def get_trackbar_values_morphology():
    refine_ksize = cv2.getTrackbarPos("REFINE_KSIZE", "Trackbars")
    erode_ksize = cv2.getTrackbarPos("ERODE_KSIZE", "Trackbars")
    erode_iter = cv2.getTrackbarPos("ERODE_ITER", "Trackbars")

    return max(refine_ksize, 1), erode_ksize, erode_iter
