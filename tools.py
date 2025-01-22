import cv2
import json
import time
from constants import CONFIG_FILE, BALL_CONF, TARGET_CONF, BALL_HIT_WAIT_SEC, \
                      REFINE_KSIZE, ERODE_KSIZE


# 输出当前时间和日志信息
def log_with_timestamp(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# 定义保存目标框信息的函数
def save_target_to_config(target_data):
    # 清空 config 内容
    config = {}

    # 将新的目标框信息添加到配置文件中
    config.update(target_data)

    # 保存更新后的信息
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file, indent=4)

    return config

# 创建滑块更新 V 范围
def create_trackbar():
    def nothing(x):
        pass
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("REFINE_KSIZE", "Trackbars", REFINE_KSIZE, 32, nothing)
    cv2.createTrackbar("ERODE_KSIZE", "Trackbars", ERODE_KSIZE, 32, nothing)
    cv2.createTrackbar("BALL_CONF", "Trackbars", int(BALL_CONF * 100), 100, nothing)
    cv2.createTrackbar("TARGET_CONF", "Trackbars", int(TARGET_CONF * 100), 100, nothing)
    cv2.createTrackbar("BALL_HIT_WAIT_SEC", "Trackbars", int(BALL_HIT_WAIT_SEC * 100), 100, nothing)
    cv2.createTrackbar("RETARGET_WAIT_SEC", "Trackbars", int(BALL_HIT_WAIT_SEC * 100), 100, nothing)

def get_trackbar_values_morphology():
    refine_ksize = cv2.getTrackbarPos("REFINE_KSIZE", "Trackbars")
    erode_ksize = cv2.getTrackbarPos("ERODE_KSIZE", "Trackbars")

    return max(refine_ksize, 1), max(erode_ksize, 1)

def get_trackbar_values_confidence():
    ball_conf = cv2.getTrackbarPos("BALL_CONF", "Trackbars")
    target_conf = cv2.getTrackbarPos("TARGET_CONF", "Trackbars")

    return float(ball_conf / 100), float(target_conf / 100)

def get_trackbar_values_wait_sec():
    ball_hit = cv2.getTrackbarPos("BALL_HIT_WAIT_SEC", "Trackbars")
    retarget = cv2.getTrackbarPos("RETARGET_WAIT_SEC", "Trackbars")

    return float(ball_hit / 100), float(retarget / 100)
