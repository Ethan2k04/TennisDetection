import numpy as np

# 网络配置项
MAX_RETRY = 3           # 最大重传次数
RETRY_INTERVAL = 5      # 重传时间间隔（秒）
SALT = "learningxm"     # 固定盐值
API_URL = "https://wq.hoeagri.com/api/Training/receiveInput"

# CONFIG 文件名为 config.json
CONFIG_FILE = "meta/config.json"

# SETTINGS 文件名为 settings.json
SETTINGS_FILE = "meta/settings.json"

# 终端显示器大小和中心点（可选）
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CENTER_X = (SCREEN_WIDTH - SCREEN_HEIGHT) // 2
CENTER_Y = (SCREEN_WIDTH - SCREEN_HEIGHT) // 2

# 每隔多少秒重新检测标靶
RETARGET_WAIT_SEC = 3

# 等待多少秒判断碰撞
BALL_HIT_WAIT_SEC = 3

NONLINEAR_THRESHOLD = 30000

# 黑色的HSV范围
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 30])

# 白色的HSV范围
LOWER_WHITE = np.array([0, 0, 150])
UPPER_WHITE = np.array([180, 50, 255])

# 标靶检测形态学操作参数
REFINE_KSIZE = 12
ERODE_KSIZE = 4
ERODE_ITER = 3

# 随机数种子
RANDOM_STATE = 42

# 框线和文字的粗细
LINE_THICKNESS = 5
FONT_SCALE = 0.8

# 框线和文字颜色
BALL_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 0, 255)

# 标签文字显示偏移
TEXT_MARGIN = 10

# 分数文字的粗细
SCORE_THICKNESS = 3
SCORE_SCALE = 1

FPS_THICKNESS = 3
FPS_SCALE = 1

# 分数文字的颜色
SCORE_COLOR = (0, 255, 255)

FPS_COLOR = (0, 255, 0)

# 分数文字的位置
SCORE_ORG = (30, 50)

# 判断椭圆最少的多边形边数
MIN_POLY = 5

# 非椭圆容忍度
PERI_BIAS = 0.3

# 网球置信度
BALL_CONF = 0.05

# 标靶置信度
TARGET_CONF = 0.5

# 模型相关参数
OBJ_THRESH = 0.10
NMS_THRESH = 0.45
TENNIS_MODEL_PATH = "model/best-tennis-s.rknn"
TARGET_MODEL_PATH = "model/best-digit-n.rknn"
MODEL_IMGSIZE = 320
IMG_SIZE = (640, 640)
TENNIS_CLASSES = ("ball")
TENNIS_ID_LIST = [1]
TARGET_CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
TARGET_ID_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]

AREA_THRESHOLD_PERCENTAGE = 10
