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

# 每隔多少秒重新检测标靶
RETARGET_WAIT_SEC = 1

# 等待多少秒判断碰撞
BALL_HIT_WAIT_SEC = 3

# 至少多少个点开启碰撞检测
MIN_DETECTION_SAMPLE = 6

# 判断是否为碰撞的直线拟合阈值
NONLINEAR_THRESHOLD = 6

ANGLE_THRESHOLD = 25

VELOCITY_RATIO_THRESHOLD = 1.5

TRAJECTORY_SPLIT_INTERVAL = 0.2

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

# 靶标排序权重
X_COOR_WEIGHT = 3
Y_COOR_WEIGHT = 10

# 随机数种子
RANDOM_STATE = 42

# 框线和文字的粗细
LINE_THICKNESS = 5
FONT_SCALE = 1

# 框线和文字颜色
BALL_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 0, 255)

# 轨迹点半径
TRACE_RADIUS = 10

# 标签文字显示偏移
TEXT_MARGIN = 30

# 标题文字的参数
TITLE_THICKNESS = 5
TITLE_SCALE = 1.75
TITLE_COLOR = (255, 255, 255)
TITLE_ORG = (30, 60)

# 分数文字的参数
SCORE_THICKNESS = 3
SCORE_SCALE = 1
SCORE_COLOR = (0, 200, 200)
SCORE_ORG = (30, 110)

# 帧率文字的参数
FPS_THICKNESS = 3
FPS_SCALE = 1
FPS_COLOR = (0, 255, 0)
FPS_ORG = (210, 110)

# 提示文字的参数
HINT_THICKNESS = 2
HINT_SCALE = 0.5
HINT_COLOR = (0, 255, 0)
HINT_1_ORG = (30, 400)
HINT_2_ORG = (30, 425)
HINT_3_ORG = (30, 450)

LOG_THICKNESS = 2
LOG_SCALE = 0.5
LOG_VALID_COLOR = (0, 255, 0)
LOG_INVALID_COLOR = (0, 200, 200)
LOG_VALID_ORG = (250, 450)
LOG_INVALID_ORG = (275, 450)

# 判断椭圆最少的多边形边数
MIN_POLY = 5

# 非椭圆容忍度
PERI_BIAS = 0.3

# 网球置信度
BALL_CONF = 0.05

# 标靶置信度
TARGET_CONF = 0.3

# 过滤小面积靶标识别结果（占最大识别结果的百分比）
AREA_THRESHOLD_PERCENTAGE = 10

# 模型相关参数
OBJ_THRESH = 0.10
NMS_THRESH = 0.45
TENNIS_MODEL_PATH = "model/best-tennis-s-relu.rknn"
TARGET_MODEL_PATH = "model/best-digit-s-relu.rknn"
MODEL_IMGSIZE = 320
IMG_SIZE = (640, 640)
TENNIS_CLASSES = ("ball")
TENNIS_ID_LIST = [1]
TARGET_CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
TARGET_ID_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
