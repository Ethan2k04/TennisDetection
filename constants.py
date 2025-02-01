# 网络配置项
MAX_RETRY = 3  # 最大重传次数
RETRY_INTERVAL = 5  # 重传时间间隔（秒）
SALT = "learningxm"  # 固定盐值
API_URL = "https://wq.hoeagri.com/api/Training/receiveInput"
HANDLER_URL = "10.128.51.10"

# CONFIG 文件名为 config.json
CONFIG_FILE = "meta/config.json"

# SETTINGS 文件名为 settings.json
SETTINGS_FILE = "meta/settings.json"

# 存储 size 数组的文件
SIZE_FILE = "size_data.txt"

# yolo 检测开启的线程数
NUM_THREAD = 4

# yolo检测开启的进程数
NUM_PROCESSES = 4

# 进程间传输队列的大小
MAX_QUEUE = 100

# 每隔多少秒重新检测标靶
RETARGET_WAIT_SEC = 1

# 等待多少秒判断碰撞
BALL_HIT_WAIT_SEC = 3

# 轨迹分割时间间隔
TRAJECTORY_SPLIT_INTERVAL = 0.5

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

# 屏幕分辨率相关的比例系数
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
TITLE_ORG_RATIO = (0.03, 0.15)
SCORE_ORG_RATIO = (0.03, 0.25)
FPS_ORG_RATIO = (0.03, 0.35)
HINT_1_ORG_RATIO = (0.03, 0.80)
HINT_2_ORG_RATIO = (0.03, 0.85)
LOG_VALID_ORG_RATIO = (0.4, 0.85)
LOG_INVALID_ORG_RATIO = (0.4, 0.85)

# 标签文字显示偏移
TEXT_MARGIN = 30

# 标题文字的参数
TITLE_THICKNESS = 5
TITLE_SCALE = 1.75
TITLE_COLOR = (255, 255, 255)

# 分数文字的参数
SCORE_THICKNESS = 3
SCORE_SCALE = 1
SCORE_COLOR = (0, 200, 200)

# 帧率文字的参数
FPS_THICKNESS = 3
FPS_SCALE = 1
FPS_COLOR = (0, 255, 0)

# 提示文字的参数
HINT_THICKNESS = 2
HINT_SCALE = 0.5
HINT_COLOR = (0, 255, 0)

# 日志文字的参数
LOG_THICKNESS = 2
LOG_SCALE = 0.5
LOG_VALID_COLOR = (0, 255, 0)
LOG_INVALID_COLOR = (0, 200, 200)

# 判断椭圆最少的多边形边数
MIN_POLY = 5

# 非椭圆容忍度
PERI_BIAS = 0.2

# 网球置信度
BALL_CONF = 0.05

# 模型相关参数
OBJ_THRESH = 0.10
NMS_THRESH = 0.45
TENNIS_MODEL_PATH = "model/best-tennis-s-relu.rknn"
IMG_SIZE = (640, 640)
TENNIS_CLASSES = ("ball",)
TENNIS_ID_LIST = [1]
