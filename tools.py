import json
import time
from constants import CONFIG_FILE


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
