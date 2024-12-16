import requests
import hashlib
import time
from constants import SALT, API_URL
from tools import log_with_timestamp

def generate_sign(data: dict, salt: str) -> str:
    """
    生成 sign 值，方法是将参数排序后拼接，加上盐值并计算 MD5 值。
    """
    sorted_items = sorted(data.items())
    sign_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    sign_string += f"&salt={salt}"
    return hashlib.md5(sign_string.encode('utf-8')).hexdigest()

def push_data(payload: dict, max_retry: int, retry_interval: int):
    """
    推送标注数据到小程序接口。
    自动添加当前时间戳、重传次数和盐值到 payload，
    并通过 POST 请求将数据作为 JSON 发送。
    """
    retry_count = 0
    success = False

    while retry_count <= max_retry:
        # 自动生成时间戳
        payload['timestamp'] = int(time.time())

        # 将 retry_count 添加到 payload 中
        payload['retry'] = retry_count

        # 生成 sign 值并添加到 payload 中
        payload['sign'] = generate_sign(payload, SALT)

        # 打印推送地址（仅供调试）
        log_with_timestamp(f"\033[93m推送到地址: {API_URL}\033[0m")

        try:
            print(payload)
            # 使用 POST 方法发送 JSON 数据
            response = requests.post(API_URL, json=payload)
            log_with_timestamp(f"\033[93m状态码: {response.status_code}\033[0m")
            log_with_timestamp(f"\033[93m响应内容: {response.text}\033[0m")

            if response.status_code == 200:
                log_with_timestamp("\033[92m推送成功\033[0m")
                success = True
                break
            else:
                log_with_timestamp(f"\033[91m推送失败，状态码: {response.status_code}, 内容: {response.text}\033[0m")
        except requests.RequestException as e:
            # 捕获请求异常并记录日志
            log_with_timestamp(f"\033[91m请求异常: {e}\033[0m")

        # 增加重试次数
        retry_count += 1

        if retry_count <= max_retry:
            # 如果未达到最大重试次数，等待指定时间后重试
            log_with_timestamp(f"\033[93m重试 {retry_count}/{max_retry} 次，等待 {retry_interval} 秒...\033[0m")
            time.sleep(retry_interval)

    if not success:
        # 如果达到最大重试次数仍失败，记录日志
        log_with_timestamp("\033[91m推送失败，达到最大重传次数。\033[0m")
