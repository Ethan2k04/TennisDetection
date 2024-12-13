import requests
import hashlib
import time
from constants import SALT, API_URL


def generate_sign(data: dict, salt: str) -> str:
    """
    生成 sign 值，方法是将参数排序后拼接，加上盐值并计算 MD5 值。
    """
    sorted_items = sorted(data.items())
    sign_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    sign_string += f"&salt={salt}"
    return hashlib.md5(sign_string.encode('utf-8')).hexdigest()


def build_query_string(data: dict) -> str:
    """
    根据数据构建查询字符串。
    """
    return "&".join(f"{key}={value}" for key, value in data.items())


def push_data(payload: dict, max_retry: int, retry_interval: int):
    """
    推送标注数据到小程序接口。
    自动添加当前时间戳、重传次数和盐值到 payload。
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

        # 构建 URL
        query_string = build_query_string(payload)
        full_url = f"{API_URL}?{query_string}"
        print(f"\033[93m推送到地址: {full_url}\033[0m")

        try:
            # TODO 好像服务器那边要求必须有一个 json 字段，不太清楚里面要填啥
            response = requests.post(full_url, json={})

            print(f"\033[93m状态码: {response.status_code}\033[0m")
            print(f"\033[93m响应内容: {response.text}\033[0m")  # 打印返回的原始内容

            if response.status_code == 200:
                print("\033[92m推送成功\033[0m")
                success = True
                break
            else:
                print(f"\033[91m推送失败，状态码: {response.status_code}, 内容: {response.text}\033[0m")
        except requests.RequestException as e:
            print(f"\033[91m请求异常: {e}\033[0m")

        # 增加重试次数
        retry_count += 1

        if retry_count <= max_retry:
            print(f"\033[93m重试 {retry_count}/{max_retry} 次，等待 {retry_interval} 秒...\033[0m")
            time.sleep(retry_interval)

    if not success:
        print("\033[91m推送失败，达到最大重传次数。\033[0m")
