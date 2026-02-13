import socket

# 必须与 inference_robot.py 中的 ROBOT_IP 和 ROBOT_PORT 一致
HOST = '127.0.0.1'
PORT = 8888


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"🤖 模拟机械臂服务器启动，监听 {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"✅ 成功连接到 BCI 客户端：{addr}")

    with conn:
        while True:
            try:
                # 接收数据块
                data = conn.recv(1024)
                if not data:
                    print("客户端断开连接。")
                    break

                # 打印接收到的 JSON 指令
                received_json = data.decode('utf-8').strip()
                print(f"← 接收指令: {received_json}")

            except ConnectionResetError:
                print("客户端强制关闭连接。")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                break


if __name__ == '__main__':
    start_server()