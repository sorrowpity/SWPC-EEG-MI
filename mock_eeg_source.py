# mock_eeg_source.py
import socket
import time
import numpy as np
import random

# 必须与 inference_robot.py 中的 EEG_SOURCE_IP 和 EEG_SOURCE_PORT 一致
HOST = '127.0.0.1'
PORT = 12345

CHANNELS = 22
SAMPLE_RATE = 250
SAMPLES_PER_SECOND = SAMPLE_RATE  # 每次发送 1秒的数据量，用于模拟稳定流


def start_eeg_source():
    # 使用 TCP 协议
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"🧠 模拟EEG数据源启动，监听 {HOST}:{PORT}...")

        conn, addr = s.accept()
        print(f"✅ 成功连接到 BCI 客户端：{addr}")

        with conn:
            i = 0
            while True:
                i += 1

                # 1. 模拟生成 CHANNELS x SAMPLES_PER_SECOND 的数据
                # 随机生成一个 22x250 的矩阵，模拟 1 秒的数据
                # 添加一些随机的 "MI" 信号来测试 BCI 反应
                noise = np.random.randn(CHANNELS, SAMPLES_PER_SECOND).astype(np.float32) * 0.1
                if i % 10 == 0:
                    # 每 10 秒模拟一次运动想象信号
                    signal = np.random.randn(CHANNELS, SAMPLES_PER_SECOND).astype(np.float32) * 0.5
                    data = noise + signal
                else:
                    data = noise

                # 2. 将数据转换为原始 float32 字节流 (Channels x Time)
                byte_data = data.tobytes()

                # 3. 发送数据
                conn.sendall(byte_data)
                i += 1
                print(f"→ 循环 {i}: 发送 {data.size} 个 float32 采样点 (总 {len(byte_data)} 字节)...", end='\r')

                # 4. 模拟实时发送速度：发送 1 秒的数据，然后等待 1 秒
                time.sleep(0.001)

    except Exception as e:
        print(f"\n❌ EEG Source 发生错误: {e}")
    finally:
        s.close()
        print("\nEEG Source 关闭。")


if __name__ == '__main__':
    start_eeg_source()