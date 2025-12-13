# inference_robot.py - BCI 异步控制中间件 V3.2 (Final Fix)
import numpy as np
import time
import json
import torch
import torch.nn.functional as F
import socket
import sys
import os
from collections import deque
import random
from typing import Optional

# =================================================================
# I. 模块导入 (IMPORTS)
# =================================================================
# 假设 EEGNet 和 ShallowConvNet 位于同级目录
from EEGNet import EEGNet
from ShallowConvNet import ShallowConvNet

# 假设你的数据集加载器位于同级目录
try:
    from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
except ImportError:
    print("错误：无法导入 EEG_cross_subject_loader_MI_resting.py。请确认文件路径。")
    sys.exit(1)


# =================================================================
# II. 配置参数 (CONFIGURATION)
# =================================================================

# --- 运行环境配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPLOY_MODE = True  # True: 发送指令到机械臂；False: 仅在本地打印指令

# DATA_SOURCE_MODE: "REAL_DATASET", "LIVE_SOCKET", "RANDOM_NOISE"
DATA_SOURCE_MODE = "REAL_DATASET"

# --- BCI/数据参数 ---
CHANNELS = 22  # 导联数
TIME_POINTS = 250  # 窗口长度 (1s @ 250Hz)
SAMPLE_RATE = 250  # 采样率
STEP_SIZE = 50  # 滑动步长 (200ms)
THRESHOLD = 0.7  # 预筛选阈值 (SWPC tau)
DROPOUT_RATE = 0.5  # Dropout率

# --- 机械臂通信参数 (仅用于 DEPLOY_MODE = True) ---
ROBOT_IP = '127.0.0.1'
ROBOT_PORT = 8888
# 实时 EEG 设备通信参数 (仅用于 DATA_SOURCE_MODE = "LIVE_SOCKET")
EEG_SOURCE_IP = '127.0.0.1'
EEG_SOURCE_PORT = 12345


# =================================================================
# III. 模块 A: 机械臂通信接口 (RobotInterface)
# =================================================================

class RobotInterface:
    def __init__(self, deploy_mode, ip, port):
        self.DEPLOY_MODE = deploy_mode
        self.IP = ip
        self.PORT = port
        self.socket = None
        self.instruction_map = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "GRASP", 4: "STOP"}
        if self.DEPLOY_MODE:
            self._connect_to_robot()

    def _connect_to_robot(self):
        """尝试连接到机械臂服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.IP, self.PORT))
            print(f"✅ 机械臂服务器连接成功：{self.IP}:{self.PORT}")
        except Exception as e:
            print(f"❌ 错误：无法连接到机械臂服务器。请检查 IP/端口和服务器状态。错误信息: {e}")
            self.socket = None
            self.DEPLOY_MODE = False  # 如果连接失败，强制降级为测试模式

    def send_command(self, command_id: int, confidence: float):
        """发送指令到机械臂，或在本地打印"""
        command_str = self.instruction_map.get(command_id, "UNKNOWN")

        # 格式化指令，兼容机械臂接收的 JSON 格式
        output_msg = {
            "cmd": command_str,
            "conf": float(f"{confidence:.4f}")
        }
        json_data = json.dumps(output_msg)

        if self.DEPLOY_MODE and self.socket:
            try:
                self.socket.sendall((json_data + '\n').encode('utf-8'))
            except Exception as e:
                print(f"❌ 警告：发送指令失败。连接断开？错误: {e}")
                self.DEPLOY_MODE = False  # 自动断开连接
        else:
            # 测试模式下，打印指令到控制台
            print(f"→ [TEST MODE] 指令: {command_str} (Conf: {confidence:.4f})")

        return command_str


# =================================================================
# IV. 模块 B: 数据输入接口 (Data Input)
# =================================================================

# --- B-1: 真实模拟数据加载器 (DataLoaderInterface) ---
class DataLoaderInterface:
    def __init__(self, channels: int, time_points: int, test_subj: int = 1):
        """加载真实数据集并拼接成连续数据流"""
        self.CHANNELS = channels
        self.TIME_POINTS = time_points
        self.current_idx = 0
        self.full_eeg_stream: Optional[np.ndarray] = None
        self.full_label_stream: Optional[np.ndarray] = None
        self.stream_length = 0

        self.loader = EEG_loader_resting(test_subj=test_subj)
        print(f"Prescreening数据形状: 训练X {self.loader.train_x.shape}, 训练Y {self.loader.train_y.shape}")

        # 打印训练集标签分布 (0: 静息态, 1-4: MI动作)
        labels, counts = np.unique(self.loader.train_y, return_counts=True)
        label_map = {1: 'Left', 2: 'Right', 3: 'Rest', 4: 'Feet/Tongue'}  # 假设标签映射
        print(f"训练集标签分布: {', '.join([f'{label_map.get(l, l)}({l})={c}' for l, c in zip(labels, counts)])}")

        self._prepare_stream()

    def _prepare_stream(self):
        """将所有测试试次拼接成一个连续的数据流"""
        test_data = self.loader.test_x  # 形状: (N_trials, Channels, Time_points)
        test_labels = self.loader.test_y

        # 将试次在时间轴上拼接起来
        # 假设每个试次是 250 个时间点，现在将 (N, 22, 250) 变成 (22, N*250)
        # 确保数据为 (Channels, Total_Time)

        # 1. 堆叠数据 (N_trials * 250)
        eeg_trials = [trial for trial in test_data]
        label_trials = [np.full(self.TIME_POINTS, label) for label in test_labels]

        # 2. 拼接数据和标签
        self.full_eeg_stream = np.concatenate(eeg_trials, axis=1)
        self.full_label_stream = np.concatenate(label_trials)
        self.stream_length = self.full_eeg_stream.shape[1]

        print(f"加载真实测试数据 (被试 {self.loader.test_subj})...")
        print(f"✅ 数据流加载完成。总长度: {self.stream_length} 个时间点")

    def get_next_window(self, step_size: int) -> tuple[Optional[np.ndarray], Optional[str]]:
        """从数据流中提取下一个窗口"""

        # 检查是否到达数据流末尾
        if self.current_idx + self.TIME_POINTS > self.stream_length:
            return None, None

        # 提取窗口数据 (Channels, Time_points)
        eeg_window = self.full_eeg_stream[:, self.current_idx:self.current_idx + self.TIME_POINTS]

        # 获取窗口中点的标签（用于调试打印）
        mid_label_index = self.current_idx + self.TIME_POINTS // 2
        raw_label = int(self.full_label_stream[mid_label_index])

        label_map_print = {1: '左手', 2: '右手', 3: '静息', 4: '脚/舌'}
        action_label = label_map_print.get(raw_label, '未知')

        # 滑动索引
        self.current_idx += step_size

        # 确保数据类型为 float32
        return eeg_window.astype(np.float32), action_label


# --- B-2: 随机噪声生成器 (RandomNoiseGenerator) ---
class RandomNoiseGenerator:
    def __init__(self, channels: int, time_points: int):
        self.CHANNELS = channels
        self.TIME_POINTS = time_points
        print(f"✅ 使用随机噪声作为数据源 ({self.CHANNELS}x{self.TIME_POINTS})")

    def get_next_window(self, step_size: int) -> tuple[Optional[np.ndarray], Optional[str]]:
        """生成随机噪声作为窗口数据"""
        # 生成一个模拟的 EEG 窗口 (Channels, Time_points)
        # 使用随机高斯噪声，模拟静息态
        eeg_window = np.random.randn(self.CHANNELS, self.TIME_POINTS).astype(np.float32) * 0.1

        # 随机模拟一个动作，用于测试系统的反应
        if random.random() < 0.2:  # 20% 的概率模拟 MI 动作
            eeg_window += np.random.randn(self.CHANNELS, self.TIME_POINTS).astype(np.float32) * 0.5
            action_label = random.choice(['左手', '右手'])
        else:
            action_label = '静息'

        # 在随机噪声模式下，我们不需要等待，直接返回
        return eeg_window, action_label


# --- B-3: 实时 EEG Socket 读取器 (RealTimeEEGReader) ---
class RealTimeEEGReader:
    def __init__(self, channels: int, time_points: int, ip: str, port: int):
        self.CHANNELS = channels
        self.TIME_POINTS = time_points
        self.IP = ip
        self.PORT = port
        self.socket = None
        self.buffer = deque(maxlen=2000)
        self.samples_per_chunk = 0

        self._partial_bytes = b''

        print(f"初始化实时EEG数据接口：{self.IP}:{self.PORT}...")
        self._connect_to_eeg_source()

    def _connect_to_eeg_source(self):
        """连接到脑电设备驱动程序/服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.socket.settimeout(3)
            self.socket.connect((self.IP, self.PORT))
            self.socket.setblocking(False)
            print("✅ 实时EEG数据源连接成功！")
        except Exception as e:
            print(f"❌ 警告：无法连接到EEG数据源 ({e})。请检查设备驱动程序是否运行。")
            self.socket = None

    def _receive_and_parse_data(self):
        """尝试从socket接收原始数据并解析/追加到缓冲区"""
        if self.socket is None:
            return 0

        received_samples = 0
        try:
            # 1. 尝试接收数据
            data_chunk = self.socket.recv(4096)
            if not data_chunk:
                return 0

            # 2. 将新接收的数据追加到上一次剩余的片段后
            full_bytes_stream = self._partial_bytes + data_chunk

            # 3. 计算一个 EEG 完整帧 (22个通道) 所需的字节数
            BYTES_PER_FRAME = self.CHANNELS * 4  # 4 bytes per float32

            # 4. 计算完整的 EEG 帧总共占用了多少字节
            # (确保只处理能被 BYTES_PER_FRAME 整除的那部分)
            full_bytes_length = (len(full_bytes_stream) // BYTES_PER_FRAME) * BYTES_PER_FRAME

            # 如果没有足够的数据构成一个完整的帧，则存储并等待下次接收
            if full_bytes_length == 0:
                self._partial_bytes = full_bytes_stream
                return 0

            # 5. 提取完整的帧数据用于解析
            data_to_parse = full_bytes_stream[:full_bytes_length]

            # 6. 存储剩余的片段，等待下次接收
            self._partial_bytes = full_bytes_stream[full_bytes_length:]

            # 7. 开始解析 (现在 data_to_parse 保证是完整的帧数据)
            new_data_flat = np.frombuffer(data_to_parse, dtype=np.float32)

            # 重新塑形为 (Channels, New_Time_Points)
            new_samples = new_data_flat.size // self.CHANNELS
            new_eeg_data = new_data_flat.reshape(self.CHANNELS, new_samples)

            # 8. 追加到缓冲区
            for t in range(new_samples):
                self.buffer.append(new_eeg_data[:, t])
            received_samples += 1

            # 打印调试信息（如果你还保留了之前的调试代码）
            # print(f"DEBUG RCV: 接收到 {new_samples} 新样本，当前缓冲 {len(self.buffer)}", file=sys.stderr)

            return received_samples

        except socket.error as e:
            if e.errno == 10035 or e.errno == 11:
                return 0
            print(f"⚠️ Socket错误，连接可能已断开：{e}")
            self.socket = None
            return 0

    def get_next_window(self, step_size: int) -> tuple[Optional[np.ndarray], Optional[str]]:
        """接收数据，如果缓冲区足够长，则滑动窗口并返回。"""
        self._receive_and_parse_data()
        if len(self.buffer) < self.TIME_POINTS:
            return None, None

        eeg_window_list = list(self.buffer)[-self.TIME_POINTS:]
        eeg_window = np.array(eeg_window_list, dtype=np.float32).T

        for _ in range(step_size):
            if self.buffer:
                self.buffer.popleft()

        return eeg_window, '实时'


# =================================================================
# V. 模块 C: BCI 控制逻辑 (BCI_Controller)
# =================================================================

class BCI_Controller:
    def __init__(self, channels, time_points, device):
        self.CHANNELS = channels
        self.TIME_POINTS = time_points
        self.DEVICE = device

        # 初始化模型
        # Stage 1: ShallowConvNet (2分类)
        self.model_prescreen = ShallowConvNet(
            num_classes=2,
            channels=channels,
            time_points=time_points,
            dropout_rate=DROPOUT_RATE
        ).to(device)

        # Stage 2: EEGNet (4分类)
        self.model_classify = EEGNet(
            num_classes=4,  # <--- 修正：这里必须是 4 (Left, Right, Feet, Tongue)
            channels=channels,
            time_points=time_points,
            dropout_rate=DROPOUT_RATE
        ).to(device)

        # 尝试加载模型权重
        self._load_weights()

        # 状态管理
        self.last_command_time = time.time()
        self.command_interval = 0.0  # 禁止强制去抖

    def _load_weights(self):
        """加载模型预训练权重"""
        try:
            # 获取当前脚本所在目录，确保路径正确
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 使用正确的文件名：prescreen_model.pth 和 classifier_model.pth
            prescreen_path = os.path.join(script_dir, 'prescreen_model.pth')
            classify_path = os.path.join(script_dir, 'classifier_model.pth')

            print(f"正在加载模型权重...\n 预筛选: {prescreen_path}\n 分类器: {classify_path}")

            self.model_prescreen.load_state_dict(torch.load(prescreen_path, map_location=self.DEVICE))
            self.model_prescreen.eval()
            print("- 预筛选模型已加载 (ShallowConvNet)")

            self.model_classify.load_state_dict(torch.load(classify_path, map_location=self.DEVICE))
            self.model_classify.eval()
            print("- 分类模型已加载 (EEGNet)")
        except FileNotFoundError as e:
            print(f"致命错误：模型加载失败！找不到权重文件。错误: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"致命错误：模型加载失败！{e}")
            sys.exit(1)

    def process_data(self, eeg_window: np.ndarray) -> tuple[int, float, float]:
        """
        核心处理逻辑：SWPC 两阶段分类
        """
        # 1. 数据预处理
        data_tensor = torch.from_numpy(eeg_window).float().unsqueeze(0).unsqueeze(0).to(self.DEVICE)

        # 2. 预筛选 (Prescreening)
        with torch.no_grad():
            output_prescreen = self.model_prescreen(data_tensor)
            prob_prescreen = F.softmax(output_prescreen, dim=1)
            mi_confidence = prob_prescreen[0, 1].item()

        command_id = 4  # 默认指令为 STOP
        classify_confidence = 0.0

        if mi_confidence >= THRESHOLD:
            # 3. 分类 (Classification)
            output_classify = self.model_classify(data_tensor)
            prob_classify = F.softmax(output_classify, dim=1)

            # 直接取4个类别的最大值
            max_conf, predicted_class_id = torch.max(prob_classify, dim=1)

            command_id = predicted_class_id.item()  # 0-3
            classify_confidence = max_conf.item()

        else:
            # 预筛选未通过：输出 STOP
            command_id = 4  # STOP
            classify_confidence = mi_confidence

        # 4. 指令平滑
        current_time = time.time()
        if command_id != 4 and (current_time - self.last_command_time) < self.command_interval:
            command_id = 4
            classify_confidence = 0.0

        self.last_command_time = current_time

        return command_id, mi_confidence, classify_confidence


# =================================================================
# VI. 模块 D: 主运行函数 (run_simulation)
# =================================================================

def run_simulation(data_source_mode: str, deploy_mode: bool):
    """
    BCI 系统的核心控制循环。
    """

    # 1. 初始化 BCI 控制器
    controller = BCI_Controller(CHANNELS, TIME_POINTS, DEVICE)
    print("初始化脑控系统...")

    # 2. 初始化数据输入接口
    data_io = None
    if data_source_mode == "REAL_DATASET":
        data_io = DataLoaderInterface(CHANNELS, TIME_POINTS)
    elif data_source_mode == "LIVE_SOCKET":
        data_io = RealTimeEEGReader(CHANNELS, TIME_POINTS, EEG_SOURCE_IP, EEG_SOURCE_PORT)
    elif data_source_mode == "RANDOM_NOISE":
        data_io = RandomNoiseGenerator(CHANNELS, TIME_POINTS)
    else:
        print(f"致命错误：未知的 DATA_SOURCE_MODE: {data_source_mode}")
        sys.exit(1)

    # 3. 初始化机械臂输出接口
    robot_io = RobotInterface(deploy_mode, ROBOT_IP, ROBOT_PORT)

    # 4. 运行循环
    print(f"\n======== BCI 控制循环启动 ========")
    print(
        f"模式: [{'部署模式 (发送网络)' if deploy_mode else '测试模式 (本地打印)'}] | 数据源: {data_source_mode} | 步长: {STEP_SIZE} 采样点")

    i = 0
    start_time = time.time()
    wait_step_count = 0  # <--- 新增：等待计数器

    while True:
        i += 1
        loop_start = time.perf_counter()

        # === 1. 获取数据窗口 ===
        eeg_window, raw_action_label = data_io.get_next_window(STEP_SIZE)

        if eeg_window is None:
            if data_source_mode == "REAL_DATASET":
                print("\n数据流处理完毕。")
                break

            current_buffer_size = len(data_io.buffer)  # 获取当前缓冲区大小
            wait_step_count += 1

            # 每等待 10 次（即 200ms * 10 = 2秒）打印一次状态
            if wait_step_count % 10 == 0:
                print(f"DEBUG WAIT: 正在等待数据... 缓冲: {current_buffer_size}/{TIME_POINTS}", end='\r')

            sleep_duration = STEP_SIZE / SAMPLE_RATE
            time.sleep(sleep_duration)
            continue

        # === 2. BCI 推理 ===
        command_id, mi_conf, classify_conf = controller.process_data(eeg_window)

        # === 3. 发送指令 ===
        confidence_to_send = classify_conf if command_id != 4 else mi_conf
        predicted_command = robot_io.send_command(command_id, confidence_to_send)

        # === 4. 计时和打印 ===
        loop_end = time.perf_counter()
        delay_ms = (loop_end - loop_start) * 1000

        print(
            f"Step {i:03d} | 动作: {raw_action_label} | Conf: {mi_conf:.2f} | 延迟: {delay_ms:.1f}ms | {predicted_command}",
            end='\r')
        if i % 100 == 0:
            print(
                f"Step {i:03d} | 动作: {raw_action_label} | Conf: {mi_conf:.2f} | 延迟: {delay_ms:.1f}ms | {predicted_command}")

        time_elapsed = loop_end - loop_start
        required_time = STEP_SIZE / SAMPLE_RATE
        if time_elapsed < required_time:
            time.sleep(required_time - time_elapsed)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n======== 模拟结束，总运行时间: {total_time:.2f}s ========")


# =================================================================
# VII. 主程序入口
# =================================================================

if __name__ == "__main__":
    deploy_mode = DEPLOY_MODE
    data_source_mode = DATA_SOURCE_MODE

    try:
        run_simulation(data_source_mode=data_source_mode, deploy_mode=deploy_mode)
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序发生异常：{e}")