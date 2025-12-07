import torch
import numpy as np
import time
import socket
import json
from collections import deque
import os

# 导入模型和数据加载器（确保这些文件在你当前目录下）
from EEGNet import EEGNet
from ShallowConvNet import ShallowConvNet
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting

# === 全局配置参数 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNELS = 22  # 导联数
TIME_POINTS = 250  # 时间点数 (滑动窗口长度)
DROPOUT_RATE = 0.5  # Dropout率
THRESHOLD = 0.7  # 预筛选阈值 (SWPC tau)

# 动作映射
ACTION_MAP = {
    0: "左手",
    1: "右手",
    2: "双脚",
    3: "舌头",
    4: "静息"  # 预留给 Resting
}

# 抽象命令映射 (你和机械臂队友的通信合同)
# 0-3 对应 MI 动作，STOP 对应静息态
CMD_MAP = {
    0: "LEFT",
    1: "RIGHT",
    2: "FORWARD",
    3: "GRASP",
    4: "STOP"
}


# =================================================================
#                         模块 A: BCI 控制器 (核心推理逻辑)
# =================================================================

class BCI_Controller:
    def __init__(self):
        print("初始化脑控系统...")

        # 1. 加载预筛选模型 (Stage 1: ShallowConvNet)
        self.model_prescreen = ShallowConvNet(
            num_classes=2,
            channels=CHANNELS,  # ShallowConvNet的参数名称可能与你原始的参数名匹配
            time_points=TIME_POINTS,
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        self.model_prescreen.load_state_dict(torch.load('prescreen_model.pth', map_location=DEVICE))
        self.model_prescreen.eval()
        print("- 预筛选模型已加载 (ShallowConvNet)")

        # 2. 加载分类模型 (Stage 2: EEGNet)
        self.model_classify = EEGNet(
            num_classes=4,
            channels=CHANNELS,
            time_points=TIME_POINTS,
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        self.model_classify.load_state_dict(torch.load('classifier_model.pth', map_location=DEVICE))
        self.model_classify.eval()
        print("- 分类模型已加载 (EEGNet)")

        # 平滑队列 (Moving Average)
        self.prob_queue = deque(maxlen=3)

    def predict(self, eeg_segment):
        """
        接收一段 EEG 数据，输出动作名称、预测类别索引和置信度。
        eeg_segment shape: (Channels, Time_Points) -> (22, 250)
        """
        input_tensor = torch.from_numpy(eeg_segment).float().to(DEVICE)
        # 形状必须是 (Batch, 1, Channels, Time) -> (1, 1, 22, 250)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # NOTE: 你的 ShallowConvNet/EEGNet 的输入形状需要是 (B, C, T, E) 或 (B, 1, E, T)
        # 你的模型定义似乎使用了 (B, 1, E, T) 即 (Batch, 1, Channels, Time)，所以这里形状转换正确。

        with torch.no_grad():
            # === Stage 1: 预筛选 (Prescreening) ===
            prescreen_out = self.model_prescreen(input_tensor)
            prescreen_prob = torch.softmax(prescreen_out, dim=1)
            # MI (标签1) 的置信度
            mi_confidence = prescreen_prob[0][1].item()

            # 如果是静息态，直接返回
            if mi_confidence < THRESHOLD:
                return ACTION_MAP[4], 4, mi_confidence  # 4 对应 CMD_MAP["STOP"]

            # === Stage 2: 具体分类 (Classification) ===
            class_out = self.model_classify(input_tensor)
            class_prob = torch.softmax(class_out, dim=1)  # 4个MI类别的概率

            # === 平滑处理 ===
            self.prob_queue.append(class_prob.cpu().numpy())
            # 平均化最近 N 个预测的概率
            avg_prob = np.mean(np.array(self.prob_queue), axis=0)

            # 最终预测类别 (0, 1, 2, or 3)
            predicted_class = np.argmax(avg_prob)
            action_name = ACTION_MAP[predicted_class]

            return action_name, predicted_class, mi_confidence  # 返回 0-3 动作类别索引


# =================================================================
#                         模块 B: 数据输入接口 (DataLoaderInterface)
# =================================================================

class DataLoaderInterface:
    def __init__(self, channels, time_points):
        self.CHANNELS = channels
        self.TIME_POINTS = time_points
        self.loader = EEG_loader_resting(test_subj=1)  # 默认测试被试1
        self.full_eeg_stream = self._prepare_stream()
        self.current_step = 0

    def _prepare_stream(self):
        """将测试集的试次拼接成一个长的连续流 (Channels, Total_Time)"""
        print(f"加载真实测试数据 (被试 {self.loader.test_subj})...")
        # test_x 形状 (N_trials, Channels, Time)
        test_data = self.loader.test_x

        # 将所有试次在时间轴上拼接到一起：(Channels, Total_Time)
        full_stream = np.concatenate(test_data, axis=-1)
        print(f"✅ 数据流加载完成。总长度: {full_stream.shape[1]} 个时间点")
        return full_stream

    def get_next_window(self, step_size):
        """返回下一个滑动窗口的数据 (形状: (22, 250))"""
        WINDOW_LENGTH = self.TIME_POINTS

        start_idx = self.current_step * step_size
        end_idx = start_idx + WINDOW_LENGTH

        # 检查是否到达数据末尾
        if end_idx > self.full_eeg_stream.shape[1]:
            return None

        # 提取窗口数据 (形状: (Channels, Time))
        eeg_window = self.full_eeg_stream[:, start_idx:end_idx]
        self.current_step += 1
        return eeg_window


# =================================================================
#                         模块 C: 机械臂输出接口 (RobotInterface)
# =================================================================

class RobotInterface:
    def __init__(self, deploy_mode=False, ip='127.0.0.1', port=8888):
        self.DEPLOY_MODE = deploy_mode
        self.IP = ip
        self.PORT = port
        self.socket = None

        if self.DEPLOY_MODE:
            self._connect_to_robot()

    def _connect_to_robot(self):
        """部署模式下：建立TCP连接或初始化串口连接"""
        print(f"尝试连接机械臂接口：{self.IP}:{self.PORT}...")
        try:
            # 这是一个示例，你需要根据队友2的实际接口进行修改
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2)  # 设置连接超时
            self.socket.connect((self.IP, self.PORT))
            print("✅ 机械臂接口连接成功！")
        except Exception as e:
            print(f"❌ 警告：连接失败 ({e})，将退回测试模式。请检查机械臂服务器是否开启。")
            self.DEPLOY_MODE = False

    def send_command(self, predicted_class, confidence):
        """发送控制命令"""
        # 获取抽象命令，STOP 对应类别 4
        abstract_cmd = CMD_MAP.get(predicted_class, "STOP")

        # 组装最终指令（例如 JSON 格式）
        robot_command = {
            "action": abstract_cmd,
            "confidence": round(confidence, 4),
        }

        if self.DEPLOY_MODE:
            # 实际部署模式：发送网络数据包给队友2
            try:
                message = json.dumps(robot_command).encode('utf-8')
                self.socket.sendall(message)
                return f"[DEPLOY] 指令已发送: {robot_command['action']} (Conf: {robot_command['confidence']})"
            except Exception as e:
                # 如果发送失败，打印错误并继续 (可能连接断开)
                return f"❌ 发送失败，连接断开: {e}"
        else:
            # 调试/测试模式：只打印指令，保留测试能力
            return f"→ [TEST MODE] 指令: {abstract_cmd} (Conf: {robot_command['confidence']})"


# =================================================================
#                             主程序循环
# =================================================================

def run_simulation(deploy_mode=False, use_real_data=True):
    """
    主程序运行函数，通过 deploy_mode 切换模式。
    """
    # 初始化控制器 (模型加载)
    try:
        controller = BCI_Controller()
    except Exception as e:
        print(f"致命错误：模型加载失败！{e}")
        return

    # 初始化接口
    # 假设机械臂接口的IP和端口需要和队友2确认
    robot_io = RobotInterface(deploy_mode=deploy_mode, ip='127.0.0.1', port=8888)

    if use_real_data:
        # 使用真实数据流
        try:
            data_io = DataLoaderInterface(CHANNELS, TIME_POINTS)
        except FileNotFoundError:
            print("错误：无法加载数据集。请确保 data/BNCI2014001.mat 文件存在。")
            return

    # 定义滑动窗口步长 (决定了推理的频率)
    # 50个时间点相当于 200ms (采样率 250Hz)
    STEP_SIZE = 50

    print("\n======== BCI 控制循环启动 ========")
    mode_str = "[部署模式 (网络连接)]" if deploy_mode else "[测试模式 (本地打印)]"
    data_str = "真实数据流" if use_real_data else "随机噪声"
    print(f"模式: {mode_str} | 数据源: {data_str} | 步长: {STEP_SIZE} 采样点")

    i = 0
    while True:
        # === 1. 获取数据窗口 ===
        if use_real_data:
            eeg_window = data_io.get_next_window(STEP_SIZE)
            if eeg_window is None:
                break  # 数据流结束
        else:
            # 随机噪声测试 (保留测试能力)
            eeg_window = np.random.randn(CHANNELS, TIME_POINTS).astype(np.float32)

        # === 2. 模型推理 ===
        start_time = time.time()
        # action_name: 动作名称, predicted_class: 0-3 MI或4静息, confidence: 置信度
        action_name, predicted_class, confidence = controller.predict(eeg_window)
        latency = (time.time() - start_time) * 1000

        # === 3. 发送指令 ===
        command_log = robot_io.send_command(predicted_class, confidence)

        # === 4. 打印调试信息 ===
        print(
            f"Step {i + 1:03d} | 动作: {action_name} | Conf: {confidence:.2f} | 延迟: {latency:.1f}ms | {command_log}")

        i += 1
        # 模拟下一个窗口到来的时间间隔 (50/250Hz = 0.2s)
        time.sleep(0.01)  # 实际运行中可根据需求调整或移除，让它跑得更快

    print("\n======== 循环结束 ========")


if __name__ == "__main__":

    # **重要配置区域**
    # 默认值：保留测试模式 (False)，使用真实数据 (True)
    DEPLOY_MODE = False  # True: 尝试连接机械臂，False: 仅打印指令
    USE_REAL_DATA = True  # True: 使用 EEG_loader_resting 加载的数据流，False: 使用随机噪声


    try:
        run_simulation(deploy_mode=DEPLOY_MODE, use_real_data=USE_REAL_DATA)
    except FileNotFoundError:
        print("错误：请检查模型文件 (.pth) 和数据集文件 (.mat) 是否存在！")


    # 用于连接机械臂
    # robot_io = RobotInterface(deploy_mode=DEPLOY_MODE, ip='127.0.0.1', port=8888)