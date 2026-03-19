import numpy as np
import time
import json
import torch
import torch.nn.functional as F
import socket
import sys
import os
from collections import deque, Counter
from utils_ea import compute_EA_matrix

# =================================================================
# I. 4分类配置参数（核心修改）
# =================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPLOY_MODE = False
DATA_SOURCE_MODE = "REAL_DATASET"  # "REAL_DATASET" 或 "LIVE_SOCKET"

EEG_IP, EEG_PORT = '127.0.0.1', 12345
ROBOT_IP, ROBOT_PORT = '127.0.0.1', 8888

CHANNELS = 22
# 适配BCI IV 2a：MI段4s=1000点，静息段0.5s=125点，统一取250点（1s）
TIME_POINTS = 250
SAMPLE_RATE = 250
STEP_SIZE = 10
THRESHOLD = 0.55
DROPOUT_RATE = 0.5

# ===================== 4分类标签/指令映射（核心修改）=====================
# 4类MI + 静息态
ACTION_MAP_4CLASS = {
    0: 'REST',  # 静息态
    1: 'LEFT',  # 左手
    2: 'RIGHT',  # 右手
    3: 'FOOT',  # 脚
    4: 'TONGUE',  # 舌头
    5: 'STOP'  # 停止指令
}
CMD_MAP_4CLASS = {
    1: "LEFT",
    2: "RIGHT",
    3: "FOOT",
    4: "TONGUE",
    5: "STOP"
}


# =================================================================
# II. 4分类数据加载接口（适配新的加载类）
# =================================================================
class DataLoaderInterface:
    def __init__(self, mode, channels, time_points, test_subj=1, start_from_mi=False):
        self.mode = mode
        self.CHANNELS, self.TIME_POINTS = channels, time_points
        self.current_idx = 0
        self.R = None

        if self.mode == "REAL_DATASET":
            # 导入4分类数据加载类
            from EEG_cross_subject_loader_MI_resting_4 import EEG_loader_resting_4class
            self.loader = EEG_loader_resting_4class(test_subj=test_subj)
            self.current_idx = (70 * STEP_SIZE) if start_from_mi else 0
            self._prepare_stream()

            rest_samples = self.loader.test_x[self.loader.test_y == 0]
            if len(rest_samples) > 0:
                self.R = compute_EA_matrix(rest_samples)
                print(f"✅ EA 对齐矩阵计算完成，样本数: {len(rest_samples)}")
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setblocking(True)
            try:
                self.sock.connect((EEG_IP, EEG_PORT))
                print(f"🧠 已连接脑控设备: {EEG_IP}:{EEG_PORT}")
            except:
                print("❌ 脑控设备连线失败")
                sys.exit(1)

    def _prepare_stream(self):
        self.full_eeg = np.concatenate([t for t in self.loader.test_x], axis=1)
        labels_list = [np.full(self.TIME_POINTS, l) for l in self.loader.test_y]
        self.full_labels = np.concatenate(labels_list)
        self.stream_length = self.full_eeg.shape[1]

    def get_next_window(self, step_size: int):
        if self.mode == "REAL_DATASET":
            if self.current_idx + self.TIME_POINTS > self.stream_length:
                return None, None
            window = self.full_eeg[:, self.current_idx: self.current_idx + self.TIME_POINTS]
            raw_l = int(self.full_labels[self.current_idx + self.TIME_POINTS // 2])
            # 4分类标签名称映射
            label_name = ACTION_MAP_4CLASS.get(raw_l, f'MI-{raw_l}')
            self.current_idx += step_size
            return window.astype(np.float32), label_name
        else:
            packet_size = CHANNELS * TIME_POINTS * 4
            try:
                data = self.sock.recv(packet_size, socket.MSG_WAITALL)
                if not data: return None, None
                window = np.frombuffer(data, dtype=np.float32).reshape(CHANNELS, TIME_POINTS)
                return window, "LIVE"
            except:
                return None, None


# =================================================================
# III. 4分类BCI控制器（核心修改：分类逻辑）
# =================================================================
from EEGNet import EEGNet
from ShallowConvNet import ShallowConvNet


class BCI_Controller_4class:
    def __init__(self, channels, time_points, device, deploy_mode):
        self.DEVICE = device
        self.DEPLOY_MODE = deploy_mode
        # ===================== 模型初始化（核心修改）=====================
        # prescreen模型：仍为2分类（REST/MI）
        self.model_1 = ShallowConvNet(2, channels, time_points, DROPOUT_RATE).to(device)
        # classifier模型：改为4分类（LEFT/RIGHT/FOOT/TONGUE）
        self.model_2 = EEGNet(4, channels, time_points, DROPOUT_RATE).to(device)

        self._load_weights()
        self.vote_buffer = deque(maxlen=20)

        # 机械臂Socket初始化
        self.robot_sock = None
        if self.DEPLOY_MODE:
            try:
                self.robot_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.robot_sock.connect((ROBOT_IP, ROBOT_PORT))
                print(f"🤖 已连接机械臂服务: {ROBOT_IP}:{ROBOT_PORT}")
            except:
                print("⚠️ 机械臂连线失败，切换为本地观察模式")
                self.DEPLOY_MODE = False

    def _load_weights(self):
        root = os.path.dirname(os.path.abspath(__file__))
        # 加载prescreen模型（2分类）
        path1 = os.path.join(root, 'models/prescreen_model_Lyapunov.pth')
        state_dict_1 = torch.load(path1, map_location=self.DEVICE)
        self.model_1.load_state_dict(state_dict_1, strict=False)

        # 加载4分类classifier模型（需提前训练好4分类权重）
        path2 = os.path.join(root, 'classifier_model_22ch_4cls_final.pth')
        state_dict_2 = torch.load(path2, map_location=self.DEVICE)
        self.model_2.load_state_dict(state_dict_2, strict=False)

        self.model_1.eval()
        self.model_2.eval()
        print("✅ 4分类模型权重加载成功！")

    def process_data(self, eeg_window: np.ndarray, R_matrix=None):
        # EA对齐
        from utils_ea import apply_EA
        if R_matrix is not None:
            eeg_window = apply_EA(eeg_window, R_matrix)

        # 标准化
        eeg_std = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-6)
        tensor = torch.from_numpy(eeg_std).float().unsqueeze(0).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            # Step 1: 判断是否为运动想象（2分类）
            prob_mi = F.softmax(self.model_1(tensor), dim=1)[0, 1].item()
            raw_pred = 5  # 默认STOP
            conf = 1.0 - prob_mi if prob_mi < THRESHOLD else 0.0

            # Step 2: 4分类MI动作识别（核心修改）
            if prob_mi >= THRESHOLD:
                out2 = self.model_2(tensor)
                soft_out2 = F.softmax(out2, dim=1)
                conf2, pred2 = torch.max(soft_out2, dim=1)
                # 4分类映射：pred2=0→LEFT(1), pred2=1→RIGHT(2), pred2=2→FOOT(3), pred2=3→TONGUE(4)
                raw_pred = pred2.item() + 1
                conf = conf2.item()

            # Step 3: 投票缓冲机制（适配4分类）
            self.vote_buffer.append(raw_pred)
            counts = Counter(self.vote_buffer)
            most_common, count = counts.most_common(1)[0]

            # 4分类动作判定逻辑
            if most_common in [1, 2, 3, 4]:  # 4类MI动作
                if count >= 10:  # 投票阈值
                    final_id = most_common
                else:
                    final_id = 5  # STOP
            else:
                final_id = 5  # STOP

            # 4分类指令映射
            cmd_name = CMD_MAP_4CLASS[final_id]

            # 输出至机械臂
            if self.DEPLOY_MODE and self.robot_sock:
                msg = json.dumps({"cmd": cmd_name, "conf": round(float(conf), 4)}) + "\n"
                try:
                    self.robot_sock.sendall(msg.encode('utf-8'))
                except:
                    pass

            return cmd_name, prob_mi, conf


# =================================================================
# IV. 4分类主模拟循环
# =================================================================
def run_simulation():
    controller = BCI_Controller_4class(CHANNELS, TIME_POINTS, DEVICE, DEPLOY_MODE)
    data_io = DataLoaderInterface(DATA_SOURCE_MODE, CHANNELS, TIME_POINTS)

    # 4分类准确率统计（扩展）
    stats = {
        "rest_steps": 0, "rest_correct": 0,
        "left_steps": 0, "left_correct": 0,
        "right_steps": 0, "right_correct": 0,
        "foot_steps": 0, "foot_correct": 0,
        "tongue_steps": 0, "tongue_correct": 0
    }

    print(f"\n{'=' * 90}\n BCI 4分类异步控制与实时统计系统\n{'=' * 90}")
    header = f"{'Step':<5} | {'真实动作':<8} | {'MI 概率':<7} | {'预测指令':<8} | {'当前准确率'} | {'延迟'}"
    print(header)
    print("-" * len(header))

    try:
        step = 0
        while True:
            t_cycle_start = time.perf_counter()
            eeg, real_action = data_io.get_next_window(STEP_SIZE)
            if eeg is None: break

            pred_cmd, mi_p, conf = controller.process_data(eeg, R_matrix=data_io.R)
            step += 1

            # 4分类准确率统计
            acc_val = 0.0
            if real_action == 'REST':
                stats["rest_steps"] += 1
                if pred_cmd == 'STOP': stats["rest_correct"] += 1
                acc_val = stats["rest_correct"] / stats["rest_steps"]
            elif real_action != 'LIVE':
                if real_action == 'LEFT':
                    stats["left_steps"] += 1
                    if pred_cmd == 'LEFT': stats["left_correct"] += 1
                    acc_val = stats["left_correct"] / stats["left_steps"] if stats["left_steps"] > 0 else 0.0
                elif real_action == 'RIGHT':
                    stats["right_steps"] += 1
                    if pred_cmd == 'RIGHT': stats["right_correct"] += 1
                    acc_val = stats["right_correct"] / stats["right_steps"] if stats["right_steps"] > 0 else 0.0
                elif real_action == 'FOOT':
                    stats["foot_steps"] += 1
                    if pred_cmd == 'FOOT': stats["foot_correct"] += 1
                    acc_val = stats["foot_correct"] / stats["foot_steps"] if stats["foot_steps"] > 0 else 0.0
                elif real_action == 'TONGUE':
                    stats["tongue_steps"] += 1
                    if pred_cmd == 'TONGUE': stats["tongue_correct"] += 1
                    acc_val = stats["tongue_correct"] / stats["tongue_steps"] if stats["tongue_steps"] > 0 else 0.0

            # 输出控制
            if DEPLOY_MODE:
                if step % 10 == 0:
                    print(f">>> 已发送指令: {pred_cmd:<8} | MI Prob: {mi_p:.2f} | Step: {step}", end='\r')
            else:
                print(f"#{step:04d} | {real_action:<8} | {mi_p:7.2f} | {pred_cmd:<8} | {acc_val * 100:5.1f}%")

            # 频率控制
            elapsed = time.perf_counter() - t_cycle_start
            wait_time = max(0, (STEP_SIZE / SAMPLE_RATE) - elapsed)
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        # 4分类最终统计
        print(f"\n\n{'=' * 35} 4分类性能统计总结 {'=' * 35}")
        if stats["rest_steps"] > 0:
            print(
                f"1. 静息态准确率: {(stats['rest_correct'] / stats['rest_steps']) * 100:.2f}% ({stats['rest_correct']}/{stats['rest_steps']})")
        if stats["left_steps"] > 0:
            print(
                f"2. 左手识别准确率: {(stats['left_correct'] / stats['left_steps']) * 100:.2f}% ({stats['left_correct']}/{stats['left_steps']})")
        if stats["right_steps"] > 0:
            print(
                f"3. 右手识别准确率: {(stats['right_correct'] / stats['right_steps']) * 100:.2f}% ({stats['right_correct']}/{stats['right_steps']})")
        if stats["foot_steps"] > 0:
            print(
                f"4. 脚识别准确率: {(stats['foot_correct'] / stats['foot_steps']) * 100:.2f}% ({stats['foot_correct']}/{stats['foot_steps']})")
        if stats["tongue_steps"] > 0:
            print(
                f"5. 舌头识别准确率: {(stats['tongue_correct'] / stats['tongue_steps']) * 100:.2f}% ({stats['tongue_correct']}/{stats['tongue_steps']})")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    run_simulation()