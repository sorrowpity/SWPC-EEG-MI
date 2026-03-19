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
# I. 配置參數 (加入动态阈值+自适应投票)
# =================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPLOY_MODE = False
DATA_SOURCE_MODE = "REAL_DATASET"

EEG_IP, EEG_PORT = '127.0.0.1', 12345
ROBOT_IP, ROBOT_PORT = '127.0.0.1', 8888

CHANNELS = 22
TIME_POINTS = 250
SAMPLE_RATE = 250
STEP_SIZE = 10
# 核心优化1：动态阈值基准值（从0.6→0.75，作为动态调整的基准）
THRESHOLD_BASE = 0.82
DROPOUT_RATE = 0.5
# 新增：动态阈值滑动窗口大小
MI_P_WINDOW = 5
# 新增：自适应投票门槛（MI概率越高，门槛越低）
VOTE_THRESHOLD_HIGH = 5  # MI概率>0.8时，需要6票
VOTE_THRESHOLD_LOW = 6   # MI概率<0.8时，需要8票


# =================================================================
# II. 數據加載接口 (保持不变)
# =================================================================
class DataLoaderInterface:
    def __init__(self, mode, channels, time_points, test_subj=1, start_from_mi=False):
        self.mode = mode
        self.CHANNELS, self.TIME_POINTS = channels, time_points
        self.current_idx = 0
        self.R = None

        if self.mode == "REAL_DATASET":
            from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
            self.loader = EEG_loader_resting(test_subj=test_subj)
            self.current_idx = (70 * STEP_SIZE) if start_from_mi else 0
            self._prepare_stream()

            rest_samples = self.loader.test_x[self.loader.test_y == 0]
            if len(rest_samples) > 0:
                self.R = compute_EA_matrix(rest_samples)
                print(f"✅ EA 對齊矩陣計算完成，樣本數: {len(rest_samples)}")
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setblocking(True)
            try:
                self.sock.connect((EEG_IP, EEG_PORT))
                print(f"🧠 已連接腦控設備: {EEG_IP}:{EEG_PORT}")
            except:
                print("❌ 腦控設備連線失敗")
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
            action_map = {0: 'REST', 1: 'LEFT', 2: 'RIGHT'}
            label_name = action_map.get(raw_l, f'MI-{raw_l}')
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
# III. BCI 控制器 (核心优化：动态阈值+自适应投票)
# =================================================================
from EEGNet import EEGNet
from ShallowConvNet import ShallowConvNet


class BCI_Controller:
    def __init__(self, channels, time_points, device, deploy_mode):
        self.DEVICE = device
        self.DEPLOY_MODE = deploy_mode
        self.model_1 = ShallowConvNet(2, channels, time_points, DROPOUT_RATE).to(device)
        self.model_2 = EEGNet(2, channels, time_points, DROPOUT_RATE).to(device)
        self._load_weights()
        # 核心优化2：投票窗口大小从20→15，平衡响应速度和稳定性
        self.vote_buffer = deque(maxlen=10)
        # 新增：MI概率滑动窗口，用于计算动态阈值
        self.mi_p_buffer = deque(maxlen=MI_P_WINDOW)

        self.robot_sock = None
        if self.DEPLOY_MODE:
            try:
                self.robot_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.robot_sock.connect((ROBOT_IP, ROBOT_PORT))
                print(f"🤖 已連接機械臂服務: {ROBOT_IP}:{ROBOT_PORT}")
            except:
                print("⚠️ 機械臂連線失敗，切換為本地觀察模式")
                self.DEPLOY_MODE = False

    def _load_weights(self):
        root = os.path.dirname(os.path.abspath(__file__))
        self.model_1.load_state_dict(torch.load(os.path.join(root, 'prescreen_model_best_sl_ssl_cnet.pth'), map_location=self.DEVICE))
        self.model_2.load_state_dict(torch.load(os.path.join(root, 'classifier_model_best_sl_ssl_cnet.pth'), map_location=self.DEVICE))
        self.model_1.eval()
        self.model_2.eval()

    def process_data(self, eeg_window: np.ndarray, R_matrix=None):
        from utils_ea import apply_EA
        if R_matrix is not None:
            eeg_window = apply_EA(eeg_window, R_matrix)

        eeg_std = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-6)
        tensor = torch.from_numpy(eeg_std).float().unsqueeze(0).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            prob_mi = F.softmax(self.model_1(tensor), dim=1)[0, 1].item()
            # 核心优化3：计算动态阈值（基于MI概率的滑动平均值调整）
            self.mi_p_buffer.append(prob_mi)
            mi_p_avg = np.mean(self.mi_p_buffer) if self.mi_p_buffer else prob_mi
            # 动态阈值：基准值±0.05，根据平均MI概率调整
            dynamic_threshold = THRESHOLD_BASE + (mi_p_avg - 0.7) * 0.1
            dynamic_threshold = np.clip(dynamic_threshold, 0.7, 0.8)  # 限制在0.7~0.8之间

            raw_pred = 4
            conf = 1.0 - prob_mi if prob_mi < dynamic_threshold else 0.0

            if prob_mi >= dynamic_threshold:
                out2 = self.model_2(tensor)
                soft_out2 = F.softmax(out2, dim=1)
                conf2, pred2 = torch.max(soft_out2, dim=1)
                raw_pred = 1 if pred2.item() == 0 else 2
                conf = conf2.item()

            self.vote_buffer.append(raw_pred)
            counts = Counter(self.vote_buffer)
            most_common, count = counts.most_common(1)[0]


            # 核心优化4：自适应投票门槛（根据MI概率高低调整）
            final_id = 4  # 默认判定为STOP
            if prob_mi >= 0.7:  # 仅当MI概率≥0.7时，才判断是否为MI
                if most_common in [1, 2]:
                    vote_threshold = VOTE_THRESHOLD_HIGH if prob_mi >= 0.8 else VOTE_THRESHOLD_LOW
                    if count >= vote_threshold:
                        final_id = most_common
                # 否则保持final_id=4
            else:
                self.vote_buffer.append(4)  # MI概率<0.7，强制加STOP到投票池

            cmd_map = {1: "LEFT", 2: "RIGHT", 4: "STOP"}
            cmd_name = cmd_map[final_id]

            if self.DEPLOY_MODE and self.robot_sock:
                msg = json.dumps({"cmd": cmd_name, "conf": round(float(conf), 4)}) + "\n"
                try:
                    self.robot_sock.sendall(msg.encode('utf-8'))
                except:
                    pass

            return cmd_name, prob_mi, conf, dynamic_threshold  # 返回动态阈值，用于调试


# =================================================================
# IV. 主模擬循環 (run_simulation)
# =================================================================
def run_simulation():
    controller = BCI_Controller(CHANNELS, TIME_POINTS, DEVICE, DEPLOY_MODE)
    data_io = DataLoaderInterface(DATA_SOURCE_MODE, CHANNELS, TIME_POINTS)

    stats = {"rest_steps": 0, "rest_correct": 0, "mi_steps": 0, "mi_correct": 0}

    print(f"\n{'=' * 85}\n BCI 異步控制與實時統計系統（动态阈值版）\n{'=' * 85}")
    header = f"{'Step':<5} | {'真實動作':<8} | {'MI 概率':<7} | {'動態閾值':<7} | {'預測指令':<8} | {'當前準確率'} | {'延遲'}"
    print(header)
    print("-" * len(header))

    try:
        step = 0
        while True:
            t_cycle_start = time.perf_counter()
            eeg, real_action = data_io.get_next_window(STEP_SIZE)
            if eeg is None: break

            pred_cmd, mi_p, conf, dynamic_thresh = controller.process_data(eeg, R_matrix=data_io.R)
            step += 1

            acc_val = 0.0
            if real_action == 'REST':
                stats["rest_steps"] += 1
                if pred_cmd == 'STOP': stats["rest_correct"] += 1
                acc_val = stats["rest_correct"] / stats["rest_steps"]
            elif real_action != 'LIVE':
                stats["mi_steps"] += 1
                if pred_cmd == real_action: stats["mi_correct"] += 1
                if stats["mi_steps"] > 0:
                    acc_val = stats["mi_correct"] / stats["mi_steps"]

            if DEPLOY_MODE:
                if step % 10 == 0:
                    print(f">>> 已發送指令: {pred_cmd:<6} | MI Prob: {mi_p:.2f} | Step: {step}", end='\r')
            else:
                elapsed = time.perf_counter() - t_cycle_start
                print(f"#{step:04d} | {real_action:<8} | {mi_p:7.2f} | {dynamic_thresh:7.2f} | {pred_cmd:<8} | {acc_val * 100:5.1f}% | {elapsed*1000:4.1f}ms")

            elapsed = time.perf_counter() - t_cycle_start
            wait_time = max(0, (STEP_SIZE / SAMPLE_RATE) - elapsed)
            # time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n[用戶中斷]")
    finally:
        print(f"\n\n{'=' * 30} 性能統計總結 {'=' * 30}")
        if stats["rest_steps"] > 0:
            rest_acc = (stats["rest_correct"] / stats["rest_steps"]) * 100
            print(f"1. 靜息態穩定性 (REST Accuracy): {rest_acc:.2f}% ({stats['rest_correct']}/{stats['rest_steps']})")
        if stats["mi_steps"] > 0:
            mi_acc = (stats["mi_correct"] / stats["mi_steps"]) * 100
            print(f"2. 動作識別準確率 (MI Accuracy):   {mi_acc:.2f}% ({stats['mi_correct']}/{stats['mi_steps']})")
        print(f"{'=' * 74}\n")


if __name__ == "__main__":
    run_simulation()