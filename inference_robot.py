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
# I. 配置參數 (加入 Socket 地址)
# =================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模式切換
DEPLOY_MODE = False  # True: 輸出到機械臂；False: 僅本地打印
DATA_SOURCE_MODE = "REAL_DATASET"  # "REAL_DATASET" 或 "LIVE_SOCKET"

EEG_IP, EEG_PORT = '127.0.0.1', 12345
ROBOT_IP, ROBOT_PORT = '127.0.0.1', 8888

CHANNELS = 22
TIME_POINTS = 250
SAMPLE_RATE = 250
STEP_SIZE = 10
THRESHOLD = 0.50
DROPOUT_RATE = 0.6


# =================================================================
# II. 數據加載接口 (加入 LIVE_SOCKET 適配)
# =================================================================
class DataLoaderInterface:
    def __init__(self, mode, channels, time_points, test_subj=1, start_from_mi=False):
        self.mode = mode
        self.CHANNELS, self.TIME_POINTS = channels, time_points
        self.current_idx = 0
        self.R = None  # 存儲對齊矩陣

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
            # 初始化腦控設備 Socket
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
            # 從 Socket 讀取實時數據
            packet_size = CHANNELS * TIME_POINTS * 4
            try:
                data = self.sock.recv(packet_size, socket.MSG_WAITALL)
                if not data: return None, None
                window = np.frombuffer(data, dtype=np.float32).reshape(CHANNELS, TIME_POINTS)
                return window, "LIVE"
            except:
                return None, None


# =================================================================
# III. BCI 控制器 (加入機械臂輸出適配)
# =================================================================
# 🔥 修复：统一导入 ShallowConvNet
from ShallowConvNet_Lyapunov import ShallowConvNet


class BCI_Controller:
    def __init__(self, channels, time_points, device, deploy_mode):
        self.DEVICE = device
        self.DEPLOY_MODE = deploy_mode

        # 🔥 修复：两个模型全部换成 ShallowConvNet
        self.model_1 = ShallowConvNet(2, channels, time_points, DROPOUT_RATE).to(device)
        self.model_2 = ShallowConvNet(2, channels, time_points, DROPOUT_RATE).to(device)

        self._load_weights()
        self.vote_buffer = deque(maxlen=20)

        # 初始化機械臂 Socket
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

        # --- Stage 1 加载 ---
        path1 = os.path.join(root, 'models/prescreen_model_best_sl_ssl_cnet.pth')
        state_dict_1 = torch.load(path1, map_location=self.DEVICE)
        self.model_1.load_state_dict(state_dict_1, strict=False)

        # --- Stage 2 加载 ---
        path2 = os.path.join(root, 'models/classifier_model_best_sl_ssl_cnet.pth')
        state_dict_2 = torch.load(path2, map_location=self.DEVICE)
        self.model_2.load_state_dict(state_dict_2, strict=False)

        self.model_1.eval()
        self.model_2.eval()
        print("✅ 模型权重加载成功！(ShallowConvNet 双模型)")

    def process_data(self, eeg_window: np.ndarray, R_matrix=None):
        # --- [EA 關鍵步驟] 應用對齊 ---
        from utils_ea import apply_EA
        if R_matrix is not None:
            eeg_window = apply_EA(eeg_window, R_matrix)

        eeg_std = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-6)
        tensor = torch.from_numpy(eeg_std).float().unsqueeze(0).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            # 处理 model_1 输出（兼容双输出）
            out1 = self.model_1(tensor)
            if isinstance(out1, tuple):
                out1 = out1[0]
            prob_mi = F.softmax(out1, dim=1)[0, 1].item()

            raw_pred = 4
            conf = 1.0 - prob_mi if prob_mi < THRESHOLD else 0.0

            if prob_mi >= THRESHOLD:
                # 处理 model_2 输出
                out2 = self.model_2(tensor)
                if isinstance(out2, tuple):
                    out2 = out2[0]
                soft_out2 = F.softmax(out2, dim=1)
                conf2, pred2 = torch.max(soft_out2, dim=1)
                # 方向映射: 0->LEFT, 1->RIGHT
                raw_pred = 1 if pred2.item() == 0 else 2
                conf = conf2.item()

            self.vote_buffer.append(raw_pred)
            counts = Counter(self.vote_buffer)
            most_common, count = counts.most_common(1)[0]

            # 1. 動作判定 (LEFT/RIGHT)
            if most_common in [1, 2]:
                if count >= 5:
                    final_id = most_common
                else:
                    final_id = 4
            # 2. 靜息判定 (STOP)
            else:
                final_id = 4

            cmd_map = {1: "LEFT", 2: "RIGHT", 4: "STOP"}
            cmd_name = cmd_map[final_id]

            # 輸出至機械臂
            if self.DEPLOY_MODE and self.robot_sock:
                msg = json.dumps({"cmd": cmd_name, "conf": round(float(conf), 4)}) + "\n"
                try:
                    self.robot_sock.sendall(msg.encode('utf-8'))
                except:
                    pass

            return cmd_name, prob_mi, conf


# =================================================================
# IV. 主模擬循環 (run_simulation)
# =================================================================
def run_simulation():
    # 傳入 DEPLOY_MODE 參數
    controller = BCI_Controller(CHANNELS, TIME_POINTS, DEVICE, DEPLOY_MODE)
    # 傳入 DATA_SOURCE_MODE 參數
    data_io = DataLoaderInterface(DATA_SOURCE_MODE, CHANNELS, TIME_POINTS)

    stats = {"rest_steps": 0, "rest_correct": 0, "mi_steps": 0, "mi_correct": 0}

    print(f"\n{'=' * 85}\n BCI 異步控制與實時統計系統\n{'=' * 85}")
    header = f"{'Step':<5} | {'真實動作':<8} | {'MI 概率':<7} | {'預測指令':<8} | {'當前準確率'} | {'延遲'}"
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

            acc_val = 0.0
            if real_action == 'REST':
                stats["rest_steps"] += 1
                if pred_cmd == 'STOP': stats["rest_correct"] += 1
                acc_val = stats["rest_correct"] / stats["rest_steps"]
            elif real_action != 'LIVE':  # LIVE 模式不統計準確率
                stats["mi_steps"] += 1
                if pred_cmd == real_action: stats["mi_correct"] += 1
                if stats["mi_steps"] > 0:
                    acc_val = stats["mi_correct"] / stats["mi_steps"]

            # --- 輸出控制 ---
            if DEPLOY_MODE:
                if step % 10 == 0:
                    print(f">>> 已發送指令: {pred_cmd:<6} | MI Prob: {mi_p:.2f} | Step: {step}", end='\r')
            else:
                print(f"#{step:04d} | {real_action:<8} | {mi_p:7.2f} | {pred_cmd:<8} | {acc_val * 100:5.1f}%")

            # --- 頻率控制 ---
            elapsed = time.perf_counter() - t_cycle_start
            wait_time = max(0, (STEP_SIZE / SAMPLE_RATE) - elapsed)

    except KeyboardInterrupt:
        print("\n[用戶中斷]")
    finally:
        # 最終結算報告
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