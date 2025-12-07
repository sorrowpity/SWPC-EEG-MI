import torch
import numpy as np
import time
from EEGNet import EEGNet
from ShallowConvNet import ShallowConvNet
from collections import deque

# 这是最终交付给机械臂项目组的代码。它模拟实时接收数据，通过 SWPC 流程处理，并输出控制指令。

# === 配置参数 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNELS = 22  # 导联数
TIME_POINTS = 250  # 关键修正：必须是 250
DROPOUT_RATE = 0.5  # 训练时的Dropout率
THRESHOLD = 0.7  # 预筛选阈值 (SWPC论文中的 tau)

# 动作映射
ACTION_MAP = {
    0: "左手 (机械臂向左)",
    1: "右手 (机械臂向右)",
    2: "双脚 (机械臂前进)",
    3: "舌头 (机械臂抓取)"
}


class BCI_Controller:
    def __init__(self):
        print("初始化脑控系统...")

        # 1. 加载预筛选模型 (Stage 1: ShallowConvNet)
        # 使用最可能正确的参数名: time_points
        self.model_prescreen = ShallowConvNet(
            num_classes=2,
            channels=CHANNELS,
            time_points=TIME_POINTS,
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
        self.model_prescreen.load_state_dict(torch.load('prescreen_model.pth', map_location=DEVICE))
        self.model_prescreen.eval()
        print("- 预筛选模型已加载")

        # 2. 加载分类模型 (Stage 2: EEGNet)
        # 使用最可能正确的参数名: time_points
        self.model_classify = EEGNet(
            num_classes=4,
            channels=CHANNELS,
            time_points=TIME_POINTS,
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)

        # 为了解决架构不匹配问题 (Missing/Unexpected Keys)
        # 我们需要先加载权重，然后手动处理 key 映射。
        # 但最干净的方法是确保 EEGNet.py 结构与权重文件匹配。

        # 假设 Step 1 已经将 EEGNet.py 的尺寸修正为 112，我们直接加载
        self.model_classify.load_state_dict(torch.load('classifier_model.pth', map_location=DEVICE))
        self.model_classify.eval()
        print("- 分类模型已加载")

        # 平滑队列 (Moving Average)
        self.prob_queue = deque(maxlen=3)

    def predict(self, eeg_segment):
        """
        接收一段 EEG 数据，输出控制指令
        eeg_segment shape: (Channels, Time_Points) -> (22, 250)
        """
        # 数据预处理 (转Tensor, 加Batch维, 加Channel维)
        input_tensor = torch.from_numpy(eeg_segment).float().to(DEVICE)
        # 形状必须是 (Batch, 1, Channels, Time) -> (1, 1, 22, 250)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # === Step 1: 预筛选 (Prescreening) ===
            prescreen_out = self.model_prescreen(input_tensor)
            prescreen_prob = torch.softmax(prescreen_out, dim=1)
            mi_confidence = prescreen_prob[0][1].item()

            # 如果是静息态，直接返回
            if mi_confidence < THRESHOLD:
                return "Resting (No Action)", mi_confidence

            # === Step 2: 具体分类 (Classification) ===
            class_out = self.model_classify(input_tensor)
            class_prob = torch.softmax(class_out, dim=1)

            # === Step 3: 平滑处理 (可选) ===
            self.prob_queue.append(class_prob.cpu().numpy())
            avg_prob = np.mean(np.array(self.prob_queue), axis=0)

            predicted_class = np.argmax(avg_prob)
            action_name = ACTION_MAP[predicted_class]

            return action_name, mi_confidence


# === 模拟实时控制 ===
def run_simulation():
    try:
        controller = BCI_Controller()
    except RuntimeError as e:
        # 致命错误：架构或尺寸不匹配
        print(f"致命错误：{e}")
        print(
            "💡 **紧急修复提示：** 模型加载失败，这意味着您当前的 'EEGNet.py' 文件结构与训练时使用的模型结构不匹配！\n  请务必将 **训练时** 使用的 'EEGNet.py' 文件替换当前的 'EEGNet.py'，并确保计算的 FC 尺寸是 112。")
        return

    # 模拟生成一些随机 EEG 数据 (实际项目中这里接 EEG 设备 SDK)
    print("\n开始模拟实时数据流...\n")

    for i in range(10):
        # 模拟数据：(22, 250)
        fake_data = np.random.randn(CHANNELS, TIME_POINTS).astype(np.float32)

        # 记录开始时间
        start_time = time.time()

        # 获取预测
        command, confidence = controller.predict(fake_data)

        # 计算耗时
        latency = (time.time() - start_time) * 1000

        print(f"Frame {i + 1} | MI置信度: {confidence:.2f} | 指令: {command} | 延迟: {latency:.1f}ms")
        time.sleep(1)


if __name__ == "__main__":
    try:
        run_simulation()
    except FileNotFoundError:
        print("错误：未找到模型文件。请先运行 train_swpc.py 进行训练！")