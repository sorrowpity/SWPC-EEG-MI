import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import EEGNet  # 确保你更新了 model.py
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from ShallowConvNet import ShallowConvNet
from EEG_cross_subject_loader_MI import EEG_loader
import torch.nn.functional as F  # 尽管代码中未使用，但为常见库导入

# 这个脚本负责生成模型文件。它会运行两次训练流程：一次生成预筛选模型，一次生成分类模型

# =================================================================
# 配置参数
# =================================================================
BATCH_SIZE = 32
EPOCHS = 300
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJ = 1


# =================================================================
# 训练过程函数
# =================================================================
def train_process(model, train_loader, criterion, optimizer, save_path):
    model.train()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 确保输入维度是 (Batch, 1, Channels, Time)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            # EEGNet需要 float32
            inputs = inputs.float()
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)

            # 使用 nn.CrossEntropyLoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 增加最佳模型保存逻辑
        if total_loss < best_loss:
            best_loss = total_loss
            # 只有在训练损失下降时才保存（这是在没有验证集时的权宜之计）
            torch.save(model.state_dict(), save_path)
            print(f"** NEW BEST MODEL SAVED ** @ Epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}, Acc: {100 * correct / total:.2f}%")

    # 保存模型
    # torch.save(model.state_dict(), save_path)
    print(f"模型训练流程结束，最佳模型已保存至: {save_path}")


# =================================================================
# 主函数
# =================================================================
def main():
    print(f"正在使用设备: {DEVICE}")

    # --- 阶段 1: 训练预筛选模型 (Rest vs. MI) ---
    print("\nStarting Stage 1: Training Prescreening Model (Rest vs. MI)...")

    # 1. 加载数据 (含静息态)
    loader_rest = EEG_loader_resting(test_subj=TEST_SUBJ)
    X_train = loader_rest.train_x
    y_train = loader_rest.train_y

    # 2. 标签转换：静息态(3) -> 0, 运动想象(1,2,4) -> 1
    y_train_binary = np.where(y_train == 3, 0, 1)

    # 3. 创建 DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_binary))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 初始化模型 (2分类)
    channels = X_train.shape[1]
    time_points = X_train.shape[2]
    model_prescreen = ShallowConvNet(
        num_classes=2,
        channels=channels,
        time_points=time_points,
        dropout_rate=0.5
    ).to(DEVICE)

    # 5. 训练并保存
    criterion = nn.CrossEntropyLoss()  # 使用 CrossEntropyLoss
    optimizer = optim.Adam(
        model_prescreen.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY  # L2 正则化
    )
    train_process(model_prescreen, train_loader, criterion, optimizer, 'prescreen_model.pth')

    # --- 阶段 2: 训练分类模型 (Left/Right/Feet/Tongue) ---
    print("\nStarting Stage 2: Training Classification Model (Left/Right/Feet/Tongue)...")

    # 1. 加载数据 (只含运动想象)
    loader_cls = EEG_loader(test_subj=TEST_SUBJ)
    X_train_cls = loader_cls.train_x
    y_train_cls = loader_cls.train_y  # 已经是 0,1,2,3

    # 2. 创建 DataLoader
    train_dataset_cls = TensorDataset(torch.from_numpy(X_train_cls), torch.from_numpy(y_train_cls))
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化模型 (4分类)
    model_classifier = EEGNet(num_classes=4, channels=channels, time_points=time_points).to(DEVICE)

    # 4. 训练并保存
    # 沿用 Stage 1 的 criterion (CrossEntropyLoss)
    optimizer_cls = optim.Adam(model_classifier.parameters(), lr=LR)
    train_process(model_classifier, train_loader_cls, criterion, optimizer_cls, 'classifier_model.pth')

    print("\n所有模型训练完成！请运行 inference_robot.py 进行控制模拟。")


if __name__ == '__main__':
    main()