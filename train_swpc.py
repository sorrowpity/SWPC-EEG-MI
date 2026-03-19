import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# 導入你的模型和加載器
from model import EEGNet
from ShallowConvNet import ShallowConvNet
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from EEG_cross_subject_loader_MI import EEG_loader
from visualizer import plot_training_history

# =================================================================
# I. 配置參數 (針對魯棒性優化)
# =================================================================
BATCH_SIZE = 16
EPOCHS = 500
LR = 0.0005
WEIGHT_DECAY = 0.01  # 加強 L2 正則化
DROPOUT_RATE = 0.6  # <--- 新增這一行，建議從 0.5 提高到 0.6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJ = 1


# =================================================================
# 新增：标签归一化函数（核心修复）
# =================================================================
def normalize_labels(labels):
    """
    将任意整数标签归一化为 0 开始的连续整数（0,1,2,...）
    例如：[1,2] -> [0,1]，[-1,1] -> [0,1]，[2,3] -> [0,1]
    """
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    normalized_labels = np.array([label_map[old] for old in labels])
    print(f"标签映射: {label_map} (原始标签: {unique_labels}, 归一化后: {np.unique(normalized_labels)})")
    return normalized_labels


# =================================================================
# II. 數據增強 (解決窗口偏移敏感與盲目自信)
# =================================================================
def augment_batch(inputs, shift_limit=25, noise_level=0.01):
    """
    inputs shape: (Batch, 1, Channels, Time)
    """
    b, c, h, w = inputs.shape
    # 1. 隨機時間平移
    for i in range(b):
        shift = np.random.randint(-shift_limit, shift_limit)
        if shift > 0:
            inputs[i, :, :, shift:] = inputs[i, :, :, :-shift]
            inputs[i, :, :, :shift] = 0
        elif shift < 0:
            inputs[i, :, :, :shift] = inputs[i, :, :, -shift:]
            inputs[i, :, :, shift:] = 0
    # 2. 注入隨機高斯噪聲
    noise = torch.randn_like(inputs) * noise_level
    return inputs + noise


# =================================================================
# III. 核心訓練流程 (含驗證與早停)
# =================================================================
def train_process(model, train_loader, val_loader, criterion, optimizer, save_path):
    best_val_loss = float('inf')
    patience = 30  # 如果 30 輪驗證集 Loss 不降則停止
    counter = 0
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}  # 將 train_loss 改回 loss

    # --- 核心改進 1: 引入學習率調度器 ---
    # 當 10 輪驗證集 Loss 不降時，將學習率乘以 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print(f"開始訓練模型: {save_path}...")

    for epoch in range(EPOCHS):
        # --- 訓練階段 ---
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for inputs, labels in train_loader:
            # Z-score 標準化
            inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
            inputs = inputs.unsqueeze(1).float().to(DEVICE)

            # 訓練集才使用數據增強
            inputs = augment_batch(inputs, shift_limit=30, noise_level=0.02)
            labels = labels.long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            t_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            t_total += labels.size(0)
            t_correct += (pred == labels).sum().item()

        # --- 驗證階段 (不使用數據增強) ---
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                labels = labels.long().to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        # 記錄數據
        epoch_t_acc = 100 * t_correct / t_total
        epoch_v_acc = 100 * v_correct / v_total
        epoch_v_loss = v_loss / len(val_loader)

        # 修复：只调用一次 scheduler.step()
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_v_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # 2. 修改記錄部分 (約第 102 行)
        history['loss'].append(t_loss / len(train_loader))  # 改為 loss
        history['acc'].append(epoch_t_acc)  # 改為 acc
        history['val_loss'].append(epoch_v_loss)
        history['val_acc'].append(epoch_v_acc)

        # 保存最佳模型
        if epoch_v_loss < best_val_loss:
            best_val_loss = epoch_v_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | Train: {epoch_t_acc:5.1f}% | Val: {epoch_v_acc:5.1f}% | ValLoss: {epoch_v_loss:.4f}")

        if counter >= patience:
            print(f"早停觸發！在第 {epoch + 1} 輪停止訓練。")
            break

        if old_lr != new_lr:
            print(f"📉 學習率從 {old_lr:.6f} 下降至 {new_lr:.6f}")

    plot_training_history(history, title=save_path, save_path=f"{save_path}.png")
    print(f"✅ 模型已保存至: {save_path}\n")


from scipy.linalg import inv, sqrtm


def compute_EA_matrix(X):
    """
    計算 EA 對齊矩陣 R = mean(X * X^T)^(-1/2)
    X: (N_samples, Channels, TimePoints)
    """
    n_samples = X.shape[0]
    n_channels = X.shape[1]
    cov = np.zeros((n_channels, n_channels))
    for i in range(n_samples):
        # 計算單個樣本的協方差: (C, T) @ (T, C) -> (C, C)
        trial_cov = np.dot(X[i], X[i].T) / X.shape[2]
        cov += trial_cov
    mean_cov = cov / n_samples
    # 計算 R = mean_cov^(-1/2)
    R = inv(sqrtm(mean_cov)).real
    return R


def apply_EA_to_dataset(X, subj_indices):
    """
    按被試分別進行對齊
    subj_indices: 與 X 對應的被試編號列表
    """
    X_aligned = np.zeros_like(X)
    unique_subjs = np.unique(subj_indices)
    print(f"--- 執行 EA 對齊，總計 {len(unique_subjs)} 個被試 ---")
    for subj in unique_subjs:
        mask = (subj_indices == subj)
        # 1. 提取該被試所有數據並計算對齊矩陣
        R = compute_EA_matrix(X[mask])
        # 2. 應用對齊: X_aligned = R * X
        X_aligned[mask] = np.matmul(R, X[mask])
    return X_aligned


# =================================================================
# IV. 主函數
# =================================================================
def main():
    # 1. 訓練階段 1：預篩選 (Rest vs MI)
    print("\n>>> Stage 1: Training Prescreening (Rest vs MI)")
    loader_rest = EEG_loader_resting(test_subj=TEST_SUBJ)

    # 【添加 EA 區域】
    # 注意：假設你的 loader_rest 有提供 train_subj 標籤，如果沒有，
    # 這裡可以暫時把所有 train_x 看作一體（雖然效果略打折扣），
    # 理想情況是傳入 loader_rest.train_subj
    train_x_ea = apply_EA_to_dataset(loader_rest.train_x, loader_rest.train_subj)

    # 对 Stage 1 标签也做归一化（保险）
    train_y_norm_1 = normalize_labels(loader_rest.train_y)

    # 修改原本的 TensorDataset，使用 train_x_ea 和归一化后的标签
    full_ds_1 = TensorDataset(torch.from_numpy(train_x_ea), torch.from_numpy(train_y_norm_1))

    # 80/20 劃分
    train_size = int(0.8 * len(full_ds_1))
    val_size = len(full_ds_1) - train_size
    ds_train_1, ds_val_1 = random_split(full_ds_1, [train_size, val_size])

    train_loader_1 = DataLoader(ds_train_1, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_1 = DataLoader(ds_val_1, batch_size=BATCH_SIZE)

    model_1 = ShallowConvNet(2, loader_rest.train_x.shape[1], loader_rest.train_x.shape[2], DROPOUT_RATE).to(DEVICE)
    # 使用 Label Smoothing 抑制盲目自信
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_process(model_1, train_loader_1, val_loader_1, criterion, optimizer_1, 'prescreen_model.pth')

    # --- 分割線 ---

    # 2. 訓練階段 2：左 vs 右
    print(">>> Stage 2: Training Classifier (Left vs Right)")
    loader_cls = EEG_loader(test_subj=TEST_SUBJ)

    if hasattr(loader_cls, 'train_subj'):
        train_x_ea_2 = apply_EA_to_dataset(loader_cls.train_x, loader_cls.train_subj)
    else:
        print("⚠️ 警告: Stage 2 加載器缺少 train_subj，將使用全局 EA (效果較差)")
        R_global = compute_EA_matrix(loader_cls.train_x)
        train_x_ea_2 = np.matmul(R_global, loader_cls.train_x)

    # 核心修复：归一化 Stage 2 的标签
    train_y_norm_2 = normalize_labels(loader_cls.train_y)

    full_ds_2 = TensorDataset(torch.from_numpy(train_x_ea_2), torch.from_numpy(train_y_norm_2))

    train_size_2 = int(0.8 * len(full_ds_2))
    val_size_2 = len(full_ds_2) - train_size_2
    ds_train_2, ds_val_2 = random_split(full_ds_2, [train_size_2, val_size_2])

    train_loader_2 = DataLoader(ds_train_2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_2 = DataLoader(ds_val_2, batch_size=BATCH_SIZE)

    model_2 = EEGNet(2, loader_cls.train_x.shape[1], loader_cls.train_x.shape[2], DROPOUT_RATE).to(DEVICE)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_process(model_2, train_loader_2, val_loader_2, criterion, optimizer_2, 'classifier_model.pth')


if __name__ == '__main__':
    main()