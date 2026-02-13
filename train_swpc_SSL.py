import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import torch.nn.functional as F
import copy
from scipy.linalg import inv, sqrtm

# 導入自定義模塊
from model import EEGNet
from ShallowConvNet_SSL import ShallowConvNet
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from EEG_cross_subject_loader_MI import EEG_loader
from visualizer import plot_training_history

# =================================================================
# I. 配置參數（核心优化：更低SSL学习率+更短SSL轮数）
# =================================================================
BATCH_SIZE = 16
EPOCHS = 500
LR = 0.0005
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJ = 1
# 核心修复1：预筛选SSL学习率再降一半，避免覆盖REST特征
SSL_PRESCREEN_LR = 5e-6    # 从1e-5 → 5e-6
SSL_CLASSIFIER_LR = 8e-6
SSL_EPOCHS = 20            # 核心修复2：SSL轮数从40→20，减少过度拟合
DELTA = 0.3
SIGMA = 2.0
REST_WEIGHT = 1.5          # REST样本权重，强制保留REST特征

# =================================================================
# II. 數據增強 + 损失函数（集成维度匹配+REST权重）
# =================================================================
def augment_batch(inputs, shift_limit=25, noise_level=0.01):
    b, c, h, w = inputs.shape
    for i in range(b):
        shift = np.random.randint(-shift_limit, shift_limit)
        if shift > 0:
            inputs[i, :, :, shift:] = inputs[i, :, :, :-shift]
            inputs[i, :, :, :shift] = 0
        elif shift < 0:
            inputs[i, :, :, :shift] = inputs[i, :, :, -shift:]
            inputs[i, :, :, shift:] = 0
    noise = torch.randn_like(inputs) * noise_level
    return inputs + noise

def create_negative_samples(rest_data, mi_data, n_samples):
    neg_samples = []
    n_rest = len(rest_data)
    n_mi = len(mi_data)
    for _ in range(n_samples):
        r_idx = np.random.randint(0, n_rest)
        m_idx = np.random.randint(0, n_mi)
        r = rest_data[r_idx]
        m = mi_data[m_idx]
        neg = 0.5 * (r + m)
        neg_samples.append(neg)
    return np.array(neg_samples)

# EMA提取器（保持梯度启用）
class EMAExtractor:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.ema_params = None
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.model = self.model.to(DEVICE)

    def update(self):
        if self.ema_params is None:
            self.ema_params = copy.deepcopy(self.model.state_dict())
            print(f"✅ EMA参数初始化完成，共{len(self.ema_params)}个参数")
            return
        current_params = self.model.state_dict()
        for k in self.ema_params.keys():
            assert k in current_params, f"参数名不匹配：{k}"
            assert self.ema_params[k].shape == current_params[k].shape, f"维度不匹配：{k}"
            self.ema_params[k].data = self.decay * self.ema_params[k].data + (1 - self.decay) * current_params[k].data
        self.model.load_state_dict(self.ema_params)

# 预筛选损失函数：加入REST权重，避免遗忘REST
def contrastive_loss(f_theta_pos, f_phi_pos, f_phi_neg):
    f_theta_pos = F.normalize(f_theta_pos, p=2, dim=1)
    f_phi_pos = F.normalize(f_phi_pos, p=2, dim=1)
    f_phi_neg = F.normalize(f_phi_neg, p=2, dim=1)
    pos_dist = torch.sum((f_theta_pos - f_phi_pos) ** 2, dim=1) / (2 * SIGMA ** 2)
    neg_dist = torch.sum((f_theta_pos - f_phi_neg) ** 2, dim=1) / (2 * SIGMA ** 2)
    # 核心：给负样本（含REST）加权重，强制模型记住REST
    loss = -torch.log(torch.exp(-pos_dist) / (torch.exp(-pos_dist) + DELTA * REST_WEIGHT * torch.exp(-neg_dist)))
    return loss.mean()

# 分类模块增强+损失（保持不变）
def classifier_augment(x):
    batch_size, _, ch, time = x.shape
    x1 = x.clone()
    x2 = x.clone()
    noise1 = torch.randn_like(x1) * 0.01
    noise2 = torch.randn_like(x2) * 0.01
    x1 += noise1
    x2 += noise2
    scale1 = torch.randint(0, 2, (batch_size, 1, 1, 1), device=x.device) * 0.5 + 0.75
    scale2 = torch.randint(0, 2, (batch_size, 1, 1, 1), device=x.device) * 0.5 + 0.75
    x1 *= scale1
    x2 *= scale2
    mask_prob = 0.5
    mask_len = int(time * 0.05)
    for i in range(batch_size):
        if torch.rand(1) < mask_prob:
            start = torch.randint(0, time - mask_len, (1,)).item()
            x1[i, :, :, start:start+mask_len] = 0
        if torch.rand(1) < mask_prob:
            start = torch.randint(0, time - mask_len, (1,)).item()
            x2[i, :, :, start:start+mask_len] = 0
    shift1 = np.random.randint(-10, 10, batch_size)
    shift2 = np.random.randint(-10, 10, batch_size)
    for i in range(batch_size):
        x1[i] = torch.roll(x1[i], shift1[i], dims=-1)
        x2[i] = torch.roll(x2[i], shift2[i], dims=-1)
    return x1, x2

def classifier_contrastive_loss(f_theta, f_phi):
    f_theta = F.normalize(f_theta, p=2, dim=1)
    f_phi = F.normalize(f_phi, p=2, dim=1)
    batch_size = f_theta.shape[0]
    pos_dist = torch.sum((f_theta - f_phi) ** 2, dim=1) / (2 * SIGMA ** 2)
    neg_dist = []
    for i in range(batch_size):
        neg_mask = torch.arange(batch_size) != i
        neg_phi = f_phi[neg_mask]
        i_dist = torch.sum((f_theta[i].unsqueeze(0) - neg_phi) ** 2, dim=1) / (2 * SIGMA ** 2)
        neg_dist.append(i_dist.min())
    neg_dist = torch.stack(neg_dist)
    loss = -torch.log(torch.exp(-pos_dist) / (torch.exp(-pos_dist) + DELTA * torch.exp(-neg_dist)))
    return loss.mean()

# =================================================================
# III. EA特征对齐（保持不变）
# =================================================================
def compute_EA_matrix(X):
    n_samples, n_channels = X.shape[0], X.shape[1]
    cov = np.zeros((n_channels, n_channels))
    for i in range(n_samples):
        trial_cov = np.dot(X[i], X[i].T) / X.shape[2]
        cov += trial_cov
    mean_cov = cov / n_samples
    R = inv(sqrtm(mean_cov)).real
    return R

def apply_EA_to_dataset(X, subj_indices):
    X_aligned = np.zeros_like(X)
    unique_subjs = np.unique(subj_indices)
    print(f"--- 执行EA对齐，共{len(unique_subjs)}个被试，通道数{X.shape[1]} ---")
    for subj in unique_subjs:
        mask = (subj_indices == subj)
        R = compute_EA_matrix(X[mask])
        X_aligned[mask] = np.matmul(R, X[mask])
    return X_aligned

# =================================================================
# IV. 核心訓練流程（集成维度匹配+混合监督损失）
# =================================================================
def train_process(model, train_loader, val_loader, criterion, optimizer, save_path,
                  is_prescreen=False, rest_data=None, mi_data=None, is_classifier=False):
    best_val_loss = float('inf')
    patience = 30
    counter = 0
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    print(f"\n📌 开始训练: {save_path} | 设备: {DEVICE} | 批次大小: {BATCH_SIZE}")

    # 第一階段：監督訓練（原有逻辑）
    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
            inputs = inputs.unsqueeze(1).float().to(DEVICE)
            if not is_prescreen:
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

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                labels = labels.long().to(DEVICE)
                outputs = model(inputs)
                v_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        epoch_t_acc = 100 * t_correct / t_total
        epoch_v_acc = 100 * v_correct / v_total
        epoch_v_loss = v_loss / len(val_loader)
        scheduler.step(epoch_v_loss)

        history['loss'].append(t_loss / len(train_loader))
        history['acc'].append(epoch_t_acc)
        history['val_loss'].append(epoch_v_loss)
        history['val_acc'].append(epoch_v_acc)

        if epoch_v_loss < best_val_loss:
            best_val_loss = epoch_v_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | 训练精度: {epoch_t_acc:5.1f}% | 验证精度: {epoch_v_acc:5.1f}% | 验证损失: {epoch_v_loss:.4f}")
        if counter >= patience:
            print(f"🛑 早停触发！在第{epoch+1}轮停止监督训练")
            break

    # 第二階段：SSL微調（核心修复：维度匹配+混合监督损失）
    if is_prescreen and rest_data is not None and mi_data is not None:
        print("\n=== 开始预筛选模块 SSL 微调（Rest/MI）===")
        n_total_train = len(train_loader.dataset)
        neg_samples = create_negative_samples(rest_data, mi_data, n_total_train)
        print(f"📊 负样本生成完成：{len(neg_samples)}个（与训练集一致）")

        # 提前初始化模型特征提取
        model.eval()
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                _ = model.extract_feature(inputs)
                break

        # 冻结卷积层，仅训练全连接层
        params_to_train = []
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
                params_to_train.append(param)
                print(f"🔓 解锁SSL训练：{name} ({param.shape})")
            else:
                param.requires_grad = False
                print(f"🔒 冻结SSL训练：{name} ({param.shape})")

        ema_extractor = EMAExtractor(model)
        ssl_optimizer = optim.Adam(params_to_train, lr=SSL_PRESCREEN_LR, weight_decay=WEIGHT_DECAY)

        # SSL微调训练（核心修复：维度匹配+混合监督损失）
        for ssl_epoch in range(SSL_EPOCHS):
            model.train()
            ssl_total_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):  # 关键：同时获取labels
                # 核心修复3：动态获取当前批次实际大小，避免维度不匹配
                batch_size = inputs.shape[0]
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                labels = labels.long().to(DEVICE)  # 标签也移到设备

                # 核心修复4：按实际批次大小截取负样本，保证维度一致
                start_idx = batch_idx * BATCH_SIZE
                end_idx = start_idx + batch_size
                neg_batch = neg_samples[start_idx:end_idx]
                # 兜底：如果负样本不够，重复填充（避免维度错误）
                if len(neg_batch) < batch_size:
                    neg_batch = np.pad(neg_batch, ((0, batch_size - len(neg_batch)), (0,0), (0,0)), mode='wrap')
                neg_inputs = torch.from_numpy(neg_batch).unsqueeze(1).float().to(DEVICE)

                # 强制启用梯度
                with torch.enable_grad():
                    f_theta_pos = model.extract_feature(inputs)
                    f_phi_pos = ema_extractor.model.extract_feature(inputs)
                    f_phi_neg = ema_extractor.model.extract_feature(neg_inputs)

                # 核心修复5：混合SSL损失+监督损失（9:1），保留Rest/MI分类能力
                ssl_loss = contrastive_loss(f_theta_pos, f_phi_pos, f_phi_neg)
                supervise_output = model(inputs)
                supervise_loss = criterion(supervise_output, labels)
                # 90% SSL损失 + 10% 监督损失，避免遗忘REST
                total_loss = 0.9 * ssl_loss + 0.1 * supervise_loss

                ssl_optimizer.zero_grad()
                total_loss.backward()
                ssl_optimizer.step()
                ema_extractor.update()

                ssl_total_loss += total_loss.item()

            # 打印SSL日志
            avg_loss = ssl_total_loss / len(train_loader)
            if (ssl_epoch + 1) % 5 == 0:  # 每5轮打印一次
                print(f"SSL Epoch {ssl_epoch+1}/{SSL_EPOCHS} | 混合损失: {avg_loss:.4f}")

        # 保存SSL模型
        ssl_save_path = save_path.replace('.pth', '_ssl.pth')
        torch.save(model.state_dict(), ssl_save_path)
        print(f"✅ 预筛选模块SSL完成 | 模型保存至: {ssl_save_path}")

    elif is_classifier:
        print("\n=== 开始分类模块 SSL 微调（Left/Right）===")
        params_to_train = []
        for name, param in model.named_parameters():
            if "fc" in name or "classifier" in name:
                param.requires_grad = True
                params_to_train.append(param)
                print(f"🔓 解锁SSL训练：{name} ({param.shape})")
            else:
                param.requires_grad = False
                print(f"🔒 冻结SSL训练：{name} ({param.shape})")

        ema_extractor = EMAExtractor(model)
        ssl_optimizer = optim.Adam(params_to_train, lr=SSL_CLASSIFIER_LR, weight_decay=WEIGHT_DECAY)

        for ssl_epoch in range(SSL_EPOCHS):
            model.train()
            ssl_total_loss = 0.0
            for inputs, _ in train_loader:
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                x1, x2 = classifier_augment(inputs)

                with torch.enable_grad():
                    f_theta = model.extract_feature(x1)
                    f_phi = ema_extractor.model.extract_feature(x2)

                ssl_loss = classifier_contrastive_loss(f_theta, f_phi)
                ssl_optimizer.zero_grad()
                ssl_loss.backward()
                ssl_optimizer.step()
                ema_extractor.update()

                ssl_total_loss += ssl_loss.item()

            avg_loss = ssl_total_loss / len(train_loader)
            if (ssl_epoch + 1) % 5 == 0:
                print(f"SSL Epoch {ssl_epoch+1}/{SSL_EPOCHS} | 对比损失: {avg_loss:.4f}")

        ssl_save_path = save_path.replace('.pth', '_ssl.pth')
        torch.save(model.state_dict(), ssl_save_path)
        print(f"✅ 分类模块SSL完成 | 模型保存至: {ssl_save_path}")

    # 绘制训练曲线
    plot_training_history(history, title=os.path.basename(save_path), save_path=f"{save_path}.png")
    print(f"\n📦 训练流程结束 | 最佳监督模型: {save_path} | SSL模型: {ssl_save_path if (is_prescreen or is_classifier) else '无'}\n")

# =================================================================
# V. 主函數（保持不变）
# =================================================================
def main():
    # Stage 1: 预筛选模块训练
    print(">>> [Stage 1] 训练预筛选模型（ShallowConvNet: Rest vs MI）")
    loader_rest = EEG_loader_resting(test_subj=TEST_SUBJ)
    n_channels = int(loader_rest.train_x.shape[1])
    n_timepoints = int(loader_rest.train_x.shape[2])
    print(f"📥 预筛选模块输入维度：{n_channels}通道 × {n_timepoints}时间点")

    train_x_ea = apply_EA_to_dataset(loader_rest.train_x, loader_rest.train_subj)
    rest_data = loader_rest.train_x[loader_rest.train_y == 0]
    mi_data = loader_rest.train_x[loader_rest.train_y == 1]
    print(f"📊 预筛选训练数据：Rest{len(rest_data)}个 | MI{len(mi_data)}个")

    full_ds_1 = TensorDataset(torch.from_numpy(train_x_ea), torch.from_numpy(loader_rest.train_y))
    train_size = int(0.8 * len(full_ds_1))
    val_size = len(full_ds_1) - train_size
    ds_train_1, ds_val_1 = random_split(full_ds_1, [train_size, val_size])
    train_loader_1 = DataLoader(ds_train_1, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_1 = DataLoader(ds_val_1, batch_size=BATCH_SIZE)

    model_1 = ShallowConvNet(
        num_classes=2,
        channels=n_channels,
        time_points=n_timepoints,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_process(
        model=model_1,
        train_loader=train_loader_1,
        val_loader=val_loader_1,
        criterion=criterion,
        optimizer=optimizer_1,
        save_path='prescreen_model.pth',
        is_prescreen=True,
        rest_data=rest_data,
        mi_data=mi_data
    )

    # Stage 2: 分类模块训练
    print(">>> [Stage 2] 训练分类模型（EEGNet: Left vs Right）")
    loader_cls = EEG_loader(test_subj=TEST_SUBJ)
    print(f"原始分类标签：{np.unique(loader_cls.train_y)}")
    loader_cls.train_y = np.where(loader_cls.train_y == 1, 0, 1)
    valid_mask = (loader_cls.train_y == 0) | (loader_cls.train_y == 1)
    loader_cls.train_x = loader_cls.train_x[valid_mask]
    loader_cls.train_y = loader_cls.train_y[valid_mask]
    print(f"归一化后标签：{np.unique(loader_cls.train_y)} | 有效样本：{len(loader_cls.train_x)}")
    print(f"类别分布：Left(0){sum(loader_cls.train_y==0)}个 | Right(1){sum(loader_cls.train_y==1)}个")

    n_channels_cls = int(loader_cls.train_x.shape[1])
    n_timepoints_cls = int(loader_cls.train_x.shape[2])
    print(f"📥 分类模块输入维度：{n_channels_cls}通道 × {n_timepoints_cls}时间点")

    if hasattr(loader_cls, 'train_subj'):
        train_x_ea_2 = apply_EA_to_dataset(loader_cls.train_x, loader_cls.train_subj)
    else:
        print("⚠️ 警告：分类模块无被试索引，使用全局EA对齐")
        R_global = compute_EA_matrix(loader_cls.train_x)
        train_x_ea_2 = np.matmul(R_global, loader_cls.train_x)

    full_ds_2 = TensorDataset(
        torch.from_numpy(train_x_ea_2),
        torch.from_numpy(loader_cls.train_y).long()
    )
    train_size_2 = int(0.8 * len(full_ds_2))
    val_size_2 = len(full_ds_2) - train_size_2
    ds_train_2, ds_val_2 = random_split(full_ds_2, [train_size_2, val_size_2])
    train_loader_2 = DataLoader(ds_train_2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_2 = DataLoader(ds_val_2, batch_size=BATCH_SIZE)

    model_2 = EEGNet(
        num_classes=2,
        channels=n_channels_cls,
        time_points=n_timepoints_cls,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model_2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_process(
        model=model_2,
        train_loader=train_loader_2,
        val_loader=val_loader_2,
        criterion=criterion_cls,
        optimizer=optimizer_2,
        save_path='classifier_model.pth',
        is_classifier=True
    )

    print(">>> 🔥 所有训练完成！生成模型：prescreen_model_ssl.pth | classifier_model_ssl.pth")

if __name__ == '__main__':
    main()