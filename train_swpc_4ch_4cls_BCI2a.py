import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import torch.nn.functional as F
import copy
from scipy.linalg import inv, sqrtm

# 導入你的模型和加載器
from model import EEGNet
from ShallowConvNet_Lyapunov import ShallowConvNet
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from EEG_cross_subject_loader_MI import EEG_loader
from visualizer import plot_training_history

# =================================================================
# I. 配置參數 (新增自监督轻量配置)
# =================================================================
BATCH_SIZE = 16
SUPERVISED_EPOCHS = 500
# 轻量自监督配置（核心：小学习率+少轮数）
SSL_PRESCREEN_EPOCHS = 8  # 论文10轮，此处更保守
SSL_CLASSIFIER_EPOCHS = 5  # 论文10轮，此处更保守
SSL_PRESCREEN_LR = 5e-6  # 比监督学习小100倍
SSL_CLASSIFIER_LR = 5e-6  # 比监督学习小100倍
# 监督学习原有配置（不变）
LR = 0.0005
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SUBJ = 1
# 论文自监督超参数（严格遵循）
DELTA = 0.3
SIGMA = 2.0
EMA_DECAY = 0.9995


# =================================================================
# 新增：自监督核心组件（完全复刻论文）
# =================================================================
class EMAEncoder:
    """论文EMA辅助编码器"""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.encoder = copy.deepcopy(model).to(DEVICE)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def update(self, online_model):
        """EMA更新：φ = λφ + (1-λ)θ"""
        online_params = online_model.state_dict()
        ema_params = self.encoder.state_dict()
        for k in ema_params.keys():
            ema_params[k] = self.decay * ema_params[k] + (1 - self.decay) * online_params[k]
        self.encoder.load_state_dict(ema_params)


def create_negative_samples(rest_data, mi_data, n_samples):
    """论文式负样本构建：0.5*rest + 0.5*MI"""
    neg_samples = []
    for _ in range(n_samples):
        rest = rest_data[np.random.choice(len(rest_data))]
        mi = mi_data[np.random.choice(len(mi_data))]
        neg = 0.5 * rest + 0.5 * mi
        neg_samples.append(neg)
    return np.array(neg_samples, dtype=np.float32)


def prescreen_ssl_loss(f_theta_pos, f_phi_pos, f_phi_neg):
    """论文预筛选模块对比损失"""
    # L2归一化（论文要求）
    f_theta_pos = F.normalize(f_theta_pos, dim=1)
    f_phi_pos = F.normalize(f_phi_pos, dim=1)
    f_phi_neg = F.normalize(f_phi_neg, dim=1)

    # 论文损失公式
    pos_sim = torch.exp(-torch.sum((f_theta_pos - f_phi_pos) ** 2, 1) / (2 * SIGMA ** 2))
    neg_sim = torch.exp(-torch.sum((f_theta_pos - f_phi_neg) ** 2, 1) / (2 * SIGMA ** 2))
    loss = -torch.log(pos_sim / (pos_sim + DELTA * neg_sim))
    return loss.mean()


def classifier_ssl_loss(f_theta, f_phi):
    """论文分类模块对比损失（数据增强版）"""
    f_theta = F.normalize(f_theta, dim=1)
    f_phi = F.normalize(f_phi, dim=1)
    B = f_theta.shape[0]

    # 正样本相似度
    pos_sim = torch.exp(-torch.sum((f_theta - f_phi) ** 2, 1) / (2 * SIGMA ** 2))
    # 负样本相似度（批次内其他样本）
    neg_sim = []
    for i in range(B):
        other_samples = f_phi[torch.arange(B) != i]
        min_dist = torch.min(torch.sum((f_theta[i:i + 1] - other_samples) ** 2, 1))
        neg_sim.append(torch.exp(-min_dist / (2 * SIGMA ** 2)))
    neg_sim = torch.stack(neg_sim)

    loss = -torch.log(pos_sim / (pos_sim + DELTA * neg_sim))
    return loss.mean()


def ssl_augment_classifier(x):
    """论文分类模块数据增强（温和版，不破坏特征）"""
    b, _, ch, t = x.shape
    # 增强1：小幅噪声
    x1 = x.clone() + torch.randn_like(x) * 0.003
    # 增强2：小幅缩放
    x1 *= torch.rand(1, device=x.device) * 0.1 + 0.95

    # 增强3：时间平移（±5点，比论文更温和）
    x2 = x.clone()
    shift = np.random.randint(-5, 6, b)
    for i in range(b):
        x2[i] = torch.roll(x2[i], shift[i], dims=-1)

    return x1, x2


# =================================================================
# 修复版：自监督预训练函数（无 return_feature，兼容所有模型）
# =================================================================
def ssl_pretrain(model, loader, ema_encoder, criterion, optimizer, epochs, is_prescreen=False, neg_data=None):
    """轻量自监督预训练：仅更新特征提取器，不触碰分类头"""
    model.train()
    # 冻结分类头，只训特征提取器（核心：不破坏后续监督训练）
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'linear' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for e in range(epochs):
        total_loss = 0
        for i, data in enumerate(loader):
            if is_prescreen:
                # 预筛选模块：正样本+负样本
                x = data[0].to(DEVICE).float().unsqueeze(1)
                B = x.shape[0]
                # 取对应批次负样本
                st = i * BATCH_SIZE
                ed = st + B
                neg_batch = neg_data[st:ed]
                if len(neg_batch) < B:
                    neg_batch = np.concatenate([neg_batch, neg_data[:B - len(neg_batch)]], axis=0)
                neg_x = torch.from_numpy(neg_batch).unsqueeze(1).float().to(DEVICE)

                # --------------------------
                # 🔥 修复：去掉 return_feature
                # --------------------------
                f_theta = model(x)
                if isinstance(f_theta, tuple):
                    f_theta = f_theta[0]

                f_phi_pos = ema_encoder.encoder(x)
                if isinstance(f_phi_pos, tuple):
                    f_phi_pos = f_phi_pos[0]

                f_phi_neg = ema_encoder.encoder(neg_x)
                if isinstance(f_phi_neg, tuple):
                    f_phi_neg = f_phi_neg[0]

                loss = criterion(f_theta, f_phi_pos, f_phi_neg)
            else:
                # 分类模块：数据增强双样本
                x = data[0].to(DEVICE).float().unsqueeze(1)
                x1, x2 = ssl_augment_classifier(x)

                # --------------------------
                # 🔥 修复：去掉 return_feature
                # --------------------------
                f_theta = model(x1)
                if isinstance(f_theta, tuple):
                    f_theta = f_theta[0]

                f_phi = ema_encoder.encoder(x2)
                if isinstance(f_phi, tuple):
                    f_phi = f_phi[0]

                loss = criterion(f_theta, f_phi)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()
            ema_encoder.update(model)
            total_loss += loss.item()

        if (e + 1) % 2 == 0:
            print(f"SSL Pretrain Epoch {e+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    # 解冻所有参数，为后续监督训练做准备
    for param in model.parameters():
        param.requires_grad = True
    return model


# =================================================================
# 原有功能保留（标签归一化+Lyapunov+数据增强+EA对齐）
# =================================================================
import nolds


def jacobian_regularization(net, x, lambda_J=0.01):
    """
    深度学习雅可比正则化（域适应专用）
    计算：||d(f(x))/dx||²
    """
    x.requires_grad_(True)
    output = net(x)

    # 🔥 关键修复：模型输出是 tuple，只拿分类结果计算雅可比
    if isinstance(output, tuple):
        output = output[0]  # 只取分类分数

    # 计算雅可比范数
    grad = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]

    jac_loss = grad.pow(2).mean()
    return lambda_J * jac_loss

def compute_lyapunov_labels(data_np):
    print(f"--- 正在計算 {len(data_np)} 個樣本的 Lyapunov 複雜度標籤... ---")
    lyaps = []
    for i in range(len(data_np)):
        signal = np.mean(data_np[i], axis=0)
        try:
            l = nolds.lyap_r(signal, emb_dim=5, lag=None)
        except:
            l = 0.0
        lyaps.append(l)
    lyaps = np.array(lyaps).astype(np.float32)
    lyaps = (lyaps - lyaps.min()) / (lyaps.max() - lyaps.min() + 1e-6)
    return lyaps


def normalize_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    normalized_labels = np.array([label_map[old] for old in labels])
    print(f"标签映射: {label_map} (原始标签: {unique_labels}, 归一化后: {np.unique(normalized_labels)})")
    return normalized_labels


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


def compute_EA_matrix(X):
    n_samples = X.shape[0]
    n_channels = X.shape[1]
    cov = np.zeros((n_channels, n_channels))
    for i in range(n_samples):
        trial_cov = np.dot(X[i], X[i].T) / X.shape[2]
        cov += trial_cov
    mean_cov = cov / n_samples
    try:
        R = inv(sqrtm(mean_cov)).real
    except:
        R = np.eye(n_channels)
    return R


def apply_EA_to_dataset(X, subj_indices):
    X_aligned = np.zeros_like(X)
    unique_subjs = np.unique(subj_indices)
    print(f"--- 執行 EA 對齊，總計 {len(unique_subjs)} 個被試 ---")
    for subj in unique_subjs:
        mask = (subj_indices == subj)
        R = compute_EA_matrix(X[mask])
        X_aligned[mask] = np.matmul(R, X[mask])
    return X_aligned


# =================================================================
# 原有训练流程保留（兼容自监督预训练后的数据）
# =================================================================
def train_process(model, train_loader, val_loader, criterion, optimizer, save_path, is_stage1=False):
    best_val_loss = float('inf')
    patience = 30
    counter = 0
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    mse_criterion = nn.MSELoss()
    alpha = 0.5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(SUPERVISED_EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0

        for batch in train_loader:
            if is_stage1:
                inputs, labels, lyap_labels = batch
                lyap_labels = lyap_labels.to(DEVICE)
            else:
                inputs, labels = batch

            inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
            inputs = inputs.unsqueeze(1).float().to(DEVICE)
            inputs = augment_batch(inputs, shift_limit=30, noise_level=0.02)
            labels = labels.long().to(DEVICE)

            optimizer.zero_grad()

            if is_stage1:
                outputs, pred_lyap = model(inputs)
                loss_cls = criterion(outputs, labels)
                loss_dyn = mse_criterion(pred_lyap.squeeze(), lyap_labels)
                loss = loss_cls + alpha * loss_dyn + jacobian_regularization(model, inputs)
            else:
                outputs = model(inputs)
                # 🔥 修复这里！
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
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
            for batch in val_loader:
                if is_stage1:
                    inputs, labels, lyap_labels = batch
                    lyap_labels = lyap_labels.to(DEVICE)
                else:
                    inputs, labels = batch

                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
                inputs = inputs.unsqueeze(1).float().to(DEVICE)
                labels = labels.long().to(DEVICE)

                if is_stage1:
                    outputs, pred_lyap = model(inputs)
                    loss_cls = criterion(outputs, labels)
                    loss_dyn = mse_criterion(pred_lyap.squeeze(), lyap_labels)
                    loss = loss_cls + alpha * loss_dyn
                else:
                    outputs = model(inputs)
                    # 🔥 修复：只取第一个输出（分类结果）
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)

                v_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        epoch_t_acc = 100 * t_correct / t_total
        epoch_v_acc = 100 * v_correct / v_total
        epoch_v_loss = v_loss / len(val_loader)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_v_loss)
        new_lr = optimizer.param_groups[0]['lr']

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
            print(
                f"Epoch {epoch + 1:03d} | Train: {epoch_t_acc:5.1f}% | Val: {epoch_v_acc:5.1f}% | ValLoss: {epoch_v_loss:.4f}")

        if counter >= patience:
            print(f"早停觸發！在第 {epoch + 1} 輪停止訓練。")
            break

        if old_lr != new_lr:
            print(f"📉 學習率從 {old_lr:.6f} 下降至 {new_lr:.6f}")

    plot_training_history(history, title=save_path, save_path=f"{save_path}.png")
    print(f"✅ 模型已保存至: {save_path}\n")


# =================================================================
# IV. 主函數（新增自监督预训练流程）
# =================================================================
def main():
    # 1. 訓練階段 1：預篩選 (Rest vs MI)
    print("\n>>> Stage 1: Prescreening (Rest vs MI)")
    print("--- Step 1/2: 监督训练（先学正确答案） ---")  # 🔥 改这里
    loader_rest = EEG_loader_resting(test_subj=TEST_SUBJ)
    train_x_ea = apply_EA_to_dataset(loader_rest.train_x, loader_rest.train_subj)
    train_y_norm_1 = normalize_labels(loader_rest.train_y)

    # 先做监督训练
    train_lyap = compute_lyapunov_labels(train_x_ea)
    full_ds_1 = TensorDataset(
        torch.from_numpy(train_x_ea),
        torch.from_numpy(train_y_norm_1),
        torch.from_numpy(train_lyap)
    )
    train_size = int(0.8 * len(full_ds_1))
    val_size = len(full_ds_1) - train_size
    ds_train_1, ds_val_1 = random_split(full_ds_1, [train_size, val_size])

    train_loader_1 = DataLoader(ds_train_1, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_1 = DataLoader(ds_val_1, batch_size=BATCH_SIZE)

    model_1 = ShallowConvNet(2, loader_rest.train_x.shape[1], loader_rest.train_x.shape[2], DROPOUT_RATE).to(DEVICE)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    # 先监督训练
    train_process(model_1, train_loader_1, val_loader_1, criterion, optimizer_1, 'prescreen_model_best.pth',
                  is_stage1=True)

    print("--- Step 2/2: 自监督微调（强化动作特征） ---")  # 🔥 后做自监督
    # 构建自监督数据集
    ssl_ds_1 = TensorDataset(torch.from_numpy(train_x_ea))
    ssl_loader_1 = DataLoader(ssl_ds_1, BATCH_SIZE, shuffle=True)

    # 加载刚才监督训练好的模型
    model_1 = ShallowConvNet(2, loader_rest.train_x.shape[1], loader_rest.train_x.shape[2], DROPOUT_RATE).to(DEVICE)
    model_1.load_state_dict(torch.load('prescreen_model_best.pth'))  # 🔥 加载监督模型

    ema_1 = EMAEncoder(model_1)
    ssl_optimizer_1 = optim.Adam(model_1.parameters(), lr=SSL_PRESCREEN_LR, weight_decay=WEIGHT_DECAY)

    rest_data = train_x_ea[train_y_norm_1 == 0]
    mi_data = train_x_ea[train_y_norm_1 == 1]
    neg_samples_1 = create_negative_samples(rest_data, mi_data, len(train_x_ea))

    # 自监督精调
    model_1 = ssl_pretrain(
        model=model_1,
        loader=ssl_loader_1,
        ema_encoder=ema_1,
        criterion=prescreen_ssl_loss,
        optimizer=ssl_optimizer_1,
        epochs=SSL_PRESCREEN_EPOCHS,
        is_prescreen=True,
        neg_data=neg_samples_1
    )

    # 保存最终模型（监督 + 自监督）
    torch.save(model_1.state_dict(), 'prescreen_model_best.pth')
    print("✅ Stage1 最终模型保存完成：监督训练 → 自监督精调\n")

    # ==============================
    # 🔥 最强 Stage2：继承 Stage1 特征 + 迁移学习
    # ==============================
    print("\n>>> Stage 2: Classifier (Left vs Right)")
    print("--- Step 1/1: 加载 Stage1 模型 → 微调左右手分类 ---")

    # 1. 读取真实左右手数据（必须用 EEG_loader）
    loader_cls = EEG_loader(test_subj=TEST_SUBJ)

    # EA 对齐
    if hasattr(loader_cls, 'train_subj'):
        train_x_ea_2 = apply_EA_to_dataset(loader_cls.train_x, loader_cls.train_subj)
    else:
        R_global = compute_EA_matrix(loader_cls.train_x)
        train_x_ea_2 = np.matmul(R_global, loader_cls.train_x)

    # 左右手标签
    train_y_2 = loader_cls.train_y
    train_y_norm_2 = normalize_labels(train_y_2)

    # 过滤
    mask = (train_y_norm_2 == 0) | (train_y_norm_2 == 1)
    train_x_final = train_x_ea_2[mask]
    train_y_final = train_y_norm_2[mask]

    # 构建数据集
    full_ds_2 = TensorDataset(torch.from_numpy(train_x_final), torch.from_numpy(train_y_final))
    train_size_2 = int(0.8 * len(full_ds_2))
    val_size_2 = len(full_ds_2) - train_size_2
    ds_train_2, ds_val_2 = random_split(full_ds_2, [train_size_2, val_size_2])

    train_loader_2 = DataLoader(ds_train_2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_2 = DataLoader(ds_val_2, batch_size=BATCH_SIZE)

    # 2. ✅ 关键：加载 Stage1 训练好的模型！！！
    model_2 = ShallowConvNet(2, train_x_final.shape[1], train_x_final.shape[2], DROPOUT_RATE).to(DEVICE)
    model_2.load_state_dict(torch.load('prescreen_model_best_sl_ssl_cnet.pth'))  # 🔥 继承 Stage1

    # 3. 训练
    optimizer_2 = optim.Adam(model_2.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)  # 小学习率微调
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    train_process(model_2, train_loader_2, val_loader_2, criterion, optimizer_2,
                  'classifier_model_best_sl_ssl_cnet.pth')

if __name__ == '__main__':
    main()