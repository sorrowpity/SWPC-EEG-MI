# utils.py
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def CSPfeature_train(X, y, n_components=4):
    """
    X: 输入EEG数据，形状 (n_trials, channels, time_steps)
    y: 标签
    返回CSP权重矩阵W和训练特征
    """
    # 计算两类数据的协方差矩阵
    covs = []
    for cls in np.unique(y):
        cls_data = X[y == cls]
        cov = np.mean([np.cov(trial) for trial in cls_data], axis=0)
        covs.append(cov)
    # 求解广义特征值问题，得到CSP权重
    eig_vals, eig_vecs = np.linalg.eigh(covs[0], covs[0] + covs[1])
    # 选取前n_components和后n_components个特征向量
    W = np.hstack([eig_vecs[:, -n_components:], eig_vecs[:, :n_components]])
    # 提取特征（投影后计算方差）
    features = []
    for trial in X:
        projected = W.T @ trial
        feat = np.log(np.var(projected, axis=1))
        features.append(feat)
    return np.array(features), W

# utils.py
from torch.utils.data import Dataset

class EEG_loader_augment_cross(Dataset):
    def __init__(self, data_x, data_y, augment=True):
        self.x = data_x  # (试次数, 导联数, 时间点)
        self.y = data_y
        self.augment = augment

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            # 生成两个增强样本（如加高斯噪声）
            aug1 = x + np.random.normal(0, 0.01, size=x.shape)
            aug2 = x + np.random.normal(0, 0.01, size=x.shape)
            return aug1, aug2, self.y[idx]
        return x, x, self.y[idx]  # 不增强时返回相同样本


import numpy as np
import torch
from torch.utils.data import Dataset
import random


class EEG_loader_augment_resting_cross(Dataset):
    """
    加载包含静息态数据的EEG数据集，生成带数据增强的样本对（用于自监督学习）
    输入：
        data_x: EEG数据，形状为(试次数, 导联数, 时间点)
        data_y: 标签，形状为(试次数,)（包含运动想象和静息态标签）
        resting_label: 静息态数据的标签（默认3，根据实际数据集调整）
        augment: 是否进行数据增强（默认True）
    输出：
        aug1: 第一个增强样本
        aug2: 第二个增强样本
        label: 原始标签（用于监督学习或过滤）
    """

    def __init__(self, data_x, data_y, resting_label=3, augment=True):
        self.data_x = data_x  # 原始EEG数据，形状：(N, 导联数, 时间点)
        self.data_y = data_y  # 标签，形状：(N,)
        self.resting_label = resting_label  # 静息态数据的标签（根据你的数据集调整）
        self.augment = augment  # 是否启用数据增强
        # 预定义数据增强方法（针对EEG信号的常用增强）
        self.augmentations = [
            self.add_gaussian_noise,  # 加高斯噪声
            self.time_shift,  # 时间平移
            self.scale_amplitude,  # 幅度缩放
            self.time_reverse  # 时间反转（部分场景适用）
        ]

    def __len__(self):
        """返回数据集长度"""
        return len(self.data_x)

    def __getitem__(self, idx):
        """获取索引为idx的样本，并生成两个增强版本"""
        # 获取原始数据和标签
        x = self.data_x[idx]  # 形状：(导联数, 时间点)
        label = self.data_y[idx]

        # 如果不增强，直接返回两个相同的样本
        if not self.augment:
            return x.astype(np.float32), x.astype(np.float32), label

        # 生成两个不同的增强样本
        aug1 = self.apply_random_augmentation(x.copy())
        aug2 = self.apply_random_augmentation(x.copy())

        return aug1.astype(np.float32), aug2.astype(np.float32), label

    def apply_random_augmentation(self, x):
        """随机选择一种数据增强方法并应用"""
        # 随机选择1-2种增强方法（避免过度增强）
        num_aug = random.choice([1, 2])  # 随机选1或2种增强
        chosen_augs = random.sample(self.augmentations, num_aug)

        # 依次应用选中的增强
        for aug in chosen_augs:
            x = aug(x)
        return x

    # -------------------------- 以下是具体的数据增强方法 --------------------------
    def add_gaussian_noise(self, x, noise_std=0.01):
        """添加高斯噪声（模拟EEG信号中的生理噪声）"""
        # 噪声强度根据数据范围调整，这里假设数据已归一化到[-1,1]
        noise = np.random.normal(loc=0, scale=noise_std, size=x.shape)
        return x + noise

    def time_shift(self, x, max_shift=10):
        """时间平移（沿时间轴平移一定步数，边缘补0）"""
        shift = random.randint(-max_shift, max_shift)  # 随机平移步数（正负表示方向）
        if shift == 0:
            return x
        # 沿时间轴（第1维）平移
        return np.roll(x, shift, axis=1)

    def scale_amplitude(self, x, scale_range=(0.8, 1.2)):
        """幅度缩放（乘以一个随机因子，模拟信号强度变化）"""
        scale = random.uniform(scale_range[0], scale_range[1])
        return x * scale

    def time_reverse(self, x):
        """时间反转（将时间序列倒序，适用于无明显时间方向性的信号）"""
        return x[:, ::-1]  # 沿时间轴（第1维）反转