# data_loader.py 完整代码
import numpy as np
import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


class EEG_loader:
    def __init__(self, test_subj, dataset='BNCI2014001'):
        self.test_subj = test_subj  # 测试被试编号（1-5）
        self.dataset = dataset
        self.data_dir = './data/'  # 数据存储目录（需确认实际路径）
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()

    def convert_labels(self, labels):
        """转换标签为数字（0:left_hand, 1:right_hand）"""
        label_map = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }
        labels = np.squeeze(labels)
        processed_labels = []
        for lbl in labels:
            lbl_clean = str(lbl).strip().lower()
            processed_labels.append(label_map[lbl_clean])
        return np.array(processed_labels, dtype=np.int64)

    def process_eeg_data(self, eeg_data, labels):
        """统一处理EEG数据维度为(试次数, 导联数, 时间点)"""
        labels_flat = labels.flatten()
        label_trial_num = len(labels_flat)
        trial_dim = None
        if len(eeg_data.shape) == 3:
            for i in range(3):
                if eeg_data.shape[i] == label_trial_num:
                    trial_dim = i
                    break
        if trial_dim is None:
            raise ValueError(f"EEG数据维度{eeg_data.shape}与标签数{label_trial_num}不匹配！")

        if trial_dim == 0:
            if eeg_data.shape[1] > eeg_data.shape[2]:
                eeg_data = np.transpose(eeg_data, (0, 2, 1))
        elif trial_dim == 1:
            eeg_data = np.transpose(eeg_data, (1, 0, 2))
            if eeg_data.shape[1] > eeg_data.shape[2]:
                eeg_data = np.transpose(eeg_data, (0, 2, 1))
        else:
            eeg_data = np.transpose(eeg_data, (2, 0, 1))
            if eeg_data.shape[1] > eeg_data.shape[2]:
                eeg_data = np.transpose(eeg_data, (0, 2, 1))
        return eeg_data

    def load_data(self):
        """加载所有训练被试（排除测试被试）和测试被试数据"""
        all_train_data = []
        all_train_labels = []
        # 遍历训练被试（1-5，排除test_subj）
        for subj in range(1, 6):
            if subj == self.test_subj:
                continue
            train_file = os.path.join(self.data_dir, f'train_A{subj}.mat')
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"训练文件不存在: {train_file}")
            mat_data = loadmat(train_file)
            eeg_data = mat_data['X']
            labels = mat_data['y']
            # 处理数据维度和标签
            eeg_data = self.process_eeg_data(eeg_data, labels)
            labels = self.convert_labels(labels)
            all_train_data.append(eeg_data)
            all_train_labels.append(labels)
        # 合并训练数据
        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0)
        # 加载测试被试数据
        test_file = os.path.join(self.data_dir, f'test_A{self.test_subj}.mat')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        mat_data = loadmat(test_file)
        test_x = mat_data['X']
        test_y = mat_data['y']
        test_x = self.process_eeg_data(test_x, test_y)
        test_y = self.convert_labels(test_y)
        return train_x, train_y, test_x, test_y


class EEGDataset(Dataset):
    def __init__(self, data_x, data_y, augment=False):
        self.data_x = data_x  # 形状: (N, 导联数, 时间点)
        self.data_y = data_y  # 形状: (N,)
        self.augment = augment

    def __len__(self):
        return len(self.data_x)

    def augment_eeg(self, x):
        """EEG数据增强（对比学习用）"""
        # 1. 时间轴随机平移（±5个采样点）
        shift = np.random.randint(-5, 6)
        x = np.roll(x, shift, axis=-1)
        # 2. 幅值随机缩放（0.9~1.1倍）
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        # 3. 加轻微高斯噪声（标准差为原数据的1%）
        noise = np.random.normal(0, x.std() * 0.01, x.shape)
        x = x + noise
        return x

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        if self.augment:
            # 生成两个增强视图
            x1 = self.augment_eeg(x)
            x2 = self.augment_eeg(x)
            return torch.from_numpy(x1).float(), torch.from_numpy(x2).float(), torch.tensor(y, dtype=torch.long)
        else:
            return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)