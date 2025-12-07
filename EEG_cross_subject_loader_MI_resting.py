# EEG_cross_subject_loader_MI_resting.py (已修订 - 启用数据切割)
import numpy as np
import os
from scipy.io import loadmat
from typing import Tuple, List

# 假设你的.mat文件包含 string 标签: 'left_hand', 'right_hand', 'feet', 'tongue'
MI_ACTION_MAP = {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}


class EEG_loader_resting:
    def __init__(self, test_subj: int, data_dir: str = './data/'):
        self.test_subj = test_subj
        self.data_dir = data_dir
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()

    def load_data(self):
        all_train_data = []
        all_train_labels = []

        # 1. 加载训练数据 (Cross-Subject: 排除测试被试)
        for s in range(1, 6):  # 假设被试编号 A1 到 A5
            if s == self.test_subj:
                continue

            train_file = os.path.join(self.data_dir, f'train_A{s}.mat')
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"训练文件不存在，请检查路径: {train_file}")

            mat_data = loadmat(train_file)
            eeg_data = mat_data['X']  # 假设形状: (N_trials, Channels, Time_points)
            labels = mat_data['y']

            # 找到所有MI试次的索引
            labels_sq = np.squeeze(labels)
            valid_mi_indices = [i for i, lbl in enumerate(labels_sq) if str(lbl).strip().lower() in MI_ACTION_MAP]

            if not valid_mi_indices:
                continue

            eeg_mi = eeg_data[valid_mi_indices]  # 仅MI试次数据

            # === 数据切割，创建Resting Class ===

            # 提取 Rest segment (时间点 250:500，对应 1s 到 2s，长度 250)
            X_rest_segment = eeg_mi[:, :, 250:500]
            y_rest_labels = np.zeros(X_rest_segment.shape[0], dtype=np.int64)  # 标签 0: Rest

            # 提取 MI segment (时间点 750:1000，对应 3s 到 4s，长度 250)
            X_mi_segment = eeg_mi[:, :, 750:1000]
            y_mi_labels = np.ones(X_mi_segment.shape[0], dtype=np.int64)  # 标签 1: MI

            # 合并 Rest 和 MI segments
            X_train_subj = np.concatenate([X_rest_segment, X_mi_segment], axis=0)
            y_train_subj = np.concatenate([y_rest_labels, y_mi_labels], axis=0)

            all_train_data.append(X_train_subj)
            all_train_labels.append(y_train_subj)

        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0)

        # 2. 加载测试数据 (Test Data Slicing - 相同逻辑)
        test_file = os.path.join(self.data_dir, f'test_A{self.test_subj}.mat')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在，请检查路径: {test_file}")

        mat_test = loadmat(test_file)
        test_x_raw = mat_test['X']
        test_y_raw = mat_test['y']

        # 找到所有MI试次的索引
        labels_sq = np.squeeze(test_y_raw)
        valid_mi_indices = [i for i, lbl in enumerate(labels_sq) if str(lbl).strip().lower() in MI_ACTION_MAP]

        test_x_mi = test_x_raw[valid_mi_indices]

        # 提取测试集的 Rest 和 MI segments
        test_x_rest_segment = test_x_mi[:, :, 250:500]
        test_y_rest_labels = np.zeros(test_x_rest_segment.shape[0], dtype=np.int64)

        test_x_mi_segment = test_x_mi[:, :, 750:1000]
        test_y_mi_labels = np.ones(test_x_mi_segment.shape[0], dtype=np.int64)

        test_x = np.concatenate([test_x_rest_segment, test_x_mi_segment], axis=0)
        test_y = np.concatenate([test_y_rest_labels, test_y_mi_labels], axis=0)

        print(f"Prescreening数据形状: 训练X {train_x.shape}, 训练Y {train_y.shape}")

        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0)

        # 检查平衡性
        count_rest = np.sum(train_y == 0)
        count_mi = np.sum(train_y == 1)
        print(f"训练集标签分布: Rest(0)={count_rest}, MI(1)={count_mi}")

        return train_x, train_y, test_x, test_y