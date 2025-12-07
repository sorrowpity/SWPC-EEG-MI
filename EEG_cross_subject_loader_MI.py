# EEG_cross_subject_loader_MI.py (已修订 - 启用数据切割)
import numpy as np
import os
from scipy.io import loadmat
from typing import Tuple, List


class EEG_loader:
    def __init__(self, test_subj: int, data_dir: str = './data/'):
        self.test_subj = test_subj
        self.data_dir = data_dir
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()

    def convert_labels(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将字符串标签转换为四分类标签 (0-3) 并返回有效数据索引"""
        # 4-class MI classification labels
        label_map = {
            'left_hand': 0,
            'right_hand': 1,
            'feet': 2,
            'tongue': 3
        }
        labels = np.squeeze(labels)

        valid_indices: List[int] = []
        mi_labels: List[int] = []

        for i, lbl in enumerate(labels):
            lbl_clean = str(lbl).strip().lower()
            if lbl_clean in label_map:
                valid_indices.append(i)
                mi_labels.append(label_map[lbl_clean])

        return np.array(valid_indices), np.array(mi_labels, dtype=np.int64)

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
            eeg_data = mat_data['X']
            labels = mat_data['y']

            # 获取有效 MI 试次索引和四分类标签
            valid_idx, processed_labels = self.convert_labels(labels)

            # 根据索引筛选EEG数据和标签
            filtered_eeg_data = eeg_data[valid_idx]

            # === 数据切割：仅MI segment ===
            # 提取 MI segment (时间点 750:1000，长度 250)
            X_mi_segment = filtered_eeg_data[:, :, 750:1000]

            all_train_data.append(X_mi_segment)
            all_train_labels.append(processed_labels)

        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0)

        # 2. 加载测试数据 (Test Data Slicing - 相同逻辑)
        test_file = os.path.join(self.data_dir, f'test_A{self.test_subj}.mat')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在，请检查路径: {test_file}")

        mat_test = loadmat(test_file)
        test_x_raw = mat_test['X']
        test_y_raw = mat_test['y']

        # 获取有效 MI 试次索引和四分类标签
        test_valid_idx, test_y = self.convert_labels(test_y_raw)
        test_x_filtered = test_x_raw[test_valid_idx]

        # 提取 MI segment (时间点 750:1000，长度 250)
        test_x = test_x_filtered[:, :, 750:1000]

        return train_x, train_y, test_x, test_y