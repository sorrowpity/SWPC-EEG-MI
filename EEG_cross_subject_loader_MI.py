# EEG_cross_subject_loader_MI.py (已修訂 - 啟用 EA 對齊支持)
import numpy as np
import os
from scipy.io import loadmat
from typing import Tuple, List


class EEG_loader:
    def __init__(self, test_subj: int, data_dir: str = './data/'):
        self.test_subj = test_subj
        self.data_dir = data_dir
        # 修正：接收 5 個返回值
        self.train_x, self.train_y, self.train_subj, self.test_x, self.test_y = self.load_data()

    def convert_labels(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        label_map = {'left_hand': 1, 'right_hand': 2, 'stop': 4}
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
        all_train_subjs = []  # 關鍵新增：記錄被試ID

        # 1. 加載訓練數據 (Cross-Subject)
        for s in range(1, 6):  # 確保範圍涵蓋你所有的 A1-A5/A9 文件
            if s == self.test_subj:
                continue

            train_file = os.path.join(self.data_dir, f'train_A{s}.mat')
            if not os.path.exists(train_file):
                continue

            mat_data = loadmat(train_file)
            eeg_data = mat_data['X']
            labels = mat_data['y']

            valid_idx, processed_labels = self.convert_labels(labels)
            filtered_eeg_data = eeg_data[valid_idx]

            # 提取 MI segment
            X_mi_segment = filtered_eeg_data[:, :, 750:1000]

            all_train_data.append(X_mi_segment)
            all_train_labels.append(processed_labels)

            # --- 關鍵修復：為每個樣本打上被試標籤 s ---
            all_train_subjs.append(np.full(len(processed_labels), s))

        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0)
        train_subj = np.concatenate(all_train_subjs, axis=0)  # 輸出被試標籤

        # 2. 加載測試數據
        test_file = os.path.join(self.data_dir, f'test_A{self.test_subj}.mat')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"測試文件不存在: {test_file}")

        mat_test = loadmat(test_file)
        test_x_raw = mat_test['X']
        test_y_raw = mat_test['y']

        test_valid_idx, test_y = self.convert_labels(test_y_raw)
        test_x_filtered = test_x_raw[test_valid_idx]
        test_x = test_x_filtered[:, :, 750:1000]

        # 修正：返回 5 個變量
        return train_x, train_y, train_subj, test_x, test_y