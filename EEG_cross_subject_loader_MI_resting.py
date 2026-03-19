# EEG_cross_subject_loader_MI_resting.py
import numpy as np
import os
from scipy.io import loadmat
from typing import Tuple, List
from scipy.signal import welch

MI_ACTION_MAP = {'left_hand': 1, 'right_hand': 2, 'stop': 4}


class EEG_loader_resting:
    def __init__(self, test_subj: int, data_dir: str = './data/'):
        self.test_subj_id = test_subj  # 區分測試被試ID
        self.data_dir = data_dir
        # 增加存儲被試標籤的屬性
        self.train_x, self.train_y, self.train_subj, self.test_x, self.test_y = self.load_data()

    def _filter_by_erd(self, X_rest, X_mi, threshold=1.0):
        keep_indices = []
        motor_channels = [6, 10, 14]
        for i in range(len(X_rest)):
            ratios = []
            for ch in motor_channels:
                f_r, p_r = welch(X_rest[i, ch], fs=250, nperseg=128)
                f_m, p_m = welch(X_mi[i, ch], fs=250)
                band = np.where((f_r >= 8) & (f_r <= 30))[0]
                res = np.mean(p_m[band]) / np.mean(p_r[band])
                ratios.append(res)
            if np.mean(ratios) < threshold:
                keep_indices.append(i)
        return keep_indices

    def load_data(self):
        all_train_data = []
        all_train_labels = []
        all_train_subjs = []  # 新增：記錄訓練集被試ID

        for s in range(1, 10):  # 假設你有9個被試 A1-A9
            if s == self.test_subj_id: continue

            train_file = os.path.join(self.data_dir, f'train_A{s}.mat')
            if not os.path.exists(train_file): continue

            mat_data = loadmat(train_file)
            eeg_data = mat_data['X']
            labels = np.squeeze(mat_data['y'])

            valid_indices = [i for i, lbl in enumerate(labels) if str(lbl).strip().lower() in MI_ACTION_MAP]
            eeg_mi = eeg_data[valid_indices]

            X_rest = eeg_mi[:, :, 0:250]
            X_mi = eeg_mi[:, :, 850:1100]

            good_idx = self._filter_by_erd(X_rest, X_mi, threshold=1.0)
            if not good_idx: continue

            X_rest_clean = X_rest[good_idx]
            X_mi_clean = X_mi[good_idx]

            X_subj = np.concatenate([X_rest_clean, X_mi_clean], axis=0)
            y_subj = np.concatenate([np.zeros(len(good_idx)), np.ones(len(good_idx))], axis=0)

            # --- 關鍵修改：記錄被試標籤 ---
            # 每個樣本都標上它是來自被試 s
            s_labels = np.full(len(X_subj), s)

            all_train_data.append(X_subj)
            all_train_labels.append(y_subj)
            all_train_subjs.append(s_labels)

        train_x = np.concatenate(all_train_data, axis=0)
        train_y = np.concatenate(all_train_labels, axis=0).astype(np.int64)
        train_subj = np.concatenate(all_train_subjs, axis=0)  # 輸出為 (N_samples,)

        # --- 測試集處理 ---
        test_file = os.path.join(self.data_dir, f'test_A{self.test_subj_id}.mat')
        mat_test = loadmat(test_file)
        test_x_raw = mat_test['X']
        test_y_raw = np.squeeze(mat_test['y'])

        t_idx = [i for i, lbl in enumerate(test_y_raw) if str(lbl).strip().lower() in MI_ACTION_MAP]
        test_x_mi_raw = test_x_raw[t_idx]

        T_rest = test_x_mi_raw[:, :, 0:250]
        T_mi = test_x_mi_raw[:, :, 850:1100]

        t_good_idx = self._filter_by_erd(T_rest, T_mi, threshold=1.0)

        test_x = np.concatenate([T_rest[t_good_idx], T_mi[t_good_idx]], axis=0)
        test_y = np.concatenate([np.zeros(len(t_good_idx)), np.ones(len(t_good_idx))], axis=0).astype(np.int64)

        print(f"篩選完成！訓練樣本: {len(train_x)}, 測試樣本: {len(test_x)}")
        # 返回 5 個值
        return train_x, train_y, train_subj, test_x, test_y