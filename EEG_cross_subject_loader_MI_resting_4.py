import numpy as np
import os
import mne
from typing import Tuple, List
from scipy.signal import welch

# ===================== 4分类标签映射（核心修改）=====================
# BCI IV 2a标签：770=左手,771=右手,772=脚,773=舌头 | 静息态单独映射为0
MI_ACTION_MAP_4CLASS = {
    770: 1,  # 左手
    771: 2,  # 右手
    772: 3,  # 脚
    773: 4,  # 舌头
    'stop': 5  # 静息态
}
SAMPLING_FREQ = 250
# BCI IV 2a时间窗：刺激后0.5s-4.5s（运动想象段），前0.5s为静息段
MI_TMIN, MI_TMAX = 0.5, 4.5
REST_TMIN, REST_TMAX = -0.5, 0.0


class EEG_loader_resting_4class:
    def __init__(self, test_subj: int, data_dir: str = './data/BCI_IV_2a/'):
        self.test_subj_id = test_subj  # 测试被试ID（1-9）
        self.data_dir = data_dir
        # 加载4分类数据（训练集=T.gdf，测试集=E.gdf）
        self.train_x, self.train_y, self.train_subj, self.test_x, self.test_y = self.load_data()

    def _filter_by_erd(self, X_rest, X_mi, threshold=1.0):
        """保留ERD过滤逻辑，适配4分类MI样本"""
        keep_indices = []
        motor_channels = [6, 10, 14]  # 运动相关通道，适配BCI IV 2a
        for i in range(len(X_rest)):
            ratios = []
            for ch in motor_channels:
                f_r, p_r = welch(X_rest[i, ch], fs=SAMPLING_FREQ, nperseg=128)
                f_m, p_m = welch(X_mi[i, ch], fs=SAMPLING_FREQ)
                band = np.where((f_r >= 8) & (f_r <= 30))[0]  # 8-30Hz节律
                res = np.mean(p_m[band]) / np.mean(p_r[band])
                ratios.append(res)
            if np.mean(ratios) < threshold:
                keep_indices.append(i)
        return keep_indices

    def _load_gdf_data(self, file_path):
        """加载GDF文件并提取4分类EEG数据"""
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        # 提取事件标注（BCI IV 2a：769=休息,770=左手,771=右手,772=脚,773=舌头）
        events, event_ids = mne.events_from_annotations(raw, verbose=False)

        # 筛选4分类有效事件（排除休息标签769，后续单独处理）
        valid_mi_events = events[np.isin(events[:, 2], [770, 771, 772, 773])]
        if len(valid_mi_events) == 0:
            return None, None

        # 提取MI段数据（0.5-4.5s）
        epochs_mi = mne.Epochs(
            raw, valid_mi_events, event_id={770: 770, 771: 771, 772: 772, 773: 773},
            tmin=MI_TMIN, tmax=MI_TMAX, baseline=None, preload=True, verbose=False
        )
        mi_data = epochs_mi.get_data()  # (n_samples, 22, 1000) -> 4s*250Hz=1000点
        mi_labels = epochs_mi.events[:, 2]  # 原始标签770/771/772/773

        # 提取静息段数据（-0.5-0.0s，每个MI事件前的静息）
        epochs_rest = mne.Epochs(
            raw, valid_mi_events, event_id={770: 770, 771: 771, 772: 772, 773: 773},
            tmin=REST_TMIN, tmax=REST_TMAX, baseline=None, preload=True, verbose=False
        )
        rest_data = epochs_rest.get_data()  # (n_samples, 22, 125) -> 0.5s*250Hz=125点
        rest_labels = np.zeros(len(rest_data))  # 静息态标签为0

        # 标签映射（770→1,771→2,772→3,773→4）
        mapped_mi_labels = [MI_ACTION_MAP_4CLASS[lbl] for lbl in mi_labels]

        return rest_data, rest_labels, mi_data, mapped_mi_labels

    def load_data(self):
        all_train_data = []
        all_train_labels = []
        all_train_subjs = []

        # 遍历1-9被试（A01-A09）
        for s in range(1, 10):
            if s == self.test_subj_id: continue

            # 训练集：A0{s}T.gdf
            train_file = os.path.join(self.data_dir, f'A0{s}T.gdf')
            if not os.path.exists(train_file):
                print(f"⚠️ 训练文件不存在: {train_file}")
                continue

            # 加载GDF数据
            rest_data, rest_labels, mi_data, mi_labels = self._load_gdf_data(train_file)
            if rest_data is None: continue

            # ERD过滤无效样本
            good_idx = self._filter_by_erd(rest_data, mi_data, threshold=1.0)
            if not good_idx: continue

            # 筛选有效样本
            rest_clean = rest_data[good_idx]
            rest_labels_clean = rest_labels[good_idx]
            mi_clean = mi_data[good_idx]
            mi_labels_clean = np.array(mi_labels)[good_idx]

            # 拼接静息+MI数据
            X_subj = np.concatenate([rest_clean, mi_clean], axis=0)
            y_subj = np.concatenate([rest_labels_clean, mi_labels_clean], axis=0)
            s_labels = np.full(len(X_subj), s)  # 标记被试ID

            all_train_data.append(X_subj)
            all_train_labels.append(y_subj)
            all_train_subjs.append(s_labels)

        # 拼接训练集
        train_x = np.concatenate(all_train_data, axis=0) if all_train_data else np.array([])
        train_y = np.concatenate(all_train_labels, axis=0).astype(np.int64) if all_train_labels else np.array([])
        train_subj = np.concatenate(all_train_subjs, axis=0) if all_train_subjs else np.array([])

        # 加载测试集（A0{s}E.gdf）
        test_file = os.path.join(self.data_dir, f'A0{self.test_subj_id}E.gdf')
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"❌ 测试文件不存在: {test_file}")

        rest_test, rest_labels_test, mi_test, mi_labels_test = self._load_gdf_data(test_file)
        t_good_idx = self._filter_by_erd(rest_test, mi_test, threshold=1.0)

        test_x = np.concatenate([rest_test[t_good_idx], mi_test[t_good_idx]], axis=0)
        test_y = np.concatenate([rest_labels_test[t_good_idx], np.array(mi_labels_test)[t_good_idx]], axis=0).astype(
            np.int64)

        print(f"✅ 4分类数据加载完成！训练样本: {len(train_x)}, 测试样本: {len(test_x)}")
        return train_x, train_y, train_subj, test_x, test_y