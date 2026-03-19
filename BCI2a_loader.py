# BCI2a_loader.py（新增，4分类数据加载器）
import mne
import numpy as np
import os


def get_bci2a_channel_indices(raw, target_channels=['C3', 'Cz', 'C4', 'Pz']):
    """获取BCI2a目标通道索引"""
    ch_names = raw.info['ch_names']
    indices = []
    for ch in target_channels:
        for idx, name in enumerate(ch_names):
            if ch in name and 'EEG' in name:
                indices.append(idx)
                break
    return indices


def load_bci2a_subject(subj_id, data_path='./data/BCI_IV_2a/'):
    """加载单个被试的BCI2a数据（4分类）"""
    e_file = os.path.join(data_path, f'A0{subj_id}E.gdf')
    t_file = os.path.join(data_path, f'A0{subj_id}T.gdf')

    raw_e = mne.io.read_raw_gdf(e_file, preload=True, verbose=False)
    raw_t = mne.io.read_raw_gdf(t_file, preload=True, verbose=False)
    raw = mne.concatenate_raws([raw_e, raw_t])

    # 选择4个通道
    ch_indices = get_bci2a_channel_indices(raw)
    raw.pick(ch_indices)

    # 提取4类MI事件（1=左手,2=右手,3=双脚,4=舌头）
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    mi_events = events[np.isin(events[:, 2], [1, 2, 3, 4])]

    # 提取0.5-4.5s时窗（1001时间点）
    epochs = mne.Epochs(
        raw, mi_events, tmin=0.5, tmax=4.5, baseline=None, preload=True, verbose=False
    )

    x = epochs.get_data()  # (n_trials, 4, 1001)
    y = np.array([ev - 1 for ev in mi_events[:, 2]])  # 归一化到0-3（4分类）
    subj_indices = np.ones(len(x)) * subj_id

    print(f"被试 {subj_id} 加载完成: 样本数={len(x)}, 通道数={x.shape[1]}, 时间点={x.shape[2]}")
    return x, y, subj_indices


def load_bci2a_cross_subject(test_subj=1, data_path='./data/BCI_IV_2a/'):
    """加载跨被试BCI2a数据（排除测试被试，其余为训练集）"""
    train_x_list = []
    train_y_list = []
    train_subj_list = []

    for subj_id in range(1, 10):
        if subj_id == test_subj:
            continue
        x, y, subj_indices = load_bci2a_subject(subj_id, data_path)
        train_x_list.append(x)
        train_y_list.append(y)
        train_subj_list.append(subj_indices)

    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    train_subj_indices = np.concatenate(train_subj_list, axis=0)

    print(f"\n跨被试数据加载完成: 训练样本={len(train_x)}, 测试被试={test_subj}")
    return train_x, train_y, train_subj_indices