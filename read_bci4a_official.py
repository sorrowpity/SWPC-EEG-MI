# read_convert_bci4a_ultimate.py（终极兼容版）
import os
import re
import mne
import numpy as np
from scipy.io import savemat

# 配置
GDF_PATH = "./data/BCI_IV_2a/"
MAT_PATH = "./data/"
SUBJECTS = ["A01"]  # 先处理A01，验证成功后再扩展
EEG_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2']
SAMPLE_RATE = 250

# 兼容所有可能的事件标注格式
EVENT_PATTERNS = {
    'stop': [768, '768', 'Id0', 'S  0', 'rest', 'REST'],
    'left_hand': [769, '769', 'Id1', 'S  1', 'left', 'LEFT'],
    'right_hand': [770, '770', 'Id2', 'S  2', 'right', 'RIGHT'],
    'feet': [771, '771', 'Id3', 'S  3', 'feet', 'FEET'],
    'tongue': [772, '772', 'Id4', 'S  4', 'tongue', 'TONGUE']
}


def extract_events_from_annotations(annotations):
    """自动识别所有格式的事件标注"""
    events = []
    event_labels = []

    # 遍历所有标注
    for i, ann in enumerate(annotations):
        desc = str(ann['description']).strip()
        onset = int(ann['onset'] * SAMPLE_RATE)

        # 匹配事件类型
        matched_label = None
        for label, patterns in EVENT_PATTERNS.items():
            # 数字匹配/字符串匹配/正则匹配
            if desc in patterns:
                matched_label = label
                break
            # 提取数字部分匹配（比如"S768" → 768）
            num_match = re.search(r'(\d+)', desc)
            if num_match and int(num_match.group(1)) in patterns:
                matched_label = label
                break

        if matched_label:
            events.append([onset, 0, i])
            event_labels.append(matched_label)

    return np.array(events), event_labels


def convert_gdf_to_mat_4cls(subj_id):
    """终极兼容版：转换为4分类.mat文件"""
    # 忽略无关警告
    mne.set_log_level('ERROR')

    for suffix, mat_prefix in [("T", "train"), ("E", "test")]:
        # 文件路径
        gdf_file = os.path.join(GDF_PATH, f"{subj_id}{suffix}.gdf")
        mat_file = os.path.join(MAT_PATH, f"{mat_prefix}_A{subj_id[2:]}.mat")

        if not os.path.exists(gdf_file):
            print(f"❌ 找不到文件: {gdf_file}")
            continue

        try:
            # 读取GDF文件（兼容所有版本）
            raw = mne.io.read_raw_gdf(
                gdf_file,
                preload=True,
                verbose=False,
                stim_channel=None  # 禁用自动刺激通道识别
            )

            # 只保留22个EEG通道（忽略通道名重复警告）
            try:
                raw.pick_channels(EEG_CHANNELS)
            except:
                # 如果通道名不匹配，取前22个通道
                raw = raw.pick_channels(raw.ch_names[:22])

            # 提取事件（核心：兼容所有标注格式）
            events, event_labels = extract_events_from_annotations(raw.annotations)

            if len(events) == 0:
                print(f"⚠️ {gdf_file} 未提取到任何事件")
                continue

            # 提取EEG数据（静息段+运动段）
            eeg_data = []
            final_labels = []

            for i, (ev, label) in enumerate(zip(events, event_labels)):
                ev_time = ev[0]

                # 提取静息段（0-250ms）
                rest_start = ev_time
                rest_end = rest_start + 250
                if rest_end <= len(raw):
                    rest_data = raw.get_data()[:, rest_start:rest_end]
                    eeg_data.append(rest_data)
                    final_labels.append('stop')

                # 提取运动段（850-1100ms）
                mi_start = ev_time + 850
                mi_end = mi_start + 250
                if mi_end <= len(raw):
                    mi_data = raw.get_data()[:, mi_start:mi_end]
                    eeg_data.append(mi_data)
                    final_labels.append(label)

            # 转为numpy数组
            eeg_data = np.array(eeg_data)
            final_labels = np.array(final_labels)

            # 保存为.mat文件（和现有代码完全兼容）
            savemat(mat_file, {
                'X': eeg_data,
                'y': final_labels
            })

            # 打印成功信息
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            print(f"✅ 成功生成: {mat_file}")
            print(f"   总样本数: {len(eeg_data)}")
            print(f"   标签分布:")
            for lbl, cnt in zip(unique_labels, counts):
                print(f"      {lbl}: {cnt}个样本")
            print("-" * 50)

        except Exception as e:
            print(f"❌ 处理{gdf_file}失败: {str(e)[:100]}")
            continue


# 执行转换（A01）
if __name__ == "__main__":
    convert_gdf_to_mat_4cls("A01")