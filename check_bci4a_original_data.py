# check_bci4a_original_data.py
import os
import mne
import numpy as np

# 你的原始.gdf文件路径
GDF_PATH = "./data/BCI_IV_2a/"
SUBJECTS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

# BCI IV 2a标准事件ID映射
EVENT_MAP = {
    768: 'REST',
    769: 'LEFT',
    770: 'RIGHT',
    771: 'FEET',
    772: 'TONGUE'
}


def check_gdf_file(file_path):
    """检查单个.gdf文件的标签分布"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}

    try:
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        events, event_ids = mne.events_from_annotations(raw, verbose=False)

        # 统计各标签数量
        label_counts = {v: 0 for v in EVENT_MAP.values()}
        for event in events:
            event_id = event[2]
            if event_id in EVENT_MAP:
                label_counts[EVENT_MAP[event_id]] += 1

        return label_counts
    except Exception as e:
        print(f"⚠️ 读取{file_path}失败: {e}")
        return {}


# 检查所有被试的训练/测试文件
print(f"{'=' * 80}")
print(f"{'被试':<5} | {'文件类型':<6} | REST | LEFT | RIGHT | FEET | TONGUE | 总计")
print(f"{'=' * 80}")

total_all = {v: 0 for v in EVENT_MAP.values()}
for subj in SUBJECTS:
    # 检查训练文件（T.gdf）
    train_file = os.path.join(GDF_PATH, f"{subj}T.gdf")
    train_counts = check_gdf_file(train_file)

    # 检查测试文件（E.gdf）
    test_file = os.path.join(GDF_PATH, f"{subj}E.gdf")
    test_counts = check_gdf_file(test_file)

    # 打印单个被试结果
    if train_counts:
        total_train = sum(train_counts.values())
        print(
            f"{subj:<5} | {'训练':<6} | {train_counts['REST']:<4} | {train_counts['LEFT']:<4} | {train_counts['RIGHT']:<5} | {train_counts['FEET']:<4} | {train_counts['TONGUE']:<6} | {total_train}")
    if test_counts:
        total_test = sum(test_counts.values())
        print(
            f"{subj:<5} | {'测试':<6} | {test_counts['REST']:<4} | {test_counts['LEFT']:<4} | {test_counts['RIGHT']:<5} | {test_counts['FEET']:<4} | {test_counts['TONGUE']:<6} | {total_test}")

    # 累计总数
    for key in total_all:
        total_all[key] += train_counts.get(key, 0) + test_counts.get(key, 0)

# 打印总计
total_total = sum(total_all.values())
print(f"{'=' * 80}")
print(
    f"{'总计':<5} | {'-':<6} | {total_all['REST']:<4} | {total_all['LEFT']:<4} | {total_all['RIGHT']:<5} | {total_all['FEET']:<4} | {total_all['TONGUE']:<6} | {total_total}")

# 关键判断
if total_all['FEET'] == 0 or total_all['TONGUE'] == 0:
    print(f"\n❌ 原始.gdf文件中缺少FEET/TONGUE样本！请检查：")
    print(f"   1. 是否下载了完整的BCI IV 2a数据集")
    print(f"   2. .gdf文件是否损坏")
else:
    print(f"\n✅ 原始.gdf文件包含完整4分类数据！问题出在预处理步骤")