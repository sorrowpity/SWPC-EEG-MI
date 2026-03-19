import mne
import numpy as np
import os

# 1. 读取单个被试数据（以被试1为例）
data_dir = './data/BCI_IV_2a/'  # 替换为你的数据路径
raw = mne.io.read_raw_gdf(os.path.join(data_dir, 'A09T.gdf'), preload=True, verbose=False)

# 2. 提取原始事件注释与MNE映射
events, event_ids = mne.events_from_annotations(raw, verbose=False)
print("原始事件注释 → MNE映射值：")
for event_str, mapped_val in event_ids.items():
    print(f"  '{event_str}' → {mapped_val}")

# 3. 验证4类MI事件是否完整（原始码770-773）
mi_event_strs = ['770', '771', '772', '773']
mi_present = {s: s in event_ids for s in mi_event_strs}
print("\n4类MI事件存在性：", mi_present)

# 4. 统计各MI事件数量
for s in mi_event_strs:
    if s in event_ids:
        count = len(events[events[:, 2] == event_ids[s]])
        print(f"  原始码{s}（映射值{event_ids[s]}）事件数：{count}")