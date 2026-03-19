import scipy.io as sio
import numpy as np

# ====================== 核心配置 ======================
MAT_FILE_PATH = "F:/software_learn/python/MI/SWPC/data/k6b.mat"

# BCI III 3a 4类标签映射（Classlabel的1-4对应）
LABEL_MAP = {
    1.0: "左手运动想象",
    2.0: "右手运动想象",
    3.0: "双脚运动想象",
    4.0: "舌头运动想象"
}


# ====================== 解析核心逻辑 ======================
def test_4cls_mat(file_path):
    try:
        # 1. 加载文件
        mat_data = sio.loadmat(file_path)
        print("✅ 成功加载.mat文件，文件包含的键：")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  - {key}: {mat_data[key].shape}")

        # 2. 解析EEG数据
        if 's' in mat_data:
            sample_num, ch_num = mat_data['s'].shape
            print(f"\n📈 EEG数据：采样点数={sample_num}，通道数={ch_num}")
            print(f"✅ 通道数验证：{'符合' if ch_num == 60 else '不符合'}BCI III 3a（60导）特征")

        # 3. 解析核心信息（关键：区分TRIG和Classlabel）
        hdr = mat_data['HDR'][0, 0]

        # 3.1 TRIG：试次起始采样点（不是标签！）
        if 'TRIG' in hdr.dtype.fields:
            trig_pos = hdr['TRIG'].flatten()  # 每个试次的起始采样点位置
            print(f"\n📍 TRIG字段：试次起始采样点（共{len(trig_pos)}个），前5个：{trig_pos[:5]}")

        # 3.2 Classlabel：真正的4类标签（含nan，需过滤）
        if 'Classlabel' in hdr.dtype.fields:
            class_labels = hdr['Classlabel'].flatten()
            # 过滤nan值，只保留有效标签（1-4）
            valid_labels = class_labels[~np.isnan(class_labels)]
            print(f"\n📊 Classlabel字段：")
            print(f"   原始总数：{len(class_labels)}（含nan）")
            print(f"   有效标签数：{len(valid_labels)}（过滤nan后）")

            # 统计4类数量
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            print(f"\n🎯 4类标签分布（BCI III 3a标准应为每类60个）：")
            total_valid = 0
            for label in [1.0, 2.0, 3.0, 4.0]:
                if label in unique_labels:
                    count = counts[np.where(unique_labels == label)][0]
                    total_valid += count
                else:
                    count = 0
                print(f"   - {LABEL_MAP[label]}（标签{int(label)}）：{count}个试次")

            # 最终验证
            print(f"\n✅ 总计有效试次数：{total_valid}")
            if len(unique_labels) == 4 and total_valid == 240:
                print("🎉 验证通过！该数据集包含完整的4分类（左/右/脚/舌头），共240个有效试次")
            elif len(unique_labels) == 4:
                print(f"⚠️ 包含4类，但总试次数{total_valid}（标准应为240）")
            else:
                missing = [k for k in LABEL_MAP if k not in unique_labels]
                print(f"❌ 缺失标签：{[int(m) for m in missing]}，仅识别到{len(unique_labels)}类")

        # 3.3 EVENT字段：补充验证事件位置
        if 'EVENT' in hdr.dtype.fields:
            events = hdr['EVENT'][0, 0]
            if 'POS' in events.dtype.fields:
                event_pos = events['POS'][0, 0].flatten()
                print(f"\n📌 EVENT.POS：事件位置（前5个）：{event_pos[:5]}")

    except Exception as e:
        print(f"\n❌ 解析出错：{str(e)}")
        import traceback
        traceback.print_exc()


# ====================== 运行 ======================
if __name__ == '__main__':
    test_4cls_mat(MAT_FILE_PATH)