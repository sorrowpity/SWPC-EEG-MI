#check_gap.py
import torch
import numpy as np
from EEG_cross_subject_loader_MI_resting import EEG_loader_resting
from ShallowConvNet import ShallowConvNet
import torch.nn.functional as F

# 1. 加載模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 確保模型參數與訓練時一致 (22通道, 250長度)
model = ShallowConvNet(2, 22, 250, 0.5).to(device)
model.load_state_dict(torch.load('prescreen_model.pth', map_location=device))
model.eval()

# 2. 加載數據
loader = EEG_loader_resting(test_subj=1)
test_x = loader.test_x  # 形狀應該是 (N_samples, 22, 250)
test_y = loader.test_y  # 標籤 0 為 Rest, 1 為 MI

rest_probs = []
mi_probs = []

print(f"數據加載完成，測試集樣本數: {len(test_x)}")
print("開始分析數據特徵差距...")

with torch.no_grad():
    for i in range(len(test_x)):
        sample = test_x[i]  # (22, 250)
        label = test_y[i]

        # 標準化
        s_std = (sample - sample.mean()) / (sample.std() + 1e-6)
        t = torch.from_numpy(s_std).float().unsqueeze(0).unsqueeze(0).to(device)

        # 推斷 MI 概率
        prob = F.softmax(model(t), dim=1)[0, 1].item()

        if label == 0:
            rest_probs.append(prob)
        else:
            mi_probs.append(prob)

# 3. 輸出統計結果
if len(rest_probs) > 0 and len(mi_probs) > 0:
    mean_rest = np.mean(rest_probs)
    mean_mi = np.mean(mi_probs)
    std_rest = np.std(rest_probs)
    std_mi = np.std(mi_probs)

    print("\n" + "=" * 30)
    print(f"靜息段 (Rest) 平均 MI 概率: {mean_rest:.4f} (±{std_rest:.4f})")
    print(f"運動段 (MI)   平均 MI 概率: {mean_mi:.4f} (±{std_mi:.4f})")
    print("-" * 30)

    gap = mean_mi - mean_rest
    print(f"兩者平均差距 (Gap): {gap:.4f}")

    if gap < 0.1:
        print("\n結論: 【警告】兩者幾乎沒區別！")
        print("這意味著模型在判斷時完全是在猜。請檢查數據切片位置。")
    elif gap > 0.2:
        print("\n結論: 【良好】區分度明顯。")
        print("模型性能不錯，只需調整 inference_robot.py 的門檻。")
    else:
        print("\n結論: 【一般】有一定區分度，但容易誤觸。")
    print("=" * 30)
else:
    print("錯誤：未能收集到足夠的分類數據，請檢查標籤分布。")