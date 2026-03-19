import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, channels=25, time_points=1001, dropout_rate=0.5):
        super(ShallowConvNet, self).__init__()

        # --- 必须与训练脚本完全一致的结构 ---
        # Block 1: Temporal
        self.conv1_temporal = nn.Conv2d(1, 40, (1, 25), padding=(0, 12), bias=False)
        # Block 1: Spatial (注意：训练时是 channels=25)
        self.conv1_spatial = nn.Conv2d(40, 40, (channels, 1), groups=40, bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()

        # Block 2: Pooling (必须使用训练时的 0.3 倍窗口逻辑)
        pool_window = int(time_points * 0.3)  # 1001点时为 300
        pool_stride = int(pool_window / 5)  # 1001点时为 60
        self.pool1 = nn.AvgPool2d((1, pool_window), stride=(1, pool_stride))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 动态计算展平维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_points)
            dummy = self.conv1_temporal(dummy)
            dummy = self.conv1_spatial(dummy)
            dummy = self.pool1(dummy)
            self.flatten_size = dummy.view(1, -1).size(1)

        # 分类头
        self.fc = nn.Linear(self.flatten_size, num_classes)

        # 混沌度预测头 (训练脚本里有这个，load_state_dict 时才不会报错)
        self.dyn_head = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.elu1(self.bn1(self.conv1_spatial(self.conv1_temporal(x))))
        x = self.pool1(x)
        x = self.dropout1(x)
        feat = x.view(-1, self.flatten_size)
        cls_out = self.fc(feat)
        dyn_out = self.dyn_head(feat)
        return cls_out, dyn_out  # 返回两个值，与推理脚本中的解包逻辑一致