# ShallowConvNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, channels=22, time_points=250, dropout_rate=0.5):
        super(ShallowConvNet, self).__init__()

        # Block 1: Temporal + Spatial Filter
        self.conv1_temporal = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.conv1_spatial = nn.Conv2d(40, 40, (channels, 1), groups=40, bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()

        # Block 2: Pooling and Dropout
        # (1, 75) is about 0.3s pooling window for 250Hz data
        self.pool1 = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Classifier (动态计算展平维度)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_points)
            dummy = self.conv1_temporal(dummy) # (1, 40, 22, 250)
            dummy = self.conv1_spatial(dummy) # (1, 40, 1, 250)
            dummy = self.pool1(dummy) # (1, 40, 1, ~15)
            self.flatten_size = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(self.flatten_size, num_classes)

        # 辅助任务：Lyapunov 指数预测头 (回归任务)
        self.dyn_head = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.elu1(self.bn1(self.conv1_spatial(self.conv1_temporal(x))))
        x = self.pool1(x)
        x = self.dropout1(x)
        feat = x.view(-1, self.flatten_size)

        cls_out = self.fc(feat)
        dyn_out = self.dyn_head(feat)
        return cls_out, dyn_out

