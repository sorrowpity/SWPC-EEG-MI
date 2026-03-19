# ShallowConvNet_SSL.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    # 【关键修改：单任务版本参数，仅保留必要参数，避免冲突】
    def __init__(self, num_classes=2, channels=22, time_points=250, dropout_rate=0.5):
        super(ShallowConvNet, self).__init__()

        # Block 1: Temporal + Spatial Filter
        self.conv1_temporal = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.conv1_spatial = nn.Conv2d(40, 40, (channels, 1), groups=40, bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()

        # Block 2: Pooling and Dropout
        self.pool1 = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 动态计算展平维度（确保输入是整数）
        with torch.no_grad():
            # 这里的 channels 和 time_points 已强制为整数，不会报错
            dummy = torch.zeros(1, 1, channels, time_points)
            dummy = self.conv1_temporal(dummy)
            dummy = self.conv1_spatial(dummy)
            dummy = self.pool1(dummy)
            self.flatten_size = dummy.view(1, -1).size(1)

        # 单任务分类头（Rest/MI二分类）
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Channels, Time)
        x = self.conv1_temporal(x)
        x = self.conv1_spatial(x)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x

    def extract_feature(self, x):
        # 提取全连接层之前的特征（用于SSL对比损失）
        x = self.conv1_temporal(x)
        x = self.conv1_spatial(x)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        feature = x.view(-1, self.flatten_size)
        return feature